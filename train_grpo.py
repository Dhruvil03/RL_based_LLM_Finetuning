import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import yaml
from tqdm.auto import tqdm

from reward_model import MathRewardModel


@dataclass
class Config:
    model_name: str
    max_prompt_length: int
    max_completion_length: int
    train_file: str
    eval_file: str
    num_train_steps: int
    eval_every: int
    save_every: int
    output_dir: str
    group_size: int
    learning_rate: float
    batch_size: int
    gamma: float
    kl_coeff: float
    max_grad_norm: float
    temperature: float
    top_p: float
    seed: int


class JSONLinesDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"No data found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Force conversion (fix hidden unicode issues)
    cfg_dict["learning_rate"] = float(cfg_dict["learning_rate"])
    cfg_dict["batch_size"] = int(cfg_dict["batch_size"])
    cfg_dict["group_size"] = int(cfg_dict["group_size"])
    cfg_dict["num_train_steps"] = int(cfg_dict["num_train_steps"])

    return Config(**cfg_dict)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_prompts(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # For this project, we just pass through the dicts
    return batch


def generate_completions(
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    cfg: Config,
    device: torch.device,
):
    """Generate group_size completions for each prompt.

    Returns:
        all_texts: List[str] of generated completions
        all_prompts_text: List[str] of prompt texts repeated group_size times
    """
    model.eval()
    all_completions = []
    all_prompts_text = []

    with torch.no_grad():
        for sample in prompts:
            prompt_text = sample["prompt"]
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_prompt_length,
            ).to(device)

            for _ in range(cfg.group_size):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_completion_length,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]

                # Slice off the prompt part to keep only the completion
                completion_ids = output_ids[inputs["input_ids"].shape[-1]:]
                completion_text = tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )
                all_completions.append(completion_text)
                all_prompts_text.append(prompt_text)

    return all_completions, all_prompts_text

def compute_logprobs(
    model,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute log-probability of each completion under the given model.

    Returns:
        Tensor of shape (N,) with log π(completion | prompt), where N is
        len(completions). This version is differentiable if called without
    torch.no_grad() by the caller.
    """
    model.eval()
    logprobs = []

    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length + cfg.max_completion_length,
        ).to(device)

        outputs = model(**enc)
        logits = outputs.logits  # (1, seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Identify completion token positions
        input_ids = enc["input_ids"][0]

        comp_ids = tokenizer(
            completion,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_completion_length,
        )["input_ids"][0]

        comp_len = comp_ids.shape[0]
        seq_len = input_ids.shape[0]

        if comp_len >= seq_len:
            # Fall back: treat all except the first token as completion
            start_idx = 1
        else:
            start_idx = seq_len - comp_len

        token_logprobs = log_probs[0, torch.arange(start_idx, seq_len, device=device), input_ids[start_idx:seq_len]]
        # IMPORTANT: no .item() here — keep it as a tensor so it stays in the graph
        logprobs.append(token_logprobs.sum())

    return torch.stack(logprobs)  # shape: (N,)


def compute_kl_penalty(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
) -> torch.Tensor:
    # Approximate KL as difference in log-probs
    return (logprobs_new - logprobs_old)


def main():
    cfg = load_config("config.yaml")
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.to(device)

    # Clone a reference model for KL penalty (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_ds = JSONLinesDataset(cfg.train_file)
    eval_ds = JSONLinesDataset(cfg.eval_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_prompts,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = cfg.num_train_steps
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    reward_model = MathRewardModel()

    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    progress = tqdm(total=cfg.num_train_steps, desc="Training")

    while global_step < cfg.num_train_steps:
        for batch in train_loader:
            if global_step >= cfg.num_train_steps:
                break

            # 1. Generate group_size completions per prompt
            completions, prompts_text = generate_completions(
                model, tokenizer, batch, cfg, device
            )

            # 2. Compute rewards for each completion
            rewards = []
            for sample in batch:
                for _ in range(cfg.group_size):
                    # Because generate_completions repeats prompts in same order,
                    # we can iterate in the same pattern.
                    idx = len(rewards)
                    completion = completions[idx]
                    r = reward_model.compute_reward(sample, completion)
                    rewards.append(r)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # 3. Compute log-probs under current policy and reference policy
            logprobs_new = compute_logprobs(
                model, tokenizer, prompts_text, completions, cfg, device
            )
            with torch.no_grad():
                logprobs_ref = compute_logprobs(
                    ref_model, tokenizer, prompts_text, completions, cfg, device
                )

            # 4. Group-relative advantages
            # For each prompt, subtract the mean reward of its group
            advantages = []
            for i in range(0, len(rewards), cfg.group_size):
                group_rewards = rewards[i : i + cfg.group_size]
                mean_r = group_rewards.mean()
                group_adv = group_rewards - mean_r
                advantages.append(group_adv)
            advantages = torch.cat(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 5. Policy loss (REINFORCE with group-relative baseline)
            #    plus KL penalty to keep policy close to reference model
            kl = compute_kl_penalty(logprobs_new, logprobs_ref)
            policy_loss = -(advantages * logprobs_new).mean()
            kl_loss = (kl ** 2).mean()  # simple L2 penalty in log-prob space
            loss = policy_loss + cfg.kl_coeff * kl_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optim.step()
            scheduler.step()

            global_step += 1
            progress.update(1)
            progress.set_postfix(
                {
                    "loss": loss.item(),
                    "policy": policy_loss.item(),
                    "kl": kl_loss.item(),
                    "reward_mean": rewards.mean().item(),
                }
            )

            if global_step % cfg.eval_every == 0:
                eval_reward = evaluate(model, tokenizer, eval_ds, cfg, device, reward_model)
                print(f"Step {global_step}: eval mean reward = {eval_reward:.4f}")

            if global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

            if global_step >= cfg.num_train_steps:
                break

    # Final checkpoint
    ckpt_path = os.path.join(cfg.output_dir, "final")
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"Training complete. Final model saved to {ckpt_path}")


def evaluate(model, tokenizer, eval_ds, cfg, device, reward_model):
    model.eval()
    rewards_all = []

    # Use a small subset for quick eval
    num_eval = min(32, len(eval_ds))
    indices = random.sample(range(len(eval_ds)), num_eval)

    with torch.no_grad():
        for idx in indices:
            sample = eval_ds[idx]
            prompt = sample["prompt"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_prompt_length,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_completion_length,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
            completion_ids = output_ids[inputs["input_ids"].shape[-1]:]
            completion_text = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )

            r = reward_model.compute_reward(sample, completion_text)
            rewards_all.append(r)

    if not rewards_all:
        return 0.0
    return sum(rewards_all) / len(rewards_all)


if __name__ == "__main__":
    main()
