import os
import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import yaml
from tqdm.auto import tqdm


@dataclass
class Config:
    model_name: str
    max_length: int
    train_file: str
    eval_file: str
    num_epochs: int
    eval_every: int
    save_every: int
    output_dir: str
    learning_rate: float
    batch_size: int
    max_grad_norm: float
    seed: int


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Concatenate prompt + answer for supervised learning
                    full_text = data["prompt"] + " " + data["answer"]
                    self.samples.append(full_text)
        
        if not self.samples:
            raise ValueError(f"No data found in {path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels are the same as input_ids for causal LM
        labels = input_ids.clone()
        # Set padding tokens to -100 so they're ignored in loss
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    # Extract only SFT-relevant parameters
    sft_config = {
        "model_name": cfg_dict["model_name"],
        "max_length": cfg_dict["max_prompt_length"] + cfg_dict["max_completion_length"],
        "train_file": cfg_dict["train_file"],
        "eval_file": cfg_dict["eval_file"],
        "num_epochs": 3,  # Default for SFT
        "eval_every": 50,
        "save_every": 100,
        "output_dir": "checkpoints_sft",
        "learning_rate": 2e-5,  # Higher LR for SFT
        "batch_size": cfg_dict["batch_size"],
        "max_grad_norm": cfg_dict["max_grad_norm"],
        "seed": cfg_dict["seed"],
    }
    
    return Config(**sft_config)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    return total_loss / total_samples if total_samples > 0 else 0.0


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
    
    print("Loading datasets...")
    train_dataset = SFTDataset(cfg.train_file, tokenizer, cfg.max_length)
    eval_dataset = SFTDataset(cfg.eval_file, tokenizer, cfg.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    global_step = 0
    progress = tqdm(total=total_steps, desc="SFT Training")
    
    for epoch in range(cfg.num_epochs):
        model.train()
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            progress.update(1)
            progress.set_postfix({"loss": loss.item(), "epoch": epoch + 1})
            
            if global_step % cfg.eval_every == 0:
                eval_loss = evaluate(model, eval_loader, device)
                print(f"\nStep {global_step}: eval loss = {eval_loss:.4f}")
                model.train()
            
            if global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
    
    # Final checkpoint
    ckpt_path = os.path.join(cfg.output_dir, "final")
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"\nSFT training complete. Final model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
