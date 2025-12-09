# Reinforcement Fine-Tuning of an LLM with Group-Relative Policy Optimization (GRPO-style)

> ⚠️ Note  
> This project is **not** the official final project code from DeepLearning.AI's
> *Reinforcement Fine-Tuning LLMs with GRPO* course.  
> It is an original, self-contained example that implements a **GRPO-style**
> reinforcement fine-tuning loop on a toy arithmetic task, inspired by the ideas
> from that course.

## Project Overview

This repository demonstrates how to fine-tune a causal language model with a
**group-relative policy gradient** method similar in spirit to GRPO:

- The model is asked to solve small arithmetic problems, e.g.  
  *"What is 2 + 3? Answer with an integer."*
- For each prompt, we:
  1. Sample **multiple completions** (a *group*).
  2. Score each completion with a **reward model** (rule-based here).
  3. Compute **group-relative advantages** by subtracting each completion's
     reward from the **mean reward of the group**.
  4. Update the policy (the LLM) to increase log-probs of higher-reward
     completions and decrease log-probs of lower-reward completions.
  5. Add a **KL penalty** to keep the updated model close to a frozen
     reference model (similar to PPO-style regularization).

This is a compact end-to-end example of **reinforcement fine-tuning (RFT)**
for LLMs.

## Repository Structure

```text
.
├── config.yaml                 # Training & model configuration
├── requirements.txt            # Python dependencies
├── reward_model.py             # Rule-based reward model for arithmetic
├── train_grpo.py               # Main training script (GRPO-style loop)
└── data
    ├── math_tasks_train.jsonl  # Small toy training set
    └── math_tasks_eval.jsonl   # Small toy eval set
```

### `config.yaml`

All important hyperparameters live here:

- **model_name**: Hugging Face model name (e.g. `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`).  
  You can swap this for any causal LM with an auto-regressive interface.
- **max_prompt_length / max_completion_length**: Truncation and generation lengths.
- **train_file / eval_file**: Paths to JSONL datasets with `{"prompt", "answer"}`.
- **num_train_steps**: Number of gradient steps for RL fine-tuning.
- **eval_every / save_every**: How often to run evaluation and save checkpoints.
- **output_dir**: Where checkpoints are written.
- **group_size**: Number of completions sampled *per prompt* in each RL step.
- **learning_rate, batch_size, max_grad_norm**: Standard optimization hyperparameters.
- **gamma**: Discount factor (kept at 1.0 here since tasks are single-step).
- **kl_coeff**: Weight of the KL penalty that keeps the new policy near the
  reference policy.
- **temperature, top_p**: Sampling parameters for generation.
- **seed**: Random seed for reproducibility.

### `requirements.txt`

Basic dependencies:

- `torch` – core deep learning library.
- `transformers` – for loading and running Hugging Face LLMs.
- `datasets` – (optional) if you want to scale the dataset beyond the toy example.
- `accelerate` – (optional) if you extend the script to use multi-GPU or distributed training.
- `tqdm` – nice progress bars.
- `pyyaml` – reads `config.yaml`.

### `data/math_tasks_*.jsonl`

Simple **JSON Lines** files where each line is a JSON object:

```json
{"prompt": "What is 2 + 3? Answer with an integer.", "answer": "5"}
```

- `math_tasks_train.jsonl` – used for RL training.
- `math_tasks_eval.jsonl` – used for quick evaluation (mean reward).

You can replace these with your own tasks (e.g. code debugging, summarization,
style transfer), as long as each sample has `prompt` and `answer`.

### `reward_model.py`

Contains a minimal **rule-based reward model**:

- Class: `MathRewardModel`
- For each completion:
  1. Extract the **last integer** appearing in the generated text.
  2. Extract the integer from the `answer` field of the sample.
  3. If they match → reward = `correct_reward` (default 1.0).  
     If not / missing → reward = `wrong_reward` (default 0.0).

This mimics having an **automatic grader**: the model gets higher reward
when it outputs the correct integer.

You can swap this for:
- A learned **reward model** (e.g. another transformer that scores answers).
- Heuristic or programmatic rewards (e.g. unit tests for code).

### `train_grpo.py`

This is the main training script and the core of the project.

#### 1. Config & Setup

- Reads `config.yaml` into a `Config` dataclass.
- Sets random seeds for reproducibility.
- Loads:
  - **Policy model**: `AutoModelForCausalLM` from `model_name`.
  - **Tokenizer**: `AutoTokenizer`.
  - **Reference model**: a *frozen* clone of the initial policy, used to
    compute KL penalties.

#### 2. Data Loading

- `JSONLinesDataset` reads a JSONL file and exposes a list of samples.
- `train_loader` is a `DataLoader` that yields batches of samples.  
  Each sample is a dict like:
  ```python
  {"prompt": "...", "answer": "..."}
  ```

#### 3. Generation: `generate_completions(...)`

For each batch:

- For every sample `prompt`:
  - Generate `group_size` completions with `model.generate(...)`.
  - Store:
    - `completions`: list of generated strings.
    - `prompts_text`: list of prompts of the same length (each prompt
      repeated `group_size` times).

This implements the **"group"** in Group-Relative Policy Optimization.

#### 4. Log-Probabilities: `compute_logprobs(...)`

For each `(prompt, completion)` pair:

1. Concatenate `full_text = prompt + completion`.
2. Tokenize and run the model to get `logits`.
3. Apply `log_softmax` over the vocabulary to get token-level log-probs.
4. Identify which tokens belong to the **completion** (not the prompt).
5. Sum their log-probs to get **log π(completion | prompt)**.

This is done for:
- The **current policy model** (`logprobs_new`).
- The **frozen reference model** (`logprobs_ref`).

#### 5. Rewards & Advantages

- Rewards are computed via `MathRewardModel.compute_reward(sample, completion)`.
- To make them **group-relative**:
  - For each contiguous block of `group_size` completions (same prompt),
    compute the mean reward.
  - Subtract this mean from each reward in the group to form **advantages**:
    ```python
    group_adv = group_rewards - mean_r
    ```
  - All advantages are then normalized (zero mean, unit variance) to
    stabilize training.

Intuition:
- Within a group, completions compete **relative to each other**, so the
  model learns which answers are better, even if absolute rewards are noisy.

#### 6. Loss Function (GRPO-style)

The script uses a simple **REINFORCE-style** loss with a KL penalty:

- **Policy loss**:
  ```python
  policy_loss = -(advantages * logprobs_new).mean()
  ```

- **KL term** (approximate):
  ```python
  kl = logprobs_new - logprobs_ref
  kl_loss = (kl ** 2).mean()
  loss = policy_loss + cfg.kl_coeff * kl_loss
  ```

So the final loss encourages:
- Higher log-prob for **high-advantage** completions.
- Lower log-prob for **low-advantage** completions.
- A small KL divergence from the reference model (to avoid the policy
  drifting too far or collapsing).

In a more advanced implementation, you could:
- Add PPO-style **ratio clipping**.
- Use token-level KL rather than sequence-level.
- Use value functions / baselines learned with a critic network.

#### 7. Optimization Loop

- Standard `AdamW` optimizer + cosine LR scheduler.
- Gradient clipping (`max_grad_norm`) for stability.
- Every `eval_every` steps:
  - Call `evaluate(...)`:
    - Sample a handful of eval prompts.
    - Generate one completion per prompt.
    - Compute mean reward.
- Every `save_every` steps:
  - Save a checkpoint (model + tokenizer) to `output_dir`.

At the end, a final checkpoint is saved to `output_dir/final`.

## How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Edit `config.yaml`**

   - Change `model_name` if desired.
   - Adjust `num_train_steps`, `group_size`, `learning_rate`, etc.

3. **Run training**

   ```bash
   python train_grpo.py
   ```

   > Note: For real models (billions of parameters), you will need a GPU
   > with sufficient VRAM. For quick experimentation, choose a small model
   > on Hugging Face.

4. **Use the fine-tuned model**

   After training, the fine-tuned model is saved under `checkpoints/final`:

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("checkpoints/final")
   tokenizer = AutoTokenizer.from_pretrained("checkpoints/final")

   prompt = "What is 12 - 7? Answer with an integer."
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=16)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

## Extending This Project

Here are some ideas to turn this into a more serious final project:

- **New task**: use code generation, reasoning chain-of-thought, or
  summarization instead of arithmetic.
- **LLM-as-a-judge reward**: replace `MathRewardModel` with a reward model
  powered by another LLM (e.g. scoring style or helpfulness).
- **Better GRPO implementation**:
  - Implement true PPO-style ratio clipping between old and new policies.
  - Use per-token rewards and mask out non-relevant tokens.
- **Logging & visualization**:
  - Log rewards, KL, losses to TensorBoard or Weights & Biases.
  - Track how the distribution of answers changes over training.

This codebase is intentionally compact so you can fully understand and
modify each component, mirroring the spirit of the DeepLearning.AI GRPO
course while remaining completely original.
