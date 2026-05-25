"""
NOTEBOOK 2 — PEAR SFT TRAINING
=================================
Reads: pear_sft_final.jsonl (produced by trace generation notebook)
Saves: /kaggle/working/pear_sft_checkpoint  (single save, end of training)

All trace generation logic removed — this notebook only does the SFT.
"""

import os, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

os.environ["WANDB_DISABLED"]       = "true"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"]    = "/kaggle/working/cache"
os.makedirs("/kaggle/working/cache", exist_ok=True)

BASE_MODEL  = "/kaggle/input/models/mahmoodavaram/qwen2-5-7b/transformers/default/1/qwen2.5-7b"
TRACES_PATH = "/kaggle/input/datasets/mahmoodavaram/fpeargrpo"
OUT         = "/kaggle/working"

# PEAR hyperparams (paper §4.2)
GAMMA         = 0.999
LOG_G_MIN     = -10.0
LOG_G_MAX     = 5.0
LOG_DELTA_MIN = -0.08
LOG_DELTA_MAX = 0.3
LR            = 1e-5
EPOCHS        = 1
GRAD_ACCUM    = 4     # effective batch = 4*4 = 16
NEG_LAMBDA    = 0.3   # repulsion coefficient; reduced from original 1.0
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 4

random.seed(42)
torch.manual_seed(42)

# ── Dataset ────────────────────────────────────────────────────────────
class PEARDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tok     = tokenizer
        self.max_len = max_len
        self.items   = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                # Only keep records that have both logprob vectors
                if len(r.get("behavior_logprobs", [])) > 0 and \
                   len(r.get("base_logprobs", [])) > 0:
                    self.items.append(r)
        pos = sum(1 for x in self.items if x["role"] == "positive")
        neg = sum(1 for x in self.items if x["role"] == "negative")
        print(f"PEARDataset: {len(self.items)} records — pos: {pos}, neg: {neg}, ratio: {neg/max(1,pos):.2f}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item       = self.items[idx]
        prompt     = item["prompt"]
        trace      = item["trace"]
        role       = item.get("role", "positive")

        full_enc   = self.tok(prompt + trace, max_length=self.max_len,
                              truncation=True, return_tensors="pt")
        prompt_enc = self.tok(prompt, max_length=self.max_len,
                              truncation=True, return_tensors="pt")

        input_ids  = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"].shape[1]
        trace_len  = len(input_ids) - prompt_len

        if trace_len <= 0:
            return {
                "input_ids":    input_ids,
                "prompt_len":   prompt_len,
                "pear_weights": torch.ones(1),
                "role":         role,
            }

        T     = min(trace_len, len(item["behavior_logprobs"]), len(item["base_logprobs"]))
        b_lps = item["behavior_logprobs"][:T]
        t_lps = item["base_logprobs"][:T]

        # Clip per-token log-ratios (paper §3.7)
        deltas = [
            max(LOG_DELTA_MIN, min(LOG_DELTA_MAX, t_lps[j] - b_lps[j]))
            for j in range(T)
        ]

        # Token-level suffix-based weighting (PEAR §3.4, Algorithm 1 suffix mode)
        # G_t = gamma^(T-1-t) * prod_{j=t+1}^{T-1} Delta_j
        # Backward scan: suffix_log_sum = cumulative sum of log-deltas right of t
        log_gamma      = torch.tensor(GAMMA).log().item()
        pear_weights   = torch.zeros(T)
        suffix_log_sum = 0.0

        for t in reversed(range(T)):
            # Weight for position t uses deltas from positions AFTER t
            log_g = (T - 1 - t) * log_gamma + suffix_log_sum
            log_g = max(LOG_G_MIN, min(LOG_G_MAX, log_g))
            pear_weights[t] = torch.exp(torch.tensor(log_g))
            # Now add delta[t] for positions that come before t
            suffix_log_sum += deltas[t]

        # Negatives use sequence-level weight only (PEAR §3.6)
        if role == "negative":
            seq_w = pear_weights.mean().item()
            pear_weights = torch.full((T,), seq_w)

        return {
            "input_ids":    input_ids,
            "prompt_len":   prompt_len,
            "pear_weights": pear_weights,
            "role":         role,
        }

def pear_collate(batch):
    max_len       = max(x["input_ids"].shape[0] for x in batch)
    input_ids_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn_mask     = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].shape[0]
        input_ids_pad[i, :L] = x["input_ids"]
        attn_mask[i, :L]     = 1
    return {
        "input_ids":      input_ids_pad,
        "attention_mask": attn_mask,
        "items":          batch,
    }

# ── Load model ─────────────────────────────────────────────────────────
print("Loading base model for SFT...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.gradient_checkpointing_enable()
model.train()

# ── Dataloader + optimizer ─────────────────────────────────────────────
dataset    = PEARDataset(f"{TRACES_PATH}/pear_sft_final.jsonl", tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=pear_collate, num_workers=0)

total_steps = len(dataloader) * EPOCHS
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = max(1, int(0.05 * total_steps // GRAD_ACCUM)),
    num_training_steps = max(2, total_steps // GRAD_ACCUM),
)

print(f"Rows: {len(dataset)} | Batches: {len(dataloader)} | "
      f"Optimizer steps: {total_steps // GRAD_ACCUM}")
print("Starting PEAR SFT training...\n")

# ── Training loop ──────────────────────────────────────────────────────
optimizer.zero_grad()
running_loss  = 0.0
n_pos_trained = 0
n_neg_trained = 0

for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to("cuda")
    attn_mask = batch["attention_mask"].to("cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits  = outputs.logits

    batch_loss  = torch.tensor(0.0, device="cuda")
    valid_items = 0

    for i, item in enumerate(batch["items"]):
        prompt_len = item["prompt_len"]
        weights    = item["pear_weights"].to("cuda")
        role       = item["role"]

        t_start = prompt_len
        t_end   = item["input_ids"].shape[0]
        T       = t_end - t_start
        if T <= 0:
            continue

        trace_logits  = logits[i, t_start - 1:t_end - 1, :]
        trace_targets = input_ids[i, t_start:t_end]
        log_probs     = torch.log_softmax(trace_logits, dim=-1)
        token_nll     = -log_probs[range(T), trace_targets[:T]]
        w             = weights[:T].detach()

        if role == "positive":
            item_loss = (w * token_nll).mean()
            n_pos_trained += 1
        else:
            # Repulsive: push model away from negative traces
            # Sequence-level weight (mean of PEAR weights), clamped
            seq_w     = w.mean()
            item_loss = -NEG_LAMBDA * seq_w * token_nll.mean()
            item_loss = torch.clamp(item_loss, -5.0, 5.0)
            n_neg_trained += 1

        batch_loss  += item_loss
        valid_items += 1

    if valid_items == 0:
        continue

    loss = (batch_loss / valid_items) / GRAD_ACCUM
    loss.backward()
    running_loss += loss.item() * GRAD_ACCUM

    if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        update_step = (step + 1) // GRAD_ACCUM
        if update_step % 50 == 0 or update_step == 1:
            avg_loss = running_loss / min(50, update_step)
            print(f"Step {update_step}/{total_steps // GRAD_ACCUM} | "
                  f"loss: {avg_loss:.4f} | pos: {n_pos_trained} | neg: {n_neg_trained}")
            running_loss = 0.0

# ── Save — once, at the end ────────────────────────────────────────────
ckpt = f"{OUT}/pear_sft_checkpoint"
model.save_pretrained(ckpt)
tokenizer.save_pretrained(ckpt)
print(f"\nPEAR SFT complete. Checkpoint saved: {ckpt}")
print("Expected: loss stays positive and stable (0.3-1.5 range)")
print("If loss goes negative: NEG_LAMBDA is still too high, reduce to 0.1")
