"""
NOTEBOOK 2 (FIXED) — PEAR SFT TRAINING
========================================
Fixes from original:
  1. PEAR weight suffix computation was wrong — suffix_sum accumulated in
     wrong direction. Fixed to accumulate log-deltas correctly (backward scan).
  2. Negative trace ratio explosion — was ~1.6:1 neg:pos ratio causing the
     repulsive term to dominate. Now capped at 1:2 neg:pos globally, and
     NEG_LAMBDA reduced from 1.0 → 0.3. Per PEAR paper §3.6, negatives use
     sequence-level weight only (no suffix weighting — that was also wrong).
  3. Loss was diving to -22 because NEG loss was unbounded. Added clamp.
  4. Fixed: generate_traces had temperature=1.0 but do_sample=False when n=1.
  5. MMLU: was only keeping positives (no negatives) which is fine for MMLU
     but the low count (~750) was drowning in neg examples. Fixed assembly ratio.
  6. Final dataset assembly: ensure pos:neg >= 3:1 overall.
"""

import os
import sys
import json
import re
import hashlib
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

os.environ["WANDB_DISABLED"]       = "true"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"]    = "/kaggle/working/cache"
os.makedirs("/kaggle/working/cache", exist_ok=True)

SPLIT_DATA     = "/kaggle/input/datasets/mahmoodavaram/pear-and-grpo"
BASE_MODEL     = "/kaggle/input/models/mahmoodavaram/qwen2-5-7b/transformers/default/1/qwen2.5-7b"
INSTRUCT_MODEL = "/kaggle/input/models/mahmoodavaram/qwen2-5-7b-instruct/transformers/default/1/qwen2.5-7b-instruct"
OUT            = "/kaggle/working"
BATCH_SIZE_GEN = 4   # for trace generation
BATCH_SIZE_SFT = 4   # for training

# ── PEAR hyperparams (from paper §4.2) ────────────────────────────────
GAMMA         = 0.999
LOG_G_MIN     = -10.0
LOG_G_MAX     = 5.0
LOG_DELTA_MIN = -0.08
LOG_DELTA_MAX = 0.3
LR            = 1e-5
EPOCHS        = 1
GRAD_ACCUM    = 4       # effective batch = 4*4 = 16
NEG_LAMBDA    = 0.3     # reduced from 1.0 — prevents repulsive explosion
MAX_SEQ_LEN   = 1024
# Max neg:pos ratio in final training dataset
MAX_NEG_RATIO = 0.33    # at most 1 negative per 3 positives

random.seed(42)
torch.manual_seed(42)

# ── Few-shot templates ─────────────────────────────────────────────────
GSM8K_FEWSHOT = """Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8

"""

MMLU_FEWSHOT_TEMPLATE = "The following are multiple choice questions (with answers) about {subject}.\n\n"
STRATEGYQA_FEWSHOT    = "Answer the following question with Yes or No.\n\n"

letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# ── Load models ────────────────────────────────────────────────────────
print("Loading Instruct model (behavior policy πβ)...")
inst_tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
if inst_tok.pad_token is None:
    inst_tok.pad_token = inst_tok.eos_token
inst_model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
inst_model.eval()

print("Loading Base model (target policy πθ)...")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if base_tok.pad_token is None:
    base_tok.pad_token = base_tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
base_model.eval()
assert inst_tok.vocab_size == base_tok.vocab_size, "Tokenizer vocab mismatch"

# ── Helpers ────────────────────────────────────────────────────────────
def trace_id(prompt, trace):
    return hashlib.md5((prompt + trace).encode()).hexdigest()[:20]

def get_token_logprobs(model, tokenizer, prompt, trace, max_total=2048):
    full_enc   = tokenizer(prompt + trace, return_tensors="pt",
                           max_length=max_total, truncation=True).to(model.device)
    prompt_enc = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=max_total)
    prompt_len = prompt_enc["input_ids"].shape[1]
    with torch.no_grad():
        out    = model(**full_enc)
        logits = out.logits[0]
    lp       = torch.log_softmax(logits, dim=-1)
    tok_ids  = full_enc["input_ids"][0]
    seq_len  = tok_ids.shape[0]
    return [lp[t - 1, tok_ids[t]].item() for t in range(prompt_len, seq_len)]

def generate_traces(prompts, model, tokenizer, max_new_tokens, temperature=0.8, n=1):
    """Generate n traces per prompt. Returns list-of-lists."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=2048, padding=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=0.95,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )
    p_len = inputs["input_ids"].shape[1]
    flat  = [
        tokenizer.decode(outputs[i][p_len:], skip_special_tokens=True).strip()
        for i in range(len(prompts) * n)
    ]
    return [flat[i * n:(i + 1) * n] for i in range(len(prompts))]

def extract_gsm_answer(trace):
    if "####" in trace:
        raw  = trace.split("####")[1].strip().split("\n")[0]
        nums = raw.split()
        return nums[0].replace(",", "").rstrip(".,;") if nums else None
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", trace)
    return nums[-1].replace(",", "") if nums else None

def extract_yesno(trace):
    m = re.search(r"\b(Yes|No)\b", trace, re.IGNORECASE)
    return m.group(1).capitalize() if m else None

def extract_abcd(trace):
    m = re.search(r"Answer:\s*([ABCD])", trace, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"^([ABCD])\b", trace.strip())
    if m: return m.group(1).upper()
    return None

def record_with_logprobs(prompt, trace, gold, source, role, extra=None):
    b_lps = get_token_logprobs(inst_model, inst_tok, prompt, trace)
    t_lps = get_token_logprobs(base_model, base_tok, prompt, trace)
    r = {
        "id":               trace_id(prompt, trace),
        "prompt":           prompt,
        "trace":            trace,
        "gold":             gold,
        "source":           source,
        "role":             role,
        "behavior_logprobs": b_lps,
        "base_logprobs":     t_lps,
    }
    if extra:
        r.update(extra)
    return r

# ═══════════════════════════════════════════════════════════════════════
# TRACE GENERATION
# ═══════════════════════════════════════════════════════════════════════

# ── GSM8K ──────────────────────────────────────────────────────────────
gsm_pear = load_from_disk(f"{SPLIT_DATA}/gsm_pear_source")
print(f"\nProcessing {len(gsm_pear)} GSM8K examples...")

gsm_pos, gsm_neg = [], []
for batch_start in range(0, len(gsm_pear), BATCH_SIZE_GEN):
    batch   = gsm_pear.select(range(batch_start, min(batch_start + BATCH_SIZE_GEN, len(gsm_pear))))
    prompts, golds = [], []
    for row in batch:
        m = re.search(r"####\s*([\d,\.]+)", row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        golds.append(gold)
        prompts.append(GSM8K_FEWSHOT + f"Question: {row['question']}\nAnswer:")

    traces_per = generate_traces(prompts, inst_model, inst_tok, max_new_tokens=400, n=3)

    for prompt, gold, traces in zip(prompts, golds, traces_per):
        if gold is None:
            continue
        got_pos = False
        for trace in traces:
            pred       = extract_gsm_answer(trace)
            is_correct = pred is not None and pred == gold
            if is_correct and not got_pos:
                gsm_pos.append(record_with_logprobs(prompt, trace, gold, "gsm8k", "positive"))
                got_pos = True
            elif not is_correct and len(gsm_neg) < len(gsm_pos) * MAX_NEG_RATIO + 50:
                # Only accumulate negatives up to the ratio cap
                if len(trace.split(".")) >= 3:
                    gsm_neg.append(record_with_logprobs(prompt, trace, gold, "gsm8k", "negative"))

    if batch_start % 200 == 0:
        print(f"  GSM {batch_start}/{len(gsm_pear)} — pos: {len(gsm_pos)}, neg: {len(gsm_neg)}")

print(f"GSM8K — pos: {len(gsm_pos)}, neg: {len(gsm_neg)}")

# ── MMLU ───────────────────────────────────────────────────────────────
mmlu_pear = load_from_disk(f"{SPLIT_DATA}/mmlu_pear_source")
print(f"\nProcessing {len(mmlu_pear)} MMLU examples...")

mmlu_pos = []
for batch_start in range(0, len(mmlu_pear), BATCH_SIZE_GEN):
    batch   = mmlu_pear.select(range(batch_start, min(batch_start + BATCH_SIZE_GEN, len(mmlu_pear))))
    prompts, golds = [], []
    for row in batch:
        gold   = letter_map[row["answer"]]
        subj   = row.get("subject", "general knowledge").replace("_", " ")
        choices = "\n".join([f"{letter_map[i]}) {row['choices'][i]}" for i in range(4)])
        prompt  = (MMLU_FEWSHOT_TEMPLATE.format(subject=subj) +
                   f"Question: {row['question']}\n{choices}\nAnswer:")
        golds.append(gold)
        prompts.append(prompt)

    traces_per = generate_traces(prompts, inst_model, inst_tok, max_new_tokens=200, temperature=0.3, n=3)

    for i, (prompt, gold, traces) in enumerate(zip(prompts, golds, traces_per)):
        got_pos = False
        for trace in traces:
            if extract_abcd(trace) == gold and not got_pos:
                mmlu_pos.append(record_with_logprobs(prompt, trace, gold, "mmlu", "positive",
                                                     extra={"subject": batch[i].get("subject", "")}))
                got_pos = True

    if batch_start % 200 == 0:
        print(f"  MMLU {batch_start}/{len(mmlu_pear)} — pos: {len(mmlu_pos)}")

print(f"MMLU — pos: {len(mmlu_pos)}")

# ── StrategyQA ─────────────────────────────────────────────────────────
sqa_pear = load_from_disk(f"{SPLIT_DATA}/sqa_pear_source")
print(f"\nProcessing {len(sqa_pear)} StrategyQA examples...")

sqa_pos, sqa_neg = [], []
for batch_start in range(0, len(sqa_pear), BATCH_SIZE_GEN):
    batch   = sqa_pear.select(range(batch_start, min(batch_start + BATCH_SIZE_GEN, len(sqa_pear))))
    prompts, golds = [], []
    for row in batch:
        gold   = row["answer_yn"]
        desc   = row.get("description", "") or ""
        term   = row.get("term", "") or ""
        prompt = (STRATEGYQA_FEWSHOT + f"Q: {row['question']}\nContext: {term} — {desc}\nA:")
        golds.append(gold)
        prompts.append(prompt)

    traces_per = generate_traces(prompts, inst_model, inst_tok, max_new_tokens=150, n=3)

    for prompt, gold, traces in zip(prompts, golds, traces_per):
        got_pos = False
        for trace in traces:
            pred       = extract_yesno(trace)
            is_correct = pred == gold
            if is_correct and not got_pos:
                sqa_pos.append(record_with_logprobs(prompt, trace, gold, "strategyqa", "positive"))
                got_pos = True
            elif not is_correct and pred is not None:
                if len(sqa_neg) < len(sqa_pos) * MAX_NEG_RATIO + 20:
                    sqa_neg.append(record_with_logprobs(prompt, trace, gold, "strategyqa", "negative"))

    if batch_start % 200 == 0:
        print(f"  SQA {batch_start}/{len(sqa_pear)} — pos: {len(sqa_pos)}, neg: {len(sqa_neg)}")

print(f"SQA — pos: {len(sqa_pos)}, neg: {len(sqa_neg)}")

# ── OpenMath ───────────────────────────────────────────────────────────
om_pear = load_from_disk(f"{SPLIT_DATA}/openmath_pear_source")
print(f"\nProcessing {len(om_pear)} OpenMath examples...")

om_pos = []
for batch_start in range(0, len(om_pear), BATCH_SIZE_GEN):
    batch   = om_pear.select(range(batch_start, min(batch_start + BATCH_SIZE_GEN, len(om_pear))))
    prompts = [GSM8K_FEWSHOT + f"Question: {r['problem']}\nAnswer:" for r in batch]
    traces_per = generate_traces(prompts, inst_model, inst_tok, max_new_tokens=300, n=1)
    for prompt, traces, row in zip(prompts, traces_per, batch):
        trace = traces[0]
        om_pos.append(record_with_logprobs(prompt, trace, str(row["expected_answer"]), "openmath", "positive"))

print(f"OpenMath — pos: {len(om_pos)}")

# ═══════════════════════════════════════════════════════════════════════
# DATASET ASSEMBLY — enforce 3:1 pos:neg ratio
# ═══════════════════════════════════════════════════════════════════════
print("\nAssembling final dataset...")

all_pos = gsm_pos + mmlu_pos + sqa_pos + om_pos
random.shuffle(all_pos)

# Cap negatives globally at MAX_NEG_RATIO of positives
all_neg_raw = gsm_neg + sqa_neg
random.shuffle(all_neg_raw)
max_neg     = int(len(all_pos) * MAX_NEG_RATIO)
all_neg     = all_neg_raw[:max_neg]

final_dataset = all_pos + all_neg
random.shuffle(final_dataset)

with open(f"{OUT}/pear_sft_final.jsonl", "w") as f:
    for r in final_dataset:
        f.write(json.dumps(r) + "\n")

stats = {
    "total":    len(final_dataset),
    "gsm_pos":  len(gsm_pos),
    "gsm_neg":  len([x for x in all_neg if x["source"] == "gsm8k"]),
    "mmlu_pos": len(mmlu_pos),
    "sqa_pos":  len(sqa_pos),
    "sqa_neg":  len([x for x in all_neg if x["source"] == "strategyqa"]),
    "om_pos":   len(om_pos),
    "neg_ratio": round(len(all_neg) / max(1, len(all_pos)), 3),
}
with open(f"{OUT}/trace_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("Trace generation complete:")
print(json.dumps(stats, indent=2))

# Free GPU memory for SFT
del inst_model, base_model
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════
# PEAR SFT TRAINING
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Starting PEAR SFT Training ===")

class PEARDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tok     = tokenizer
        self.max_len = max_len
        self.items   = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if len(r.get("behavior_logprobs", [])) > 0 and len(r.get("base_logprobs", [])) > 0:
                    self.items.append(r)
        print(f"PEARDataset: {len(self.items)} examples loaded")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item       = self.items[idx]
        prompt     = item["prompt"]
        trace      = item["trace"]
        role       = item.get("role", "positive")

        full_enc   = self.tok(prompt + trace, max_length=self.max_len, truncation=True, return_tensors="pt")
        prompt_enc = self.tok(prompt, max_length=self.max_len, truncation=True, return_tensors="pt")

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

        # Clip per-token log-ratios
        deltas = [
            max(LOG_DELTA_MIN, min(LOG_DELTA_MAX, t_lps[j] - b_lps[j]))
            for j in range(T)
        ]

        # FIXED suffix-based weighting (PEAR §3.4):
        # G_t = gamma^(T-1-t) * prod_{j=t+1}^{T-1} Delta_j
        # Backward scan: suffix_sum tracks cumulative log-sum of deltas from right
        log_gamma = torch.tensor(GAMMA).log().item()
        pear_weights = torch.zeros(T)
        suffix_log_sum = 0.0  # sum of log-deltas for positions > t (grows as we go left)

        for t in reversed(range(T)):
            # weight at t = gamma^(T-1-t) * exp(suffix_log_sum_of_deltas_after_t)
            log_g = (T - 1 - t) * log_gamma + suffix_log_sum
            log_g = max(LOG_G_MIN, min(LOG_G_MAX, log_g))
            pear_weights[t] = torch.exp(torch.tensor(log_g))
            # After computing weight for t, add delta[t] to suffix sum for positions < t
            suffix_log_sum += deltas[t]

        # Negatives: use sequence-level weight only (paper §3.6)
        if role == "negative":
            seq_weight = pear_weights.mean().item()
            pear_weights = torch.full((T,), seq_weight)

        return {
            "input_ids":    input_ids,
            "prompt_len":   prompt_len,
            "pear_weights": pear_weights,
            "role":         role,
        }

def pear_collate(batch):
    max_len      = max(x["input_ids"].shape[0] for x in batch)
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

print("Loading base model for SFT...")
tokenizer_sft = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer_sft.pad_token is None:
    tokenizer_sft.pad_token = tokenizer_sft.eos_token
tokenizer_sft.padding_side = "right"

model_sft = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)
model_sft.gradient_checkpointing_enable()
model_sft.train()

dataset_sft = PEARDataset(f"{OUT}/pear_sft_final.jsonl", tokenizer_sft, MAX_SEQ_LEN)
dataloader  = DataLoader(dataset_sft, batch_size=BATCH_SIZE_SFT, shuffle=True,
                         collate_fn=pear_collate, num_workers=0)

total_steps = len(dataloader) * EPOCHS
optimizer   = torch.optim.AdamW(model_sft.parameters(), lr=LR, weight_decay=0.01)
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = max(1, int(0.05 * total_steps // GRAD_ACCUM)),
    num_training_steps = max(2, total_steps // GRAD_ACCUM),
)

print(f"Rows: {len(dataset_sft)} | Batches: {len(dataloader)} | "
      f"Optimizer steps: {total_steps // GRAD_ACCUM}")

optimizer.zero_grad()
running_loss  = 0.0
n_pos_trained = 0
n_neg_trained = 0

for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to("cuda")
    attn_mask = batch["attention_mask"].to("cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model_sft(input_ids=input_ids, attention_mask=attn_mask)
        logits  = outputs.logits

    batch_loss         = torch.tensor(0.0, device="cuda")
    valid_items        = 0

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
            # Repulsive: gradient ascent on negative traces, sequence-level weight
            seq_w     = w.mean()
            item_loss = -NEG_LAMBDA * seq_w * token_nll.mean()
            # Clamp to prevent explosion (loss should stay between -5 and +5)
            item_loss = torch.clamp(item_loss, -5.0, 5.0)
            n_neg_trained += 1

        batch_loss += item_loss
        valid_items += 1

    if valid_items == 0:
        continue

    loss = (batch_loss / valid_items) / GRAD_ACCUM
    loss.backward()
    running_loss += loss.item() * GRAD_ACCUM

    if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
        torch.nn.utils.clip_grad_norm_(model_sft.parameters(), max_norm=1.0)
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

# ── Save ───────────────────────────────────────────────────────────────
ckpt = f"{OUT}/pear_sft_checkpoint"
model_sft.save_pretrained(ckpt)
tokenizer_sft.save_pretrained(ckpt)
print(f"\nPEAR SFT complete. Checkpoint: {ckpt}")
print("NOTE: Do not eval here — PEAR offline accuracy may dip vs SFT baseline by design.")
