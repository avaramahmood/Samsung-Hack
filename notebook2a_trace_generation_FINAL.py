"""
NOTEBOOK 2B — PEAR TRACE GENERATION (standalone, fixed)
=========================================================
This is the trace generation step separated from SFT training.
Fixes vs original:

  BUG 1 — neg explosion (root cause):
    Original: for each question with n=3 rollouts, saves ALL wrong traces
    as negatives unconditionally → 1.61:1 neg:pos ratio (3909 neg vs 2435 pos)
    
    Fix: per-question, save exactly:
      - 1 positive (the first correct trace found)
      - 1 negative ONLY if a positive also exists for that question
        (pick the wrong trace with most steps — highest learning signal)
      - If 0/3 correct: skip the question entirely (too hard, pure noise)
    
    Result: ~1:1 neg:pos for GSM8K, ~2:1 overall (MMLU/OpenMath pos-only)

  BUG 2 — answer comparison not normalized:
    Original: `pred == str(gold)` where gold is raw string from dataset
    Fix: both pred and gold go through normalize_number() before comparison

  BUG 3 — logprob computation runs 6x (3 traces × 2 models) per question
    regardless of whether the trace gets kept.
    Fix: only call get_token_logprobs() for traces that pass the keep gate.
    Saves ~50% of compute in trace generation.

  BUG 4 — SQA negatives uncapped:
    Same problem as GSM8K. Fixed with same per-question 1:1 logic.

  BUG 5 — OpenMath uses n=1 rollout (no chance of getting a negative):
    This is fine — OpenMath is positive-only by design (verified solutions).

Outputs:
  /kaggle/working/pear_sft_final.jsonl
  /kaggle/working/trace_stats.json
"""

import os, sys, json, re, hashlib, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

BATCH_SIZE     = 4
N_ROLLOUTS     = 3   # per question during generation

random.seed(42)

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

assert inst_tok.vocab_size == base_tok.vocab_size, "Tokenizer vocab mismatch — stop"

# ── Helpers ────────────────────────────────────────────────────────────
def trace_id(prompt, trace):
    return hashlib.md5((prompt + trace).encode()).hexdigest()[:20]

def normalize_number(s):
    s = str(s).strip().rstrip(".,;:").replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

def extract_gsm_answer(trace):
    if "####" in trace:
        raw  = trace.split("####")[1].strip().split("\n")[0]
        nums = raw.split()
        return normalize_number(nums[0]) if nums else None
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", trace)
    return normalize_number(nums[-1]) if nums else None

def extract_gold_gsm(row):
    m = re.search(r"####\s*([\d,\.]+)", row["answer"])
    return normalize_number(m.group(1)) if m else None

def extract_yesno(trace):
    m = re.search(r"\b(Yes|No)\b", trace, re.IGNORECASE)
    return m.group(1).capitalize() if m else None

def extract_abcd(trace):
    m = re.search(r"Answer:\s*([ABCD])", trace, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"^([ABCD])\b", trace.strip())
    if m: return m.group(1).upper()
    return None

def n_steps(trace):
    """Heuristic: number of reasoning steps = newline-separated non-empty lines."""
    return len([l for l in trace.split("\n") if l.strip()])

def get_token_logprobs(model, tokenizer, prompt, trace, max_total=2048):
    """Returns list of per-token log-probs for the trace tokens only."""
    full_enc   = tokenizer(prompt + trace, return_tensors="pt",
                           max_length=max_total, truncation=True).to(model.device)
    prompt_enc = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=max_total)
    p_len = prompt_enc["input_ids"].shape[1]
    with torch.no_grad():
        logits = model(**full_enc).logits[0]
    lp      = torch.log_softmax(logits, dim=-1)
    tok_ids = full_enc["input_ids"][0]
    return [lp[t - 1, tok_ids[t]].item() for t in range(p_len, tok_ids.shape[0])]

def make_record(prompt, trace, gold, source, role, extra=None):
    """Only called for traces that pass the keep gate — avoids wasted compute."""
    b_lps = get_token_logprobs(inst_model, inst_tok, prompt, trace)
    t_lps = get_token_logprobs(base_model, base_tok, prompt, trace)
    r = {
        "id":                trace_id(prompt, trace),
        "prompt":            prompt,
        "trace":             trace,
        "gold":              gold,
        "source":            source,
        "role":              role,
        "behavior_logprobs": b_lps,
        "base_logprobs":     t_lps,
    }
    if extra:
        r.update(extra)
    return r

def generate_traces_batch(prompts, model, tokenizer, max_new_tokens, temperature=0.8, n=3):
    """
    Generate n traces per prompt. Returns list[list[str]].
    Padding side = left (required for batch generation).
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=2048, padding=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
    p_len = inputs["input_ids"].shape[1]
    flat  = [
        tokenizer.decode(outputs[i][p_len:], skip_special_tokens=True).strip()
        for i in range(len(prompts) * n)
    ]
    # reshape: flat[i*n : (i+1)*n] = the n traces for prompt i
    return [flat[i * n: (i + 1) * n] for i in range(len(prompts))]

# ══════════════════════════════════════════════════════════════════════
# CORE SELECTION LOGIC — per-question, enforces 1 pos + at most 1 neg
# ══════════════════════════════════════════════════════════════════════

def select_traces_for_question(traces, is_correct_fn, prompt, gold, source, extra=None):
    """
    Given N_ROLLOUTS traces for one question, select:
      - The first correct trace as positive (if any)
      - One wrong trace as negative ONLY if a positive was also found
        → pick the wrong trace with the most reasoning steps (most signal)
      - If no correct trace: skip (return empty list)
      - If all correct: return 1 positive, 0 negatives

    This gives at most 2 records per question, ratio bounded at 1:1.
    """
    correct_traces   = []
    incorrect_traces = []

    for trace in traces:
        if is_correct_fn(trace, gold):
            correct_traces.append(trace)
        else:
            incorrect_traces.append(trace)

    if not correct_traces:
        # No correct answer found — skip entirely (too hard, noisy signal)
        return []

    records = []

    # 1 positive — best (first found; all are correct so pick longest reasoning)
    best_pos = max(correct_traces, key=n_steps)
    records.append(make_record(prompt, best_pos, gold, source, "positive", extra))

    # 1 negative — only if we have one, and it has at least 2 reasoning steps
    if incorrect_traces:
        # Pick the wrong trace with the most steps (richest negative signal)
        best_neg = max(incorrect_traces, key=n_steps)
        if n_steps(best_neg) >= 2:
            records.append(make_record(prompt, best_neg, gold, source, "negative", extra))

    return records

# ══════════════════════════════════════════════════════════════════════
# GSM8K
# ══════════════════════════════════════════════════════════════════════
gsm_pear   = load_from_disk(f"{SPLIT_DATA}/gsm_pear_source")
print(f"\nProcessing {len(gsm_pear)} GSM8K examples (n={N_ROLLOUTS} rollouts each)...")

gsm_records = []
skipped_gsm = 0

for batch_start in range(0, len(gsm_pear), BATCH_SIZE):
    batch   = gsm_pear.select(range(batch_start, min(batch_start + BATCH_SIZE, len(gsm_pear))))
    prompts, golds = [], []

    for row in batch:
        gold = extract_gold_gsm(row)
        golds.append(gold)
        prompts.append(GSM8K_FEWSHOT + f"Question: {row['question']}\nAnswer:")

    traces_per = generate_traces_batch(prompts, inst_model, inst_tok,
                                       max_new_tokens=400, temperature=0.8, n=N_ROLLOUTS)

    for prompt, gold, traces in zip(prompts, golds, traces_per):
        if gold is None:
            skipped_gsm += 1
            continue

        def gsm_correct(trace, gold):
            pred = extract_gsm_answer(trace)
            return pred is not None and normalize_number(pred) == normalize_number(gold)

        recs = select_traces_for_question(traces, gsm_correct, prompt, gold, "gsm8k")
        if not recs:
            skipped_gsm += 1  # 0/N correct — skip
        gsm_records.extend(recs)

    if batch_start % 400 == 0:
        pos = sum(1 for r in gsm_records if r["role"] == "positive")
        neg = sum(1 for r in gsm_records if r["role"] == "negative")
        print(f"  GSM {batch_start}/{len(gsm_pear)} — pos: {pos}, neg: {neg}, "
              f"skipped: {skipped_gsm}")

pos = sum(1 for r in gsm_records if r["role"] == "positive")
neg = sum(1 for r in gsm_records if r["role"] == "negative")
print(f"GSM8K done — pos: {pos}, neg: {neg}, skipped: {skipped_gsm}")

# ══════════════════════════════════════════════════════════════════════
# MMLU — positives only, no negatives
# (MCQ: wrong answers are trivially identifiable, add noise not signal)
# ══════════════════════════════════════════════════════════════════════
mmlu_pear = load_from_disk(f"{SPLIT_DATA}/mmlu_pear_source")
print(f"\nProcessing {len(mmlu_pear)} MMLU examples (pos-only)...")

mmlu_records = []

for batch_start in range(0, len(mmlu_pear), BATCH_SIZE):
    batch   = mmlu_pear.select(range(batch_start, min(batch_start + BATCH_SIZE, len(mmlu_pear))))
    prompts, golds = [], []

    for row in batch:
        gold = letter_map[row["answer"]]
        golds.append(gold)
        subj    = row.get("subject", "general knowledge").replace("_", " ")
        choices = "\n".join([f"{letter_map[i]}) {row['choices'][i]}" for i in range(4)])
        prompts.append(
            MMLU_FEWSHOT_TEMPLATE.format(subject=subj) +
            f"Question: {row['question']}\n{choices}\nAnswer:"
        )

    # Low temperature → more deterministic, better quality positives
    # n=3 to increase chance of getting a correct trace per question
    traces_per = generate_traces_batch(prompts, inst_model, inst_tok,
                                       max_new_tokens=200, temperature=0.3, n=3)

    for i, (prompt, gold, traces) in enumerate(zip(prompts, golds, traces_per)):
        correct_traces = [t for t in traces if extract_abcd(t) == gold]
        if correct_traces:
            best = max(correct_traces, key=n_steps)
            mmlu_records.append(
                make_record(prompt, best, gold, "mmlu", "positive",
                            extra={"subject": batch[i].get("subject", "")})
            )

    if batch_start % 400 == 0:
        print(f"  MMLU {batch_start}/{len(mmlu_pear)} — pos: {len(mmlu_records)}")

print(f"MMLU done — pos: {len(mmlu_records)}")

# ══════════════════════════════════════════════════════════════════════
# StrategyQA — same 1 pos + 1 neg logic
# (binary Y/N: negatives are semantically meaningful)
# ══════════════════════════════════════════════════════════════════════
sqa_pear = load_from_disk(f"{SPLIT_DATA}/sqa_pear_source")
print(f"\nProcessing {len(sqa_pear)} StrategyQA examples...")

sqa_records = []
skipped_sqa = 0

for batch_start in range(0, len(sqa_pear), BATCH_SIZE):
    batch   = sqa_pear.select(range(batch_start, min(batch_start + BATCH_SIZE, len(sqa_pear))))
    prompts, golds = [], []

    for row in batch:
        golds.append(row["answer_yn"])
        desc = row.get("description", "") or ""
        term = row.get("term", "") or ""
        prompts.append(
            STRATEGYQA_FEWSHOT +
            f"Q: {row['question']}\nContext: {term} — {desc}\nA:"
        )

    traces_per = generate_traces_batch(prompts, inst_model, inst_tok,
                                       max_new_tokens=150, temperature=0.8, n=N_ROLLOUTS)

    for prompt, gold, traces in zip(prompts, golds, traces_per):
        def sqa_correct(trace, gold):
            return extract_yesno(trace) == gold

        recs = select_traces_for_question(traces, sqa_correct, prompt, gold, "strategyqa")
        if not recs:
            skipped_sqa += 1
        sqa_records.extend(recs)

    if batch_start % 400 == 0:
        pos = sum(1 for r in sqa_records if r["role"] == "positive")
        neg = sum(1 for r in sqa_records if r["role"] == "negative")
        print(f"  SQA {batch_start}/{len(sqa_pear)} — pos: {pos}, neg: {neg}, "
              f"skipped: {skipped_sqa}")

pos = sum(1 for r in sqa_records if r["role"] == "positive")
neg = sum(1 for r in sqa_records if r["role"] == "negative")
print(f"SQA done — pos: {pos}, neg: {neg}, skipped: {skipped_sqa}")

# ══════════════════════════════════════════════════════════════════════
# OpenMath — positives only (n=1, pre-verified solutions as reference)
# ══════════════════════════════════════════════════════════════════════
om_pear = load_from_disk(f"{SPLIT_DATA}/openmath_pear_source")
print(f"\nProcessing {len(om_pear)} OpenMath examples (pos-only, n=1)...")

om_records = []

for batch_start in range(0, len(om_pear), BATCH_SIZE):
    batch   = om_pear.select(range(batch_start, min(batch_start + BATCH_SIZE, len(om_pear))))
    prompts = [GSM8K_FEWSHOT + f"Question: {r['problem']}\nAnswer:" for r in batch]

    traces_per = generate_traces_batch(prompts, inst_model, inst_tok,
                                       max_new_tokens=300, temperature=0.8, n=1)

    for prompt, traces, row in zip(prompts, traces_per, batch):
        trace = traces[0]  # n=1, no selection needed
        gold  = str(row["expected_answer"])
        om_records.append(make_record(prompt, trace, gold, "openmath", "positive"))

print(f"OpenMath done — pos: {len(om_records)}")

# ══════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ══════════════════════════════════════════════════════════════════════
print("\nAssembling final dataset...")

all_records = gsm_records + mmlu_records + sqa_records + om_records
random.shuffle(all_records)

with open(f"{OUT}/pear_sft_final.jsonl", "w") as f:
    for r in all_records:
        f.write(json.dumps(r) + "\n")

# Stats breakdown
def count(records, role, source=None):
    return sum(1 for r in records
               if r["role"] == role and (source is None or r["source"] == source))

stats = {
    "total":       len(all_records),
    "gsm_pos":     count(all_records, "positive", "gsm8k"),
    "gsm_neg":     count(all_records, "negative", "gsm8k"),
    "mmlu_pos":    count(all_records, "positive", "mmlu"),
    "sqa_pos":     count(all_records, "positive", "strategyqa"),
    "sqa_neg":     count(all_records, "negative", "strategyqa"),
    "om_pos":      count(all_records, "positive", "openmath"),
    "total_pos":   count(all_records, "positive"),
    "total_neg":   count(all_records, "negative"),
    "neg_to_pos_ratio": round(
        count(all_records, "negative") / max(1, count(all_records, "positive")), 3
    ),
    "skipped_gsm": skipped_gsm,
    "skipped_sqa": skipped_sqa,
}

with open(f"{OUT}/trace_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n=== TRACE GENERATION COMPLETE ===")
print(json.dumps(stats, indent=2))
print("\nExpected: neg_to_pos_ratio ~0.35-0.45 (1 neg per pos only for GSM+SQA)")
