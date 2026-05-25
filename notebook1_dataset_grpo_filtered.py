"""
NOTEBOOK 1 (FIXED) — DATASET PREPARATION + GRPO DIFFICULTY FILTERING
======================================================================
Changes from original:
  - Added pass@8 difficulty filtering for GRPO data:
      keep questions where Qwen-7B-base answers 2–5/8 correctly
      (the Goldilocks zone: not trivially easy, not impossible)
  - Target: exactly 300 filtered GRPO questions saved as grpo_filtered_300
  - PEAR data unchanged (instruct model generates traces in nb2)
  - OpenMath GRPO also pass@k filtered for difficulty
  - All other splits unchanged
"""

import os
import json
import re
import random
import torch
from collections import defaultdict
import datasets
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

datasets.disable_caching()
random.seed(42)
torch.manual_seed(42)

BASE      = "/kaggle/input/datasets/mahmoodavaram/riva-training-datasets"
BASE_MODEL = "/kaggle/input/models/mahmoodavaram/qwen2-5-7b/transformers/default/1/qwen2.5-7b"
OUT       = "/kaggle/working"

# ── Difficulty filter hyperparameters ────────────────────────────────
N_ROLLOUTS       = 8        # rollouts per question for difficulty estimation
MIN_CORRECT      = 2        # minimum correct out of N_ROLLOUTS to keep
MAX_CORRECT      = 5        # maximum correct out of N_ROLLOUTS to keep
TARGET_QUESTIONS = 300      # final GRPO pool size
FILTER_BATCH     = 4        # batch size for difficulty scoring
MAX_GEN_TOKENS   = 512

GSM8K_FEWSHOT = """Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8

"""

# ── Helpers ────────────────────────────────────────────────────────────
def normalize_number(s):
    s = str(s).strip().rstrip(".,;:")
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

def extract_gsm_answer(trace):
    if "####" in trace:
        raw = trace.split("####")[1].strip().split("\n")[0]
        nums = raw.split()
        return normalize_number(nums[0].replace(",", "").rstrip(".,;") if nums else raw)
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", trace)
    return normalize_number(nums[-1]) if nums else None

def extract_gold_from_gsm_row(row):
    m = re.search(r"####\s*([\d,\.]+)", row["answer"])
    return normalize_number(m.group(1).replace(",", "")) if m else None

def extract_openmath_answer(trace, expected):
    # For openmath just do exact string match after normalizing
    if "####" in trace:
        raw = trace.split("####")[1].strip().split("\n")[0].strip()
        return normalize_number(raw) == normalize_number(str(expected))
    # fallback: check if expected answer appears near end of trace
    return str(expected).strip() in trace[-200:]

# ── Load model for difficulty scoring ─────────────────────────────────
print("Loading Qwen-7B-base for difficulty scoring...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Model loaded.\n")

# ── Core: pass@k difficulty filter ─────────────────────────────────────
def score_difficulty_batch(prompts, golds, check_fn, n_rollouts=N_ROLLOUTS, batch_size=FILTER_BATCH):
    """
    For each prompt: generate n_rollouts responses, count correct ones.
    Returns list of (n_correct, traces) per prompt.
    """
    results = [[] for _ in range(len(prompts))]

    for _ in range(n_rollouts):
        for bs in range(0, len(prompts), batch_size):
            batch_prompts = prompts[bs:bs + batch_size]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt",
                truncation=True, max_length=2048,
                padding=True, padding_side="left"
            ).to("cuda")
            p_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_GEN_TOKENS,
                    temperature=1.0, top_p=1.0, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for i, out in enumerate(outs):
                global_i = bs + i
                if global_i >= len(prompts):
                    break
                trace = tokenizer.decode(out[p_len:], skip_special_tokens=True)
                is_correct = check_fn(trace, golds[global_i])
                results[global_i].append(is_correct)

    return [sum(r) for r in results]

# ═══════════════════════════════════════════════════════════════════════
# PEAR SPLITS — unchanged, no filtering needed
# ═══════════════════════════════════════════════════════════════════════
print("=== Loading PEAR source splits (unchanged) ===")
gsm_train = load_from_disk(f"{BASE}/gsm8k/train")
gsm_shuffled = gsm_train.shuffle(seed=42, keep_in_memory=True)
gsm_pear_source = gsm_shuffled.select(range(3000), keep_in_memory=True)
gsm_pear_source.save_to_disk(f"{OUT}/gsm_pear_source")
print(f"GSM PEAR: {len(gsm_pear_source)}")

mmlu_val = load_from_disk(f"{BASE}/mmlu/validation")
letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
by_subj = defaultdict(list)
for i, row in enumerate(mmlu_val):
    by_subj[row["subject"]].append(i)
rng = random.Random(42)
mmlu_pear_indices = []
for subj, indices in by_subj.items():
    mmlu_pear_indices.extend(rng.sample(indices, min(30, len(indices))))
mmlu_pear_source = mmlu_val.select(mmlu_pear_indices, keep_in_memory=True)
mmlu_pear_source.save_to_disk(f"{OUT}/mmlu_pear_source")
print(f"MMLU PEAR: {len(mmlu_pear_source)}")

sqa_train = load_from_disk(f"{BASE}/strategyqa/train")
def normalize_sqa(row):
    ans = row["answer"]
    if isinstance(ans, bool):
        row["answer_yn"] = "Yes" if ans else "No"
    elif isinstance(ans, str):
        row["answer_yn"] = "Yes" if ans.lower() in ["true", "yes", "1"] else "No"
    else:
        row["answer_yn"] = "Yes" if int(ans) == 1 else "No"
    return row
sqa_normalized = sqa_train.map(normalize_sqa, keep_in_memory=True)
sqa_shuffled = sqa_normalized.shuffle(seed=42, keep_in_memory=True)
sqa_shuffled.save_to_disk(f"{OUT}/sqa_pear_source")
print(f"SQA PEAR: {len(sqa_shuffled)}")

openmath_ds = load_from_disk(f"{BASE}/openmath")
openmath_shuffled = openmath_ds.shuffle(seed=42, keep_in_memory=True)
def count_words_om(row):
    sol = row.get("generated_solution", "") or ""
    row["sol_words"] = len(sol.split())
    return row
openmath_with_len = openmath_shuffled.map(count_words_om, num_proc=1, keep_in_memory=True)
om_pear_indices = [i for i, r in enumerate(openmath_with_len) if r["sol_words"] <= 150][:600]
om_pear = openmath_with_len.select(om_pear_indices, keep_in_memory=True)
om_pear.save_to_disk(f"{OUT}/openmath_pear_source")
print(f"OpenMath PEAR: {len(om_pear)}")

# ═══════════════════════════════════════════════════════════════════════
# GRPO FILTERING — pass@8 difficulty filter
# ═══════════════════════════════════════════════════════════════════════
print("\n=== GRPO Difficulty Filtering (pass@8, keep 2-5/8 correct) ===")

# ── GSM8K GRPO candidate pool ─────────────────────────────────────────
gsm_grpo_pool = gsm_shuffled.select(range(3000, len(gsm_shuffled)), keep_in_memory=True)
print(f"GSM GRPO candidate pool: {len(gsm_grpo_pool)} questions")

# We don't need to filter all 4473 — stop once we have 250 good ones
gsm_grpo_prompts = []
gsm_grpo_golds   = []
gsm_grpo_rows    = []

for row in gsm_grpo_pool:
    gold = extract_gold_from_gsm_row(row)
    if gold is None:
        continue
    prompt = GSM8K_FEWSHOT + f"Question: {row['question']}\nAnswer:"
    gsm_grpo_prompts.append(prompt)
    gsm_grpo_golds.append(gold)
    gsm_grpo_rows.append(row)

print(f"Valid GSM GRPO candidates: {len(gsm_grpo_prompts)}")

# Score in chunks — exit early once we collect 250 good questions
gsm_kept = []
CHUNK    = 200   # score 200 at a time to save time

for chunk_start in range(0, len(gsm_grpo_prompts), CHUNK):
    if len(gsm_kept) >= 250:
        break
    chunk_end  = min(chunk_start + CHUNK, len(gsm_grpo_prompts))
    c_prompts  = gsm_grpo_prompts[chunk_start:chunk_end]
    c_golds    = gsm_grpo_golds[chunk_start:chunk_end]
    c_rows     = gsm_grpo_rows[chunk_start:chunk_end]

    print(f"  Scoring GSM chunk {chunk_start}:{chunk_end} ...")

    def gsm_check(trace, gold):
        pred = extract_gsm_answer(trace)
        return pred is not None and pred == gold

    n_corrects = score_difficulty_batch(c_prompts, c_golds, gsm_check)

    for i, n_c in enumerate(n_corrects):
        if MIN_CORRECT <= n_c <= MAX_CORRECT:
            gsm_kept.append({
                "question": c_rows[i]["question"],
                "answer":   c_rows[i]["answer"],
                "prompt":   c_prompts[i],
                "gold":     c_golds[i],
                "pass_count": n_c,
                "source":   "gsm8k"
            })

    print(f"  GSM kept so far: {len(gsm_kept)} (chunk pass rate: "
          f"{sum(MIN_CORRECT <= x <= MAX_CORRECT for x in n_corrects)}/{len(n_corrects)})")

print(f"GSM GRPO filtered: {len(gsm_kept)}")

# ── OpenMath GRPO filtering ────────────────────────────────────────────
om_grpo_candidates = [
    r for r in openmath_with_len
    if 150 < r["sol_words"] <= 400
][:2000]

print(f"\nOpenMath GRPO candidate pool: {len(om_grpo_candidates)}")

om_prompts = []
om_golds   = []
om_rows    = []
for row in om_grpo_candidates:
    prompt = GSM8K_FEWSHOT + f"Question: {row['problem']}\nAnswer:"
    om_prompts.append(prompt)
    om_golds.append(str(row["expected_answer"]))
    om_rows.append(row)

om_kept = []
for chunk_start in range(0, len(om_prompts), CHUNK):
    if len(om_kept) >= 100:   # we only need 50-100 openmath in pool
        break
    chunk_end = min(chunk_start + CHUNK, len(om_prompts))
    c_prompts = om_prompts[chunk_start:chunk_end]
    c_golds   = om_golds[chunk_start:chunk_end]
    c_rows    = om_rows[chunk_start:chunk_end]

    print(f"  Scoring OpenMath chunk {chunk_start}:{chunk_end} ...")

    def om_check(trace, gold):
        return extract_openmath_answer(trace, gold)

    n_corrects = score_difficulty_batch(c_prompts, c_golds, om_check, n_rollouts=N_ROLLOUTS)

    for i, n_c in enumerate(n_corrects):
        # OpenMath harder — accept 1-4/8 range
        if 1 <= n_c <= 4:
            om_kept.append({
                "problem":           c_rows[i]["problem"],
                "expected_answer":   c_rows[i]["expected_answer"],
                "prompt":            c_prompts[i],
                "gold":              c_golds[i],
                "pass_count":        n_c,
                "source":            "openmath"
            })

    print(f"  OpenMath kept so far: {len(om_kept)}")

print(f"OpenMath GRPO filtered: {len(om_kept)}")

# ── Combine and cap to TARGET_QUESTIONS = 300 ─────────────────────────
combined = gsm_kept + om_kept
random.shuffle(combined)
grpo_final = combined[:TARGET_QUESTIONS]

print(f"\nFinal GRPO pool: {len(grpo_final)}")
gsm_count = sum(1 for x in grpo_final if x["source"] == "gsm8k")
om_count  = sum(1 for x in grpo_final if x["source"] == "openmath")
print(f"  GSM8K: {gsm_count}, OpenMath: {om_count}")

pass_hist = defaultdict(int)
for x in grpo_final:
    pass_hist[x["pass_count"]] += 1
print(f"  Pass@8 distribution: {dict(sorted(pass_hist.items()))}")

# Save as HuggingFace dataset
grpo_ds = Dataset.from_list(grpo_final)
grpo_ds.save_to_disk(f"{OUT}/grpo_filtered_300")
print(f"Saved grpo_filtered_300 ({len(grpo_ds)} rows)")

# Also save the separate source splits for openmath (GRPO notebook uses these)
gsm_grpo_ds = Dataset.from_list([x for x in grpo_final if x["source"] == "gsm8k"])
om_grpo_ds  = Dataset.from_list([x for x in grpo_final if x["source"] == "openmath"])
gsm_grpo_ds.save_to_disk(f"{OUT}/gsm_grpo_source")
om_grpo_ds.save_to_disk(f"{OUT}/openmath_grpo_source")

# ── Summary ────────────────────────────────────────────────────────────
summary = {
    "gsm_pear":            len(gsm_pear_source),
    "mmlu_pear":           len(mmlu_pear_source),
    "sqa_pear":            len(sqa_shuffled),
    "om_pear":             len(om_pear),
    "grpo_filtered_total": len(grpo_final),
    "grpo_gsm":            gsm_count,
    "grpo_openmath":       om_count,
    "pass_distribution":   dict(sorted(pass_hist.items())),
}
with open(f"{OUT}/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nNotebook 1 complete. Publish /kaggle/working as dataset: pear-and-grpo")
