# ═══════════════════════════════════════════════════════════════════════
# NEW CELL — GRPO Difficulty Filter (add after existing notebook 1 code)
# ═══════════════════════════════════════════════════════════════════════
# Reads: gsm_grpo_source (4473 rows), openmath_grpo_source (2000 rows)
# Writes: grpo_filtered_300 — 300 questions where base model gets 2-5/8
# Also overwrites gsm_grpo_source + openmath_grpo_source with filtered versions
# so notebook 3 (PROF-GRPO) can read them directly without any path changes.
# ═══════════════════════════════════════════════════════════════════════

import re, random, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, Dataset

random.seed(42)

BASE_MODEL = "/kaggle/input/models/mahmoodavaram/qwen2-5-7b/transformers/default/1/qwen2.5-7b"
OUT        = "/kaggle/working"

# Difficulty filter settings
N_ROLLOUTS  = 8   # rollouts per question
MIN_CORRECT = 2   # keep if base model gets at least this many right
MAX_CORRECT = 5   # keep if base model gets at most this many right
TARGET      = 300 # final pool size
GEN_BATCH   = 4   # generation batch size

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

def extract_openmath_answer(trace, expected):
    expected_norm = normalize_number(str(expected))
    if "####" in trace:
        raw = trace.split("####")[1].strip().split("\n")[0].strip()
        return normalize_number(raw) == expected_norm
    # fallback: check last 300 chars
    return expected_norm in trace[-300:]

# ── Load base model ────────────────────────────────────────────────────
print("Loading Qwen-7B-base for pass@8 difficulty scoring...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)
mdl.eval()
print("Model loaded.\n")

# ── Core: score one batch of prompts, N_ROLLOUTS times each ───────────
def count_correct(prompts, gold_list, check_fn, n=N_ROLLOUTS, batch=GEN_BATCH):
    """
    Returns list of int: how many of the n rollouts were correct per prompt.
    Runs n full passes over the prompt list, each time in mini-batches.
    """
    counts = [0] * len(prompts)
    for _ in range(n):
        for bs in range(0, len(prompts), batch):
            bp = prompts[bs:bs + batch]
            bg = gold_list[bs:bs + batch]
            inputs = tok(bp, return_tensors="pt", truncation=True,
                         max_length=1024, padding=True).to("cuda")
            p_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outs = mdl.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1.0, top_p=1.0, do_sample=True,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            for i, out in enumerate(outs):
                trace = tok.decode(out[p_len:], skip_special_tokens=True)
                if check_fn(trace, bg[i]):
                    counts[bs + i] += 1
            del outs, inputs
            torch.cuda.empty_cache()
    return counts

# ── GSM8K ──────────────────────────────────────────────────────────────
print("=== Filtering GSM8K GRPO pool ===")
gsm_grpo = load_from_disk(f"{OUT}/gsm_grpo_source")
print(f"Input: {len(gsm_grpo)} questions")

gsm_prompts, gsm_golds, gsm_rows = [], [], []
for row in gsm_grpo:
    m = re.search(r"####\s*([\d,\.]+)", row["answer"])
    if not m:
        continue
    gold = normalize_number(m.group(1))
    gsm_golds.append(gold)
    gsm_prompts.append(GSM8K_FEWSHOT + f"Question: {row['question']}\nAnswer:")
    gsm_rows.append(row)

print(f"Valid GSM candidates: {len(gsm_prompts)}")

# Score in chunks — stop early once we have enough
gsm_kept = []
CHUNK = 200

for chunk_start in range(0, len(gsm_prompts), CHUNK):
    if len(gsm_kept) >= 270:  # leave room for openmath
        break
    chunk_end = min(chunk_start + CHUNK, len(gsm_prompts))
    cp = gsm_prompts[chunk_start:chunk_end]
    cg = gsm_golds[chunk_start:chunk_end]
    cr = gsm_rows[chunk_start:chunk_end]

    print(f"  Scoring chunk [{chunk_start}:{chunk_end}]...")
    n_corrects = count_correct(cp, cg, lambda t, g: (
        (pred := extract_gsm_answer(t)) is not None and
        normalize_number(pred) == normalize_number(g)
    ))

    n_kept_this_chunk = 0
    for i, nc in enumerate(n_corrects):
        if MIN_CORRECT <= nc <= MAX_CORRECT:
            gsm_kept.append({**cr[i], "prompt": cp[i], "gold": cg[i],
                             "pass_count": nc, "source": "gsm8k"})
            n_kept_this_chunk += 1

    dist = {}
    for nc in n_corrects:
        dist[nc] = dist.get(nc, 0) + 1
    print(f"  Chunk pass distribution: {dict(sorted(dist.items()))}")
    print(f"  Kept this chunk: {n_kept_this_chunk} | Total GSM kept: {len(gsm_kept)}")

print(f"\nGSM8K filtered: {len(gsm_kept)}")

# ── OpenMath ───────────────────────────────────────────────────────────
print("\n=== Filtering OpenMath GRPO pool ===")
om_grpo = load_from_disk(f"{OUT}/openmath_grpo_source")
print(f"Input: {len(om_grpo)} questions")

om_prompts, om_golds, om_rows = [], [], []
for row in om_grpo:
    gold = str(row["expected_answer"])
    om_prompts.append(GSM8K_FEWSHOT + f"Question: {row['problem']}\nAnswer:")
    om_golds.append(gold)
    om_rows.append(row)

om_kept = []
needed  = TARGET - len(gsm_kept)  # only score as many as we need

for chunk_start in range(0, len(om_prompts), CHUNK):
    if len(om_kept) >= needed:
        break
    chunk_end = min(chunk_start + CHUNK, len(om_prompts))
    cp = om_prompts[chunk_start:chunk_end]
    cg = om_golds[chunk_start:chunk_end]
    cr = om_rows[chunk_start:chunk_end]

    print(f"  Scoring OpenMath chunk [{chunk_start}:{chunk_end}]...")
    n_corrects = count_correct(cp, cg,
                               lambda t, g: extract_openmath_answer(t, g),
                               n=N_ROLLOUTS)

    n_kept_this_chunk = 0
    for i, nc in enumerate(n_corrects):
        if 1 <= nc <= 4:   # slightly wider window for harder math
            om_kept.append({**cr[i], "prompt": cp[i], "gold": cg[i],
                           "pass_count": nc, "source": "openmath"})
            n_kept_this_chunk += 1

    print(f"  Kept this chunk: {n_kept_this_chunk} | Total OM kept: {len(om_kept)}")

print(f"OpenMath filtered: {len(om_kept)}")

# ── Combine, cap, save ─────────────────────────────────────────────────
combined = gsm_kept + om_kept
random.shuffle(combined)
final    = combined[:TARGET]

print(f"\nFinal GRPO pool: {len(final)}")

from collections import defaultdict
src_counts  = defaultdict(int)
pass_dist   = defaultdict(int)
for x in final:
    src_counts[x["source"]] += 1
    pass_dist[x["pass_count"]] += 1
print(f"  Sources: {dict(src_counts)}")
print(f"  Pass@8 distribution: {dict(sorted(pass_dist.items()))}")

# Save as unified dataset
grpo_ds = Dataset.from_list(final)
grpo_ds.save_to_disk(f"{OUT}/grpo_filtered_300")
print(f"\nSaved: {OUT}/grpo_filtered_300 ({len(grpo_ds)} rows)")

# Also overwrite the individual source files so notebook 3
# can read them by their original names without any changes
gsm_final  = [x for x in final if x["source"] == "gsm8k"]
om_final   = [x for x in final if x["source"] == "openmath"]

Dataset.from_list(gsm_final).save_to_disk(f"{OUT}/gsm_grpo_source")
Dataset.from_list(om_final).save_to_disk(f"{OUT}/openmath_grpo_source")
print(f"Overwrote gsm_grpo_source ({len(gsm_final)}) and openmath_grpo_source ({len(om_final)})")

# Free GPU
del mdl
torch.cuda.empty_cache()

# Update summary
import json
with open(f"{OUT}/dataset_summary.json") as f:
    summary = json.load(f)
summary["grpo_filtered_total"] = len(final)
summary["grpo_gsm"]            = len(gsm_final)
summary["grpo_openmath"]       = len(om_final)
summary["pass_distribution"]   = dict(sorted(pass_dist.items()))
with open(f"{OUT}/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDataset summary updated.")
print(json.dumps(summary, indent=2))
print("\nNotebook 1 complete. Publish /kaggle/working as dataset: pear-and-grpo")
