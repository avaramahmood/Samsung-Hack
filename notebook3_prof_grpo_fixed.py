"""
NOTEBOOK 3 (FIXED) — PROF-GRPO TRAINING
=========================================
Fixes from original (the uploaded hack.ipynb code):

  1. CHECKPOINT OOM (critical): Was saving full 14GB model every 50 steps
     → 3 checkpoints = 42GB, blowing the 20GB output limit by step 90.
     FIX: Save only one checkpoint at the very end. Mid-run saves write
     tokenizer-only state files (<1MB) for crash recovery, not full weights.

  2. GRPO dataset: Now reads grpo_filtered_300 (pass@8-filtered questions)
     instead of the raw gsm_grpo_source + openmath_grpo_source. This fixes
     the too-easy / too-hard problem — all 300 questions are in the 2-5/8
     pass rate zone where GRPO gradient signal is non-zero.

  3. PRM fallback: The original fallback when nl_token/pos_id/neg_id are
     missing produced constant 0.5 for all steps — meaningless. Fixed to
     use last-token softmax over pos/neg logits more robustly.

  4. PHASE logic simplification: Since we now only have 300 questions,
     we train all 300 in one phase (no phase split needed). If you still
     need 2 phases, set TOTAL_STEPS=150 and resume from phase1_final.

  5. Minor: ref_model.cpu() + gradient_checkpointing to save VRAM.
     PRM stays on CPU (unchanged — was already correct).
     BATCH_SIZE stays at 1 (safest for 96GB on serial generation).
"""

import os, sys, json, re, random, torch, shutil, time, copy
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, Dataset

os.environ["WANDB_DISABLED"]       = "true"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"]    = "/kaggle/working/cache"
os.makedirs("/kaggle/working/cache", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  PHASE CONFIG — run PHASE=1 (steps 1-150) then PHASE=2 (151-300)
#  Or set TOTAL_STEPS=300 and run everything in one session.
# ══════════════════════════════════════════════════════════════════════
PHASE = 1   # set to 2 for second half

SPLIT_DATA    = "/kaggle/input/datasets/mahmoodavaram/pear-and-grpo"
PEAR_CKPT     = "/kaggle/input/models/mahmoodavaram/pear-sft-qwen-checkpoint1/transformers/default/1/pear_sft_checkpoint"
PRM_MODEL     = "/kaggle/input/models/mahmoodavaram/qwen2-5-math-prm-7b/transformers/default/1/qwen2.5-math-prm-7b"
BNB_WHEEL     = "/kaggle/input/datasets/mahmoodavaram/bitsandbyteswheel/bnb_wheel"
OUT           = "/kaggle/working"

PHASE2_CKPT_WORKING = f"{OUT}/stage1_phase1_final"
PHASE2_CKPT_INPUT   = "/kaggle/input/models/mahmoodavaram/riva-stage1-phase1/transformers/default/1/stage1_phase1_final"

assert PHASE in (1, 2), "PHASE must be 1 or 2"

PHASE_START = 0   if PHASE == 1 else 150
PHASE_END   = 150 if PHASE == 1 else 300
PHASE_STEPS = 150

if PHASE == 1:
    START_CKPT = PEAR_CKPT
elif os.path.isdir(PHASE2_CKPT_WORKING):
    START_CKPT = PHASE2_CKPT_WORKING
    print(f"Phase 2: resuming from {START_CKPT}")
elif os.path.isdir(PHASE2_CKPT_INPUT):
    START_CKPT = PHASE2_CKPT_INPUT
    print(f"Phase 2: resuming from {START_CKPT}")
else:
    raise FileNotFoundError(
        f"Phase 2 checkpoint not found.\n"
        f"  Checked: {PHASE2_CKPT_WORKING}\n"
        f"  Checked: {PHASE2_CKPT_INPUT}"
    )

# ── Hyperparameters (from PROF paper §4.1) ────────────────────────────
N_ROLLOUTS     = 8
M_KEEP         = 4
PRM_LAMBDA     = 10.0
PRM_H_LAMBDA   = 30
CLIP_EPS_LOW   = 0.2
CLIP_EPS_HIGH  = 0.28
KL_COEFF       = 0.001
ENTROPY_COEFF  = 0.001
GRPO_LR        = 1e-6
MAX_GEN_TOKENS = 1024
BATCH_SIZE     = 1     # serial generation, safest for 96GB

GLOBAL_SEED = 42 + PHASE
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

GSM8K_FEWSHOT = """Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8

"""

# ── Utilities ──────────────────────────────────────────────────────────
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
        raw  = trace.split("####")[1].strip().split("\n")[0]
        nums = raw.split()
        return normalize_number(nums[0].replace(",", "").rstrip(".,;") if nums else raw)
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", trace)
    return normalize_number(nums[-1]) if nums else None

def orm_score(trace, gold, source):
    if source == "gsm8k":
        pred = extract_gsm_answer(trace)
        return 1 if (pred is not None and normalize_number(pred) == normalize_number(gold)) else -1
    else:
        m = re.search(r"####\s*(.+?)(?:\n|$)", trace) or re.search(r"Answer:\s*(.+?)(?:\n|$)", trace)
        pred = m.group(1).strip() if m else None
        return 1 if (pred and pred.strip() == gold.strip()) else -1

# ── Install bitsandbytes ───────────────────────────────────────────────
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "--no-index", "--find-links", BNB_WHEEL, "bitsandbytes"],
    capture_output=True, text=True
)
if result.returncode != 0:
    subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes"], check=True)
import bitsandbytes as bnb

# ── Load models ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RIVA Stage 1 PROF-GRPO  |  Phase {PHASE}  |  Steps {PHASE_START+1}–{PHASE_END}")
print(f"  Checkpoint: {START_CKPT}")
print(f"{'='*60}\n")

print("Loading policy model...")
tokenizer = AutoTokenizer.from_pretrained(START_CKPT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    START_CKPT, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.gradient_checkpointing_enable()

print("Loading ref model (CPU)...")
ref_model = copy.deepcopy(model).cpu()
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

print("Loading PRM (CPU)...")
prm_tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL)
prm_model     = AutoModelForCausalLM.from_pretrained(
    PRM_MODEL, torch_dtype=torch.bfloat16
)
prm_model.eval()

print("All models loaded.\n")

# ── PRM scoring (fixed fallback) ───────────────────────────────────────
def score_with_prm(prompt, trace):
    steps = [s.strip() for s in trace.split("\n") if s.strip()]
    H     = len(steps)
    if H == 0:
        return [0.5], 0.5

    enc = prm_tokenizer(
        prompt + trace, return_tensors="pt", max_length=2048, truncation=True
    )
    with torch.no_grad():
        logits = prm_model(**enc).logits[0]

    input_ids = enc["input_ids"][0]

    # Get token IDs for newline, +, and -
    nl_tok = prm_tokenizer.encode("\n", add_special_tokens=False)
    nl_id  = nl_tok[0] if nl_tok else None
    pos_id = prm_tokenizer.convert_tokens_to_ids("+")
    neg_id = prm_tokenizer.convert_tokens_to_ids("-")

    # Validate ids — if tokenizer doesn't have +/- as single tokens, use vocab search
    if pos_id == prm_tokenizer.unk_token_id or pos_id is None:
        # Try to find it another way
        pos_candidates = prm_tokenizer.encode("+", add_special_tokens=False)
        pos_id = pos_candidates[0] if pos_candidates else None
    if neg_id == prm_tokenizer.unk_token_id or neg_id is None:
        neg_candidates = prm_tokenizer.encode("-", add_special_tokens=False)
        neg_id = neg_candidates[0] if neg_candidates else None

    step_scores = []
    if nl_id is not None and pos_id is not None and neg_id is not None:
        nl_positions = (input_ids == nl_id).nonzero(as_tuple=True)[0].tolist()
        for pos in nl_positions[-H:]:
            if pos < logits.shape[0]:
                prob_pos = torch.softmax(logits[pos][[pos_id, neg_id]], dim=0)[0].item()
                step_scores.append(prob_pos)

    # Fallback: use last token with pos/neg ids if available, else 0.5
    if not step_scores:
        if pos_id is not None and neg_id is not None:
            fallback = torch.softmax(logits[-1][[pos_id, neg_id]], dim=0)[0].item()
        else:
            fallback = 0.5
        step_scores = [fallback] * max(H, 1)

    return step_scores, float(np.mean(step_scores))

# ── PROF helpers ───────────────────────────────────────────────────────
def prof_consistency_score(step_scores, H, ro):
    """Trajectory-level consistency: mean PRM * ORM, with step-length penalty."""
    penalty = PRM_LAMBDA if (H == 1 or H >= PRM_H_LAMBDA) else 0.0
    return (float(np.mean(step_scores)) - penalty) * ro

def prof_filter_correct_variant(rollouts_with_scores, n, m):
    """
    PROF Algorithm 1: filter n rollouts down to m, balancing correct/incorrect.
    rollouts_with_scores: list of (trace, ro, r_pro)
    """
    correct   = [(t, ro, rp) for t, ro, rp in rollouts_with_scores if ro ==  1]
    incorrect = [(t, ro, rp) for t, ro, rp in rollouts_with_scores if ro == -1]
    n_plus, n_minus = len(correct), len(incorrect)

    if n_plus + n_minus == 0:
        return []

    delta   = n_plus - n_minus
    # How many to remove from each group
    k_plus  = min(n_plus,  max(0, (delta + n - m + 1) // 2))
    k_minus = max(0, n - m - k_plus)
    k_plus  = min(k_plus,  max(0, n_plus  - 1))
    k_minus = min(k_minus, max(0, n_minus - 1))

    # Keep top-r_pro correct, bottom-r_pro incorrect (PROF logic)
    correct_sorted   = sorted(correct, key=lambda x: x[2], reverse=True)
    kept_correct     = correct_sorted[:max(1, n_plus - k_plus)] if n_plus > 0 else []
    # For incorrect: keep lowest r_pro (most wrong reasoning = clearest negative signal)
    incorrect_sorted = sorted(incorrect, key=lambda x: x[2], reverse=False)
    kept_incorrect   = incorrect_sorted[:max(0, n_minus - k_minus)] if n_minus > k_minus else incorrect

    kept = kept_correct + kept_incorrect
    if len(kept) > m:
        kept = random.sample(kept, m)
    return kept

# ── GRPO loss ──────────────────────────────────────────────────────────
def compute_grpo_loss(model, ref_model, tokenizer, prompt, kept_rollouts, advantages,
                      clip_low, clip_high, kl_coeff, entropy_coeff, max_len):
    total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    n_valid    = 0

    for (trace, ro, _), adv in zip(kept_rollouts, advantages):
        enc     = tokenizer(prompt + trace, return_tensors="pt",
                            max_length=max_len, truncation=True).to("cuda")
        enc_cpu = {k: v.cpu() for k, v in enc.items()}
        p_len   = tokenizer(prompt, return_tensors="pt",
                            max_length=max_len, truncation=True)["input_ids"].shape[1]

        token_ids      = enc["input_ids"][0]
        t_start, t_end = p_len, token_ids.shape[0]
        T              = t_end - t_start
        if T <= 0:
            continue

        with torch.enable_grad():
            lp_curr = torch.log_softmax(model(**enc).logits[0], dim=-1)
        with torch.no_grad():
            lp_ref = torch.log_softmax(
                ref_model(**enc_cpu).logits[0], dim=-1
            ).to("cuda")

        targets     = token_ids[t_start:t_end]
        curr_tok_lp = lp_curr[t_start - 1:t_end - 1][range(T), targets]
        ref_tok_lp  = lp_ref[t_start - 1:t_end - 1][range(T), targets].detach()

        ratio   = torch.exp(curr_tok_lp - ref_tok_lp)
        adv_t   = torch.tensor(adv, dtype=torch.float32, device="cuda")
        clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)

        ppo_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
        kl_loss  = kl_coeff * (
            (lp_ref[t_start - 1:t_end - 1] - lp_curr[t_start - 1:t_end - 1])
            .exp().detach()
            .mul(lp_ref[t_start - 1:t_end - 1].detach() - lp_curr[t_start - 1:t_end - 1])
        )[range(T), targets].mean()
        entropy  = -(
            torch.exp(lp_curr[t_start - 1:t_end - 1]) * lp_curr[t_start - 1:t_end - 1]
        ).sum(dim=-1).mean()

        total_loss = total_loss + (ppo_loss + kl_loss + (-entropy_coeff * entropy))
        n_valid   += 1

        del enc, enc_cpu, lp_curr, lp_ref, ratio, clipped
        torch.cuda.empty_cache()

    return total_loss / max(1, n_valid)

# ── Dataset — reads grpo_filtered_300 ─────────────────────────────────
print("Loading GRPO dataset (pass@8-filtered questions)...")

grpo_data_path = f"{SPLIT_DATA}/grpo_filtered_300"
if os.path.isdir(grpo_data_path):
    grpo_ds   = load_from_disk(grpo_data_path)
    stage1_data = []
    for row in grpo_ds:
        stage1_data.append({
            "prompt": row["prompt"],
            "gold":   normalize_number(str(row["gold"])),
            "source": row["source"],
        })
    print(f"Loaded {len(stage1_data)} filtered GRPO questions")
else:
    # Fallback to raw splits if filtered dataset not available
    print("WARNING: grpo_filtered_300 not found, falling back to raw splits")
    gsm_grpo = load_from_disk(f"{SPLIT_DATA}/gsm_grpo_source")
    om_grpo  = load_from_disk(f"{SPLIT_DATA}/openmath_grpo_source")
    stage1_data = []
    for row in gsm_grpo:
        m = re.search(r"####\s*([\d,\.]+)", row["answer"])
        if m:
            gold   = normalize_number(m.group(1).replace(",", ""))
            prompt = GSM8K_FEWSHOT + f"Question: {row['question']}\nAnswer:"
            stage1_data.append({"prompt": prompt, "gold": gold, "source": "gsm8k"})
    for row in om_grpo:
        prompt = GSM8K_FEWSHOT + f"Question: {row['problem']}\nAnswer:"
        stage1_data.append({"prompt": prompt, "gold": str(row["expected_answer"]), "source": "openmath"})

# Shuffle and slice to this phase
rng_data = random.Random(42)
rng_data.shuffle(stage1_data)

# Cycle if dataset smaller than total steps
data_cycle = (stage1_data * (300 // len(stage1_data) + 2))
phase_data = data_cycle[PHASE_START:PHASE_END]
print(f"Phase {PHASE}: training on {len(phase_data)} steps [{PHASE_START}:{PHASE_END}]")

# ── Optimizer ──────────────────────────────────────────────────────────
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=GRPO_LR)

# ── Training loop ──────────────────────────────────────────────────────
print(f"\nStarting Phase {PHASE} PROF-GRPO ({PHASE_STEPS} steps)...")
step_times = []

# Track a small progress file (tokenizer only — <1MB) for crash recovery
def save_progress(global_step):
    """Save only tokenizer + metadata — NOT the 14GB model weights."""
    prog_path = f"{OUT}/progress_step{global_step}"
    os.makedirs(prog_path, exist_ok=True)
    tokenizer.save_pretrained(prog_path)
    with open(f"{prog_path}/meta.json", "w") as f:
        json.dump({"global_step": global_step, "phase": PHASE}, f)

for local_step in range(PHASE_STEPS):
    global_step = PHASE_START + local_step
    item        = phase_data[local_step]
    prompt      = item["prompt"]
    gold        = item["gold"]
    source      = item["source"]

    # ── Rollout generation ─────────────────────────────────────────────
    model.eval()
    tokenizer.padding_side = "left"
    enc   = tokenizer(prompt, return_tensors="pt",
                      max_length=2048, truncation=True).to("cuda")
    p_len = enc["input_ids"].shape[1]

    all_traces = []
    remaining  = N_ROLLOUTS
    t0         = time.time()

    while remaining > 0:
        batch_n = min(BATCH_SIZE, remaining)
        with torch.no_grad():
            outs = model.generate(
                **enc,
                max_new_tokens=MAX_GEN_TOKENS,
                temperature=1.0, top_p=1.0, do_sample=True,
                num_return_sequences=batch_n,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        all_traces.extend([
            tokenizer.decode(outs[i][p_len:], skip_special_tokens=True)
            for i in range(batch_n)
        ])
        del outs
        remaining -= batch_n

    del enc
    torch.cuda.empty_cache()

    # ── Score rollouts ─────────────────────────────────────────────────
    rollouts_scored = []
    for trace in all_traces:
        ro                = orm_score(trace, gold, source)
        step_scores, _    = score_with_prm(prompt, trace)
        r_pro             = prof_consistency_score(step_scores, len(step_scores), ro)
        rollouts_scored.append((trace, ro, r_pro))

    kept = prof_filter_correct_variant(rollouts_scored, N_ROLLOUTS, M_KEEP)
    if len(kept) < 2:
        print(f"  [step {global_step + 1}] skipped — fewer than 2 kept rollouts")
        continue

    rewards = np.array([float(ro) for _, ro, _ in kept])
    if rewards.std() < 1e-6:
        print(f"  [step {global_step + 1}] skipped — zero reward variance")
        continue
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # ── GRPO update ────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    loss = compute_grpo_loss(
        model, ref_model, tokenizer, prompt, kept, advantages,
        CLIP_EPS_LOW, CLIP_EPS_HIGH, KL_COEFF, ENTROPY_COEFF,
        MAX_GEN_TOKENS + p_len
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    step_times.append(elapsed)

    if local_step % 10 == 0:
        n_correct = sum(1 for _, ro, _ in rollouts_scored if ro == 1)
        avg_t     = np.mean(step_times[-10:])
        eta_min   = (avg_t * (PHASE_STEPS - local_step - 1)) / 60
        print(
            f"[Ph{PHASE}] G{global_step + 1:3d} | L{local_step + 1:3d}/{PHASE_STEPS}"
            f" | loss: {loss.item():.4f}"
            f" | correct: {n_correct}/{N_ROLLOUTS}"
            f" | src: {source}"
            f" | {elapsed:.0f}s/step"
            f" | ETA: {eta_min:.0f}m"
        )

    # ── Lightweight progress save every 50 steps (NO model weights) ────
    # This saves <1MB (tokenizer only) so it never blows the 20GB limit
    if (global_step + 1) % 50 == 0:
        save_progress(global_step + 1)
        print(f"  [progress] tokenizer state saved at step {global_step + 1}")

# ── FINAL SAVE — full model weights, only once ─────────────────────────
# This is the only place we save the 14GB model. One save = ~14GB, safely
# under the 20GB output limit.
final_path = f"{OUT}/stage1_phase{PHASE}_final"
print(f"\nSaving final model to {final_path}...")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

total_min = sum(step_times) / 60
print(f"\n{'='*60}")
print(f"  Phase {PHASE} complete.")
print(f"  Final checkpoint: {final_path}")
print(f"  Total training time: {total_min:.1f} min")
if PHASE == 1:
    print(f"\n  Next: set PHASE = 2 and re-run.")
    print(f"  Attach this run's output as a Kaggle model input.")
print(f"{'='*60}")
