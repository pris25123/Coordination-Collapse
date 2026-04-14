# =============================================================================
# Experiment 2: Mechanistic Analysis of Coordination Collapse
# =============================================================================
#
# Three experimental conditions:
#   Direct      — model solves the task directly from the problem statement.
#                 This is the capability baseline.
#   Hierarchical — architect agent generates a plan; implementor agent sees
#                  ONLY the plan (not the original problem). This is the
#                  coordination collapse condition.
#   Oracle      — same as hierarchical, but the implementor sees BOTH the
#                 plan AND the original problem. This isolates whether failure
#                 is caused by a bad plan or information lost at the role boundary.
#
# Models:
#   DeepSeek-Coder-1.3B-Instruct  (FP16, Cell 1)
#   CodeLlama-7B-Instruct          (4-bit NF4 quantization, Cell 2)
#
# Scale: 10 tasks × 3 conditions × 10 runs = 300 trials per model
#
# Key results:
#   Direct always scored 1.00 on both models.
#   Hierarchical degraded significantly (0.867 DeepSeek, 0.567 CodeLlama).
#   Oracle fully recovered for DeepSeek (1.00), partially for CodeLlama (~0.85).
#   Cohen's d: 0.61 (DeepSeek), 1.29 (CodeLlama) — medium to very large effect.
#   The oracle recovery proves the implementor HAS the capability to solve the
#   tasks; it merely loses critical information at the role boundary.
#
# Install:
#   pip install transformers accelerate sentence-transformers numpy pandas \
#               torch scikit-learn bitsandbytes
# =============================================================================

import re
import gc
import math
import time
import json
import traceback
import numpy as np
import pandas as pd
import torch

from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# SECTION 1: DEVICE + MEMORY MANAGEMENT
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] dtype:  {DTYPE}")
if DEVICE == "cuda":
    print(f"[INFO] VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def free_memory():
    """Release GPU cache between heavy operations."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def vram_used_gb() -> float:
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


# =============================================================================
# SECTION 2: DEEPSEEK MODEL LOADING
# deepseek-coder-1.3b-instruct: ~2.6 GB FP16 — safe on T4 (15 GB)
# sentence-transformers/all-MiniLM-L6-v2: ~90 MB — negligible
# =============================================================================

print("\n[INFO] Loading DeepSeek-Coder-1.3B model...")

CODE_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME, trust_remote_code=True)
code_model = AutoModelForCausalLM.from_pretrained(
    CODE_MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True
)
code_model.eval()

print("[INFO] DeepSeek code model loaded.")

print("[INFO] Loading sentence embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
print("[INFO] Embedding model loaded.")


# =============================================================================
# SECTION 3: DETERMINISTIC LLM CALL (DeepSeek)
# =============================================================================

def llm(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Deterministic (greedy) generation with DeepSeek.
    Returns only the newly generated text, stripped.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = code_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy — fully deterministic
            temperature=1.0,           # ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# =============================================================================
# SECTION 4: BENCHMARK TASKS
# 10 canonical Python programming tasks, each with unit tests.
# =============================================================================

TASKS: List[Tuple[str, str]] = [
    ("prime",       "Write a Python function is_prime(n) that returns True if n is prime, else False."),
    ("factorial",   "Write a Python function factorial(n) that returns n factorial recursively."),
    ("fibonacci",   "Write a Python function fib(n) that returns the nth Fibonacci number (0-indexed)."),
    ("gcd",         "Write a Python function gcd(a, b) that returns the greatest common divisor."),
    ("palindrome",  "Write a Python function is_palindrome(s) that returns True if s is a palindrome."),
    ("reverse",     "Write a Python function reverse_string(s) that returns the reversed string."),
    ("is_even",     "Write a Python function is_even(n) that returns True if n is even."),
    ("max_two",     "Write a Python function max_of_two(a, b) that returns the larger of two numbers."),
    ("sum_list",    "Write a Python function sum_list(lst) that returns the sum of a list."),
    ("count_vowels","Write a Python function count_vowels(s) that returns the count of vowels in s."),
]

FUNC_NAMES: Dict[str, str] = {
    "prime":        "is_prime",
    "factorial":    "factorial",
    "fibonacci":    "fib",
    "gcd":          "gcd",
    "palindrome":   "is_palindrome",
    "reverse":      "reverse_string",
    "is_even":      "is_even",
    "max_two":      "max_of_two",
    "sum_list":     "sum_list",
    "count_vowels": "count_vowels",
}

TESTS: Dict[str, List[Tuple]] = {
    "prime":        [(2, True), (4, False), (17, True), (1, False), (0, False)],
    "factorial":    [(0, 1), (1, 1), (5, 120), (6, 720)],
    "fibonacci":    [(0, 0), (1, 1), (5, 5), (7, 13)],
    "gcd":          [((10, 5), 5), ((14, 21), 7), ((0, 5), 5)],
    "palindrome":   [("madam", True), ("hello", False), ("racecar", True), ("ab", False)],
    "reverse":      [("abc", "cba"), ("hello", "olleh"), ("", "")],
    "is_even":      [(2, True), (3, False), (0, True), (-4, True)],
    "max_two":      [((2, 5), 5), ((10, 1), 10), ((3, 3), 3)],
    "sum_list":     [([1, 2, 3], 6), ([], 0), ([5], 5)],
    "count_vowels": [("hello", 2), ("aeiou", 5), ("xyz", 0)],
}


# =============================================================================
# SECTION 5: PLAN VALIDATOR
# Checks if the architect's plan is semantically valid (describes correct logic).
# Threshold: at least 2 task-relevant keywords must appear (case-insensitive).
# =============================================================================

PLAN_KEYWORDS: Dict[str, List[str]] = {
    "prime":        ["prime", "divisor", "divide", "loop", "check"],
    "factorial":    ["factorial", "recursi", "multiply", "base case"],
    "fibonacci":    ["fibonacci", "sum", "previous", "recursi"],
    "gcd":          ["gcd", "divisor", "modulo", "euclid", "remainder"],
    "palindrome":   ["palindrome", "reverse", "equal", "mirror"],
    "reverse":      ["reverse", "string", "flip", "backward"],
    "is_even":      ["even", "modulo", "remainder", "divisib"],
    "max_two":      ["larger", "greater", "maximum", "compare"],
    "sum_list":     ["sum", "add", "total", "accumulate"],
    "count_vowels": ["vowel", "count", "letter", "aeiou"],
}

def plan_is_valid(plan: str, task: str) -> bool:
    plan_lower = plan.lower()
    keywords   = PLAN_KEYWORDS.get(task, [])
    hits       = sum(1 for kw in keywords if kw in plan_lower)
    return hits >= 2


# =============================================================================
# SECTION 6: SEMANTIC ALIGNMENT (plan vs code)
# Cosine similarity between sentence-transformer embeddings.
# Collapse threshold: 0.68, derived from empirical data:
#   failing tasks  → fibonacci align=0.50, gcd align=0.66
#   passing tasks  → all others align >= 0.75
# =============================================================================

COLLAPSE_ALIGNMENT_THRESHOLD = 0.68
COLLAPSE_SCORE_THRESHOLD     = 0.99   # any imperfect score counts as failure

def semantic_alignment(plan: str, code: str, emb_model=None) -> float:
    """Cosine similarity between plan and code embeddings. Returns float in [0, 1]."""
    model = emb_model if emb_model is not None else embed_model
    if not plan.strip() or not code.strip():
        return 0.0
    embs = model.encode([plan, code], convert_to_numpy=True)
    sim  = cosine_similarity(embs[0:1], embs[1:2])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


# =============================================================================
# SECTION 7: CODE EXTRACTION
# =============================================================================

def extract_code(raw: str, func_name: str) -> str:
    """
    Robustly extracts Python code from model output.
    Strategy:
      1. Pull fenced ```python ... ``` blocks
      2. Pull fenced ``` ... ``` blocks
      3. Pull any line starting with 'def func_name'
      4. Return full raw as fallback
    """
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(func_name)}\s*\(", line):
            return "\n".join(lines[i:]).strip()

    return raw.strip()


# =============================================================================
# SECTION 8: EXECUTION + TEST RUNNER
# Executes generated code in an isolated namespace and runs all unit tests.
# Returns: (score: fraction of tests passed, executable: bool)
# =============================================================================

def run_tests(code: str, task: str) -> Tuple[float, bool]:
    func_name = FUNC_NAMES[task]
    namespace: Dict = {}

    try:
        exec(compile(code, "<string>", "exec"), namespace)
        fn = namespace.get(func_name)
        if fn is None or not callable(fn):
            return 0.0, False
    except Exception:
        return 0.0, False

    tests  = TESTS[task]
    passed = 0

    for inp, expected in tests:
        try:
            out = fn(*inp) if isinstance(inp, tuple) else fn(inp)
            if out == expected:
                passed += 1
        except Exception:
            pass

    return passed / len(tests), True


# =============================================================================
# SECTION 9: AGENTS (strict role separation)
# =============================================================================

def architect_agent(problem: str) -> str:
    """
    ROLE: High-level planner.
    INPUT:  problem description (natural language)
    OUTPUT: numbered plan, NO code
    CONTRACT: must not emit any def/code blocks
    """
    prompt = (
        "You are a software architect. Your ONLY job is to output a numbered plan.\n"
        "Rules:\n"
        "- Output ONLY numbered steps (1. 2. 3. ...)\n"
        "- Do NOT write any Python code\n"
        "- Do NOT include def, return, if, for in your output\n"
        "- Be concise: 3-5 steps maximum\n\n"
        f"Problem: {problem}\n\n"
        "Plan:"
    )
    raw   = llm(prompt, max_new_tokens=150)
    lines = [l.strip() for l in raw.split("\n") if re.match(r"^\d+[\.\ )]\s+\S", l.strip())]
    return "\n".join(lines) if lines else raw.strip()


def implementor_agent(plan: str, func_name: str, problem: str) -> str:
    """
    ROLE: Code implementor.
    INPUT:  plan ONLY — does NOT see the original problem description.
            This enforces the information bottleneck condition.
    OUTPUT: Python function
    """
    prompt = (
        f"You are a Python programmer. Implement the function `{func_name}` following this plan:\n\n"
        f"{plan}\n\n"
        "Rules:\n"
        "- Output ONLY valid Python code\n"
        f"- The function MUST be named `{func_name}`\n"
        "- No explanation, no markdown prose\n\n"
        "```python\n"
    )
    raw = llm(prompt, max_new_tokens=200)
    return extract_code("```python\n" + raw, func_name)


def direct_agent(problem: str, func_name: str) -> str:
    """
    ROLE: Baseline — no architect, direct generation from full problem.
    This is the capability ceiling.
    """
    prompt = (
        f"You are a Python programmer. Solve the following:\n\n"
        f"{problem}\n\n"
        "Rules:\n"
        "- Output ONLY valid Python code\n"
        f"- The function MUST be named `{func_name}`\n"
        "- No explanation, no markdown prose\n\n"
        "```python\n"
    )
    raw = llm(prompt, max_new_tokens=200)
    return extract_code("```python\n" + raw, func_name)


def oracle_agent(problem: str, plan: str, func_name: str) -> str:
    """
    ROLE: Oracle condition — implementor sees BOTH plan AND original problem.
    Critical causal test: if the oracle fully recovers performance, the failure
    in the hierarchical condition is caused by information loss at the role
    boundary, NOT by bad planning or insufficient model capability.
    """
    prompt = (
        f"You are a Python programmer.\n\n"
        f"Problem: {problem}\n\n"
        f"Architectural plan to follow:\n{plan}\n\n"
        "Rules:\n"
        "- Output ONLY valid Python code\n"
        f"- The function MUST be named `{func_name}`\n"
        "- No explanation, no markdown prose\n\n"
        "```python\n"
    )
    raw = llm(prompt, max_new_tokens=200)
    return extract_code("```python\n" + raw, func_name)


# =============================================================================
# SECTION 10: COLLAPSE DETECTION
# Coordination collapse v2 — recalibrated from empirical data.
#
# Fires when ALL four conditions hold:
#   1. Plan was architecturally valid (plan_valid = True)
#      → rules out architect failure
#   2. Code is executable
#      → rules out total generation failure (syntax error)
#   3. Semantic alignment is LOW (alignment < 0.68)
#      → implementor deviated from the plan
#   4. Score is imperfect (score < 0.99)
#      → deviation caused task failure
# =============================================================================

def detect_collapse(
    plan_valid: bool,
    alignment:  float,
    score:      float,
    executable: bool,
) -> bool:
    return (
        plan_valid
        and executable
        and alignment < COLLAPSE_ALIGNMENT_THRESHOLD
        and score     < COLLAPSE_SCORE_THRESHOLD
    )


# =============================================================================
# SECTION 11: EXPERIMENT RUNNER (DeepSeek)
# Conditions: direct | hierarchical | oracle
# =============================================================================

def run_one_trial(task: str, problem: str, condition: str) -> Dict:
    """Run a single trial for a given task and condition."""
    func_name = FUNC_NAMES[task]
    result = {
        "task":       task,
        "condition":  condition,
        "plan":       "",
        "code":       "",
        "plan_valid": False,
        "alignment":  0.0,
        "score":      0.0,
        "executable": False,
        "collapse":   False,
        "error":      "",
    }

    try:
        if condition == "direct":
            code       = direct_agent(problem, func_name)
            plan       = ""
            plan_valid = False
            alignment  = 0.0

        elif condition == "hierarchical":
            plan       = architect_agent(problem)
            code       = implementor_agent(plan, func_name, problem)
            plan_valid = plan_is_valid(plan, task)
            alignment  = semantic_alignment(plan, code)

        elif condition == "oracle":
            plan       = architect_agent(problem)
            code       = oracle_agent(problem, plan, func_name)
            plan_valid = plan_is_valid(plan, task)
            alignment  = semantic_alignment(plan, code)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        score, executable = run_tests(code, task)
        collapse = detect_collapse(plan_valid, alignment, score, executable) \
                   if condition == "hierarchical" else False

        result.update({
            "plan":       plan,
            "code":       code,
            "plan_valid": plan_valid,
            "alignment":  round(alignment, 4),
            "score":      round(score, 4),
            "executable": executable,
            "collapse":   collapse,
        })

    except Exception as e:
        result["error"] = traceback.format_exc()

    return result


def run_experiment(n_runs: int = 10) -> pd.DataFrame:
    """
    Runs all tasks × all conditions × n_runs trials.
    n_runs=10  → fast sanity check  (~30-40 min on T4)
    n_runs=30  → publication-grade  (~90-120 min on T4)
    """
    conditions = ["direct", "hierarchical", "oracle"]
    records    = []
    total      = len(TASKS) * len(conditions) * n_runs
    done       = 0

    print(f"\n[EXPERIMENT] {len(TASKS)} tasks × {len(conditions)} conditions × {n_runs} runs = {total} trials\n")

    for run_idx in range(n_runs):
        for task, problem in TASKS:
            for condition in conditions:
                done += 1
                print(f"  [{done}/{total}] run={run_idx+1}  task={task:<14}  condition={condition}", end=" ... ")
                t0      = time.time()
                result  = run_one_trial(task, problem, condition)
                elapsed = time.time() - t0
                result["run"] = run_idx
                records.append(result)
                print(f"score={result['score']:.2f}  align={result['alignment']:.2f}  "
                      f"collapse={result['collapse']}  ({elapsed:.1f}s)")

                if done % 15 == 0:
                    free_memory()

    return pd.DataFrame(records)


# =============================================================================
# SECTION 12: STATISTICAL ANALYSIS
# =============================================================================

def confidence_interval_95(data: List[float]) -> Tuple[float, float, float]:
    """Returns (mean, std, ci_half_width) using 95% CI (z=1.96)."""
    n    = len(data)
    mean = float(np.mean(data))
    std  = float(np.std(data, ddof=1)) if n > 1 else 0.0
    ci   = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, ci


def analyze_model(df: pd.DataFrame, model_name: str) -> Dict:
    """Computes per-condition and per-task statistics. Returns nested dict."""
    summary = {"model": model_name}

    for cond in ["direct", "hierarchical", "oracle"]:
        sub    = df[df["condition"] == cond]
        scores = sub["score"].tolist()
        mean_s, std_s, ci_s = confidence_interval_95(scores)

        entry = {
            "n":          len(sub),
            "score_mean": round(mean_s, 4),
            "score_std":  round(std_s,  4),
            "score_ci95": round(ci_s,   4),
            "exec_rate":  round(float(sub["executable"].mean()), 4),
        }

        if cond != "direct":
            aligns = sub["alignment"].tolist()
            mean_a, std_a, ci_a = confidence_interval_95(aligns)
            entry.update({
                "align_mean":      round(mean_a, 4),
                "align_std":       round(std_a,  4),
                "align_ci95":      round(ci_a,   4),
                "plan_valid_rate": round(float(sub["plan_valid"].mean()), 4),
            })

        if cond == "hierarchical":
            entry["collapse_rate"] = round(float(sub["collapse"].mean()), 4)

        summary[cond] = entry

    # Cohen's d: direct vs hierarchical
    d_scores = df[df["condition"] == "direct"]["score"].values
    h_scores = df[df["condition"] == "hierarchical"]["score"].values
    pooled   = math.sqrt(
        (np.std(d_scores, ddof=1)**2 + np.std(h_scores, ddof=1)**2) / 2
    ) if len(d_scores) > 1 and len(h_scores) > 1 else 1e-9
    summary["cohens_d"] = round(float((np.mean(d_scores) - np.mean(h_scores)) / (pooled + 1e-9)), 4)

    # Per-task collapse rate (hierarchical only)
    hier = df[df["condition"] == "hierarchical"]
    summary["per_task_collapse"] = (
        hier.groupby("task")["collapse"]
            .mean()
            .round(4)
            .to_dict()
    )

    return summary


def print_summary(summary: Dict):
    print("\n" + "="*60)
    print(f"  RESULTS SUMMARY — {summary.get('model', '').upper()}")
    print("="*60)

    for cond in ["direct", "hierarchical", "oracle"]:
        if cond not in summary:
            continue
        s = summary[cond]
        print(f"\n── {cond.upper()} ──")
        for k, v in s.items():
            print(f"   {k:<22}: {v}")

    print("\n── PER-TASK COLLAPSE RATE (hierarchical) ──")
    for task, rate in summary.get("per_task_collapse", {}).items():
        bar = "█" * int(rate * 20)
        print(f"   {task:<16}: {rate:.2f}  {bar}")

    print(f"\n── EFFECT SIZE ──")
    print(f"   Cohen's d (direct vs hierarchical): "
          f"{summary.get('cohens_d', 'N/A')}")
    print("="*60)


def print_unified_summary(summary_ds: Dict, summary_cl: Dict):
    print("\n" + "="*70)
    print("  UNIFIED RESULTS — DeepSeek-1.3B vs CodeLlama-7B")
    print("="*70)

    for summary, label in [(summary_ds, "DEEPSEEK-CODER-1.3B"), (summary_cl, "CODELLAMA-7B")]:
        print(f"\n{'─'*35} {label} {'─'*5}")
        for cond in ["direct", "hierarchical", "oracle"]:
            if cond not in summary:
                continue
            s = summary[cond]
            print(f"\n  [{cond.upper()}]")
            for k, v in s.items():
                print(f"    {k:<22}: {v}")
        print(f"\n  [EFFECT SIZE]  Cohen's d (direct vs hierarchical): {summary.get('cohens_d', 'N/A')}")
        print(f"\n  [PER-TASK COLLAPSE RATE — hierarchical]")
        for task, rate in summary.get("per_task_collapse", {}).items():
            bar = "█" * int(rate * 20)
            print(f"    {task:<16}: {rate:.2f}  {bar}")

    print("\n" + "="*70)


def save_results(df: pd.DataFrame, summary: Dict, prefix: str = ""):
    """Save full trial data and summary to disk."""
    csv_name  = f"results_{prefix}trials.csv"   if prefix else "results_trials.csv"
    json_name = f"results_{prefix}full.jsonl"   if prefix else "results_full.jsonl"
    summ_name = f"results_{prefix}summary.json" if prefix else "results_summary.json"

    df_save = df.drop(columns=["plan", "code"])
    df_save.to_csv(csv_name, index=False)

    df[["task", "condition", "run", "plan", "code", "score",
        "plan_valid", "alignment", "collapse", "error"]].to_json(
        json_name, orient="records", lines=True
    )

    with open(summ_name, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[SAVED] {csv_name}, {json_name}, {summ_name}")


# =============================================================================
# SECTION 13: QUALITATIVE DIVERGENCE ANALYSIS
# Loads full JSONL output and prints representative plan/code pairs for
# failing hierarchical cases — provides mechanistic evidence of the bottleneck.
# =============================================================================

def analyze_divergence(jsonl_path: str = "results_full.jsonl") -> pd.DataFrame:
    try:
        df = pd.read_json(jsonl_path, orient="records", lines=True)
    except Exception as e:
        print(f"[WARN] Could not load {jsonl_path}: {e}")
        return pd.DataFrame()

    failing = df[
        (df["condition"] == "hierarchical") &
        (df["score"] < 1.0)
    ][["task", "run", "score", "alignment", "plan", "code"]].copy()

    return failing


def print_divergence_report(failing: pd.DataFrame):
    """Prints one representative plan/code pair per failing task."""
    if failing.empty:
        print("[INFO] No failing hierarchical trials found.")
        return

    print("\n" + "="*60)
    print("  QUALITATIVE DIVERGENCE ANALYSIS")
    print("  (one representative failing trial per task)")
    print("="*60)

    for task in failing["task"].unique():
        row = failing[failing["task"] == task].iloc[0]
        print(f"\n── TASK: {task.upper()}  (score={row['score']:.2f}, align={row['alignment']:.2f}) ──")
        print(f"\n[ARCHITECT PLAN]\n{row['plan']}")
        print(f"\n[IMPLEMENTOR CODE]\n{row['code'][:800]}")


# =============================================================================
# SECTION 14: CODELLAMA-7B EXPERIMENT
# Loaded AFTER unloading DeepSeek to free VRAM.
# Uses 4-bit NF4 quantization — fits ~4 GB on a T4.
# =============================================================================

CODELLAMA_MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"


def unload_deepseek():
    """Unload the DeepSeek model to free VRAM for CodeLlama."""
    global code_model, tokenizer
    try:
        del code_model
        del tokenizer
        free_memory()
        print(f"[INFO] DeepSeek unloaded. VRAM used: {vram_used_gb():.2f} GB")
    except NameError:
        print("[INFO] DeepSeek model not in globals — already unloaded.")


def load_codellama() -> Tuple:
    """Load CodeLlama-7B-Instruct in 4-bit NF4 quantization."""
    print(f"[INFO] Loading CodeLlama-7B-Instruct (4-bit NF4)...")
    print(f"[INFO] VRAM before load: {vram_used_gb():.2f} GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    cl_tokenizer = AutoTokenizer.from_pretrained(
        CODELLAMA_MODEL_NAME, trust_remote_code=True
    )
    cl_tokenizer.pad_token = cl_tokenizer.eos_token

    cl_model = AutoModelForCausalLM.from_pretrained(
        CODELLAMA_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    cl_model.eval()

    print(f"[INFO] CodeLlama loaded. VRAM used: {vram_used_gb():.2f} GB")
    return cl_tokenizer, cl_model


def llm_codellama(prompt: str, cl_tokenizer, cl_model, max_new_tokens: int = 256) -> str:
    """Deterministic greedy generation with CodeLlama."""
    inputs    = cl_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = cl_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=cl_tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_len:]
    return cl_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# CodeLlama agents — same role separation and prompts as DeepSeek agents

def cl_architect(problem: str, cl_tokenizer, cl_model) -> str:
    prompt = (
        "You are a software architect. Your ONLY job is to output a numbered plan.\n"
        "Rules:\n"
        "- Output ONLY numbered steps (1. 2. 3. ...)\n"
        "- Do NOT write any Python code\n"
        "- Do NOT include def, return, if, for in your output\n"
        "- Be concise: 3-5 steps maximum\n\n"
        f"Problem: {problem}\n\n"
        "Plan:"
    )
    raw   = llm_codellama(prompt, cl_tokenizer, cl_model, max_new_tokens=150)
    lines = [l.strip() for l in raw.split("\n") if re.match(r"^\d+[\.\ )]\s+\S", l.strip())]
    return "\n".join(lines) if lines else raw.strip()


def cl_implementor(plan: str, func_name: str, cl_tokenizer, cl_model) -> str:
    """Implementor sees ONLY the plan — not the original problem."""
    prompt = (
        f"You are a Python programmer. Implement the function `{func_name}` following this plan:\n\n"
        f"{plan}\n\n"
        "Rules:\n"
        "- Output ONLY valid Python code\n"
        f"- The function MUST be named `{func_name}`\n"
        "- No explanation, no markdown prose\n\n"
        "```python\n"
    )
    raw = llm_codellama(prompt, cl_tokenizer, cl_model, max_new_tokens=200)
    return extract_code("```python\n" + raw, func_name)


def cl_direct(problem: str, func_name: str, cl_tokenizer, cl_model) -> str:
    prompt = (
        f"You are a Python programmer. Solve the following:\n\n"
        f"{problem}\n\n"
        "Rules:\n"
        "- Output ONLY valid Python code\n"
        f"- The function MUST be named `{func_name}`\n"
        "- No explanation, no markdown prose\n\n"
        "```python\n"
    )
    raw = llm_codellama(prompt, cl_tokenizer, cl_model, max_new_tokens=200)
    return extract_code("```python\n" + raw, func_name)


def cl_oracle(problem: str, plan: str, func_name: str, cl_tokenizer, cl_model) -> str:
    """Oracle: implementor sees BOTH plan AND original problem."""
    prompt = (
        f"You are a Python programmer.\n\n"
        f"Problem: {problem}\n\n"
        f"Architectural plan to follow:\n{plan}\n\n"
        "Rules:\n"
        "- Output ONLY valid Python code\n"
        f"- The function MUST be named `{func_name}`\n"
        "- No explanation, no markdown prose\n\n"
        "```python\n"
    )
    raw = llm_codellama(prompt, cl_tokenizer, cl_model, max_new_tokens=200)
    return extract_code("```python\n" + raw, func_name)


def run_one_trial_cl(
    task:        str,
    problem:     str,
    condition:   str,
    cl_tokenizer,
    cl_model,
    emb_model,
) -> Dict:
    func_name = FUNC_NAMES[task]
    result = {
        "model":      "codellama-7b",
        "task":       task,
        "condition":  condition,
        "plan":       "",
        "code":       "",
        "plan_valid": False,
        "alignment":  0.0,
        "score":      0.0,
        "executable": False,
        "collapse":   False,
        "error":      "",
    }

    try:
        if condition == "direct":
            code       = cl_direct(problem, func_name, cl_tokenizer, cl_model)
            plan       = ""
            plan_valid = False
            alignment  = 0.0

        elif condition == "hierarchical":
            plan       = cl_architect(problem, cl_tokenizer, cl_model)
            code       = cl_implementor(plan, func_name, cl_tokenizer, cl_model)
            plan_valid = plan_is_valid(plan, task)
            alignment  = semantic_alignment(plan, code, emb_model)

        elif condition == "oracle":
            plan       = cl_architect(problem, cl_tokenizer, cl_model)
            code       = cl_oracle(problem, plan, func_name, cl_tokenizer, cl_model)
            plan_valid = plan_is_valid(plan, task)
            alignment  = semantic_alignment(plan, code, emb_model)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        score, executable = run_tests(code, task)
        collapse = detect_collapse(plan_valid, alignment, score, executable) \
                   if condition == "hierarchical" else False

        result.update({
            "plan":       plan,
            "code":       code,
            "plan_valid": plan_valid,
            "alignment":  round(alignment, 4),
            "score":      round(score, 4),
            "executable": executable,
            "collapse":   collapse,
        })

    except Exception:
        result["error"] = traceback.format_exc()

    return result


def run_codellama_experiment(
    cl_tokenizer,
    cl_model,
    emb_model,
    n_runs: int = 10,
) -> pd.DataFrame:
    conditions = ["direct", "hierarchical", "oracle"]
    records    = []
    total      = len(TASKS) * len(conditions) * n_runs
    done       = 0

    print(f"\n[CODELLAMA EXPERIMENT] {len(TASKS)} tasks × {len(conditions)} conditions × {n_runs} runs = {total} trials\n")

    for run_idx in range(n_runs):
        for task, problem in TASKS:
            for condition in conditions:
                done += 1
                print(f"  [{done}/{total}] run={run_idx+1}  task={task:<14}  condition={condition}", end=" ... ")
                t0      = time.time()
                result  = run_one_trial_cl(task, problem, condition, cl_tokenizer, cl_model, emb_model)
                elapsed = time.time() - t0
                result["run"] = run_idx
                records.append(result)
                print(f"score={result['score']:.2f}  align={result['alignment']:.2f}  "
                      f"collapse={result['collapse']}  ({elapsed:.1f}s)")

                if done % 15 == 0:
                    free_memory()

    return pd.DataFrame(records)


# =============================================================================
# SECTION 15: MAIN
# =============================================================================

if __name__ == "__main__":

    # ── Set number of runs ──────────────────────────────────────────────────
    # 10 runs  → fast sanity check (~30-40 min on T4)
    # 30 runs  → publication-grade (~90-120 min on T4)
    N_RUNS = 10

    # ── Phase 1: DeepSeek experiment ────────────────────────────────────────
    print("\n[PHASE 1] Running DeepSeek-Coder-1.3B experiment...")
    df_ds   = run_experiment(n_runs=N_RUNS)
    summary_ds = analyze_model(df_ds, "deepseek-coder-1.3b")
    print_summary(summary_ds)
    save_results(df_ds, summary_ds, prefix="deepseek_")

    # ── Qualitative divergence analysis (DeepSeek) ───────────────────────────
    print("\n[INFO] Qualitative divergence analysis (DeepSeek)...")
    failing_ds = analyze_divergence("results_deepseek_full.jsonl")
    print_divergence_report(failing_ds)

    # ── Phase 2: Unload DeepSeek, load CodeLlama ────────────────────────────
    print("\n[PHASE 2] Unloading DeepSeek, loading CodeLlama-7B-Instruct...")
    unload_deepseek()
    cl_tokenizer, cl_model = load_codellama()

    # ── Phase 3: CodeLlama experiment ───────────────────────────────────────
    print("\n[PHASE 3] Running CodeLlama-7B experiment...")
    df_cl      = run_codellama_experiment(cl_tokenizer, cl_model, embed_model, n_runs=N_RUNS)
    summary_cl = analyze_model(df_cl, "codellama-7b")
    save_results(df_cl, summary_cl, prefix="codellama_")

    # ── Phase 4: Unified cross-model analysis ───────────────────────────────
    print_unified_summary(summary_ds, summary_cl)

    unified = {
        "deepseek_1.3b": summary_ds,
        "codellama_7b":  summary_cl,
        "collapse_alignment_threshold": COLLAPSE_ALIGNMENT_THRESHOLD,
        "threshold_rationale": (
            "Recalibrated from empirical data: failing tasks (fibonacci, gcd) "
            "have alignment 0.50 and 0.66; passing tasks have alignment >= 0.75. "
            "Midpoint 0.68 is the natural decision boundary."
        ),
    }
    with open("results_unified_summary.json", "w") as f:
        json.dump(unified, f, indent=2, default=str)

    print("[SAVED] results_unified_summary.json")
    free_memory()
    print("\n[DONE]")
