# The Coordination Tax: Coordination Collapse in Hierarchical Multi-Agent Code Generation

**Dataset:** [HumanEval on HuggingFace](https://huggingface.co/datasets/openai/openai_humaneval)

---

## What We Found

Introducing an intermediate planning agent between a problem statement and a code-writing agent causes a dramatic drop in performance, a phenomenon we call **coordination collapse**.

| Model & System | Pass@1 | Collapse |
|---|---|---|
| DeepSeek-1.3B (Single-Agent) | 62.8% | — |
| DeepSeek-1.3B (2-Agent) | 16.5% | **47.6%** |
| DeepSeek-1.3B (3-Agent) | 16.5% | **47.6%** |
| Qwen-3B (Single-Agent) | 82.9% | — |
| Qwen-3B (2-Agent) | 41.5% | **43.3%** |
| Qwen-3B (3-Agent) | 52.4% | 34.2% |

Adding a Validator agent or scaling from 1.3B → 6.7B parameters does **not** fix this. The root cause is information loss at the agent boundary, not model capacity.

---

## Repo Structure

```
NLP/
├── CoordinationCollapse_Full_Evaluation.ipynb                         # Full Evaluation
│
├── Experiment1/                    # Main HumanEval benchmark experiments
│   ├── experiment1.ipynb           # Single-agent, 2-agent, 3-agent pipelines
│   ├── experiment1_results/        # CSVs, per-task results, reports
│   ├── experiment1_figures/        # All paper figures
│   ├── ablation1/
│   │   └── ablation1.ipynb         # Model scaling ablation (1.3B vs 6.7B)
│   └── ablation2/
│       ├── ablation2_qwen_agents.ipynb   # Cross-model ablation (Qwen-3B)
│       ├── ablation2_report
│       └── ablation2_results/      # Qwen ablation outputs and analysis
│
└── Experiment2/                    # Controlled mechanistic (Oracle) experiment
    ├── Experiment2.ipynb           # Direct / Hierarchical / Oracle conditions
    ├── experiment2.py
    ├── visualization.py
    ├── report_exp2.md
    ├── experiment2_figures/        # Oracle experiment figures
    └── results/                    # JSONL and CSV results per condition
```

---

## Models Used

All models are open-source and available on HuggingFace — no paid APIs required.

- [`deepseek-ai/deepseek-coder-1.3b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- [`deepseek-ai/deepseek-coder-6.7b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
- [`Qwen/Qwen2.5-Coder-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
- [`codellama/CodeLlama-7b-Instruct-hf`](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)

---

## Requirements

```bash
pip install transformers bitsandbytes torch scipy numpy
```

The full experiment suite runs on a single **NVIDIA RTX 4090 (24GB VRAM)** in ~2.5 GPU hours. Larger models (6.7B, 7B) use 4-bit quantization via BitsAndBytes.

---

## Reproducing Results

```bash
# Main experiment (Single / 2-Agent / 3-Agent on HumanEval)
jupyter notebook Experiment1/experiment1.ipynb

# Oracle mechanistic experiment
jupyter notebook Experiment2/Experiment2.ipynb

# Model scaling ablation (DeepSeek 1.3B vs 6.7B)
jupyter notebook Experiment1/ablation1/ablation1.ipynb

# Cross-model ablation (Qwen-3B)
jupyter notebook Experiment1/ablation2/ablation2_qwen_agents.ipynb
```

All experiments use a fixed seed (`42`) and greedy decoding (`temperature=0`).
