import json
import matplotlib.pyplot as plt

with open("results_unified_summary.json") as f:
    data = json.load(f)

models = ["deepseek_1.3b", "codellama_7b"]

# ---------------- FIGURE 1: PERFORMANCE ----------------
plt.figure()

labels = ["Direct", "Hierarchical", "Oracle"]
x = list(range(len(labels)))

for model in models:
    scores = [
        data[model]["direct"]["score_mean"],
        data[model]["hierarchical"]["score_mean"],
        data[model]["oracle"]["score_mean"]
    ]

    errors = [
        data[model]["direct"]["score_ci95"],
        data[model]["hierarchical"]["score_ci95"],
        data[model]["oracle"]["score_ci95"]
    ]

    plt.errorbar(x, scores, yerr=errors, marker='o', capsize=5, label=model)

    # Annotate values
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)

plt.xticks(x, labels)
plt.ylabel("Accuracy")
plt.title("Performance Degradation Under Hierarchical Planning")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("figure1_performance.png", dpi=300)
plt.close()


# ---------------- FIGURE 2: ALIGNMENT VS SCORE ----------------
plt.figure()

for model in models:
    align = data[model]["hierarchical"]["align_mean"]
    score = data[model]["hierarchical"]["score_mean"]

    plt.scatter(align, score, s=80, label=model)
    plt.text(align, score + 0.03, model, ha='center', fontsize=9)

# Threshold line
threshold = data["collapse_threshold_v2"]
plt.axvline(x=threshold, linestyle='--')
plt.text(threshold + 0.01, 0.1, f"Threshold = {threshold:.2f}", fontsize=9)

plt.xlabel("Alignment Score")
plt.ylabel("Task Accuracy")
plt.title("Alignment Threshold Governs Execution Collapse")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("figure2_alignment_vs_score.png", dpi=300)
plt.close()


# ---------------- FIGURE 3: TASK COLLAPSE ----------------
for model in models:
    plt.figure()

    collapse = data[model]["per_task_collapse"]
    tasks = list(collapse.keys())
    values = list(collapse.values())

    plt.bar(tasks, values)

    # Annotate values
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.1f}", ha='center', fontsize=8)

    plt.xticks(rotation=45)
    plt.ylabel("Collapse Rate")
    plt.title(f"Task-Specific Collapse Patterns ({model})")

    plt.tight_layout()
    plt.savefig(f"figure3_{model}_collapse.png", dpi=300)
    plt.close()


# ---------------- FIGURE 4: EFFECT SIZE ----------------
plt.figure()

cohen_values = [data[m]["cohens_d"] for m in models]

plt.bar(models, cohen_values)

# Annotate values
for i, v in enumerate(cohen_values):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=9)

plt.ylabel("Cohen's d")
plt.title("Effect Size: Direct vs Hierarchical")

plt.tight_layout()
plt.savefig("figure4_effect_size.png", dpi=300)
plt.close()