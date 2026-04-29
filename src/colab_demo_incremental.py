import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Import primary NTM Tensor Core V3
from ntm_tensor_core import NTMTensorGraph

print("\n" + "="*80)
print("  🚀 EVALUATION: CLASS-INCREMENTAL CONTINUAL LEARNING (ZERO-REPLAY) 🚀  ")
print("="*80)
print("Objective: Sequentially optimize 4 independent domains.")
print("Static SGD-based models will exhibit subsequent Catastrophic Forgetting.")
print("The NTM will allocate latent topological branches to mitigate overwriting.\n")
print(f"Hardware: {'✅ CUDA Enabled' if torch.cuda.is_available() else '⚠️  CPU (no CUDA detected)'}\n")

# =====================================================================
# 1. ENVIRONMENT SETUP
# =====================================================================
DOMAINS = ["TECHNOLOGY", "CULINARY", "ASTRONOMY", "MEDICINE"]
N_DOMAINS = len(DOMAINS)

config = {
    "R_MAX": 900.0,
    "R_MIN": 1.0,
    "DECAY_RATE": 1.00,   # No biological temporal decay — isolates structural retention
    "N_NEURONS": 50000,
    "MAX_EDGES": 50000 * 50,
    "MEMBRANE_LIMIT": 15.0,
    "RUPTURE_HEAT": 10.0,
    "PLASTICITY_GAIN": 0.05
}

# Input dimensionality after QJL projection
N_INPUTS = 512

ntm = NTMTensorGraph(n_inputs=N_INPUTS, actions=DOMAINS, config=config)

# =====================================================================
# 2. ORTHOGONAL PROJECTION (LLM -> TURBOQUANT PIPELINE)
#
# Each domain vector has:
#   - A shared overlapping region  [indices 0:14]  → simulates semantic overlap
#   - A unique separability signal [index 14+i]    → domain identity "whisper"
#
# Vector length = 14 + N_DOMAINS = 18 (safe for 4+ domains without aliasing)
# =====================================================================
VEC_DIM = 14 + N_DOMAINS  # 18 — scales cleanly with any number of domains

latent_vectors = {}
for i, dom in enumerate(DOMAINS):
    vec = torch.zeros(VEC_DIM)
    vec[0:10] = 1.0          # Heavy semantic overlap region (induces interference)
    vec[14 + i] = 4.0        # Domain-unique separability spike — no index collision
    latent_vectors[dom] = vec

# Static QJL rotation matrix — frozen seed for full reproducibility
SEED = 42
torch.manual_seed(SEED)
QJL_MATRIX = torch.randn(VEC_DIM, N_INPUTS) / math.sqrt(N_INPUTS)

def transform_to_sparse_cortex(vec):
    """
    TurboQuant pipeline:
      1. QJL rotation: projects VEC_DIM -> N_INPUTS into orthogonal sub-space
      2. Quantization: rounds absolute values to discrete magnitudes
      3. Z-Score filter (threshold = mean + 2*std): zeros out low activations
      4. Normalization: maps survivors to hydraulic tension units (total = 1200.0)
    """
    proj    = torch.matmul(vec, QJL_MATRIX)
    t_quant = torch.round(torch.abs(proj) * 10.0)
    threshold = torch.mean(t_quant) + torch.std(t_quant) * 2.0
    t_quant[t_quant < threshold] = 0.0
    total = torch.sum(t_quant) + 1e-9
    return ((t_quant / total) * 1200.0).tolist()

sparse_patterns = {dom: transform_to_sparse_cortex(vec) for dom, vec in latent_vectors.items()}

# =====================================================================
# UTILITY: Reset node tension between propagation calls
#
# Zeroes the internal tension tensor T directly — avoids iterating over
# TensorDictProxy keys and guarantees a clean CUDA state every time.
# =====================================================================
def reset_tension(model: NTMTensorGraph):
    model.T.zero_()

# =====================================================================
# 3. RAW SEQUENTIAL LEARNING (ZERO-REPLAY TEST)
# =====================================================================
print(">>> INITIATING TEMPORAL SEQUENTIAL TRAINING <<<\n")

EPOCHS_PER_DOMAIN = 25

# Track per-domain accuracy right after each domain finishes training
# Used later to compute Forward Transfer (FWT)
post_train_scores = {}

for dom in DOMAINS:
    print(f"[{dom}] Optimizing topological parameters for the block...")
    for epoch in range(EPOCHS_PER_DOMAIN):
        ntm.propagate(
            {"Input_Lobe": sparse_patterns[dom]},
            target_action_by_pid={"Input_Lobe": dom},
            read_only=False
        )

    # Immediate self-evaluation right after training this domain
    # (uses half-amplitude recall pattern — simulates partial/degraded cue)
    reset_tension(ntm)
    immediate_res = ntm.propagate(
        {"Input_Lobe": [v / 2 for v in sparse_patterns[dom]]},
        read_only=True
    )
    total = sum(immediate_res.values()) + 1e-9
    conf  = (immediate_res[dom] / total) * 100
    post_train_scores[dom] = conf / 100.0   # store as 0–1 for BWT/FWT calc
    print(f"  -> Immediate Retention ({dom}): {conf:.1f}%")
    reset_tension(ntm)

print("\n" + "-"*50)
print(f"✓ Topology Consolidated. Total dynamic edges allocated: {ntm.num_edges}.")
print("-"*50)

# =====================================================================
# 4. LATE-STAGE MULTI-CLASS INFERENCE EVALUATION
# =====================================================================
print("\n>>> GLOBAL COMPARATIVE INFERENCE <<<\n")
print("Semantic amnesia report under Zero-Replay constraint.\n")

confusion_matrix = {}
final_scores = {}  # domain -> fraction correct (diagonal element)

for test_dom in DOMAINS:
    reset_tension(ntm)
    inference_result = ntm.propagate(
        {"Input_Lobe": [v / 2 for v in sparse_patterns[test_dom]]},
        read_only=True
    )
    total = sum(inference_result.values()) + 1e-9
    row   = {d: (inference_result[d] / total) * 100 for d in DOMAINS}
    confusion_matrix[test_dom] = row
    final_scores[test_dom] = row[test_dom] / 100.0

reset_tension(ntm)

# =====================================================================
# 5. BWT / FWT METRICS
#
# Backward Transfer (BWT): measures how much training later domains
#   degraded retention of earlier ones. Negative = forgetting.
#   BWT = (1/N) * Σ [final_score(i) - post_train_score(i)]  for i < last domain
#
# Forward Transfer (FWT): measures how much training earlier domains
#   boosted zero-shot performance on later ones. Positive = transfer.
#   FWT = (1/N-1) * Σ [post_train_score(i) - random_baseline]  for i > first
#   (random baseline = 1/N_DOMAINS for uniform prior)
# =====================================================================
print("\n>>> CONTINUAL LEARNING METRICS <<<\n")

bwt_values = []
fwt_values = []
random_baseline = 1.0 / N_DOMAINS

for i, dom in enumerate(DOMAINS):
    delta = final_scores[dom] - post_train_scores[dom]
    bwt_values.append(delta)

    if i > 0:   # FWT: how much earlier training helped this domain at first eval
        fwt_values.append(post_train_scores[dom] - random_baseline)

BWT = sum(bwt_values) / len(bwt_values)
FWT = sum(fwt_values) / len(fwt_values) if fwt_values else 0.0
AVG_FINAL_ACC = sum(final_scores.values()) / N_DOMAINS

print(f"  Average Final Accuracy  : {AVG_FINAL_ACC*100:.2f}%")
print(f"  Backward Transfer (BWT) : {BWT:+.4f}  {'✅ Structural retention holding' if BWT >= -0.05 else '⚠️  Signs of interference'}")
print(f"  Forward Transfer  (FWT) : {FWT:+.4f}  {'✅ Cross-domain priming detected' if FWT > 0 else '➖ No positive transfer'}")

per_domain_bwt = {dom: final_scores[dom] - post_train_scores[dom] for dom in DOMAINS}
print("\n  Per-domain BWT:")
for dom, bwt in per_domain_bwt.items():
    bar = "█" * int(abs(bwt) * 20)
    sign = "+" if bwt >= 0 else "-"
    print(f"    {dom:<12}: {sign}{abs(bwt)*100:.1f}% {bar}")

# =====================================================================
# 6. CONFUSION MATRIX — TERMINAL TABLE
# =====================================================================
print("\n>>> CONFUSION MATRIX <<<\n")

short = [d[:4] for d in DOMAINS]
header = f"{'Injected Domain':<16} | " + " | ".join([f"{s:<5}" for s in short])
print("=" * len(header))
print(header)
print("=" * len(header))

for test_dom in DOMAINS:
    row_str = f"{test_dom:<16} | "
    for col_dom in DOMAINS:
        val = confusion_matrix[test_dom][col_dom]
        if test_dom == col_dom:
            row_str += f"\033[92m{val:>4.1f}%\033[0m | "   # green diagonal
        else:
            row_str += f"{val:>4.1f}% | "
    print(row_str)

print("=" * len(header))

# =====================================================================
# 7. VISUAL HEATMAP (SEABORN — Colab / GitHub)
# =====================================================================
matrix_data = [
    [confusion_matrix[test_dom][col_dom] for col_dom in DOMAINS]
    for test_dom in DOMAINS
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Confusion Heatmap ---
ax_cm = axes[0]
sns.heatmap(
    matrix_data, annot=True, fmt=".1f", cmap="Blues",
    xticklabels=DOMAINS, yticklabels=DOMAINS,
    cbar=True, vmin=0, vmax=100, ax=ax_cm
)
for t in ax_cm.texts:
    if "%" not in t.get_text():
        t.set_text(t.get_text() + "%")
ax_cm.set_title("NTM V3 — Continual Learning Retention Matrix", pad=14, fontsize=13, fontweight="bold")
ax_cm.set_ylabel("Injected Domain (Actual)",    fontsize=11, fontweight="bold")
ax_cm.set_xlabel("Cortex Activation (Predicted)", fontsize=11, fontweight="bold")

# --- Right: BWT bar chart ---
ax_bwt = axes[1]
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in per_domain_bwt.values()]
bars   = ax_bwt.barh(
    list(per_domain_bwt.keys()),
    [v * 100 for v in per_domain_bwt.values()],
    color=colors, edgecolor="white", height=0.5
)
ax_bwt.axvline(0, color="white", linewidth=0.8, linestyle="--")
ax_bwt.set_xlabel("BWT (Δ Accuracy % after full training)", fontsize=11, fontweight="bold")
ax_bwt.set_title("Backward Transfer per Domain", pad=14, fontsize=13, fontweight="bold")
ax_bwt.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
for bar, val in zip(bars, per_domain_bwt.values()):
    ax_bwt.text(
        bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
        f"{val*100:+.1f}%", va="center", color="white", fontsize=10
    )

plt.tight_layout()

image_path = "ntm_retention_matrix.png"
plt.savefig(image_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[+] Graph saved: '{image_path}'")

try:
    plt.show(block=False)
except Exception:
    pass

print("\n" + "="*80)
print("  ✅ EVALUATION COMPLETE")
print(f"     Avg Accuracy : {AVG_FINAL_ACC*100:.2f}%")
print(f"     BWT          : {BWT:+.4f}")
print(f"     FWT          : {FWT:+.4f}")
print(f"     Total Edges  : {ntm.num_edges}")
print("="*80 + "\n")
