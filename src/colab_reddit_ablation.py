import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ntm_tensor_core import NTMTensorGraph


DOMAINS = ["TECHNOLOGY", "CULINARY", "ASTRONOMY", "MEDICINE"]
SEEDS = list(range(10))

BASE_CONFIG = {
    "R_MAX": 900.0,
    "R_MIN": 1.0,
    "DECAY_RATE": 1.00,
    "N_NEURONS": 50000,
    "MAX_EDGES": 50000 * 50,
    "MEMBRANE_LIMIT": 15.0,
    "RUPTURE_HEAT": 10.0,
    "PLASTICITY_GAIN": 0.05,
}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_tension(model):
    model.T.zero_()


def make_latent_vectors(num_domains):
    latent_vectors = {}
    for i, dom in enumerate(DOMAINS[:num_domains]):
        vec = torch.zeros(20)
        vec[4:14] = 1.0
        vec[15 + i] = 4.0
        latent_vectors[dom] = vec
    return latent_vectors


def turboquant(vec, qjl_matrix):
    proj = torch.matmul(vec, qjl_matrix)
    t_quant = torch.round(torch.abs(proj) * 10.0)
    threshold = torch.mean(t_quant) + torch.std(t_quant) * 2.0
    t_quant[t_quant < threshold] = 0.0
    return ((t_quant / (torch.sum(t_quant) + 1e-9)) * 1200.0).tolist()


def raw_overlap_pattern(vec):
    scaled = vec * 120.0
    return scaled.tolist()


def evaluate_ntm(model, patterns, domains):
    matrix = {}
    for test_dom in domains:
        clear_tension(model)
        res = model.propagate(
            {"Input_Lobe": [v / 2 for v in patterns[test_dom]]},
            read_only=True,
        )
        total = sum(res.values()) + 1e-9
        matrix[test_dom] = {dom: (res[dom] / total) * 100.0 for dom in domains}
    clear_tension(model)
    return matrix


def matrix_stats(matrix, domains):
    diagonal = [matrix[dom][dom] for dom in domains]
    off_diag = [
        matrix[row][col]
        for row in domains
        for col in domains
        if row != col
    ]
    return {
        "diag_mean": float(np.mean(diagonal)),
        "diag_min": float(np.min(diagonal)),
        "leak_mean": float(np.mean(off_diag)),
        "leak_max": float(np.max(off_diag)),
    }


def run_ntm_variant(seed, variant_name):
    set_all_seeds(seed)
    domains = DOMAINS
    latent_vectors = make_latent_vectors(len(domains))

    use_qjl = variant_name != "ntm_no_qjl"
    config = dict(BASE_CONFIG)

    if variant_name == "ntm_no_neurogenesis":
        config["RUPTURE_HEAT"] = 1e12
    if variant_name == "ntm_no_plasticity":
        config["PLASTICITY_GAIN"] = 0.0

    if use_qjl:
        qjl_matrix = torch.randn(20, 512) / math.sqrt(512)
        patterns = {
            dom: turboquant(vec, qjl_matrix)
            for dom, vec in latent_vectors.items()
        }
        model = NTMTensorGraph(n_inputs=512, actions=domains, config=config)
    else:
        patterns = {
            dom: raw_overlap_pattern(vec)
            for dom, vec in latent_vectors.items()
        }
        model = NTMTensorGraph(n_inputs=20, actions=domains, config=config)

    immediate = {}
    edges_after_domain = {}

    for dom in domains:
        for _ in range(25):
            model.propagate(
                {"Input_Lobe": patterns[dom]},
                target_action_by_pid={"Input_Lobe": dom},
                read_only=False,
            )

        clear_tension(model)
        res = model.propagate(
            {"Input_Lobe": [v / 2 for v in patterns[dom]]},
            read_only=True,
        )
        total = sum(res.values()) + 1e-9
        immediate[dom] = (res[dom] / total) * 100.0
        edges_after_domain[dom] = model.num_edges
        clear_tension(model)

    final_matrix = evaluate_ntm(model, patterns, domains)
    stats = matrix_stats(final_matrix, domains)

    bwt_by_domain = {
        dom: final_matrix[dom][dom] - immediate[dom]
        for dom in domains
    }

    return {
        **stats,
        "bwt_mean": float(np.mean(list(bwt_by_domain.values()))),
        "bwt_min": float(np.min(list(bwt_by_domain.values()))),
        "immediate_mean": float(np.mean(list(immediate.values()))),
        "final_edges": float(model.num_edges),
        "new_edges": float(model.num_edges - (512 if use_qjl else 20)),
    }


class DenseMLP(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


def run_mlp_baseline(seed):
    set_all_seeds(seed)
    domains = DOMAINS
    latent_vectors = make_latent_vectors(len(domains))
    qjl_matrix = torch.randn(20, 512) / math.sqrt(512)
    patterns = {
        dom: torch.tensor(turboquant(vec, qjl_matrix), dtype=torch.float32)
        for dom, vec in latent_vectors.items()
    }

    model = DenseMLP(512, len(domains))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    immediate = {}

    for class_idx, dom in enumerate(domains):
        target = torch.tensor([class_idx])
        x = patterns[dom].unsqueeze(0)

        model.train()
        for _ in range(120):
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            immediate[dom] = probs[class_idx].item() * 100.0

    final_matrix = {}
    model.eval()
    with torch.no_grad():
        for row_dom in domains:
            probs = torch.softmax(model(patterns[row_dom].unsqueeze(0)), dim=1)[0]
            final_matrix[row_dom] = {
                col_dom: probs[col_idx].item() * 100.0
                for col_idx, col_dom in enumerate(domains)
            }

    stats = matrix_stats(final_matrix, domains)
    bwt_by_domain = {
        dom: final_matrix[dom][dom] - immediate[dom]
        for dom in domains
    }

    return {
        **stats,
        "bwt_mean": float(np.mean(list(bwt_by_domain.values()))),
        "bwt_min": float(np.min(list(bwt_by_domain.values()))),
        "immediate_mean": float(np.mean(list(immediate.values()))),
        "final_edges": 0.0,
        "new_edges": 0.0,
    }


def summarize(name, rows):
    keys = [
        "immediate_mean",
        "diag_mean",
        "diag_min",
        "leak_mean",
        "leak_max",
        "bwt_mean",
        "bwt_min",
        "new_edges",
    ]

    summary = {}
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        summary[key] = (float(vals.mean()), float(vals.std(ddof=1)))

    print(f"\n{name}")
    print("-" * len(name))
    for key in keys:
        mean, std = summary[key]
        print(f"{key:>16}: {mean:8.2f} +/- {std:6.2f}")

    return summary


def main():
    print("=" * 80)
    print("NTM V3 CONTINUAL LEARNING ABLATION")
    print("=" * 80)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Seeds: {SEEDS}")
    print("Domains:", ", ".join(DOMAINS))

    variants = [
        ("NTM Full", lambda seed: run_ntm_variant(seed, "ntm_full")),
        ("NTM No QJL", lambda seed: run_ntm_variant(seed, "ntm_no_qjl")),
        ("NTM No Neurogenesis", lambda seed: run_ntm_variant(seed, "ntm_no_neurogenesis")),
        ("NTM No Plasticity", lambda seed: run_ntm_variant(seed, "ntm_no_plasticity")),
        ("MLP Baseline", run_mlp_baseline),
    ]

    all_summaries = {}
    start = time.time()

    for label, runner in variants:
        rows = []
        for seed in SEEDS:
            print(f"[running] {label} seed={seed}")
            rows.append(runner(seed))
        all_summaries[label] = summarize(label, rows)

    elapsed = time.time() - start
    print("\n" + "=" * 80)
    print(f"Finished in {elapsed:.1f}s")
    print("=" * 80)
    print("\nReading guide:")
    print("- immediate_mean: acquisition right after each domain is trained.")
    print("- diag_mean/diag_min: final retained correct-class resonance.")
    print("- leak_mean/leak_max: final off-diagonal interference.")
    print("- bwt_mean/bwt_min: final retention minus immediate acquisition.")
    print("- new_edges: structural expansion beyond initial input edges.")


if __name__ == "__main__":
    main()
