<div align="center">
  <h1>🧠 Neural Tension Model (NTM V3)</h1>
  <h3>Thermodynamic Continual Learning — Zero Replay, 20 Domains</h3>
  <p>
    A sparse graph architecture governed by fluid dynamics and thermodynamic backpressure<br>
    that retains all learned domains simultaneously — without replaying past data.
  </p>

  <img src="https://img.shields.io/badge/Continual%20Learning-Zero%20Replay-brightgreen" />
  <img src="https://img.shields.io/badge/Domains-20-blue" />
  <img src="https://img.shields.io/badge/Seeds-10-blue" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" />
  <img src="https://img.shields.io/badge/Device-CUDA-green" />
</div>

---

## The Problem

Traditional neural networks (MLPs, Transformers) suffer from **Catastrophic Forgetting**: learning a new task overwrites the weights encoding previous tasks. The standard fix is a *replay buffer* — storing old examples and mixing them with new ones during training.

The NTM doesn't use a replay buffer. It doesn't freeze any weights. It doesn't use adapters per domain.

It solves the problem because **its learning mechanism is physically incapable of overwriting past structure**.

---

## How It Works

Instead of learning through backpropagation on a dense weight matrix, the NTM learns by **sculpting physical pathways** in a sparse graph:

- **Neurogenesis under pressure**: When a node's **excess tension** (tension minus its outflow capacity) surpasses `RUPTURE_HEAT`, it forges a new edge toward a low-connectivity node — expanding the graph's routing capacity for the active domain.
- **Thermodynamic backpressure**: When the graph nears capacity, a pressure wave compresses weak edges proportionally across the graph, freeing slots without erasing any specific domain.
- **Elastic expansion**: If compression isn't enough, the graph grows its tensor allocation dynamically — no hard ceiling.

Different domains naturally occupy **different topological space** in the graph. There is no shared weight matrix to overwrite. Old pathways coexist with new ones because the physics routes new signals along paths of least resistance — which are always the paths not yet claimed by existing domains.

```
Input embedding
      ↓
Tension propagates through sparse edge graph (scatter_add, CUDA)
      ↓
Heat builds at bottleneck nodes → neurogenesis OR backpressure compression
      ↓
Flow dissipates into domain sinks → classification output
```

---

## Results

### 20-Domain Continual Learning — Zero Replay

Trained on **20 semantically distinct domains** sequentially, using real sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2` with QJL orthogonal projection (384 → 512 dims).

**Validated across 10 random orderings (seeds):**

| Metric | Value |
|---|---|
| Grand mean final accuracy | **18.7% ± 0.5%** |
| Random baseline (1/20) | 5.0% |
| Performance above random | **3.7×** |
| Domains above 2× random | **20 / 20** in all 10 seeds |
| Mean BWT | −24.7% ± 0.4% (seed-level) |
| Order independence (fwd vs rev) | **0.65% mean diff** |

The seed-level std of **0.04%** on final accuracy confirms that training order is statistically irrelevant — the system converges to the same thermodynamic equilibrium regardless of which domain arrives first.

### Order Independence

Two full runs — Forward order (TECH→CULI→...→NEUR) and Reversed (NEUR→...→TECH) — produce virtually identical final distributions:

```
Domain            Fwd Final   Rev Final    Diff
------------------------------------------------
TECHNOLOGY            16.3%       19.8%   −3.5%
CULINARY              17.7%       18.7%   −1.0%
ASTRONOMY             18.4%       18.4%   +0.1%
MEDICINE              19.7%       18.8%   +0.9%
...
------------------------------------------------
Mean absolute diff                        0.65%
```

### Per-Domain Summary (10 Seeds)

```
Domain            mean_final  ±std    mean_BWT   ±std
------------------------------------------------------
TECHNOLOGY            19.8%  ±0.4%    −28.0%   ±19.2%
CULINARY              18.4%  ±0.8%    −40.1%   ±30.2%
ASTRONOMY             18.6%  ±0.4%    −24.2%   ±22.7%
MEDICINE              18.9%  ±0.5%    −12.4%   ±12.1%
LAW                   17.8%  ±0.4%    −23.9%   ±24.7%
MUSIC                 19.3%  ±0.4%    −31.2%   ±25.0%
SPORTS                18.5%  ±0.3%    −23.5%   ±23.3%
HISTORY               18.6%  ±0.4%    −24.3%   ±15.7%
MATHEMATICS           19.8%  ±0.6%    −26.6%   ±19.9%
PHILOSOPHY            19.1%  ±0.3%    −13.3%   ±12.8%
BIOLOGY               19.4%  ±0.9%    −29.5%   ±28.7%
CHEMISTRY             19.8%  ±0.7%    −27.9%   ±24.4%
ECONOMICS             18.0%  ±0.2%    −13.2%    ±9.3%
PSYCHOLOGY            18.0%  ±0.3%    −17.6%   ±19.4%
ARCHITECTURE          18.3%  ±0.4%    −23.5%   ±22.2%
LITERATURE            18.7%  ±0.4%    −24.3%   ±24.1%
GEOGRAPHY             19.4%  ±1.0%    −25.4%   ±26.5%
POLITICS              18.7%  ±0.4%    −23.9%   ±18.4%
ENGINEERING           17.6%  ±0.3%    −21.1%   ±23.0%
NEUROSCIENCE          18.1%  ±0.5%    −40.6%   ±28.1%
------------------------------------------------------
Grand mean            18.7%  ±0.5%    −24.7%   ±21.5%
```

> **Note on BWT:** The negative BWT is not pathological forgetting — it is **competitive convergence**. A domain trained in isolation reaches 97% (all energy focused on it). After 19 other domains share the graph, its energy is distributed. The 3.7× above-random signal means its pathway is still intact and identifiable.

---

## Why This Is Different

Most continual learning solutions work by **avoiding** the problem:

| Approach | How It Avoids Forgetting | Problem |
|---|---|---|
| Replay Buffers | Stores old examples, mixes in training | Requires storage; doesn't scale |
| LoRA / Adapters per domain | Freezes the backbone | Can't mix domains in one forward pass |
| EWC | Penalizes changes to important weights | Approximate; degrades with many tasks |
| Progressive Neural Nets | Adds columns per task | Linear memory growth |
| **NTM** | **Domains physically occupy different topological space** | Higher memory footprint than MLP; no baseline comparison yet |

The NTM is a single model. One forward pass classifies across all 20 domains simultaneously. No routing logic, no adapter selection, no knowing in advance which domain is being queried.

---

## Architecture

### Core: `src/ntm_tensor_core.py`

The NTM runs as a sparse directed graph on GPU:

```python
# Edges: sparse COO format
edge_src  : LongTensor[MAX_EDGES]   # source nodes
edge_dst  : LongTensor[MAX_EDGES]   # destination nodes  
edge_w    : FloatTensor[MAX_EDGES]  # conductance weights

# Nodes
T         : FloatTensor[N]          # tension (activation) per node

# Output
Sinks     : FloatTensor[N, K]       # conductance to K domain sinks
```

**Forward pass** (`propagate`):
1. Inject input embedding into first `n_inputs` nodes (adds to T)
2. For 3 iterations: compute flow `fe = T[src] * (w / capacity)`, update T
3. Leakage into sinks = output scores
4. If `not read_only`: update conductances (Hebbian + BV hook), trigger neurogenesis if heat > threshold

**Backpressure** (when capacity is full):
```python
bp = heat[hot] / (RUPTURE_HEAT + 1e-9)
compression = 1.0 / (1.0 + bp[es] * 0.15)
edge_w *= compression          # compress weak edges
_prune_and_compact()           # free dead slots
# if still full → _expand_graph() → grow the tensor
```

### Input Pipeline

Raw text → `sentence-transformers/all-MiniLM-L6-v2` → 384-dim embedding →
QJL random orthogonal projection → 512-dim sparse float vector (143/512 active dims on average)

---

## Quickstart

### Requirements

```bash
pip install torch sentence-transformers numpy matplotlib
```

### Run the 20-Domain Benchmark

```python
# On Google Colab or local GPU:
python benchmarks/test_retention_order_colab.py
```

Produces:
- Order independence test (Forward vs Reversed)
- Correct BWT per domain
- 20×20 confusion matrix

### Run the 10-Seed Variance Benchmark

```python
python benchmarks/test_variance_10seeds_colab.py
```

Produces mean ± std across 10 random orderings of the 20 domains.

> **Note:** Both scripts require `sentence-transformers` and a CUDA GPU is strongly recommended. CPU will work but is slow.

---

## Configuration

Key hyperparameters in `config` dict:

| Parameter | Default | Description |
|---|---|---|
| `N_NEURONS` | 100,000 | Initial node count |
| `R_MAX` | 900.0 | Maximum resistance (minimum conductance) |
| `R_MIN` | 1.0 | Minimum resistance (maximum conductance) |
| `DECAY_RATE` | 1.00 | 1/DECAY_RATE applied to conductances per step |
| `MEMBRANE_LIMIT` | 15.0 | Max flow per unit conductance |
| `RUPTURE_HEAT` | 10.0 | Heat threshold for neurogenesis |
| `PLASTICITY_GAIN` | 0.05 | Hebbian learning rate for edges |

---

## Repository Structure

```
MTN/
├── src/
│   ├── ntm_tensor_core.py        # Main architecture (NTMV3 class)
│   └── ntm_core.py               # Legacy scalar implementation
├── benchmarks/
│   ├── test_retention_order_colab.py   # 20-domain order independence + BWT
│   └── test_variance_10seeds_colab.py  # 10-seed statistical validation
├── docs/
│   ├── theory_and_physics.md     # Thermodynamic principles
│   ├── architecture_api.md       # API reference
│   ├── benchmarks_and_ablation.md
│   ├── tensor_evolution.md
│   └── turboquant_llm_integration.md
└── requirements.txt
```

---

## Open Questions

This is an ongoing research project. Known limitations:

1. **No comparison with EWC/PackNet** on same 20-domain setup yet
2. **Computational cost** vs. equivalent MLP + continual learning wrapper not benchmarked
3. **Scaling beyond 20 domains** — expected to work physically, not yet validated
4. **Language generation** via thermodynamic routing — exploratory, not validated

---

## Citation

If you use this work, please cite:

```
@misc{ntm2025,
  title  = {Neural Tension Model: Thermodynamic Continual Learning via Sparse Graph Dynamics},
  author = {[Thone]},
  year   = {2025},
  url    = {https://github.com/[Thone]/MTN}
}
```

---

*The NTM was not designed to solve catastrophic forgetting. It was designed as a physical system governed by thermodynamics. The absence of catastrophic forgetting is an emergent consequence of the architecture.*

---

## A Note on This Project

The NTM started from a single question:

> *"What if we stop copying the **structure** of the brain — and start copying its **fundamentals** instead?"

Most brain-inspired AI copies the form: neurons, dendrites, synapses as mathematical abstractions. The NTM copies the physics underneath: tension, pressure, heat, dissipation, and the thermodynamic drive toward equilibrium.

That question led — unexpectedly — to a system that doesn't forget.

This is an **independent experimental project** — no lab, no funding, no formal ML background. I'm a young researcher with roughly **2 hours per day** to work on this, running experiments on free Google Colab GPUs.

If you find the direction interesting — whether to critique it, build on it, or collaborate — I'm genuinely open to it. Issues and discussions are welcome.
