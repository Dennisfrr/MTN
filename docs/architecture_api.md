# Architecture API & Hyperparameter Manual

The **NTM Tensor Core V3** operates under a fundamentally different paradigm compared to classical dense neural networks. It does not possess Learning Rates, Weight Decays, or regularizer functions. Instead, its learning capabilities are governed by physical thermodynamic constraints. 

This manual details the instantiation parameters and internal configurations required to tune the fluid mechanics of the NTM Engine.

---

## 1. Instantiating the Engine

The architecture is encapsulated within the `NTMTensorGraph` object located in `src/ntm_tensor_core.py`.

```python
from ntm_tensor_core import NTMTensorGraph

# Define conceptual domains/targets
actions = ["CLASS_A", "CLASS_B", "CLASS_C"]

# Dictionary configuring thermodynamic physics
config = { ... }

engine = NTMTensorGraph(n_inputs=512, actions=actions, config=config)
```

### Constructor Arguments
* `n_inputs (int)`: The dimension of your pre-processed sparse vector (e.g., Output from the `TurboQuant` filter). This dictates the number of immutable absolute input gateways within the structural graph.
* `actions (list[str])`: A string list representing categorical prediction targets. The framework intrinsically translates these names into dedicated output physical `Sinks`.
* `config (dict)`: The thermodynamic hyperparameter dictionary regulating neurogenesis and physical topological rules.

---

## 2. Configuration Dictionary Parameters (`config`)

The configuration establishes the mechanical elasticity of the PyTorch framework holding the network structure.

### Topological VRAM Pre-Allocation
Unlike standard architectures, the NTM dynamically expands physical edge lists. To avoid continuous CPU memory-allocation penalties, the matrix operates using pre-allocated empty spaces in CUDA.
* `N_NEURONS (int)`: *Default: 50000*. The universal upper limit of nodes (input + hidden + output) the topology is allowed to utilize. Defines spatial boundaries.
* `MAX_EDGES (int)`: *Default: 50000 * 50*. The hard constraint on maximum structural branch connections allowed to exist concurrently within the GPU array.

### Mechanical Tension Limits
* `MEMBRANE_LIMIT (float)`: *Default: 15.0*. Acts as a topological bottleneck (`out_cap_local`). It strictly constrains how much computational water an individual node can conduct globally. It inherently prevents newly established neural pathways from parasitically destroying adjacent veteran matrices (Vital for Zero-Replay resilience).
* `RUPTURE_HEAT (float)`: *Default: 10.0*. The minimum pressure of computational water required to force the creation of an entirely new physical pathway. If surrounding pipelines are clogged or membrane restricted, accumulated pressure exceeding this heat triggers `Scatter-Add` Neurogenesis.

### Physical Plasticity & Temporal Degradation
* `PLASTICITY_GAIN (float)`: *Default: 0.05*. Replaces standard Gradient Descent "Learning Rate". Determines the physical dilation rate of an edge given continuous water throughput. Higher values speed up the engorgement of successful predictive paths.
* `DECAY_RATE (float)`: *Default: 1.00 (No Decay) or ~0.999 (Biological).* Applies strict temporal rotting. During passive usage or subsequent cycles, edges without residual traffic logically decay, eventually being pruned from memory (Mimicking biological sleep-cycle pruning).

*Note: For Sequential Benchmark Tests proving absolute Catastrophic Mitigation without replay, `DECAY_RATE` should remain `1.0` to verify structural resilience independently from time-based rotting.*

---

## 3. Propagation Syntax

Learning and Inference in the NTM share identical Forward-Pass functions (`propagate`). The system consolidates paths strictly if explicit reinforcement is triggered on the Sinks.

#### Forward Pass (Inference/Recall)
```python
# Pass the sparse threshold values with read_only flag to merely vibrate the graph 
# and assess semantic resonance without burning pathways.
results = engine.propagate({"Input_Lobe": sparse_values}, read_only=True)

# Returns a dictionary indicating chronological resonance per class (e.g., {"CLASS_A": 0.02, "CLASS_B": 88.5})
```

#### Forward Pass (Active Learning)
```python
# Informing the target_action manually drops water pressure heavily onto the specific target 
# Sink, burning physical pathways backwards through the nodes traversed.
engine.propagate(
    {"Input_Lobe": sparse_values}, 
    target_action_by_pid={"Input_Lobe": "CLASS_B"}, 
    read_only=False
)
```

> [!TIP]
> The engine handles mass input parallelism. You can send hundreds of distinct `Input_Lobes` simultaneously, and operations will batch compute safely using index-additive operations `torch.scatter_add_()` to naturally avert PyTorch CUDA Race Conditions.
