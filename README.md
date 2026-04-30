<div align="center">
  <h1>🧠 Neural Tension Model (NTM)</h1>
  <h3>The Journey to Thermodynamic & Electrochemical Intelligence</h3>
  <p>
    A continual learning system where intelligence is not programmed via gradients, <br>
    but emerges physically from excitatory and inhibitory balances in a dynamic graph.
  </p>

  <img src="https://img.shields.io/badge/Continual%20Learning-Stream%20Mode-brightgreen" />
  <img src="https://img.shields.io/badge/Physics-Electrochemical-blue" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" />
</div>

---

## 📖 The Story Behind the Project

The **Neural Tension Model (NTM)** didn't start as an attempt to solve *catastrophic forgetting* or to invent the next big Machine Learning algorithm. It started with a fundamental question:

> *"What if we stop trying to copy the **structure** of the brain (layers, dense weights, backpropagation) and start copying the **physical principles** that govern it?"*

Most neural networks (like MLPs and Transformers) rely on artificial algorithmic crutches: *epochs* (repeating the same data), *batches* (averaging errors), and artificially resetting states between examples. The brain doesn't have a reset button, nor does it separate its "training phase" from its "execution phase."

### Phase 1: The Hydraulic Model (NTMv3)
In earlier iterations (NTMv3), we modeled learning as a **hydraulic** system. Information was like water flowing through pipes (graph edges). When there was too much pressure, new pipes were forged (neurogenesis). It worked incredibly well: the model learned 20 semantic domains without forgetting any of them. But there was a hidden "hack": we were using a global reset (`T.zero_()`) after every sample to clear the graph. It was a crutch borrowed from the deterministic ML world.

### The Epiphany: From Hydraulic to Electrochemical (NTMv4)
Then we realized the physical flaw: NTMv3 was strictly *positive*. But real neurons are electrochemical—they can depolarize (excite) or hyperpolarize (inhibit). 

By removing artificial constraints (`clamp(min=0)`) and introducing **Inhibitory Sinks** via Anti-Hebbian learning, something magical happened: the need to reset the network vanished. When a concept is activated, inhibition organically grows to suppress the pathways of competing concepts. **The balance between excitation and inhibition acts as the natural signal isolator.**

Thus, **NTMv4** was born.

---

## 🧬 Core Physics & Emergent Principles

Instead of hard-coding separate learning rules, the NTM operates on the **Maximum Entropy Production Principle (MEPP)**. By forcing the network to maximize energy dissipation, four crucial phenomena emerged from a single physical mechanism:

1. **Hebbian Learning (Plasticity)**: Frequently used pathways widen and gain conductance.
2. **Natural Decay**: Unused pathways atrophy, releasing resources back to the system.
3. **Neurogenesis / Synaptogenesis**: When tension builds up in structural "bottlenecks," the localized heat breaches a threshold and forges new connections to relieve the pressure.
4. **Electrochemical Competition**: Localized inhibition prevents the contamination of pathways, forming the physical basis for durable associative memory.

### Theoretical Implications
* **Topology as Memory:** Unlike traditional ML where topology is fixed and weights change, the NTM's physical volume expands in exact proportion to the entropy of the dataset. The historical record resides in the viability of connective strains, not just scalar weights.
* **Intelligence as Resonance:** There is no standard feed-forward math (`X * W = Y`). When a stimulus hits the primary gates, flow is conducted based on historical records. Intelligence is measured by the *channel's resonance* when traversed by a known input.
* **Macro-Scalar Emergence:** While individual synapses fluctuate with chaotic entropy (micro-separability), the scaled macro-structure isolates fundamental global classes efficiently.

*(Read more in `docs/theory_and_physics.md` and `docs/tensor_evolution.md`)*

---

## 🚀 Capabilities That Defy Tradition

Because the NTM is based on physical and topological properties rather than shared dense weight matrices, it solves critical AI problems naturally.

### 1. Pure Stream Learning (Zero Replay, Zero Epochs)
The NTM learns in a continuous stream. There are no *epochs* or *batch sizes*. You can show it a sample of "medicine," then one of "architecture," and it learns on the fly. It achieves a **Backward Transfer (BWT) of +0.0%** across 20 shuffled domains, meaning it completely averts catastrophic forgetting without ever storing or replaying old examples.

### 2. "On-The-Fly" Class Expansion Without Retraining
In a production scenario where a client asks to add new classes:
- **Classical Neural Network (MLP)**: Fine-tuning a model on 3 new classes using only a few examples destroys the knowledge of the original 10 classes (accuracy drops from 50% to 0%).
- **NTMv4**: You can add new classes with just 10 examples each. The NTM retains **90%+** of its knowledge of the original classes. Why? Because new classes create *lateral* pathways in the graph topology, without overwriting the physical channels already forged by past knowledge.

*(See the `demo_new_class_colab.py` benchmark).*

---

## 📂 Repository Structure

```text
MTN/
├── src/
│   ├── ntm_tensor_core.py        # Core Engine (NTM V3 and V4)
│   └── ntm_core.py               # Legacy scalar implementation
├── benchmarks/
│   ├── demo_new_class_colab.py               # Demo: Add classes without forgetting
│   ├── test_ntmv4_shuffled_cf_colab.py       # Stream Learning (20 Domains)
│   ├── test_mepp_emergence_colab.py          # Proof of MEPP emergent properties
│   └── test_ntmv4_inhibition_colab.py        # Electrochemical balance validation
└── docs/                             # Deep dives into Architecture and Physics
```

---

## ⚡ Quickstart

### Prerequisites
```bash
pip install torch numpy matplotlib
```

### Running the Class Expansion Demo
This benchmark simulates a production system (10 classes). It then injects 3 new classes (using only 10 examples each) and compares "Catastrophic Forgetting" between a classic MLP and the NTMv4.
```bash
python benchmarks/demo_new_class_colab.py
```

### Running the Stream Learning Validation (20 Domains)
Tests continuous learning on a shuffled stream of 20 diverse semantic domains, measuring final accuracy and Backward Transfer (BWT).
```bash
python benchmarks/test_ntmv4_shuffled_cf_colab.py
```

*(Note: A CUDA-enabled GPU is strongly recommended for PyTorch tensor operations).*

---

## 🤝 About the Author & Research

This project is an **independent experimental endeavor**. I don't have a lab, Silicon Valley funding, or a formal background in classic ML research. I'm simply an indie researcher driven by curiosity, dedicating roughly 2 free hours a day to this.

All progress so far has been built by validating hypotheses and letting physics dictate the rules, running experiments on free Google Colab GPUs. 

If you find this direction intriguing—whether you want to critique it, build on it (the possibilities for Edge AI and Robotics are massive), or collaborate—the repository is open. Feel free to open *Issues*, fork the code, and reach out!
