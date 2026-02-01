# Product Requirements Document (PRD)

**Project Name:** LABS-Solv-V1  
**Team Name:** LateCommers 
**GitHub Repository:** [https://github.com/idilbagubek/2026-NVIDIA]

---

> **Note to Students:** The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist.
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities
> *(You can DM the judges this information instead of including it in the repository.)*

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | [Name] | [@handle] | [@handle] |
| **GPU Acceleration PIC** (Builder) | [Name] | [@handle] | [@handle] |
| **Quality Assurance PIC** (Verifier) | [Name] | [@handle] | [@handle] |
| **Technical Marketing PIC** (Storyteller) | [Name] | [@handle] | [@handle] |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** **QAOA as the Custom Quantum Seed** (Hamiltonian-informed ansatz built from existing 2-body and 4-body Pauli-rotation blocks).

* **Motivation:**  
  * **Metric-driven:** Quantum-seeded MTS is expected to improve time-to-solution scaling by generating higher-quality initial populations than random initialization.
  * **Skills-driven:** Reuses the existing Pauli-rotation compilation work while replacing the Milestone-1 counterdiabatic schedule with a tunable variational parameterization.

### Literature Review
> Use **valid sources** (peer-reviewed papers, arXiv preprints, official docs, reputable technical blogs).

* **Reference 1:** *Scaling advantage with quantum-enhanced memetic tabu search for LABS* — Alejandro Gomez Cadavid, Pranav Chandarana, Sebastian V. Romero, Jan Trautmann, Enrique Solano, Taylor Lee Patti, Narendra N. Hegade — (arXiv:2511.04553, PDF: `2511.04553v1.pdf`)
  * **Relevance:** Proposes **QE-MTS**, where a quantum subroutine (digitized counterdiabatic quantum optimization, DCQO) generates a **high-quality initial population** for classical MTS, and reports improved **time-to-solution scaling** for LABS (QE-MTS vs MTS) across N≈27–37. :contentReference[oaicite:0]{index=0}

* **Reference 2:** *New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search* — Zhiwei Zhang, Jiayu Shen, Niraj Kumar, Marco Pistoia — (arXiv:2504.00987, PDF: `2504.00987v2.pdf`)
  * **Relevance:** Presents a validated **GPU acceleration architecture** for memetic tabu search (block-level parallel replicas + thread-level parallel neighbor evaluation, shared-memory/bit-vector data structures) and reports substantial **A100 GPU speedups** and improved large-N results; directly supports our plan for GPU MTS implementation and correctness cross-checks. :contentReference[oaicite:1]{index=1}

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:**
  * Use a **quantum sub-routine PQ** to generate the **initial population** for MTS (Quantum Seed → MTS), matching the QE-MTS pipeline where PQ supplies bitstrings used as the initial population. :contentReference[oaicite:0]{index=0}
  * Baseline (paper-faithful): run PQ for a finite number of shots and use the **lowest-energy bitstring** to form the initial population (replicated K times), then run classical MTS. :contentReference[oaicite:1]{index=1}
  * Practical modification for diversity (implementation choice): instead of only replicating the single best bitstring, sample the **top-m** observed bitstrings (lowest energies) and replicate proportionally to counts to avoid a collapsed population.

* **Implementation Notes:**
  * **Backend/Target:** `nvidia` for single-GPU simulation/sampling; `nvidia-mgpu` only if multi-GPU is available and N/depth justify it.
  * **Sampling Plan:**
    * Start: 256–1024 shots per parameter set (fast iteration).
    * Final: 4096–8192 shots per parameter set (lower sampling variance).
    * Parameter sets per run: start with 5–20 candidate parameter sets; keep the best 1–3 for population generation.
  * **Population Extraction:**
    * Bitstring → spins: `0 → -1`, `1 → +1`.
    * Fix and document bit order (endianness) once; keep consistent between CUDA-Q sampling and classical arrays.
    * If measured bitstrings contain more bits than N, take the last N (or first N) consistently and record it.

---

### Classical Acceleration (MTS)
* **Strategy:**
  * Follow an **all-in-GPU** design: CPU host launches a **single GPU kernel** once to avoid repeated host↔device switching and data transportation. :contentReference[oaicite:2]{index=2}
  * Use **two-level parallelism** (as validated in prior GPU MTS work):
    * **Block-level:** each thread block runs one MTS replica with a different random seed. :contentReference[oaicite:3]{index=3}
    * **Thread-level:** expensive steps inside tabu search (neighborhood energy checks + data-structure updates) are parallelized across threads in the block. :contentReference[oaicite:4]{index=4}
  * Use compact data structures (**bit vectors**) in **shared memory** to speed up access and keep per-replica state local. :contentReference[oaicite:5]{index=5}
  * Enable early termination via a **global termination flag** and shared best-so-far state in global memory. :contentReference[oaicite:6]{index=6}

* **Implementation Notes:**
  * **Data layout:**
    * Spins: `int8` (values in {-1,+1}) for CPU interop; optional packed-bit representation for GPU kernels.
    * Tabu list: `int32` per index.
    * Energy helpers: adopt `tableC` / `vectorC` style structures to support fast neighbor evaluation. :contentReference[oaicite:7]{index=7}
  * **Energy kernel:**
    * Build helper structures in **O(N^2)** once per pivot; update after a flip in **O(N)** (linear) by updating one row/column. :contentReference[oaicite:8]{index=8}
  * **Neighbor evaluation:**
    * Each thread evaluates one (or a chunk of) neighbor flip energies; reduce to argmin inside the block.
    * Parallelize both (a) neighborhood energy checks and (b) structure updates, consistent with the validated design. :contentReference[oaicite:9]{index=9}
  * **Minimize CPU↔GPU transfers:**
    * Keep population, tabu state, and helper structures on-device.
    * Only copy out: best sequence, best energy, and minimal logs per run.

---

### Hardware Targets
* **Dev Environment:** CPU-only for correctness + small GPU (e.g., L4/T4) for profiling and kernel validation.
* **Production Environment:** A100/L40-class GPU for final benchmarks.
* Prior work reports running on a **single Nvidia A100** and achieving **8×–26× acceleration** vs a CPU baseline on mid-range N, demonstrating the payoff of the all-in-GPU architecture. :contentReference[oaicite:10]{index=10}

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

> We will keep verification lightweight (no heavy unit-test engineering).  
> Our goal is to catch incorrect AI-generated code and GPU bugs early using small, repeatable scripts.

### Verification Strategy (Lightweight Scripts)
* **Framework:** Simple Python scripts (optional `pytest` later if time permits)
* **AI Hallucination Guardrails:**
  * **CPU reference comparison (small N):** For N ∈ {6, 8, 10}, compare GPU energy / move-selection outputs against the CPU implementation on ~200 random sequences.
  * **Property checks (invariances):** Automatically verify invariances (sign flip, reversal) on random sequences before benchmarking.
  * **Deterministic reproduction:** Log `np.random.seed`, solver hyperparameters (K, p_comb, p_mut, tabu params), and the best-found sequence per run.

### Core Correctness Checks
* **Check 1 (Sign symmetry):** `energy(S) == energy(-S)`
  * **Test:** Generate 100 random sequences (N=20). For each `S`, assert `energy(S) == energy(-S)` for both CPU and GPU implementations.

* **Check 2 (Reversal invariance):** `energy(S) == energy(S[::-1])`
  * **Test:** Generate 100 random sequences (N=20). For each `S`, assert `energy(S) == energy(S[::-1])` for both CPU and GPU implementations.

* **Check 3 (Ground Truth for small N):** exact optimum via brute force
  * **Test:** For N ∈ {6, 7, 8, 9, 10}, brute force all `2^N` sequences once, store the optimal energy `E*`.  
    During development, assert our solver reaches `E*` at least once within a generous iteration budget.

* **Check 4 (Incremental update correctness):** delta-energy update vs full recomputation
  * **Test:** For random sequences (N=50), pick 50 random flip indices.  
    For each flip: (a) update energy via incremental formula, (b) recompute full energy from scratch, assert exact match.

* **Failure Handling:**
  1) Re-run with fixed seed  
  2) Reduce N  
  3) Print failing sequence + intermediate values  
  4) Fix before continuing

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:**
  * **Tooling:** Cursor (primary), VSCode (backup)
  * **Documentation:** Maintain a single `docs/contracts.md` containing:
    * bit ordering convention (qubit → bitstring mapping)
    * energy definition used (CPU + GPU must match)
    * population format (shape, dtype, {-1,+1} convention)
    * CUDA-Q kernel signatures and required inputs (G2/G4 indices, thetas)
  * **Debug loop:** Any runtime error or mismatch → paste logs (seed, N, params, failing sequence) into the agent → refactor → rerun sanity scripts.

### Success Metrics
* **Metric 1 (Quality):**
  * At fixed compute budget, **Quantum-seeded MTS** achieves lower **median final energy** than **Random-seeded MTS** across ≥30 independent runs.

* **Metric 2 (Speedup):**
  * End-to-end **time-to-target** is ≥ **X× faster** than CPU-only tutorial baseline (same success criterion and same target energy).

* **Metric 3 (Scale):**
  * Successfully run and report benchmarks for **N ≥ 40** (stretch goal **N = 50**) with stable throughput.

### Visualization Plan
* **Plot 1:** *Time-to-Solution vs N* (median + IQR) comparing:
  * CPU baseline MTS
  * GPU-accelerated MTS
  * QAOA-seeded GPU MTS

* **Plot 2:** *Convergence* (best energy vs iteration)
  * Quantum seed vs Random seed (mean + IQR band)

* **Plot 3 (Optional):** *Ablation study*
  * Random seed + CPU MTS
  * Random seed + GPU MTS
  * Quantum seed + CPU MTS
  * Quantum seed + GPU MTS

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC

* **Plan:**
  * **CPU-first development:** Implement and debug logic on CPU for small N (e.g., N ≤ 20) until all sanity checks pass.
  * **Cheap GPU for porting/profiling:** Use a low-cost GPU (e.g., L4 / T4) only for:
    * validating CUDA-Q target configuration
    * profiling the hottest kernels (energy + neighbor evaluation)
    * confirming device memory layouts and avoiding CPU↔GPU transfers
  * **Expensive GPU only for final benchmarking:** Use A100/L40-class GPUs only during a dedicated final window to run:
    * full N-sweep benchmarks (e.g., N=27..40 or higher)
    * time-to-solution (TTS) curves and convergence plots
    * final ablations (quantum seed vs random seed, CPU vs GPU)

  * **Hard caps (to prevent runaway costs):**
    * **Runs per N:** max `[30]` runs per N during development, max `[100]` runs per N for final paper-quality plots.
    * **Shots per circuit:** start at `[256–1024]` shots for tuning; only increase to `[4096–8192]` if variance prevents stable seeding.
    * **Parameter optimization budget:** cap to `[50–200]` objective evaluations per N for QAOA/VQE tuning; reuse best parameters (“warm start”) when increasing N or depth.

  * **Caching and reuse:**
    * Cache optimized quantum parameters and sampled populations to disk (`.npz`) so reruns do not repeat expensive quantum sampling.
    * Store every benchmark run’s seed + hyperparameters + best solution for reproducibility without rerunning.

  * **Manual shutdown policy:**
    * The GPU Acceleration PIC is responsible for stopping instances when:
      * tests fail and debugging switches back to CPU
      * the team is idle (meals, meetings)
      * a benchmark batch completes
    * Use a checklist before shutdown: results saved, logs exported, next run plan written in `notes.md`.

  * **Escalation rule:**
    * If GPU time is being consumed without improving metrics (quality or speed) for ≥ `[30]` minutes, stop, revert to CPU debugging, and only resume GPU after a clear change plan is written.
