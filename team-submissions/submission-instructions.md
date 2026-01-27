## Deliverables Checklist

**Early Submission (To obtain access to GPUs with Brev credits):**

* [ ] **Tutorial Notebook:** Completed `labs_tutorial.ipynb`.
* [ ] **PRD:** `PRD.md` defining your plan.

**Final Submission (Deadline):**

* [ ] **Final Code:** Notebooks and scripts for the Milestone 3 implementation.
* [ ] **Test Suite:** `tests.py` demonstrating >60% coverage.
* [ ] **AI Report:** `AI_REPORT.md` (The Post-mortem).
* [ ] **Presentation:** Slides (Live) or MP4 (Remote).

## Evaluation Criteria: How You Will Be Graded

This challenge is graded on **Rigorous Engineering**, not just raw speed. We are simulating a professional delivery pipeline. Your score is split between your **Planning (35%)** and your **Execution (65%)**.

### Part 1: The Plan (40% - Early Submission)
*Goal: Prove you have a viable strategy before we release GPU credits.*

* **The Self-Validation (Milestone 1):** Did you prove your tutorial baseline was correct? We look for rigorous checks (e.g., symmetry verification, small N brute-force) rather than just "it looks right."
* **The PRD Quality (Milestone 2):**
    * **Architecture:** Did you cite literature to justify your quantum algorithm choice? Deep research drives high scores. We explicitly reward teams that take Scientific Risks over those who play it safe.
    * **Acceleration:** Do you have a concrete plan for GPU memory management and parallelization?
    * **Verification:** Did you commit to specific physical correctness checks (e.g., `energy(x) == energy(-x)`)?
    * **Success Metrics:** Did you define quantifiable targets?


### Part 2: The Product (60% - Final Submission)
*Goal: Prove your solution works, scales, and is verified.*

* **Performance & Scale:**
    * Does the solution scale to larger $N$?
    * Did you successfully accelerate the *Classical* component (e.g., using `cupy` for batch neighbor evaluation) or did you only accelerate the quantum circuit?
* **Algorithmic Novelty:**
    * We reward the **rigorous implementation of creative ideas**. If your novel experiment fails to beat the baseline, document *why* in your report. A "Negative Result" backed by great engineering is a success.
* **Verification:**
    * How much of your code is covered by the `tests.py` suite?
    * Does the test suite catch physical violations?
* **Communication & Analysis:**
    * **Visualizations:** We expect professional data plotting. Do not just paste screenshots of code. We want to see generated plots (Time vs. N, Approximation Ratio vs. N) with clearly labeled axes comparing CPU vs. GPU performance.
    * **The Narrative:** Your presentation must tell the story of "The Plan & The Pivot." Did you identify *why* you had to change your strategy?
    * **The AI Report:** Your `AI_REPORT.md` must demonstrate *how* you verified AI-generated code, including specific examples of "Wins" and "Hallucinations."

