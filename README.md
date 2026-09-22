# Criticality-Focused Agentic AI Ising

## Project overview
This repository contains the Python implementation used for **criticality-focused active learning and Agentic AI for efficient critical-point localization in the two-dimensional Ising model**. The workflow treats each available temperature as an oracle query and adaptively selects informative temperatures instead of exhaustively using the complete temperature grid.

The experiments use lattice sizes **L = 32, 64, and 128**. The primary target is the finite-size full-grid pseudocritical temperature obtained from the magnetic-susceptibility peak. Exact 2D Ising values are used only for final evaluation, not for acquisition or stopping.

> **Important:** The current experiments use precomputed configurations. Therefore, reported savings quantify reductions in temperature queries and selected configuration usage, not direct Monte Carlo CPU-time savings.

## Main files

| File | Purpose |
|---|---|
| `Proposed(1).py` | Main proposed Agentic AI/active-learning pipeline, observables, adaptive temperature selection, Binder analysis, finite-size analysis, bootstrap uncertainty, and result generation. |
| `Compare.py` | Compares the proposed method with uniform sampling, random sampling, adaptive peak refinement, GP-UCB, expected improvement, and GP uncertainty sampling. |
| `Ablation(1).py` | Ablation study for uncertainty, criticality focus, physics bracketing, autonomous stopping, adaptive memory, and random acquisition. |

## Dataset format
Each CSV dataset should follow the structure:

```text
Temperature,Phase,spin_0,spin_1,...,spin_(L*L-1)
```

Each spin value must be `-1` or `+1`. Update the dataset paths near the beginning of each Python file before execution. The current scripts expect datasets for `L = 32, 64, 128`.

Example:

```python
DATASETS = {
    32: Path("../JOB5_Noise/J5Data/MCD32.csv"),
    64: Path("../JOB5_Noise/J5Data/MCD64.csv"),
    128: Path("../JOB5_Noise/J5Data/MCD128.csv"),
}
```

## Requirements
Recommended: **Python 3.10+**.

Install the required packages with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## Running the proposed method

```bash
python "Proposed(1).py"
```

Optional arguments include:

```bash
python "Proposed(1).py" --max-queries 20 --bootstrap 200
```

Important default agent parameters include 6 initial temperatures, a maximum of 20 queries, minimum 12 queries before autonomous stopping, a 5-step stability window, and a critical-temperature stability tolerance of 0.015.

## Running method comparison

```bash
python Compare.py
```

The default query budgets are:

```text
6, 8, 10, 12, 16, 20
```

To specify budgets manually:

```bash
python Compare.py --budgets 6 8 10 12 16 20
```

Comparison outputs are saved under:

```text
Results/Comparative_PRE/
```

The comparison includes uniform-grid sampling, random sampling, adaptive peak refinement, BO-GP-UCB, BO-GP-EI, active GP-uncertainty sampling, and the proposed physics-guided Agentic AI method.

## Running the ablation study

```bash
python "Ablation(1).py"
```

Outputs are saved under:

```text
Results/Ablation_PRE/
```

The ablation variants are:

1. Full Agent
2. No Uncertainty
3. No Criticality Focus
4. No Physics Critic
5. No Autonomous Stop
6. Frozen Memory
7. Random Acquisition

## Core methodology
The proposed framework combines:

- Gaussian-process modeling of magnetic susceptibility
- Predictive uncertainty for exploration
- Critical-region-focused acquisition
- Physics-based bracketing around the predicted transition
- Sequential observation memory
- Autonomous stopping after pseudocritical-temperature stabilization and physical bracketing

The principal finite-size reference is the full-grid susceptibility-peak temperature. The goal is to reproduce this value using substantially fewer temperature evaluations.

## Main observables
The main pipeline evaluates:

- Absolute magnetization
- Energy per spin
- Nearest-neighbor correlation
- Magnetic susceptibility
- Heat capacity
- Binder cumulant

## Reproducibility
The principal random seed is `42`. Bootstrap analysis in the main pipeline uses seed `2026`. The default bootstrap count is `200`.

## Typical outputs
Depending on the script, the `Results/` directory contains CSV summaries and publication-quality PDF figures for adaptive sampling, critical-temperature convergence, thermodynamic observables, susceptibility, Binder analysis, finite-size analysis, bootstrap uncertainty, method comparisons, runtime, and ablation studies.

## Scientific interpretation
For a finite lattice, the susceptibility maximum gives a **pseudocritical temperature**, which may differ from the exact thermodynamic-limit critical temperature because of finite-size effects. The full-grid finite-size value is therefore the primary reference for evaluating the adaptive sampling method. Binder-cumulant and bootstrap analyses provide complementary physical and statistical validation.

## Notes for GitHub users
- Change dataset paths before running the scripts.
- Do not upload large simulation datasets to GitHub unless necessary; provide a dataset link or instructions instead.
- Generated `Results/` files can be excluded with `.gitignore` if they are reproducible.
- Keep exact Ising values evaluation-only when reproducing the reported methodology.

## Suggested repository name
`Criticality-Focused-Agentic-AI-Ising`

## Suggested citation
If you use this code, please cite the associated manuscript:

**Criticality-Focused Active Learning for Efficient Critical-Point Localization in the Two-Dimensional Ising Model**

Add the final author list, journal reference, DOI, and year after publication.

## License
Add an appropriate open-source license (for example, MIT, BSD-3-Clause, or GPL-3.0) before public release, according to the authors' intended reuse conditions.
