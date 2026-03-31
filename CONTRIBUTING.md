# Contributing

This document describes how to work with the repository — branch structure,
the pull request workflow, coding conventions, and how CI is set up.

---

## Branch Structure

`main` is the protected production branch. It always contains working,
reviewed code. Direct pushes to `main` are blocked by branch protection rules.
All changes arrive through pull requests.

Feature and fix branches follow this naming pattern:

```
feature/<short-description>      # new capability
fix/<short-description>          # bug fix
refactor/<short-description>     # code cleanup with no behaviour change
docs/<short-description>         # documentation only
```

Examples: `feature/sensitivity-analysis`, `fix/warmup-contamination`,
`docs/update-readme`.

---

## Workflow

**1. Create a branch from `main`**

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

**2. Make your changes and commit**

Keep commits focused. Commit message format:

```
<type>: <short description>

<optional body explaining why, not what>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

Examples:

```
feat: add per-severity DTStat collectors to GP model
fix: warmup contamination guard on LOS stat record
docs: add GP posterior explanation to README
```

**3. Push and open a pull request**

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub targeting `main`. The PR description
should explain what changed and why.

**4. CI must pass**

The CI workflow (`.github/workflows/ci.yml`) runs on every PR and checks
Python syntax across all model and engine files. The PR cannot be merged
until CI passes. If CI fails, fix the error locally and push again — the
workflow reruns automatically.

**5. Merge**

Once CI is green, merge the PR using **Squash and merge** to keep `main`'s
history clean. Delete the feature branch after merging.

---

## Branch Protection Rules

`main` is protected by two mechanisms that apply even to the repository
owner:

**Ruleset** — blocks force pushes (`git push --force`) and branch deletion.
These cannot be bypassed. They protect commit history.

**Branch protection rule** — requires a pull request before merging to
`main`. Direct pushes are rejected. This applies to all contributors.

---

## CI Configuration

The workflow lives in `.github/workflows/ci.yml`. It runs on every pull
request targeting `main` and checks Python syntax on all `.py` files:

```yaml
name: CI
on:
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install numpy pandas matplotlib scipy scikit-learn
      - run: python -m py_compile ERSimulationModelGPwithSev.py
```

To run the syntax check locally before pushing:

```bash
python -m py_compile ERSimulationModelGPwithSev.py
python -m py_compile ERSimulationModelGP.py
python -m py_compile ERSimulationModelNHPP.py
```

---

## Code Conventions

These conventions are enforced across all project files.

**Naming**

| Item | Style | Example |
|---|---|---|
| Variables and functions | `camelCase` | `rateFn`, `buildHourlyCounts` |
| Constants | `UPPER_SNAKE_CASE` | `WARMUP_MIN`, `N_DOCTORS` |
| Classes | `PascalCase` | `ArrivalRateGP`, `PriorityQueue` |
| Private members | `_underscore` prefix | `_currentRateFn`, `_checkFitted` |

**File paths** — always use `os.path.join` with `BASE_DIR`. Never hardcode
absolute paths or use `pathlib`.

**Stage constants** — `stage_cols` and stage lookup dicts are defined locally
inside functions, not at module level.

**Comments** — written at the level of a developer with five years of
experience. No decorative or redundant comments. No `#===` banner separators.
Comments explain why, not what.

**Docstrings** — module docstrings describe inputs and outputs. Class
docstrings explain the contract and usage pattern. Method docstrings are
single-line where the name is self-explanatory, short prose otherwise.
No numpy-style `Parameters / Returns` sections.

**Imports** — standard library first, then third-party, then internal. One
blank line between each group. `sim_engine` modules imported as:

```python
from sim_engine import SimClasses, SimFunctions, SimRNG
from sim_engine.ArrivalRateGP import ArrivalRateGP
```

---

## Adding a New Simulation Model

If you create a new model variant (e.g. a version with ICU overflow):

1. Copy the closest existing model file as a starting point
2. Keep `sim_engine/` unchanged unless adding a genuinely new primitive
3. Update `RESULTS_DIR` output filenames so they do not overwrite existing results
4. Add the new file to the `py_compile` check in `ci.yml`
5. Document the model variant in the README table

---

## Data Files

`Sources/` is not tracked by git (it is in `.gitignore`). Data files are
not committed to the repository. To run the simulation, place the following
files in `Sources/` manually:

```
Sources/er_5000_patients.csv
Sources/simrng_parameters.csv
```

`simrng_parameters.csv` is generated by running `Fitting_Dist_Tat.py`.
It is not committed because it is a derived file.
