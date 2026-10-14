# Risk-adaptive multiplex trade theory

Minimal Python analysis code for supply-failure memory, supplier adaptation, crowding and hysteresis in food-trade networks. Code license: MIT (selected by the project owner for this new release).

## Contents and inputs

`code/` contains 13 unchanged computational scripts: both simulation cores and system builders, exact-map theory and checks, raw-panel extraction, synthetic and empirical experiments, event-study estimation, and number aggregation. `data/panels/` contains seven processed FAOSTAT input panels plus their coverage table. These panels are required inputs, not precomputed experiment results. Raw FAOSTAT bulk files are not included. No paper, cover letter, plotting script, manuscript builder, historical result table or temporary artifact is included.

Python 3.11 or later and the packages in `requirements.txt` are required. `environment_tested.txt` records the environment used for file/import and short-run checks, not a recovered lockfile for the original experiments. No dependencies are installed automatically.

## Prepare an isolated run

From this directory:

```powershell
python run.py check
python run.py prepare --work-dir "../risk_adaptive_run"
```

The work directory must not already exist and must be outside this source bundle. Preparation copies the original code and panels into the historical layouts needed by the scripts, including `handoff/` and the empirical job's `panels/`. It starts no experiment and refuses conflicting files.

## Compute in the original stage order

The following commands are opt-in. Full simulation blocks can take substantial time, especially with one worker. All algorithm parameters and fixed seeds remain in the original scripts; `--nproc` only selects the number of worker processes.

```powershell
python run.py theory --work-dir "../risk_adaptive_run"
python run.py theory-checks --work-dir "../risk_adaptive_run"
python run.py synthetic-PBF --work-dir "../risk_adaptive_run" --nproc 1
python run.py synthetic-CHRASE --work-dir "../risk_adaptive_run" --nproc 1
python run.py synthetic-QM --work-dir "../risk_adaptive_run" --nproc 1
python run.py synthetic-H --work-dir "../risk_adaptive_run" --nproc 1
python run.py empirical --work-dir "../risk_adaptive_run" --nproc 1
python run.py eventstudy --work-dir "../risk_adaptive_run"
python run.py analyze --work-dir "../risk_adaptive_run"
python run.py analyze-empirical --work-dir "../risk_adaptive_run"
```

Synthetic block outputs go to `res/rev/out_rev`, `res/rev2/code/out_rev`, `res/rev3/code/out_rev` and `res/rev4/code/out_rev`; empirical-model outputs go to `res/rev3/code/out_emp_rev`; event-study outputs go to `res/emp_rev`; theory and summary outputs go to `handoff`. This matches the unchanged aggregation scripts.

## Optional extraction from raw data

The original project records FAOSTAT Detailed Trade Matrix and Crops and Livestock Products bulk downloads on 9–10 October 2026, for analysis over 1986–2024. Raw files are public at [FAOSTAT bulk production files](https://bulks-faostat.fao.org/production/) and must be provided separately as `tm.zip` and `qcl.zip`. Code licensing and source-data terms are separate; consult the source-data metadata for its terms.

To build new panels in a new run directory, use `python run.py prepare --work-dir "../raw_run" --raw-dir "path/to/local/raw_files"`, then `python run.py extract-panels --work-dir "../raw_run"`. Newly extracted panels are synchronized only within that disposable run directory. Preserve the exact archived raw-file versions if numerical matching is required.

## Validation scope

File SHA-256, archive/directory consistency, script syntax, local imports and input integrity were checked. A short core-simulator check is distinct from reproducing the paper's full experiments. The full experiment suite was not executed during organization. This release intentionally excludes paper generation and plotting; the private research workspace retains those materials.
