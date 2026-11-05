# Beyond Win, Loss, and Tie in Paired Comparison Models (thesis code)

Code that produces the numerical results and figures in Dries Rooryck’s senior thesis on LMArena paired comparisons with a both-bad outcome.

All analyses use the **first evaluation turn** only (`FIRST_ORDER=1`), matching the 108,305 comparisons in Chapter 4.

## Setup

This needs Python 3.11. `requirements.txt` pins the exact package versions behind the thesis, because newer versions move a few results slightly (for example, one entry of the ridge table).

```bash
cd beyond-win-loss-tie-code
make setup         # creates .venv with python3.11 and installs requirements.txt
make data          # downloads lmarena-ai/arena-human-preference-140k into data/raw/
make all           # writes every result to results/ and every figure to figures/
```

The download uses about 4.3 GB in the Hugging Face cache and writes a 1.6 GB parquet file.

Or point at an existing download:

```bash
ln -s /path/to/arena-human-preference-140k.parquet data/raw/arena-human-preference-140k.parquet
```

Run everything from this directory so `lib/` is on `PYTHONPATH` (the Makefile sets that). `make` uses `.venv` if present, otherwise `../.venv`, otherwise `python3`.

Results and figures are not committed. They are all rebuilt by `make all`.

## How long things take

Measured from a clean copy on an Apple M2 Pro with 12 cores:

| Target | What it does | Time |
|---|---|---|
| `make setup` | Virtual environment and packages | 21 s |
| `make data` | Hugging Face download | about 2 min |
| `make overview` | Data summary and 2 figures | 5 s |
| `make fits` | Likelihood-ratio tests, diagnostics and 1 figure | 17 s |
| `make splits` | Held-out table, ten splits, including grounded Rao-Kupper | 4 s |
| `make optimizers` | Newton against L-BFGS and multistart | 6 s |
| `make lrt` | Null replications | 9 s |
| `make threshold` | Random-threshold fits | 10 s |
| `make rankings` | Rankings across models | 2 s |
| `make strata` | Language and code bootstrap, 200 resamples | 66 s |
| `make strata-math` | Math strata and held-out | 60 s |
| `make sims` | Recovery and misreporting, 100 replications | 8 s |
| `make ridge` | Ridge simulation and real-data ridge | 19 s |
| `make bootstrap` | Bootstrap on all 52 competitors, 200 resamples, for our two models and three Rao-Kupper fits | 5 s |
| `make figures` | The 5 plot-only figures | 5 s |
| **`make quick`** | Every figure, without the remaining tables | **about 2 min** |
| **`make all`** | Every table and figure | **about 4 min** |

## Layout

```
beyond-win-loss-tie-code/
  lib/           shared likelihood, data loading, simulation helpers
  scripts/       one script per experiment or figure
  data/raw/      parquet (gitignored; via make data)
  results/       JSON / npy / npz outputs (gitignored)
  figures/       all thesis PDFs written here (gitignored)
  Makefile
```

## Script → thesis map

### Chapter 4

| Script | Outputs | Thesis |
|---|---|---|
| `scripts/data_overview.py` | `results/data_overview.json`, `figures/comparison_graph.pdf`, `figures/bothbad_winrate.pdf` | §4.1; Tab. `tab:lmarena-outcomes`; Fig. `fig:comparison-graph`, `fig:bothbad-winrate` |
| `scripts/main_fits.py` | `results/main_fits.json`, `z53_*.npy`, `figures/bothbad_vs_strength.pdf` | §4.3 diagnostics; §4.4.3 tests; §4.4.2.1 and Fig. `fig:bothbad`, with both models and the split-ties and grounded Rao-Kupper baselines |
| `scripts/splits.py` | `results/splits.json` | §4.4.2; Tab. `tab:heldout`, including the grounded Rao-Kupper column, and the differences between models |
| `scripts/strata.py` (`main` block) | bootstrap intervals in `results/strata.json` | Tab. `tab:fitted` intervals |
| `scripts/lrt_calibration.py` | `results/lrt_calibration.json` | §4.4.3 critical values |
| `scripts/rankings.py` | `results/rankings.json` | §4.4.4 rank correlations; feeds Fig. `fig:rankings-all` |
| `scripts/boot52.py` | `results/boot52.npz` | input to Fig. `fig:rule-comparison` and `fig:both-bad-effect`; draws for split ties, grounded Rao-Kupper, both models, and Rao-Kupper without both-bad |
| `scripts/plot_rule_comparison.py` | `figures/rule_comparison_theta.pdf` | §4.4.4; Fig. `fig:rule-comparison`, one panel with both models and the two Rao-Kupper baselines |
| `scripts/simulations.py` | `results/simulations.json` | §4.5.1, §4.5.2; Tab. `tab:recovery`, `tab:misspecification` |
| `scripts/regularization.py` | `results/regularization.json` | §4.5.3; Tab. `tab:ridge` and the real-data ridge sentence |

### Appendix

| Script | Outputs | Thesis |
|---|---|---|
| `scripts/optimizer_comparison.py` | `results/optimizer_comparison.json` | §A.2; Tab. `tab:optimizers`; the multistart check in §4.3 |
| `scripts/plot_latent_plane.py` | `figures/latent_plane.pdf` (and `field_vs_threshold.pdf`, which the thesis does not use) | §A.4; Fig. `fig:latent-plane` |
| `scripts/plot_both_bad_effect.py` | `figures/both_bad_effect.pdf` | §A.4; Fig. `fig:both-bad-effect`, our two models |
| `scripts/plot_rankings_all52.py` | `figures/rankings_all52.pdf` | §A.4; Fig. `fig:rankings-all`, our two models |
| `scripts/random_threshold.py` | `results/random_threshold.json` | Random-threshold fits for §A.5; the thesis does not currently print these numbers. The grounded column of Tab. `tab:heldout` comes from `splits.py` |
| `scripts/strata.py` | `results/strata.json`, `code_theta_draws.npz` | §A.6; Tab. `tab:strata` (language, code); data for code ranks |
| `scripts/strata_math.py` | `results/strata_math.json` | §A.6; Tab. `tab:strata` (math against non-math) |
| `scripts/plot_code_ranks.py` | `figures/code_ranks.pdf` | §A.6; Fig. `fig:code-ranks` |

### Shared library

| Module | Role |
|---|---|
| `lib/exact_score.py` | Log-likelihood, score, Hessian, L-BFGS / Newton, grounded RK, RK3 |
| `lib/data.py` | Load parquet, first-order filter, pair counts, train/test split |
| `lib/simulate.py` | Outcome probabilities, `to_theta_c`, multinomial simulation |
| `lib/paths.py` | `data/raw`, `results/`, `figures/` |

### Data

| Script | Role |
|---|---|
| `scripts/download_data.py` | `make data` — Hugging Face → `data/raw/arena-human-preference-140k.parquet` |

## Run order

`make all` runs everything in a valid order. If you run scripts by hand:

1. `download_data`, then `data_overview` and `main_fits`
2. In any order: `splits`, `optimizer_comparison`, `lrt_calibration`, `random_threshold`, `rankings`, `strata`, `strata_math`, `simulations`, `regularization`, `boot52`
3. `plot_rule_comparison` and `plot_both_bad_effect` (need `boot52`)
4. `plot_latent_plane` (needs only the data)
5. `plot_rankings_all52` (needs `rankings`)
6. `plot_code_ranks` (needs `strata`)

## Notes

- Environment: `FIRST_ORDER=1` keeps only `evaluation_order == 1`. The Makefile exports this.
- Self-comparisons and `qwen3-coder-480b-a35b-instruct` are dropped (52 competitors).
- Older draft scripts (`fit_v2`, `eda_try*`, etc.) are **not** included.
- The seeds are fixed. With the pinned versions on Python 3.11, the saved numbers reproduce the thesis results exactly.
