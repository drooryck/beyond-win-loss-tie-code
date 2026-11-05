# Reproduce the numerical results and figures for the senior thesis.
# Run from this directory. Requires FIRST_ORDER=1 (default here).

# Prefer a local venv, then a .venv in the parent directory, then system python.
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; \
	else echo python3; fi)
export FIRST_ORDER := 1
export OUT_DIR := $(CURDIR)/results
export FIG_DIR := $(CURDIR)/figures
export PYTHONPATH := $(CURDIR)

.PHONY: help setup data all quick figures clean \
	overview fits optimizers lrt threshold rankings strata strata-math \
	sims ridge splits bootstrap plots

help:
	@echo "Targets:"
	@echo "  make setup       create venv and install requirements"
	@echo "  make data        download LMArena 140k parquet into data/raw/"
	@echo "  make quick       data overview, main fits and every figure"
	@echo "  make all         every table and figure in the thesis"
	@echo "  make figures     plot-only scripts (needs prior results/)"
	@echo "  make clean       remove results/ and figures/ outputs"

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py

overview:
	$(PYTHON) scripts/data_overview.py

fits:
	$(PYTHON) scripts/main_fits.py

splits:
	$(PYTHON) scripts/splits.py

optimizers:
	$(PYTHON) scripts/optimizer_comparison.py

lrt:
	$(PYTHON) scripts/lrt_calibration.py

threshold:
	$(PYTHON) scripts/random_threshold.py

rankings:
	$(PYTHON) scripts/rankings.py

strata:
	$(PYTHON) scripts/strata.py

strata-math:
	$(PYTHON) scripts/strata_math.py

sims:
	$(PYTHON) scripts/simulations.py

ridge:
	$(PYTHON) scripts/regularization.py

bootstrap:
	OMP_NUM_THREADS=1 $(PYTHON) scripts/boot52.py

figures:
	$(PYTHON) scripts/plot_rule_comparison.py
	$(PYTHON) scripts/plot_both_bad_effect.py
	$(PYTHON) scripts/plot_latent_plane.py
	$(PYTHON) scripts/plot_rankings_all52.py
	$(PYTHON) scripts/plot_code_ranks.py

# The results that the plot scripts read, then the plots.
plots: bootstrap rankings strata
	$(MAKE) figures

# Every figure, without the tables that take no part in them.
quick: overview fits plots
	@echo "quick pipeline done"

# Full reproduction of thesis tables and figures.
all: overview fits splits optimizers lrt threshold rankings strata strata-math sims ridge plots
	@echo "full pipeline done"

clean:
	rm -f results/* figures/*.pdf figures/*.png
	@touch results/.gitkeep figures/.gitkeep
