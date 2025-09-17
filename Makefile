
.PHONY: all data paper clean hier audit post

all: data paper hier audit post

data:
	python3 scripts/run_all.py
	python3 scripts/run_hier.py
	python3 scripts/cf_audit.py
	python3 scripts/run_posteriors.py
	python3 scripts/run_diagnostics.py
	python3 scripts/plot_barrier_mprime.py
	python3 scripts/run_kerr_surrogate.py
	python3 scripts/run_reomega.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex || true
	cd paper && bibtex main || true
	cd paper && pdflatex -interaction=nonstopmode main.tex || true
	cd paper && pdflatex -interaction=nonstopmode main.tex || true

clean:
	rm -f data/*.png data/*.csv
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf

hier:
	python3 scripts/run_hier.py

audit:
	python3 scripts/cf_audit.py

post:
	python3 scripts/run_posteriors.py
	python3 scripts/plot_barrier_mprime.py