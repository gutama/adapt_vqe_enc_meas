.PHONY: test validate tables paper benchmark sweep clean

test:
	python test_grouping.py

validate:
	python validate_results.py

tables:
	python generate_paper_data.py

paper: tables
	cd paper && latexmk -pdf manuscript.tex
	cp paper/manuscript.pdf paper/paper.pdf

benchmark:
	python run_benchmark.py --shared-shot-metric --out results.json

sweep:
	python run_qubit_sweep.py --orbs 4 5 6 7 8 --out qubit_sweep.json

clean:
	cd paper && latexmk -C manuscript.tex || true
	rm -f paper/manuscriptNotes.bib
