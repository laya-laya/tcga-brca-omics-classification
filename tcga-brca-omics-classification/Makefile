.PHONY: test download prepare cv analysis enrichment heatmap all

test:
	python -m pytest -q

download:
	python scripts/download_tcga_brca.py

prepare:
	python scripts/prepare_data.py

cv:
	python scripts/run_nested_cv.py

analysis:
	python scripts/run_full_analysis.py

enrichment:
	python scripts/run_enrichment.py

heatmap:
	python scripts/make_heatmap.py

