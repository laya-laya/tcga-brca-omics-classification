.PHONY: test download prepare cv analysis enrichment heatmap all \
        train predict serve demo simulate-drift \
        docker-build docker-train docker-predict docker-serve

IMAGE ?= tcga-brca-classifier
MODEL ?= models/pam50_classifier.joblib
INPUT ?= data/new/samples.tsv.gz
OUTPUT ?= predictions/predictions.csv
DOCKER_USER = --user $$(id -u):$$(id -g)

# ---------- analysis (unchanged) ----------
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

# ---------- deployment ----------
train:            ## fit the final model and save models/pam50_classifier.joblib
	python scripts/train_model.py --out $(MODEL)

predict:          ## make predict INPUT=path/to/new_samples.tsv.gz
	PYTHONPATH=src python -m tcga_brca.predict --model $(MODEL) --input $(INPUT) --output $(OUTPUT)

serve:            ## API on http://localhost:8000 (docs at /docs)
	MODEL_PATH=$(MODEL) uvicorn tcga_brca.api:app --app-dir src --port 8000

demo:             ## try the batch job on synthetic data, no TCGA download needed
	PYTHONPATH=src python -m tcga_brca.demo --model models/demo.joblib --samples data/demo/batch.tsv
	PYTHONPATH=src python -m tcga_brca.demo --samples data/demo/shifted.tsv --shift 1.5
	PYTHONPATH=src python -m tcga_brca.predict --model models/demo.joblib --input data/demo/batch.tsv --output predictions/demo_batch.csv
	PYTHONPATH=src python -m tcga_brca.predict --model models/demo.joblib --input data/demo/shifted.tsv --output predictions/demo_shifted.csv

simulate-drift:   ## false-alarm rate and sensitivity of the drift check
	python scripts/simulate_drift_check.py

docker-build:
	docker build -t $(IMAGE) .

docker-train: docker-build   ## download TCGA and train inside the pinned image
	mkdir -p data models
	docker run --rm $(DOCKER_USER) -v "$$PWD/data:/app/data" -v "$$PWD/models:/app/models" $(IMAGE) \
		sh -c "python scripts/download_tcga_brca.py && python scripts/train_model.py"

docker-predict:   ## make docker-predict INPUT=data/new/samples.tsv.gz (INPUT must be under data/)
	mkdir -p predictions
	docker run --rm $(DOCKER_USER) -v "$$PWD/data:/app/data:ro" -v "$$PWD/models:/app/models:ro" \
		-v "$$PWD/predictions:/app/predictions" $(IMAGE) \
		python -m tcga_brca.predict --input $(INPUT) --output $(OUTPUT)

docker-serve:
	docker run --rm -p 8000:8000 -v "$$PWD/models:/app/models:ro" $(IMAGE)
