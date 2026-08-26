.PHONY: help env data test baselines train eval benchmark ablation clean

PYTHON = python
PYTEST = pytest

help:
	@echo "Available commands:"
	@echo "  make env        : Inspect environment and dependencies"
	@echo "  make data       : Download and validate datasets"
	@echo "  make test       : Run unit tests"
	@echo "  make baselines  : Run baseline models (TF-IDF, GloVe, BERT un-finetuned)"
	@echo "  make train      : Train main SBERT model on NLI"
	@echo "  make eval       : Evaluate trained model on STS benchmark"
	@echo "  make benchmark  : Run computational efficiency benchmark"
	@echo "  make ablation   : Run pooling & concatenation ablations"
	@echo "  make clean      : Remove temporary build files & caches"

env:
	$(PYTHON) scripts/inspect_environment.py

data:
	$(PYTHON) scripts/prepare_data.py --config configs/sbert_stsb.yaml

test:
	$(PYTEST) -v tests/

baselines:
	$(PYTHON) scripts/run_baselines.py --config configs/baseline_tfidf.yaml

train:
	$(PYTHON) scripts/train_sbert.py --config configs/sbert_nli.yaml

eval:
	$(PYTHON) scripts/evaluate_stsb.py --config configs/sbert_stsb.yaml

benchmark:
	$(PYTHON) scripts/benchmark_similarity.py --config configs/benchmark.yaml

ablation:
	$(PYTHON) scripts/run_ablation.py --config configs/ablations.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
