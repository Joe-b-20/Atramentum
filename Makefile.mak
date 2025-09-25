# Makefile
# Build commands for the perpetually tired

.PHONY: help build data sft dpo serve clean

help:
	@echo "Atramentum build targets:"
	@echo "  make build  - Build Docker image"
	@echo "  make data   - Process journal data"  
	@echo "  make sft    - Run SFT training"
	@echo "  make dpo    - Run DPO training"
	@echo "  make serve  - Start API server"
	@echo "  make clean  - Remove generated files"

build:
	docker compose -f docker/docker-compose.yml build

data:
	docker compose -f docker/docker-compose.yml run --rm app \
		bash -c "python scripts/make_dataset.py --config configs/data_formatter.yaml && \
		         python scripts/auto_label.py && \
		         python scripts/build_index.py"

sft:
	docker compose -f docker/docker-compose.yml run --rm app \
		python scripts/train_sft.py --config configs/sft_llama3.yaml

dpo:
	docker compose -f docker/docker-compose.yml run --rm app \
		python scripts/train_dpo.py --config configs/dpo_llama3.yaml

serve:
	docker compose -f docker/docker-compose.yml up app

clean:
	rm -rf data/processed checkpoints index logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete