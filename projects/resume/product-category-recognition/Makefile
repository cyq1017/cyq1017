.PHONY: setup data offline online evaluate all clean demo

# Full pipeline: data → offline → evaluate
all: data offline evaluate
	@echo "=== Full pipeline complete. Results in results/ ==="

setup:
	@echo "=== Setting up environment ==="
	bash setup.sh

data:
	@echo "=== Phase 1: Generating data ==="
	python data/generate_data.py

offline:
	@echo "=== Phase 3: Offline embedding generation ==="
	python scripts/run_offline.py

online:
	@echo "=== Phase 4: Running online inference ==="
	python scripts/run_online.py

evaluate:
	@echo "=== Phase 5: Evaluation ==="
	python scripts/evaluate.py

demo:
	@echo "=== Running demo ==="
	python scripts/demo.py

clean:
	@echo "=== Cleaning generated files ==="
	rm -rf embeddings/ indices/ results/
	rm -f data/products.csv data/anchor_products.csv data/test_products.csv
	rm -rf data/images/
	@echo "Clean complete."
