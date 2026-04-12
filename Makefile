.PHONY: setup install test eval

setup:
	python -m venv venv
	.\venv\Scripts\activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

test:
	pytest tests/

eval:
	python src/evaluation/eval_ir.py
	python src/evaluation/eval_sentiment.py
