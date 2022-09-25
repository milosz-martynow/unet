.PHONY: test lint format install

test:
	pytest -v tests/

lint:
	isort --check --profile black .
	black --check .
	pylint .

format:
	isort --profile black .
	black .
