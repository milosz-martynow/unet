.PHONY: lint format install

lint:
	isort --check --profile black ./unet
	black --check ./unet
	pylint ./unet

format:
	isort --profile black ./unet
	black ./unet
	black ./scripts
	black ./tests

install:
	pip install -e .

test:
	pytest -v tests/