.PHONY: lint format install

lint:
	isort --check --profile black ./unet
	black --check ./unet
	pylint ./unet

format:
	isort --profile black ./unet
	black ./unet

install:
	pip install -e .