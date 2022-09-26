.PHONY: lint format

lint:
	isort --check --profile black ./unet
	black --check ./unet
	pylint ./unet

format:
	isort --profile black ./unet
	black ./unet
