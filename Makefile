## Install Dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install -r requirements.txt
	pre-commit install

## Delete all compiled Python files
clean:
	rm -rf **/__pycache__/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/

## Testing
test:
	pytest --durations=0 -vv .

## Basic linting
lint:
	black src
	isort src --profile=black
	pylint src
