## Install Dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install -r requirements.txt
	pre-commit install

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "__pycache__" -delete
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/

## Testing
test:
	pytest --cov=src --cov-report term-missing --cov-fail-under=90 --durations=0 -vv tests/

## Basic linting
lint:
	black src
	isort src --profile=black
	mypy src
	pylint src
