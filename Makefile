install:
	@echo "Installing..."
	pip install --upgrade pip &&\
		pip install -r requirements.txt
	@echo "Installation Complete"

test:
	@echo "Running tests..."
	python -m pytest -vv --cov=app test/*.py
	@echo "Tests Complete"

format:
	@echo "Formatting..."
	black *.py
	@echo "Formatting Complete"

lint:
	@echo "Linting..."
	pylint --disable=R,C *.py
	@echo "Linting Complete"

all: install format lint test