backend:
	docker-compose up backend

# Run the app in debug mode.
flask-debug:
	docker-compose run --service-ports backend bash -c "python main.py"

backend-lint:
	docker-compose run --no-deps backend bash -c "flake8 ."
	docker-compose run --no-deps backend bash -c "pep257 --match-dir '[^\.*data]' ."

# backend-test:
# 	docker-compose run --no-deps backend pytest -s tests

# backend-coverage:
# 	docker-compose run --no-deps backend pytest --cov=backend --cov-config .coveragerc --cov-fail-under=78 --cov-report term-missing

backend-coverage-ci:
    docker-compose -f docker-compose.yml -f docker-compose.override.local.yml run backend pytest --cov=backend --cov-config .coveragerc --cov-fail-under=68 --cov-report term-missing

# Run tests for all components.
test:
	$(MAKE) backend-lint
	$(MAKE) backend-coverage

# [Dummy dependency to force a make command to always run.]
FORCE:
