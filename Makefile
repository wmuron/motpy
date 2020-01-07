demo:
	python examples/2d_multi_object_tracking.py

install-develop:
	python setup.py develop

test:
	python -m pytest ./tests

env-create:
	conda create --name env_motpy python=3.7 -y

env-activate:
	# might fail from makefile
	conda activate env_motpy

env-install:
	pip install -r requirements.txt

clean:
	autoflake --in-place --remove-unused-variables ./motpy/*.py ./tests/*.py

check:
	mypy --ignore-missing-imports motpy