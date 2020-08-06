rm -rf dist/

python3 setup.py sdist bdist_wheel

python3 -m twine upload -u torchxrayvision --repository-url https://test.pypi.org/legacy/ dist/*
