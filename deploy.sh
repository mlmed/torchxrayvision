rm -rf dist/

python3 setup.py sdist bdist_wheel

python3 -m twine upload -u __token__ dist/*
