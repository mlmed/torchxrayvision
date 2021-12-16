import setuptools
from setuptools import setup, find_packages
from torchxrayvision import _version

with open("README.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setuptools.setup(
    name="torchxrayvision",
    version=_version.__version__,
    author="Joseph Paul Cohen",
    author_email="joseph@josephpcohen.com",
    description="TorchXRayVision: A library of chest X-ray datasets and models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlmed/torchxrayvision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires='>=3.6',
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    package_dir={'torchxrayvision': 'torchxrayvision'},
    package_data={'torchxrayvision': ['data/*.zip','data/*.gz','data/*.tgz','baseline_models/*/*.json','baseline_models/*/*/*.json']},
)
