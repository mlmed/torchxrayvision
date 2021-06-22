import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchxrayvision",
    version="0.0.27",
    author="Joseph Paul Cohen",
    author_email="joseph@josephpcohen.com",
    description="TorchXRayVision: A library of chest X-ray datasets and models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlmed/torchxrayvision",
#    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1',
        'torchvision>=0.5',
        'scikit-image>=0.16',
        'tqdm>=4',
        'numpy>=1',
        'pandas>=1',
        'pydicom>=1',
        'requests>=1'
    ],
    packages=find_packages(),
    package_dir={'torchxrayvision': 'torchxrayvision'},
    package_data={'torchxrayvision': ['data/*.zip','*.gz','*.tgz','*.zip']},
)
