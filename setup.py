from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

## read the contents of README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    name="neuralprophet",
    version="0.2.0",
    description="A simple yet customizable forecaster",
    author='Oskar Triebe',
    url="https://github.com/ourownstory/neural_prophet",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=requirements,
    extras_require={"dev": [""], },
    setup_requires=["flake8"],
    long_description=readme,
    long_description_content_type="text/markdown",
)
