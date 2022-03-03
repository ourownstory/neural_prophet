import os
import setuptools

dir_repo = os.path.abspath(os.path.dirname(__file__))
# read the contents of REQUIREMENTS file
with open(os.path.join(dir_repo, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()
# read the contents of README file
with open(os.path.join(dir_repo, "README.md"), encoding="utf-8") as f:
    readme = f.read()

setuptools.setup(
    name="neuralprophet",
    version="0.3.1",
    description="Explainable Forecasting at Scale",
    author="Oskar Triebe",
    author_email="trieb@stanford.edu",
    url="https://github.com/ourownstory/neural_prophet",
    license="MIT",
    packages=setuptools.find_packages(
        exclude=(
            "tests",
            "scripts",
        )
    ),
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "dev": ["livelossplot>=0.5.3", "black", "twine", "wheel", "sphinx>=4.2.0", "pytest>=6.2.3", "pytest-cov"],
        "live": ["livelossplot>=0.5.3"],
    },
    # setup_requires=[""],
    scripts=["scripts/neuralprophet_dev_setup"],
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
