import os

import setuptools

dir_repo = os.path.abspath(os.path.dirname(__file__))
# read the contents of REQUIREMENTS file
with open(os.path.join(dir_repo, "requirements/base.txt"), "r") as f:
    requirements = f.read().splitlines()
with open(os.path.join(dir_repo, "requirements/dev.txt"), "r") as f:
    requirements_dev = f.read().splitlines()
with open(os.path.join(dir_repo, "requirements/docs.txt"), "r") as f:
    requirements_dev += f.read().splitlines()
# read the contents of README file
with open(os.path.join(dir_repo, "README.md"), encoding="utf-8") as f:
    readme = f.read()
# read the version name
with open("neuralprophet/_version.py") as f:
    exec(f.read())

setuptools.setup(
    name="neuralprophet",
    version=__version__,
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
        "dev": requirements_dev,
        "live": ["livelossplot>=0.5.3"],
    },
    # setup_requires=[""],
    scripts=["scripts/neuralprophet_dev_setup.py"],
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
