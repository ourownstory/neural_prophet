import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="neuralprophet",
    version="0.2.0",
    description="A simple yet customizable forecaster",
    packages=setuptools.find_packages(["neuralprophet", "neuralprophet.*"]),
    python_requires='>=3',
    install_requires=requirements,
    extras_require={
            "dev": [''],
    }
)
