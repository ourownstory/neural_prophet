import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="neuralprophet",
    version="0.0.1",
    description="A package designed for forecasting of time series",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=install_requires,
)
