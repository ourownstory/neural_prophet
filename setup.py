import setuptools
import platform
import subprocess

# install pre-push hook of running unit tests
if platform.system() == 'Linux' or platform.system() == 'Darwin':
    subprocess.Popen(['ln', '-s', 'neuralprophet/test_debug.py', '.git/hooks/pre-push'])
elif platform.system() == 'Windows':
    subprocess.Popen(['mklink', 'neuralprophet/test_debug.py', '.git/hooks/pre-push'])

setuptools.setup(
    name="neuralprophet",
    version="0.0.1",
    description="A neural network designed for forecasting",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
