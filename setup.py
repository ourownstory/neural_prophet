import os
import setuptools
import platform
from subprocess import Popen, PIPE


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

## read the contents of README file
from os import path
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

# install pre-push hook of running unit tests
old_cwd = os.getcwd()
cwd = os.path.join(old_cwd, '.git', 'hooks')
os.chdir(cwd)

if platform.system() == 'Linux' or platform.system() == 'Darwin':
    p = Popen(['ln', '-s', '../../neuralprophet/test_integration.py', 'pre-commit'], stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    p = Popen(['ln', '-s', '../../neuralprophet/test_integration.py', 'pre-merge-commit'], stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    # print(output)
    # print(error)
elif platform.system() == 'Windows':
    pass
    # having issues adding symlinks in Windows
    # Popen(['mklink', 'pre-commit', os.path.join('..', '..', 'neuralprophet', 'test_integration.py')])
    # Popen(['mklink', 'pre-merge-commit', os.path.join('..', '..', 'neuralprophet', 'test_integration.py')])

os.chdir(old_cwd)

setuptools.setup(
    name="neuralprophet",
    version="0.2.2",
    description="A simple yet customizable forecaster",
    author='Oskar Triebe',
    url="https://github.com/ourownstory/neural_prophet",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    install_requires=requirements,
    extras_require={"dev": ["livelossplot>=0.5.3"], "live": ["livelossplot>=0.5.3"], },
    setup_requires=["flake8"],
    long_description=readme,
    long_description_content_type="text/markdown",
)
