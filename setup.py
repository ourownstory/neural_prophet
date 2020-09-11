import setuptools
import platform
from subprocess import Popen, PIPE
import os


# install pre-push hook of running unit tests
old_cwd = os.getcwd()
cwd = os.path.join(old_cwd, '.git', 'hooks')
os.chdir(cwd)

if platform.system() == 'Linux' or platform.system() == 'Darwin':
    p = Popen(['ln', '-s', '../../neuralprophet/test_debug.py', 'pre-commit'], stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    p = Popen(['ln', '-s', '../../neuralprophet/test_debug.py', 'pre-merge-commit'], stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    # print(output)
    # print(error)
elif platform.system() == 'Windows':
    subprocess.Popen(['mklink', '../../neuralprophet/test_debug.py', 'pre-commit'])
    subprocess.Popen(['mklink', '../../neuralprophet/test_debug.py', 'pre-merge-commit'])

os.chdir(old_cwd)

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
