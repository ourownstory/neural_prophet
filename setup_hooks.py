import os
import platform
from subprocess import Popen, PIPE


def add_unittest_hooks():
    # install pre-push hook of running unit tests
    old_cwd = os.getcwd()
    cwd = os.path.join(old_cwd, '.git', 'hooks')
    os.chdir(cwd)

    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        p1 = Popen(['ln', '-s', '../../tests/test_integration.py', 'pre-commit'], stdout=PIPE, stderr=PIPE)
        # output, error = p.communicate()
        p2 = Popen(['ln', '-s', '../../tests/test_integration.py', 'pre-merge-commit'], stdout=PIPE, stderr=PIPE)
        # output, error = p.communicate()
        # print(output)
        # print(error)
    elif platform.system() == 'Windows':
        pass
        # TODO: add symlinks in Windows
        # Popen(['mklink', 'pre-commit', os.path.join('..', '..', 'neuralprophet', 'test_integration.py')])
        # Popen(['mklink', 'pre-merge-commit', os.path.join('..', '..', 'neuralprophet', 'test_integration.py')])

    os.chdir(old_cwd)


if __name__ == '__main__':
    add_unittest_hooks()
