import os
import platform
import subprocess
from subprocess import PIPE
import stat


def install_hooks():
    dir_repo = os.path.abspath(os.path.dirname(__file__))

    script_files = ["run_unittests.bash", "pre_push.bash", "install_hooks.bash"]
    for script_f in script_files:
        file = os.path.join(dir_repo, "scripts", script_f)
        subprocess.check_call(['chmod', 'a+rwx', file])

    subprocess.call(os.path.join(dir_repo, "scripts", "install_hooks.bash"), shell=True)


def install_hooks_py():
    # install pre-push hook of running unit tests
    old_cwd = os.getcwd()
    cwd = os.path.join(old_cwd, '.git', 'hooks')
    os.chdir(cwd)

    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        print("adding 'pre-commit' and 'pre-merge-commit' unittest hooks")
        p1 = subprocess.Popen(['ln', '-s', '../../tests/test_integration.py', 'pre-commit'], stdout=PIPE, stderr=PIPE)
        p2 = subprocess.Popen(['ln', '-s', '../../tests/test_integration.py', 'pre-merge-commit'], stdout=PIPE, stderr=PIPE)
    elif platform.system() == 'Windows':
        pass
        # TODO: add symlinks in Windows
        # Popen(['mklink', 'pre-commit', os.path.join('..', '..', 'neuralprophet', 'test_integration.py')])
        # Popen(['mklink', 'pre-merge-commit', os.path.join('..', '..', 'neuralprophet', 'test_integration.py')])

    os.chdir(old_cwd)


if __name__ == '__main__':
    install_hooks()
