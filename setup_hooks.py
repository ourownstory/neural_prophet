import os
import subprocess


def install_hooks():
    dir_repo = os.path.abspath(os.path.dirname(__file__))

    script_files = ["install_hooks.bash", "pre_push.bash"]
    for script_f in script_files:
        file = os.path.join(dir_repo, "scripts", script_f)
        subprocess.check_call(['chmod', 'a+rwx', file])

    subprocess.call(os.path.join(dir_repo, "scripts", "install_hooks.bash"), shell=True)


if __name__ == '__main__':
    install_hooks()
