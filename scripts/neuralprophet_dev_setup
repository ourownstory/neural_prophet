#!/usr/bin/env python3

import os
import subprocess


def install_hooks():
    dir_scripts = os.path.abspath(os.path.dirname(__file__))
    script_files = [
        "install_hooks.bash",
        "pre_commit.bash",
        "pre_push.bash",
    ]
    for script_f in script_files:
        file = os.path.join(dir_scripts, script_f)
        subprocess.check_call(["chmod", "a+rwx", file])
    subprocess.call(os.path.join(dir_scripts, "install_hooks.bash"), shell=True)


if __name__ == "__main__":
    install_hooks()
