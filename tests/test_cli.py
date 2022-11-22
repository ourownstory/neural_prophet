import pytest

from neuralprophet.__main__ import parse_args
from neuralprophet._version import __version__


def test_main_file(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--version"])

    out, _ = capsys.readouterr()
    assert exit_info.value.code == 0
    assert __version__ in out
