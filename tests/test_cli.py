import pytest

from neuralprophet.__main__ import parser
from neuralprophet._version import __version__


def test_main_file(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])

    out, _ = capsys.readouterr()
    assert "<ExceptionInfo SystemExit(0) tblen=8>" == str(exit_info)
    assert __version__ in out
