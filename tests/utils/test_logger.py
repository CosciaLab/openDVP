"""Tests for the pre-configured `opendvp.utils.logger`.

`logger.py` binds its sink to `sys.stdout` at import time, which is already pytest's capture
object by then. Neither `capsys` nor `capfd` sees those writes, so the module's actual
configuration — stdout, INFO level — is checked in a subprocess, and the rest through a
temporary sink.
"""

import subprocess
import sys
from io import StringIO

import pytest

from opendvp.utils import logger

DEFAULT_FORMAT = "<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}"


def _run(statement: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"from opendvp.utils import logger\n{statement}"],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def sink() -> StringIO:
    """Capture the logger's own output without touching its stdout handler."""
    stream = StringIO()
    handler_id = logger.add(stream, format="{level} | {message}", level="DEBUG")
    yield stream
    logger.remove(handler_id)


def test_is_the_loguru_singleton():
    from loguru import logger as loguru_logger

    assert logger is loguru_logger


def test_info_goes_to_stdout_not_stderr():
    result = _run("logger.info('from a subprocess')")
    assert "from a subprocess" in result.stdout
    assert "INFO" in result.stdout
    assert result.stderr == ""


def test_debug_is_suppressed_at_the_default_level():
    result = _run("logger.debug('should not appear')\nlogger.info('should appear')")
    assert "should not appear" not in result.stdout
    assert "should appear" in result.stdout


def test_loguru_default_handler_was_removed():
    """logger.py calls logger.remove() first; without it every message would be duplicated."""
    result = _run("logger.info('once')")
    assert result.stdout.count("once") == 1


def test_messages_are_timestamped():
    result = _run("logger.info('timed')")
    stamp = result.stdout.splitlines()[0].split("|")[0].strip()
    assert len(stamp.split(":")) == 3, result.stdout


def test_success_level_is_available(sink: StringIO):
    """openDVP functions call logger.success, which is loguru-specific, not stdlib logging."""
    logger.success("finished")
    assert "SUCCESS | finished" in sink.getvalue()


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
def test_every_level_openDVP_uses_is_routable(sink: StringIO, level: str):
    getattr(logger, level)(f"a {level} message")
    assert f"a {level} message" in sink.getvalue()


def test_removing_the_handler_silences_output():
    """Documented in docs/api/utils.md, so worth pinning."""
    result = _run("logger.remove()\nlogger.info('silenced')\nprint('marker')")
    assert "silenced" not in result.stdout
    assert "marker" in result.stdout
