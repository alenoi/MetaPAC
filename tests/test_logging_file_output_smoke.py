from __future__ import annotations

import pytest

from metapac.src.utils.logging_utils import configure_logging, get_logger


@pytest.mark.smoke
def test_configurable_file_logging_writes_log_file(tmp_path) -> None:
    configure_logging(
        {
            "console_enabled": False,
            "file_enabled": True,
            "dir": str(tmp_path),
            "filename": "metapac.log",
            "level": "INFO",
        }
    )

    logger = get_logger("metapac.test.logging")
    logger.info("file logging smoke message")

    log_path = tmp_path / "metapac.log"
    assert log_path.exists()
    assert "file logging smoke message" in log_path.read_text(encoding="utf-8")

    configure_logging({"console_enabled": True, "file_enabled": False})