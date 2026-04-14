from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(output_dir: str,
                  script_name: str,
                  run_name: str | None = None) -> str:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = f"_{run_name}" if run_name else ""
    log_path = log_dir / f"{script_name}{safe_run_name}_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return str(log_path)
