from __future__ import annotations

import sys
from collections.abc import Mapping

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def configure_logging(*, quiet: bool = False) -> None:
    logger.remove()
    if quiet:
        return

    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    )


def print_output_paths(paths: Mapping[str, object]) -> None:
    table = Table(title="输出文件", show_header=True, header_style="bold")
    table.add_column("名称", no_wrap=True)
    table.add_column("路径")

    for name, path in paths.items():
        table.add_row(name, str(path))

    console.print(table)
