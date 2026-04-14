from __future__ import annotations

from utils.io import write_json


def build_main_table(st_summary: dict, tsr_summary: dict,
                     visref_summary: dict) -> dict:
    return {
        "ST": st_summary,
        "TSR": tsr_summary,
        "VisRef": visref_summary,
    }


def save_table(table: dict, output_path: str) -> None:
    write_json(output_path, table)
