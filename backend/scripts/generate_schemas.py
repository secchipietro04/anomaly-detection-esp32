#!/usr/bin/env python3
# compiles all cbor_schema/*.cddl directly into backend/app/cbor/schemas.py
import subprocess
import os
import argparse

ENUMS_BLOCK = """
from enum import IntEnum

class StreamMode(IntEnum):
    CONTINUOUS = 1
    ANOMALY_ONLY = 2
    ANOMALY_OR_PERIODIC = 3
    CONTINUOUS_SCORE_RAW_ANOMALY = 4

class ModelType(IntEnum):
    AUTOENCODER = 1
    ROUTER = 2
    MEMORY = 3

class ArchitectureTag(IntEnum):
    VA = 1
    CA_1D = 2
    CA_2D = 3
    CLSTM = 4
    DA = 5
    DENSE_R = 10
    STFT_MCNN = 12
    ONE_D_CNN_R = 13
    LSTM_MEM = 20

class LossMode(IntEnum):
    LOG_MSE = 1
    LINEAR_MSE = 2
"""

def main():
    parser = argparse.ArgumentParser(description="CDDL to Pydantic compiler")
    parser.add_argument("--cddl-dir", default=None, help="Directory containing .cddl files")
    parser.add_argument("--output", default=None, help="Output python file path")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cddl_dir = args.cddl_dir or os.path.join(repo_root, "cbor_schema")
    output_file = args.output or os.path.join(repo_root, "backend", "app", "cbor", "schemas.py")

    print(f"Compiling CDDL schemas from {cddl_dir} -> {output_file}...")
    cddl_files = ["device.cddl", "telemetry.cddl", "model.cddl", "ensemble.cddl"]

    header = "# Auto-generated from cbor_schema/*.cddl via cddl2py. DO NOT EDIT DIRECTLY.\n"
    header += "from __future__ import annotations\n"
    header += "from typing import Any, Literal, Optional, Union, List, Dict\n"
    header += "from pydantic import BaseModel, Field, ConfigDict\n\n"

    all_sections = []

    for fname in cddl_files:
        fpath = os.path.join(cddl_dir, fname)
        if not os.path.exists(fpath):
            continue
        res = subprocess.run(["npx", "--yes", "cddl2py@0.3.1", "--pydantic", fpath], capture_output=True, text=True, check=True)
        lines = [l for l in res.stdout.splitlines() if not l.startswith("from ") and not l.startswith("# compiled with")]
        text = "\n".join(lines)
        all_sections.append(f"# --- {fname} ---\n" + text)

    combined = "\n\n".join(all_sections)
    combined = combined.replace("    class: int", '    class_count: int = Field(alias="class")')

    content = header + combined + "\n\n# --- Typed IntEnums for Python ---\n" + ENUMS_BLOCK + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully generated: {output_file}")

if __name__ == "__main__":
    main()
