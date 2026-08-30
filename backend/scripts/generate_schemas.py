#!/usr/bin/env python3
# compiles cbor_schema/*.cddl into pydantic models using cddl2py
import subprocess
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CDDL_DIR = os.path.join(REPO_ROOT, "cbor_schema")
OUTPUT_FILE = os.path.join(REPO_ROOT, "backend", "app", "cbor", "generated_schemas.py")

def main():
    print("Generating Pydantic models from CDDL schemas...")
    telemetry_cddl = os.path.join(CDDL_DIR, "telemetry.cddl")
    model_cddl = os.path.join(CDDL_DIR, "model.cddl")

    res_t = subprocess.run(["npx", "--yes", "cddl2py", "--pydantic", telemetry_cddl], capture_output=True, text=True, check=True)
    res_m = subprocess.run(["npx", "--yes", "cddl2py", "--pydantic", model_cddl], capture_output=True, text=True, check=True)

    header = "# Auto-generated from cbor_schema/*.cddl via cddl2py. DO NOT EDIT DIRECTLY.\n"
    header += "from __future__ import annotations\n"
    header += "from typing import Any, Literal, Optional, Union, List, Dict\n"
    header += "from pydantic import BaseModel, Field, ConfigDict\n\n"

    # clean up redundant imports from outputs
    lines_t = [l for l in res_t.stdout.splitlines() if not l.startswith("from ") and not l.startswith("# compiled with")]
    lines_m = [l for l in res_m.stdout.splitlines() if not l.startswith("from ") and not l.startswith("# compiled with")]

    combined = "\n".join(lines_t) + "\n\n" + "\n".join(lines_m)

    # fix reserved keyword 'class: int' -> 'class_count: int = Field(alias="class")'
    combined = combined.replace("    class: int", '    class_count: int = Field(alias="class")')

    content = header + combined + "\n"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
