#!/usr/bin/env python3
"""
Ultra-ECP: Parameter-efficient fine-tuning of UltraSAM for fetal cardiac segmentation.

Usage:
  python ultra_ecp.py train --data-root DATA --checkpoint UltraSam/UltraSam.pth --config CONFIG ...
  python ultra_ecp.py infer --checkpoint CKPT --config CONFIG --data-root DATA ...
  python ultra_ecp.py eval --checkpoint CKPT --config CONFIG --data-root DATA ...
  python ultra_ecp.py convert-coco --image-dir DIR --mask-dir DIR --output FILE.json
  python ultra_ecp.py download-checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

COMMANDS = {
    "train": ("ultra_ecp.train_ultrasam_lora", "main"),
    "infer": ("ultra_ecp.inference_ultrasam_lora", "main"),
    "eval": ("ultra_ecp.evaluate_ultrasam_lora", "main"),
    "convert-coco": ("ultra_ecp.convert_to_coco_for_ultrasam", "main"),
}


def print_help() -> None:
    print(__doc__)
    print("Commands:", ", ".join(COMMANDS), ", download-checkpoint")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        return 0

    command = sys.argv[1]
    rest = sys.argv[2:]

    if command == "download-checkpoint":
        from download_checkpoint import main as download_main

        download_main()
        return 0

    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print_help()
        return 1

    module_path, func_name = COMMANDS[command]
    import importlib

    module = importlib.import_module(module_path)
    main_fn = getattr(module, func_name)

    # Child scripts use argparse on sys.argv[1:]
    sys.argv = [f"ultra_ecp_{command}"] + rest
    main_fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
