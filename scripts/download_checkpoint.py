"""Download UltraSAM pretrained checkpoint into UltraSam/UltraSam.pth."""

import ssl
import sys
import urllib.request
from pathlib import Path

URL = "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"
MIN_SIZE_MB = 300


def main() -> None:
    output_path = Path(__file__).resolve().parent / "UltraSam" / "UltraSam.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        if size_mb >= MIN_SIZE_MB:
            print(f"Checkpoint OK: {output_path} ({size_mb:.1f} MB)")
            return
        print(f"Removing incomplete file ({size_mb:.1f} MB): {output_path}")
        output_path.unlink()

    print(f"Downloading {URL}")
    print(f" -> {output_path}")

    ctx = ssl.create_default_context()

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100.0, done * 100.0 / total_size)
        print(f"\r  {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(URL, str(output_path), reporthook=progress)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("Install requests and retry, or download manually:")
        print(f"  {URL}")
        sys.exit(1)

    print()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
