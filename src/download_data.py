"""
Download the SkillCorner open data tracking files into data/raw/<match_id>/.

The tracking files are about 90 MB each (1.8 GB total), too large to keep in this
repository, so they are fetched from SkillCorner's public GitHub repo. They are stored
there with Git LFS, so this uses GitHub's media URL, which serves the real file
instead of the LFS pointer.

Match metadata and phases of play files are small and already included in data/raw/.

Usage:
    python src/download_data.py            # all matches found in data/raw/
    python src/download_data.py 1874553    # one or more specific matches
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(match_id: str):
    dest = RAW / match_id / f"{match_id}_tracking_extrapolated.jsonl"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"{match_id}: already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{match_id}/{dest.name}"
    print(f"{match_id}: downloading ...", end=" ", flush=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"{dest.stat().st_size / 1e6:.0f} MB")


def main():
    ids = sys.argv[1:] or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    for m in ids:
        download(m)


if __name__ == "__main__":
    main()
