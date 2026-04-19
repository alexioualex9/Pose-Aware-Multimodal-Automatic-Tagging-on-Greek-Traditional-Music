from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def fetch_danced_video_ids(url: str) -> set[str]:
    """Download metadata and return the IDs of videos annotated as danced."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), sep="\t")
    danced_ids = df.loc[df["is-danced"] == 1, "id"].astype(str)

    return set(danced_ids.tolist())


def list_video_files(directory: Path, extension: str = ".mp4") -> list[Path]:
    """Return sorted video files from a directory."""
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == extension
    )