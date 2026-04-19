#!/usr/bin/env python3

import os
from io import StringIO

import pandas as pd
import requests

from config import dance_scenes_dir, lyra_dir, fps


def video_dancing():
    """
    Download the Lyra dataset metadata and return the videos labeled as dancing.

    Returns
    -------
    list of str
        List of video identifiers for which the ``is-danced`` flag is equal to 1.
    """
    url = (
        "https://raw.githubusercontent.com/pxaris/lyra-dataset/"
        "refs/heads/main/data/raw.tsv"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    tsv_data = StringIO(response.text)
    df = pd.read_csv(tsv_data, delimiter="\t")

    filtered_df = df[df["is-danced"] == 1]
    danced_vids = filtered_df["id"].astype(str).tolist()

    return danced_vids


def get_vids_server():
    """
    Return all available Lyra video filenames stored locally.

    Returns
    -------
    list of str
        Sorted list of video filenames ending in ``.mp4``.
    """
    mp4_files = []

    for file in os.listdir(lyra_dir):
        if file.endswith(".mp4"):
            mp4_files.append(file)

    mp4_files.sort()
    return mp4_files


def read_scene_frames(name):
    """
    Read dance-scene boundaries from a text file and return frame indices.

    Parameters
    ----------
    name : str
        Video identifier without the ``.mp4`` suffix.

    Returns
    -------
    tuple of list
        Two parallel lists:
        - ``starts``: start frame indices
        - ``ends``: end frame indices

    Notes
    -----
    The function supports two filename conventions:
    - ``results_of_video_'{name}.mp4'.txt``
    - ``results_of_video_{name}.mp4.txt``

    Each valid line is expected to have the format:

        start-end
    """
    base1 = os.path.join(dance_scenes_dir, f"results_of_video_'{name}.mp4'.txt")
    base2 = os.path.join(dance_scenes_dir, f"results_of_video_{name}.mp4.txt")

    if os.path.exists(base1):
        path = base1
    elif os.path.exists(base2):
        path = base2
    else:
        raise FileNotFoundError(f"No scene file found for video: {name}")

    starts, ends = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or "-" not in line:
                continue

            start, end = line.split("-", 1)
            starts.append(int(start.strip()))
            ends.append(int(end.strip()))

    return starts, ends


def read_scene_seconds(name):
    """
    Compatibility helper: return scene boundaries in seconds.

    Parameters
    ----------
    name : str
        Video identifier without the ``.mp4`` suffix.

    Returns
    -------
    tuple of list
        Two parallel lists:
        - ``starts_sec``: start times in seconds
        - ``ends_sec``: end times in seconds
    """
    start_frames, end_frames = read_scene_frames(name)
    starts_sec = [frame / fps for frame in start_frames]
    ends_sec = [frame / fps for frame in end_frames]
    return starts_sec, ends_sec


def prop_type(tracking_data):
    """
    Convert tracking-data columns to numeric types.

    Parameters
    ----------
    tracking_data : pandas.DataFrame
        DataFrame containing tracking results.

    Returns
    -------
    pandas.DataFrame
        DataFrame with selected columns converted to numeric types.
    """
    tracking_data["id"] = pd.to_numeric(
        tracking_data["id"], errors="coerce", downcast="integer"
    )
    tracking_data["frame"] = pd.to_numeric(
        tracking_data["frame"], errors="coerce", downcast="integer"
    )
    tracking_data["x"] = pd.to_numeric(
        tracking_data["x"], errors="coerce", downcast="float"
    )
    tracking_data["y"] = pd.to_numeric(
        tracking_data["y"], errors="coerce", downcast="float"
    )
    tracking_data["w"] = pd.to_numeric(
        tracking_data["w"], errors="coerce", downcast="float"
    )
    tracking_data["h"] = pd.to_numeric(
        tracking_data["h"], errors="coerce", downcast="float"
    )

    return tracking_data
