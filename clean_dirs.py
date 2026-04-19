import os
import shutil

from config import (
    ALPHAPOSE_OUTPUT_DIR,
    ALPHAPOSE_DIR,
    BYTETRACK_DIR,
    frames_dir,
)


def clean_directories(directories):
    """
    Remove selected contents from a list of directories.

    For each directory in the input list, the function removes:
    - all ``.mp4`` files,
    - all ``.txt`` files,
    - all subdirectories recursively.

    Other file types are left untouched.

    Parameters
    ----------
    directories : list of str
        List of directory paths to be cleaned.

    Notes
    -----
    If a directory does not exist, it is skipped and a message is printed.
    Errors raised during file or folder deletion are caught and reported.
    """
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if os.path.isfile(item_path) and item_path.endswith(".mp4"):
                try:
                    os.remove(item_path)
                except Exception as e:
                    print(f"Error deleting file {item_path}: {e}")

            if os.path.isfile(item_path) and item_path.endswith(".txt"):
                try:
                    os.remove(item_path)
                except Exception as e:
                    print(f"Error deleting file {item_path}: {e}")

            elif os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error deleting folder {item_path}: {e}")


def delete_files(name):
    """
    Delete generated files associated with a specific sample or video name.

    Parameters
    ----------
    name : str
        Name of the target subdirectory under ``ALPHAPOSE_OUTPUT_DIR``.
    """
    directories_to_check = [
        os.path.join(ALPHAPOSE_OUTPUT_DIR, name),
    ]

    clean_directories(directories_to_check)


def clean_except(directory, items_to_keep):
    """
    Remove all items from a directory except for a specified subset.

    Parameters
    ----------
    directory : str
        Path to the directory to be cleaned.

    items_to_keep : list of str
        List of file or folder names to preserve. Only item names should be
        provided, not full paths.

    Notes
    -----
    If the target directory does not exist, the function returns immediately
    after printing a message.
    """
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if item in items_to_keep:
            continue

        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
            except Exception as e:
                print(f"Error deleting file {item_path}: {e}")

        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error deleting folder {item_path}: {e}")


def delete_folders():
    """
    Delete intermediate folders and generated files used during preprocessing.

    This function clears selected working directories, including temporary frame
    folders and ByteTrack output directories. It also cleans the AlphaPose
    output directory while preserving the ``poseflow`` subdirectory.
    """
    directories_to_check = [
        frames_dir,
#        os.path.join(ALPHAPOSE_OUTPUT_DIR, "btrack/frames"),
        os.path.join(BYTETRACK_DIR, "YOLOX_outputs/yolox_x_mix_det/track_vis"),
    ]

    clean_directories(directories_to_check)

    directory_to_clean = ALPHAPOSE_OUTPUT_DIR
    items_to_keep = ["poseflow"]

    clean_except(directory_to_clean, items_to_keep)


def delete_folders2():
    """
    Delete trimmed videos and stored cropped-frame data.

    This function removes intermediate data generated during the processing
    pipeline, including:
    - trimmed video files,
    - stored frame folders,
    - AlphaPose example frame outputs,
    - ByteTrack visualization outputs.
    """
    directories_to_check = [
        frames_dir,
        os.path.join(ALPHAPOSE_OUTPUT_DIR, "frames"),
        os.path.join(ALPHAPOSE_DIR, "examples/frames"),
        os.path.join(BYTETRACK_DIR, "YOLOX_outputs/yolox_x_mix_det/track_vis"),
    ]

    clean_directories(directories_to_check)
