import os

import cv2
import numpy as np

from extract_skeletons.config import ALPHAPOSE_OUTPUT_DIR


def enhance_frame(cropped_frame):
    """
    Enhance an image crop through denoising and sharpening.

    This function applies a two-step enhancement pipeline to an input image:
    first denoising, then spatial sharpening. It is intended to improve the
    visual quality of cropped person regions before downstream pose estimation
    or storage.

    Parameters
    ----------
    cropped_frame : numpy.ndarray
        Input image in OpenCV format, typically a BGR image array of shape
        ``(H, W, C)``.

    Returns
    -------
    numpy.ndarray
        Enhanced image after denoising and sharpening.

    Notes
    -----
    The enhancement pipeline consists of:
    1. non-local means color denoising,
    2. convolution-based sharpening using a fixed kernel.
    """
    cropped_frame = cv2.fastNlMeansDenoisingColored(
        cropped_frame, None, 10, 10, 7, 21
    )

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    cropped_frame = cv2.filter2D(cropped_frame, -1, kernel)

    return cropped_frame


def crop_frame(frame, bbox, expand_px=20):
    """
    Crop a frame around a bounding box with optional padding.

    The bounding box is expanded by a fixed number of pixels in all directions
    and then clamped to the image boundaries. The resulting cropped region is
    returned as an image array.

    Parameters
    ----------
    frame : object or numpy.ndarray
        Input frame. In the current implementation, the image data are accessed
        through ``frame.img``.

    bbox : tuple
        Bounding box in the format ``(x, y, w, h)``.

    expand_px : int, optional
        Number of pixels used to expand the bounding box on each side.
        Default is 20.

    Returns
    -------
    numpy.ndarray
        Cropped image region corresponding to the expanded bounding box.

    Raises
    ------
    ValueError
        If the resulting crop is empty.

    Notes
    -----
    The frame is enhanced before cropping by calling ``enhance_frame()``.
    """
    enhanced_frame = enhance_frame(frame.img)

    x, y, w, h = bbox
    img_h, img_w = enhanced_frame.shape[:2]

    x1 = max(0, x - expand_px)
    y1 = max(0, y - expand_px)
    x2 = min(img_w, x + w + expand_px)
    y2 = min(img_h, y + h + expand_px)

    crop = enhanced_frame[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop: {(x1, y1, x2, y2)} on image shape {frame.shape}"
        )

    return crop


def store_frames(crops, name, segment_idx, fps=25):
    """
    Store a sequence of cropped frames as an MP4 video.

    This function writes a list of cropped image arrays to a single MP4 file.
    All frames are resized, if necessary, to match the spatial resolution of
    the first crop.

    Parameters
    ----------
    crops : list of numpy.ndarray
        List of cropped image arrays, typically one crop per frame.

    name : str
        Name of the sample, video, or sequence used to define the output path.

    segment_idx : int or str
        Segment identifier used to create the output subdirectory.

    fps : int or float, optional
        Frame rate of the output video. Default is 25.

    Returns
    -------
    str
        Path to the generated MP4 file.

    Raises
    ------
    ValueError
        If the input crop list is empty.

    Notes
    -----
    The output video is stored at:

        ALPHAPOSE_OUTPUT_DIR / name / str(segment_idx) / "crop_video.mp4"

    Grayscale frames are converted to BGR before writing.
    """
    out_dir = os.path.join(ALPHAPOSE_OUTPUT_DIR, name, str(segment_idx))
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "crop_video.mp4")

    if not crops:
        raise ValueError("No crops to write.")

    h, w = crops[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for img in crops:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        writer.write(img)

    writer.release()

    return video_path
