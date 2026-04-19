from decord import VideoReader, cpu
from dataclasses import dataclass


@dataclass
class Frame:
    """
    Lightweight container for a single video frame.
    """
    img: object = None
    index: int = -1


def read_frames(cur_video):
    """
    Read all frames from a video file and store them as ``Frame`` objects.
    """
    vr = VideoReader(cur_video, ctx=cpu(0))
    list_frames = []

    for index, frame in enumerate(vr):
        if index >= 1_000_000:
            break

        list_frames.append(Frame(img=frame.asnumpy(), index=index))

    return list_frames