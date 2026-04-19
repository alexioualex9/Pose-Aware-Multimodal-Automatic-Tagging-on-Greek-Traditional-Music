import os

dance_scenes_dir = "/../DanceScenes/results_dance_scenes"
lyra_dir = "/../lyra/videos"
frames_dir = "/../AlphaPose/examples/frames"
ALPHAPOSE_DIR = "/../AlphaPose"
ALPHAPOSE_OUTPUT_DIR = "/../AlphaPose/examples/results"
BYTETRACK_DIR = "/../ByteTrack"
A_EXPERIMENT_CONFIG = os.path.join(ALPHAPOSE_DIR, "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
B_EXPERIMENT_CONFIG = os.path.join(BYTETRACK_DIR, "exps/example/mot/yolox_x_mix_det.py")
A_CHECKPOINT = os.path.join(ALPHAPOSE_DIR, "pretrained_models/fast_res50_256x192.pth")
B_CHECKPOINT = os.path.join(BYTETRACK_DIR, "pretrained/bytetrack_x_mot17.pth.tar")
COLUMNS = ["frame","id","x","y","w","h","confidence","class","visibility","flag"]
fps = 25

