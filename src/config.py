# Paths
# CLIPS_DIR = "../data/clips"
# CVAT_DIR = "../data/CVAT_Style_labels"
# OUTPUT_DIR = "../out"
CLIPS_DIR = "/content/drive/Othercomputers/Doug Desktop/ms-aai-521-final-project/data/clips"
CVAT_DIR = "/content/drive/Othercomputers/Doug Desktop/ms-aai-521-final-project/data/CVAT_Style_labels"
OUTPUT_DIR = "/content/drive/Othercomputers/Doug Desktop/ms-aai-521-final-project/out"
FRAMES_DIR = f"{OUTPUT_DIR}/frames"

# Data Split
TRAIN_PROPORTION = 0.7
VALIDATION_PROPORTION = 0.15
TEST_PROPORTION = 0.15

# Preprocessing
TARGET_FPS = 30
TARGET_RESOLUTION = (1280, 720)

# YOLO
YOLO_MODEL = 'yolo11n.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_BATCH_SIZE = 16
