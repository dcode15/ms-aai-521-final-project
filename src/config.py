# Paths
CLIPS_DIR = "../data/clips"
CVAT_DIR = "../data/CVAT_Style_labels"
OUTPUT_DIR = "D:/Repos/ms-aai-521-final-project/out"
# CLIPS_DIR = "/content/drive/Othercomputers/Desktop/data/clips"
# CVAT_DIR = "/content/drive/Othercomputers/Desktop/data/CVAT_Style_labels"
# OUTPUT_DIR = "/content/drive/MyDrive/AAI-521/out"

# Data Split
TRAIN_PROPORTION = 0.6
VALIDATION_PROPORTION = 0.20
TEST_PROPORTION = 0.20

# Preprocessing
TARGET_FPS = 30
TARGET_RESOLUTION = (1280, 720)

# YOLO Model
YOLO_MODEL = 'yolo11m.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.478
YOLO_BATCH_SIZE = 8
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 8
TRAIN_IMAGE_SIZE = 640
TRAIN_WARMUP_EPOCHS = 3
TRAIN_SAVE_PERIOD = 10
TRAIN_PATIENCE = 3

# Tracking
TRACK_BUFFER = 10
MATCH_THRESH = 0.571

# Model Evaluation
EVAL_BATCH_SIZE = 16

# Visualization
VIZ_BOX_THICKNESS = 2
VIZ_FONT_SCALE = 0.5
VIZ_DASH_LENGTH = 10
VIZ_FPS = 30
VIZ_CODEC = 'mp4v'

# Tuning
TUNING_TRIALS = 30
