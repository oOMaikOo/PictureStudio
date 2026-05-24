"""
Application-wide configuration constants: colours, image formats, and defaults
for training and SSH remote-training. Imported by most modules.
"""

APP_NAME = "Picture Studio"
APP_VERSION = "2.5.0-beta"

DEFAULT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#FF5722", "#607D8B", "#795548", "#4CAF50", "#2196F3",
]

IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

DEFAULT_TRAIN_CONFIG = {
    "model_type": "resnet18",
    "image_size": 224,
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "seed": 42,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
    "use_pretrained": True,
    "augmentation": {
        "rotation": True,
        "flip": True,
        "brightness": True,
        "contrast": True,
        "scale": False,
    },
}

DEFAULT_SSH_CONFIG = {
    "host": "",
    "username": "",
    "port": 22,
    "key_path": "",
    "password": "",
    "remote_path": "/tmp/image_labeling_project",
    "python_env": "python3",
}

MIN_IMAGES_PER_CLASS = 5   # classes with fewer samples trigger a warning
THUMBNAIL_SIZE = (120, 90)  # (width, height) in pixels for the thumbnail list
