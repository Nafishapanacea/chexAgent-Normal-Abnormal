import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np

class ApplyCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):
      
        # Convert to grayscale
        img_gray = np.array(img.convert("L"))

        img_clahe = self.clahe.apply(img_gray)

        # Back to RGB (SigLIP expects 3 channels)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(img_rgb)

train_transforms = T.Compose([

    # Rotation ±10–15 degrees
    T.RandomRotation(degrees=15),

    # Translation ±10–15%
    T.RandomAffine(
        degrees=0,
        translate=(0.1, 0.15),
        scale=(0.9, 1.1)   # 3️⃣ Scaling / zoom
    ),

    # Brightness / contrast jitter
    T.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    
])