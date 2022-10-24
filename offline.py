from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xyz.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xyz.npy
        np.save(feature_path, feature)

