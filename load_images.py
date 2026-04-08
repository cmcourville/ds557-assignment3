#DS577 - Assignment 3: Step 3 - Load 10 ImageNet Images
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import urllib.request
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

# 10 diverse ImageNet sample images from EliSchwartz/imagenet-sample-images on GitHub
# Each file is one representative image per ImageNet class (public domain JPEG)
# Class IDs selected for variety: animals, objects, vehicles, instruments, food
BASE = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"
IMAGE_URLS = [
    ("tench",           f"{BASE}/n01440764_tench.JPEG"),           # fish
    ("goldfish",        f"{BASE}/n01443537_goldfish.JPEG"),         # fish
    ("snail",           f"{BASE}/n01944390_snail.JPEG"),           # invertebrate
    ("tusker",          f"{BASE}/n02504458_African_elephant.JPEG"),# elephant
    ("tabby_cat",       f"{BASE}/n02123045_tabby.JPEG"),           # cat
    ("sports_car",      f"{BASE}/n04285008_sports_car.JPEG"),      # vehicle
    ("acoustic_guitar", f"{BASE}/n02676566_acoustic_guitar.JPEG"), # instrument
    ("banana",          f"{BASE}/n07753592_banana.JPEG"),          # food
    ("mushroom",        f"{BASE}/n13037406_gyromitra.JPEG"),       # fungus
    ("volcano",         f"{BASE}/n09472597_volcano.JPEG"),         # landscape
]

IMAGE_DIR = "images"
IMAGE_SIZE = (224, 224)


def download_images():
    """
    Download images using keras.utils.get_file which handles caching,
    redirects, and TF CDN authentication correctly.
    Images are saved to the images/ directory.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"Downloading {len(IMAGE_URLS)} images to '{IMAGE_DIR}/'...\n")

    for name, url in IMAGE_URLS:
        ext = url.split(".")[-1].split("?")[0]
        if ext not in ("jpg", "jpeg", "png"):
            ext = "jpg"
        dest = os.path.join(IMAGE_DIR, f"{name}.{ext}")

        if os.path.exists(dest):
            print(f"  [cached]  {name}")
            continue

        try:
            cached = keras.utils.get_file(
                fname=f"{name}.{ext}",
                origin=url,
                cache_dir=IMAGE_DIR,
                cache_subdir="."
            )
            # get_file may cache to its own dir — copy to images/ if needed
            if os.path.abspath(cached) != os.path.abspath(dest):
                import shutil
                shutil.copy(cached, dest)
            print(f"  [ok]      {name}")
        except Exception as e:
            print(f"  [FAILED]  {name}: {e}")


def load_images():
    """
    Load, resize, and return images as a list of (name, numpy_array) tuples.
    Images are float32 arrays of shape (224, 224, 3) in [0, 255] range.
    """
    images = []
    for name, url in IMAGE_URLS:
        ext = url.split(".")[-1].split("?")[0]
        if ext not in ("jpg", "jpeg", "png"):
            ext = "jpg"
        filepath = os.path.join(IMAGE_DIR, f"{name}.{ext}")

        if not os.path.exists(filepath):
            print(f"  [MISSING] {name} — skipping")
            continue

        img = load_img(filepath, target_size=IMAGE_SIZE)
        arr = img_to_array(img).astype(np.float32)  # shape: (224, 224, 3), range [0, 255]
        images.append((name, arr))

    return images


def verify_images(images, model):
    """Run ResNet50 predictions on all images and print results."""
    print(f"\n{'Image':<20} {'Top Prediction':<30} {'Confidence':>10}")
    print("-" * 62)
    for name, img in images:
        preds = model.predict(img[np.newaxis, ...], verbose=0)
        top = decode_predictions(preds, top=1)[0][0]
        print(f"  {name:<18} {top[1]:<30} {top[2]:>9.2%}")


def main():
    print("=" * 60)
    print("DS577 Assignment 3 - Step 3: Load ImageNet Images")
    print("=" * 60)

    # Download images
    download_images()

    # Load into numpy arrays
    images = load_images()
    print(f"\nLoaded {len(images)} images successfully.")

    # Load model (with preprocessing embedded)
    print("\nLoading ResNet50 for verification...")
    resnet = ResNet50(weights="imagenet")
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Lambda(preprocess_input)(inputs)
    outputs = resnet(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_with_preprocessing")

    # Verify predictions
    print("\nVerifying predictions on all 10 images:")
    verify_images(images, model)

    # Save as numpy array for use in the main attack script
    names = [n for n, _ in images]
    arrays = np.stack([a for _, a in images])  # shape: (10, 224, 224, 3)
    np.save(os.path.join(IMAGE_DIR, "images.npy"), arrays)
    np.save(os.path.join(IMAGE_DIR, "names.npy"), np.array(names))
    print(f"\nSaved arrays to '{IMAGE_DIR}/images.npy' and '{IMAGE_DIR}/names.npy'")
    print(f"Array shape: {arrays.shape}, dtype: {arrays.dtype}")
    print("=" * 60)


if __name__ == "__main__":
    main()
