import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import shutil

from config import IMAGE_DIR, IMAGE_SIZE, IMAGE_URLS


def download_images():
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
            if os.path.abspath(cached) != os.path.abspath(dest):
                shutil.copy(cached, dest)
            print(f"  [ok]      {name}")
        except Exception as e:
            print(f"  [FAILED]  {name}: {e}")


def load_images():
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
        arr = img_to_array(img).astype(np.float32)
        images.append((name, arr))

    return images


def verify_images(images, model):
    print(f"\n{'Image':<20} {'Top Prediction':<30} {'Confidence':>10}")
    print("-" * 62)
    for name, img in images:
        preds = model.predict(img[np.newaxis, ...], verbose=0)
        top = decode_predictions(preds, top=1)[0][0]
        print(f"  {name:<18} {top[1]:<30} {top[2]:>9.2%}")


def main():
    download_images()

    images = load_images()
    print(f"\nLoaded {len(images)} images successfully.")

    print("\nLoading ResNet50 for verification...")
    resnet = ResNet50(weights="imagenet")
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Lambda(preprocess_input)(inputs)
    outputs = resnet(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_with_preprocessing")

    print("\nVerifying predictions on all images:")
    verify_images(images, model)

    names = [n for n, _ in images]
    arrays = np.stack([a for _, a in images])
    np.save(os.path.join(IMAGE_DIR, "images.npy"), arrays)
    np.save(os.path.join(IMAGE_DIR, "names.npy"), np.array(names))
    print(f"\nSaved arrays to '{IMAGE_DIR}/images.npy' and '{IMAGE_DIR}/names.npy'")
    print(f"Array shape: {arrays.shape}, dtype: {arrays.dtype}")


if __name__ == "__main__":
    main()
