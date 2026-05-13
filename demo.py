import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# foolbox 2.4.0 was built for TF1 graph mode — disable eager execution
tf.compat.v1.disable_eager_execution()

import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import foolbox


def load_resnet50():
    print("Loading ResNet50 with ImageNet weights...")
    resnet = ResNet50(weights="imagenet")
    resnet.trainable = False

    # Embed preprocessing so foolbox receives raw [0,255] images
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Lambda(preprocess_input)(inputs)
    outputs = resnet(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_with_preprocessing")

    print(f"Model loaded: {model.name} | Input shape: {model.input_shape}")
    return model


def wrap_with_foolbox(keras_model):
    print("Wrapping model with foolbox KerasModel...")
    fmodel = foolbox.models.KerasModel(
        keras_model,
        bounds=(0, 255),
        preprocessing=(0, 1)
    )
    print("Model wrapped successfully.")
    return fmodel


def get_test_image():
    print("Generating random test image (224x224x3)...")
    np.random.seed(42)
    return np.random.randint(0, 256, size=(224, 224, 3)).astype(np.float32)


def get_prediction(keras_model, image):
    preds = keras_model.predict(image[np.newaxis, ...], verbose=0)
    return decode_predictions(preds, top=3)[0]


def run_fgsm_attack(fmodel, image, label):
    print("Running FGSM attack...")
    attack = foolbox.v1.attacks.FGSM(fmodel)
    return attack(image, label)


def main():
    keras_model = load_resnet50()
    fmodel = wrap_with_foolbox(keras_model)
    image = get_test_image()

    print("\nOriginal image predictions:")
    original_preds = get_prediction(keras_model, image)
    for rank, (id_, label, prob) in enumerate(original_preds, 1):
        print(f"  {rank}. {label:25s} ({prob:.4f})")

    original_label = int(np.argmax(
        keras_model.predict(image[np.newaxis, ...], verbose=0)
    ))
    print(f"\nUsing class index {original_label} for attack.")

    adversarial = run_fgsm_attack(fmodel, image, original_label)

    if adversarial is None:
        print("\nAttack failed to generate an adversarial example.")
    else:
        print("\nAdversarial image predictions:")
        adv_preds = get_prediction(keras_model, adversarial)
        for rank, (id_, label, prob) in enumerate(adv_preds, 1):
            print(f"  {rank}. {label:25s} ({prob:.4f})")

        noise = adversarial - image
        print(f"\nNoise  min={noise.min():.4f} | max={noise.max():.4f} | "
              f"mean abs={np.abs(noise).mean():.4f}")

        orig_class = original_preds[0][1]
        adv_class  = adv_preds[0][1]
        if orig_class != adv_class:
            print(f"\nAttack SUCCESS: '{orig_class}' -> '{adv_class}'")
        else:
            print(f"\nTop class unchanged ('{orig_class}') — adversarial example generated.")

    print("\nDemo complete. Environment verified.")


if __name__ == "__main__":
    main()
