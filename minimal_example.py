#DS577 - Assignment 3: Foolbox Minimal Working Example
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
    """
    Load ResNet50 pretrained on ImageNet and wrap it with a preprocessing layer.
    The returned model accepts raw [0, 255] pixel values and applies
    ResNet50's preprocess_input internally — required so foolbox can pass
    unmodified images within its declared bounds=(0, 255).
    """
    print("Loading ResNet50 with ImageNet weights...")
    resnet = ResNet50(weights="imagenet")
    resnet.trainable = False

    # Embed preprocessing inside the model so foolbox receives raw [0,255] images
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Lambda(preprocess_input)(inputs)
    outputs = resnet(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="resnet50_with_preprocessing")

    print(f"Model loaded: {model.name} | Input shape: {model.input_shape}")
    return model


def wrap_with_foolbox(keras_model):
    """
    Wrap a Keras model with foolbox's KerasModel.
    bounds=(0, 255) tells foolbox the valid pixel range.
    preprocessing=(0, 1) is a no-op (mean=0, std=1).
    ResNet50's preprocess_input normalization is applied separately.
    """
    print("Wrapping model with foolbox KerasModel...")
    fmodel = foolbox.models.KerasModel(
        keras_model,
        bounds=(0, 255),
        preprocessing=(0, 1)
    )
    print("Model wrapped successfully.")
    return fmodel


def get_test_image():
    """
    Generate a random RGB image in ResNet50's expected input shape (224, 224, 3).
    Values in [0, 255] float32 range, as required by foolbox bounds.
    """
    print("Generating random test image (224x224x3)...")
    np.random.seed(42)
    image = np.random.randint(0, 256, size=(224, 224, 3)).astype(np.float32)
    return image


def get_prediction(keras_model, image):
    """
    Run inference on a raw [0, 255] image and return top-3 predicted classes.
    Preprocessing is handled internally by the model's Lambda layer.
    """
    preds = keras_model.predict(image[np.newaxis, ...], verbose=0)
    return decode_predictions(preds, top=3)[0]


def run_fgsm_attack(fmodel, image, label):
    """
    Run FGSM (Fast Gradient Sign Method) attack using foolbox.
    FGSM computes the gradient of the loss w.r.t. the input and perturbs
    the image by a small epsilon in the direction that maximizes the loss,
    causing the classifier to output a wrong prediction.
    """
    print("Running FGSM attack...")
    attack = foolbox.v1.attacks.FGSM(fmodel)
    adversarial = attack(image, label)
    return adversarial


def main():
    print("=" * 60)
    print("DS577 Assignment 3 - Foolbox Minimal Example")
    print("=" * 60)

    # Step 1: Load ResNet50
    keras_model = load_resnet50()

    # Step 2: Wrap with foolbox
    fmodel = wrap_with_foolbox(keras_model)

    # Step 3: Generate a random test image
    image = get_test_image()

    # Step 4: Get original prediction
    print("\nOriginal image predictions:")
    original_preds = get_prediction(keras_model, image)
    for rank, (id_, label, prob) in enumerate(original_preds, 1):
        print(f"  {rank}. {label:25s} ({prob:.4f})")

    # foolbox v1 expects integer class index as label
    original_label = int(np.argmax(
        keras_model.predict(image[np.newaxis, ...], verbose=0)
    ))
    print(f"\nUsing class index {original_label} for attack.")

    # Step 5: Run FGSM attack
    adversarial = run_fgsm_attack(fmodel, image, original_label)

    # Step 6: Evaluate result
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
        adv_class = adv_preds[0][1]
        if orig_class != adv_class:
            print(f"\nAttack SUCCESS: '{orig_class}' -> '{adv_class}'")
        else:
            print(f"\nTop class unchanged ('{orig_class}') but adversarial "
                  "example was generated.")

    print("\nMinimal example complete. Environment verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
