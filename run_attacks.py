#DS577 - Assignment 3: Step 4 - Run All 5 Attacks on 10 ImageNet Images
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import foolbox

IMAGE_DIR = "images"
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    """ResNet50 with preprocessing embedded so foolbox sees raw [0,255] input."""
    resnet = ResNet50(weights="imagenet")
    resnet.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Lambda(preprocess_input)(inputs)
    outputs = resnet(x)
    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="resnet50_with_preprocessing")
    return model


def wrap_foolbox(keras_model):
    return foolbox.models.KerasModel(keras_model, bounds=(0, 255),
                                     preprocessing=(0, 1))


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_label_index(keras_model, image):
    """Return the integer class index of the top prediction."""
    preds = keras_model.predict(image[np.newaxis, ...], verbose=0)
    return int(np.argmax(preds))


def predict_class_name(keras_model, image):
    """Return the human-readable class name of the top prediction."""
    preds = keras_model.predict(image[np.newaxis, ...], verbose=0)
    return decode_predictions(preds, top=1)[0][0][1]


# ---------------------------------------------------------------------------
# Attack definitions
# ---------------------------------------------------------------------------

def get_attacks(fmodel):
    """
    Return the 5 required attacks as (name, attack_instance, params_str) tuples.
    SinglePixelAttack and SaliencyMapAttack require foolbox 2.4.0.
    BlendedUniformNoiseAttack, ContrastReductionAttack, and FGSM work on
    any foolbox 2.x version.
    """
    return [
        (
            "BlendedUniformNoiseAttack",
            foolbox.v1.attacks.BlendedUniformNoiseAttack(fmodel),
            "epsilons=1000, max_directions=1000"
        ),
        (
            "ContrastReductionAttack",
            foolbox.v1.attacks.ContrastReductionAttack(fmodel),
            "epsilons=1000"
        ),
        (
            "FGSM",
            foolbox.v1.attacks.FGSM(fmodel),
            "default epsilon"
        ),
        (
            "SinglePixelAttack",
            foolbox.v1.attacks.SinglePixelAttack(fmodel),
            "max_pixels=1000"
        ),
        (
            "SaliencyMapAttack",
            foolbox.v1.attacks.SaliencyMapAttack(fmodel),
            "max_iter=2000, fast=True"
        ),
    ]


# ---------------------------------------------------------------------------
# Run attacks
# ---------------------------------------------------------------------------

def run_attack(attack_instance, attack_name, image, label):
    """
    Run a single attack and return the adversarial image.
    Returns None if the attack fails to find an adversarial example.
    """
    try:
        if attack_name == "SinglePixelAttack":
            adversarial = attack_instance(image, label, max_pixels=1000)
        elif attack_name == "SaliencyMapAttack":
            adversarial = attack_instance(image, label, max_iter=2000, fast=True)
        elif attack_name == "BlendedUniformNoiseAttack":
            adversarial = attack_instance(image, label, epsilons=1000,
                                          max_directions=1000)
        elif attack_name == "ContrastReductionAttack":
            adversarial = attack_instance(image, label, epsilons=1000)
        else:
            adversarial = attack_instance(image, label)
        return adversarial
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def run_all_attacks(images, names, keras_model, fmodel):
    """
    Run all 5 attacks on all 10 images.
    Returns a list of result dicts, one per (image, attack) combination.
    Each dict contains:
        image_name, attack_name, attack_params,
        original, adversarial, noise,
        original_class, adversarial_class, success
    """
    attacks = get_attacks(fmodel)
    results = []

    for img_idx, (name, image) in enumerate(zip(names, images)):
        original_class = predict_class_name(keras_model, image)
        label_idx = predict_label_index(keras_model, image)

        print(f"\n[{img_idx+1}/10] {name} (predicted: {original_class})")

        for atk_name, atk_instance, atk_params in attacks:
            print(f"  -> {atk_name} ...", end=" ", flush=True)

            adversarial = run_attack(atk_instance, atk_name, image, label_idx)

            if adversarial is None:
                adv_class = None
                noise = None
                success = False
                print("FAILED")
            else:
                adv_class = predict_class_name(keras_model, adversarial)
                noise = adversarial - image
                success = (adv_class != original_class)
                status = f"SUCCESS ({original_class} -> {adv_class})" if success else f"no change ({adv_class})"
                print(status)

            results.append({
                "image_name":       name,
                "attack_name":      atk_name,
                "attack_params":    atk_params,
                "original":         image,
                "adversarial":      adversarial,
                "noise":            noise,
                "original_class":   original_class,
                "adversarial_class": adv_class,
                "success":          success,
            })

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results):
    """Save attack results to disk as .npy files for use in Step 5."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for r in results:
        key = f"{r['image_name']}__{r['attack_name']}"
        np.save(os.path.join(RESULTS_DIR, f"{key}__original.npy"),    r["original"])
        np.save(os.path.join(RESULTS_DIR, f"{key}__attack_params.npy"), np.array(r["attack_params"]))
        np.save(os.path.join(RESULTS_DIR, f"{key}__original_class.npy"), np.array(r["original_class"]))

        if r["adversarial"] is not None:
            np.save(os.path.join(RESULTS_DIR, f"{key}__adversarial.npy"), r["adversarial"])
            np.save(os.path.join(RESULTS_DIR, f"{key}__noise.npy"),       r["noise"])
            np.save(os.path.join(RESULTS_DIR, f"{key}__adv_class.npy"),   np.array(r["adversarial_class"]))

    print(f"\nSaved {len(results)} result sets to '{RESULTS_DIR}/'")


def print_summary(results):
    """Print a success/failure summary table."""
    attack_names = ["BlendedUniformNoiseAttack", "ContrastReductionAttack",
                    "FGSM", "SinglePixelAttack", "SaliencyMapAttack"]

    print("\n" + "=" * 70)
    print("ATTACK SUMMARY")
    print("=" * 70)
    print(f"  {'Image':<20}", end="")
    for a in attack_names:
        short = a[:6]
        print(f" {short:>6}", end="")
    print()
    print("-" * 70)

    image_names = list(dict.fromkeys(r["image_name"] for r in results))
    for name in image_names:
        print(f"  {name:<20}", end="")
        for a in attack_names:
            match = next((r for r in results
                          if r["image_name"] == name and r["attack_name"] == a), None)
            if match is None:
                print(f"  {'?':>5}", end="")
            elif match["adversarial"] is None:
                print(f"  {'FAIL':>5}", end="")
            elif match["success"]:
                print(f"  {'OK':>5}", end="")
            else:
                print(f"  {'~':>5}", end="")
        print()

    total = len(results)
    succeeded = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if r["adversarial"] is None)
    no_change = total - succeeded - failed
    print("-" * 70)
    print(f"  Total: {total} | Success: {succeeded} | No change: {no_change} | Failed: {failed}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DS577 Assignment 3 - Step 4: Run All 5 Attacks on 10 Images")
    print("=" * 70)

    # Load saved images
    images_path = os.path.join(IMAGE_DIR, "images.npy")
    names_path  = os.path.join(IMAGE_DIR, "names.npy")
    if not os.path.exists(images_path):
        print(f"ERROR: '{images_path}' not found. Run load_images.py first.")
        return

    images = np.load(images_path)           # shape: (10, 224, 224, 3)
    names  = np.load(names_path).tolist()   # list of 10 strings
    print(f"Loaded {len(images)} images: {names}")

    # Build model and foolbox wrapper
    print("\nLoading model...")
    keras_model = build_model()
    fmodel = wrap_foolbox(keras_model)

    # Run all attacks
    print("\nRunning attacks (10 images x 5 attacks = 50 total)...")
    results = run_all_attacks(images, names, keras_model, fmodel)

    # Save and summarise
    save_results(results)
    print_summary(results)
    print("\nStep 4 complete. Run visualize.py next for Step 5.")


if __name__ == "__main__":
    main()
