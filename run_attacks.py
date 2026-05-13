import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import foolbox

from config import IMAGE_DIR, RESULTS_DIR, ATTACKS as ATTACK_CONFIGS


def build_model():
    resnet = ResNet50(weights="imagenet")
    resnet.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Lambda(preprocess_input)(inputs)
    outputs = resnet(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="resnet50_with_preprocessing")


def wrap_foolbox(keras_model):
    return foolbox.models.KerasModel(keras_model, bounds=(0, 255), preprocessing=(0, 1))


def predict_label_index(keras_model, image):
    preds = keras_model.predict(image[np.newaxis, ...], verbose=0)
    return int(np.argmax(preds))


def predict_class_name(keras_model, image):
    preds = keras_model.predict(image[np.newaxis, ...], verbose=0)
    return decode_predictions(preds, top=1)[0][0][1]


def get_attacks(fmodel):
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


def run_attack(attack_instance, attack_name, image, label):
    try:
        if attack_name == "SinglePixelAttack":
            adversarial = attack_instance(image, label, max_pixels=1000)
        elif attack_name == "SaliencyMapAttack":
            adversarial = attack_instance(image, label, max_iter=2000, fast=True)
        elif attack_name == "BlendedUniformNoiseAttack":
            adversarial = attack_instance(image, label, epsilons=1000, max_directions=1000)
        elif attack_name == "ContrastReductionAttack":
            adversarial = attack_instance(image, label, epsilons=1000)
        else:
            adversarial = attack_instance(image, label)
        return adversarial
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def run_all_attacks(images, names, keras_model, fmodel):
    attacks = get_attacks(fmodel)
    results = []

    for img_idx, (name, image) in enumerate(zip(names, images)):
        original_class = predict_class_name(keras_model, image)
        label_idx = predict_label_index(keras_model, image)

        print(f"\n[{img_idx+1}/{len(names)}] {name} (predicted: {original_class})")

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
                "image_name":        name,
                "attack_name":       atk_name,
                "attack_params":     atk_params,
                "original":          image,
                "adversarial":       adversarial,
                "noise":             noise,
                "original_class":    original_class,
                "adversarial_class": adv_class,
                "success":           success,
            })

    return results


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for r in results:
        key = f"{r['image_name']}__{r['attack_name']}"
        np.save(os.path.join(RESULTS_DIR, f"{key}__original.npy"),      r["original"])
        np.save(os.path.join(RESULTS_DIR, f"{key}__attack_params.npy"), np.array(r["attack_params"]))
        np.save(os.path.join(RESULTS_DIR, f"{key}__original_class.npy"), np.array(r["original_class"]))

        if r["adversarial"] is not None:
            np.save(os.path.join(RESULTS_DIR, f"{key}__adversarial.npy"), r["adversarial"])
            np.save(os.path.join(RESULTS_DIR, f"{key}__noise.npy"),       r["noise"])
            np.save(os.path.join(RESULTS_DIR, f"{key}__adv_class.npy"),   np.array(r["adversarial_class"]))

    print(f"\nSaved {len(results)} result sets to '{RESULTS_DIR}/'")


def print_summary(results):
    attack_names = [a[0] for a in ATTACK_CONFIGS]

    print("\n" + "=" * 70)
    print("ATTACK SUMMARY")
    print("=" * 70)
    print(f"  {'Image':<20}", end="")
    for a in attack_names:
        print(f" {a[:6]:>6}", end="")
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


def main():
    images_path = os.path.join(IMAGE_DIR, "images.npy")
    names_path  = os.path.join(IMAGE_DIR, "names.npy")
    if not os.path.exists(images_path):
        print(f"ERROR: '{images_path}' not found. Run load_images.py first.")
        return

    images = np.load(images_path)
    names  = np.load(names_path).tolist()
    print(f"Loaded {len(images)} images: {names}")

    print("\nLoading model...")
    keras_model = build_model()
    fmodel = wrap_foolbox(keras_model)

    n_total = len(images) * len(ATTACK_CONFIGS)
    print(f"\nRunning attacks ({len(images)} images x {len(ATTACK_CONFIGS)} attacks = {n_total} total)...")
    results = run_all_attacks(images, names, keras_model, fmodel)

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
