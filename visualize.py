import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from config import RESULTS_DIR, IMAGE_NAMES, ATTACKS, DEFAULT_OUTPUT_PDF

ROWS_PER_PAGE = 5


def load_npy(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def load_result(image_name, attack_name):
    key  = f"{image_name}__{attack_name}"
    base = RESULTS_DIR

    original       = load_npy(os.path.join(base, f"{key}__original.npy"))
    adversarial    = load_npy(os.path.join(base, f"{key}__adversarial.npy"))
    noise          = load_npy(os.path.join(base, f"{key}__noise.npy"))
    original_class = load_npy(os.path.join(base, f"{key}__original_class.npy"))
    adv_class      = load_npy(os.path.join(base, f"{key}__adv_class.npy"))
    attack_params  = load_npy(os.path.join(base, f"{key}__attack_params.npy"))

    return {
        "original":          original,
        "adversarial":       adversarial,
        "noise":             noise,
        "original_class":    str(original_class) if original_class is not None else "unknown",
        "adversarial_class": str(adv_class) if adv_class is not None else None,
        "attack_params":     str(attack_params) if attack_params is not None else "",
    }


def to_display(image):
    return np.clip(image, 0, 255) / 255.0


def noise_to_display(noise):
    if noise is None:
        return np.ones((224, 224, 3)) * 0.5
    vmax = np.abs(noise).max()
    if vmax == 0:
        return np.ones_like(noise) * 0.5
    return np.clip((noise / (2 * vmax)) + 0.5, 0, 1)


def render_row(fig, outer_row, result, image_name, attack_name, attack_params):
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_row, wspace=0.05)

    original    = result["original"]
    adversarial = result["adversarial"]
    noise       = result["noise"]
    orig_class  = result["original_class"]
    adv_class   = result["adversarial_class"]

    ax0 = fig.add_subplot(inner[0])
    ax0.imshow(to_display(original))
    ax0.set_title(f"Original\n{orig_class}", fontsize=6, pad=2)
    ax0.axis("off")

    ax1 = fig.add_subplot(inner[1])
    if adversarial is not None:
        ax1.imshow(to_display(adversarial))
        ax1.set_title(f"Adversarial\n{adv_class}", fontsize=6, pad=2)
    else:
        ax1.imshow(np.ones((224, 224, 3)) * 0.85)
        ax1.text(0.5, 0.5, "Attack\nFailed", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=8, color="red", fontweight="bold")
        ax1.set_title("Adversarial\nN/A", fontsize=6, pad=2)
    ax1.axis("off")

    success_str = ""
    if adversarial is not None and adv_class != orig_class:
        success_str = " ✓"
    elif adversarial is not None:
        success_str = " ~"
    ax1.set_xlabel(f"{attack_name}{success_str}  [{attack_params}]", fontsize=5, labelpad=3)

    ax2 = fig.add_subplot(inner[2])
    ax2.imshow(noise_to_display(noise))
    ax2.set_title("Noise\n(amplified)", fontsize=6, pad=2)
    ax2.axis("off")


def render_image_page(pdf, image_name, results_for_image):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(f"Image: {image_name}", fontsize=11, fontweight="bold", y=0.98)

    outer = gridspec.GridSpec(
        ROWS_PER_PAGE, 1, figure=fig,
        hspace=0.55, top=0.94, bottom=0.03, left=0.04, right=0.96
    )

    for row_idx, (attack_name, attack_params) in enumerate(ATTACKS):
        result = results_for_image.get(attack_name, {})
        render_row(fig, outer[row_idx], result, image_name, attack_name, attack_params)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main(output_pdf=None):
    if output_pdf is None:
        output_pdf = DEFAULT_OUTPUT_PDF

    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: '{RESULTS_DIR}/' not found. Run run_attacks.py first.")
        return

    print("Loading results...")
    all_results = {}
    for image_name in IMAGE_NAMES:
        all_results[image_name] = {}
        for attack_name, _ in ATTACKS:
            all_results[image_name][attack_name] = load_result(image_name, attack_name)

    print(f"Generating '{output_pdf}'...")
    with PdfPages(output_pdf) as pdf:
        for image_name in IMAGE_NAMES:
            print(f"  Rendering page for: {image_name}")
            render_image_page(pdf, image_name, all_results[image_name])

    print(f"\nPDF saved: {output_pdf}")
    total = len(IMAGE_NAMES) * len(ATTACKS)
    succeeded = sum(
        1 for img in all_results.values()
        for atk in img.values()
        if atk.get("adversarial") is not None and
           atk.get("adversarial_class") != atk.get("original_class")
    )
    print(f"Visualized {total} attack rows | {succeeded} successful attacks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial example visualizations.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PDF,
                        help=f"Output PDF path (default: {DEFAULT_OUTPUT_PDF})")
    args = parser.parse_args()
    main(output_pdf=args.output)
