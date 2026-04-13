#DS577 - Assignment 3: Step 5 - Visualize Adversarial Examples & Generate PDF
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_DIR = "results"
IMAGE_DIR   = "images"
OUTPUT_PDF  = "cmcourville-assignment3.pdf"

IMAGE_NAMES = [
    "tench", "goldfish", "snail", "tusker", "tabby_cat",
    "sports_car", "acoustic_guitar", "banana", "mushroom", "volcano"
]

ATTACKS = [
    ("BlendedUniformNoiseAttack", "epsilons=1000, max_directions=1000"),
    ("ContrastReductionAttack",   "epsilons=1000"),
    ("FGSM",                      "default epsilon"),
    ("SinglePixelAttack",         "max_pixels=1000"),
    ("SaliencyMapAttack",         "max_iter=2000, fast=True"),
]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_npy(path):
    """Load a .npy file, return None if missing."""
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def load_result(image_name, attack_name):
    """
    Load all components for one (image, attack) pair from the results directory.
    Returns a dict with original, adversarial, noise, classes, and params.
    """
    key = f"{image_name}__{attack_name}"
    base = RESULTS_DIR

    original        = load_npy(os.path.join(base, f"{key}__original.npy"))
    adversarial     = load_npy(os.path.join(base, f"{key}__adversarial.npy"))
    noise           = load_npy(os.path.join(base, f"{key}__noise.npy"))
    original_class  = load_npy(os.path.join(base, f"{key}__original_class.npy"))
    adv_class       = load_npy(os.path.join(base, f"{key}__adv_class.npy"))
    attack_params   = load_npy(os.path.join(base, f"{key}__attack_params.npy"))

    return {
        "original":         original,
        "adversarial":      adversarial,
        "noise":            noise,
        "original_class":   str(original_class) if original_class is not None else "unknown",
        "adversarial_class": str(adv_class) if adv_class is not None else None,
        "attack_params":    str(attack_params) if attack_params is not None else "",
    }


# ---------------------------------------------------------------------------
# Image normalisation for display
# ---------------------------------------------------------------------------

def to_display(image):
    """Clip and normalise a float32 image to [0, 1] for matplotlib display."""
    img = np.clip(image, 0, 255) / 255.0
    return img


def noise_to_display(noise):
    """
    Normalise noise for visibility. Centre around 0.5 and scale to [0, 1]
    so that zero noise = grey, positive = bright, negative = dark.
    """
    if noise is None:
        return np.ones((224, 224, 3)) * 0.5
    vmax = np.abs(noise).max()
    if vmax == 0:
        return np.ones_like(noise) * 0.5
    normalised = (noise / (2 * vmax)) + 0.5
    return np.clip(normalised, 0, 1)


# ---------------------------------------------------------------------------
# Row rendering
# ---------------------------------------------------------------------------

def render_row(fig, outer_row, result, image_name, attack_name, attack_params):
    """
    Render one row: [Original | Adversarial | Noise] inside an outer GridSpec row.
    Labels the row with attack name, params, and class predictions.
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_row, wspace=0.05
    )

    original    = result["original"]
    adversarial = result["adversarial"]
    noise       = result["noise"]
    orig_class  = result["original_class"]
    adv_class   = result["adversarial_class"]

    # --- Original ---
    ax0 = fig.add_subplot(inner[0])
    ax0.imshow(to_display(original))
    ax0.set_title(f"Original\n{orig_class}", fontsize=6, pad=2)
    ax0.axis("off")

    # --- Adversarial ---
    ax1 = fig.add_subplot(inner[1])
    if adversarial is not None:
        ax1.imshow(to_display(adversarial))
        ax1.set_title(f"Adversarial\n{adv_class}", fontsize=6, pad=2)
    else:
        ax1.imshow(np.ones((224, 224, 3)) * 0.85)
        ax1.text(0.5, 0.5, "Attack\nFailed", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=8, color="red",
                 fontweight="bold")
        ax1.set_title("Adversarial\nN/A", fontsize=6, pad=2)
    ax1.axis("off")

    # Row label (attack name + params) on the middle panel
    success_str = ""
    if adversarial is not None and adv_class != orig_class:
        success_str = " ✓"
    elif adversarial is not None:
        success_str = " ~"
    row_label = f"{attack_name}{success_str}  [{attack_params}]"
    ax1.set_xlabel(row_label, fontsize=5, labelpad=3)

    # --- Noise ---
    ax2 = fig.add_subplot(inner[2])
    ax2.imshow(noise_to_display(noise))
    ax2.set_title("Noise\n(amplified)", fontsize=6, pad=2)
    ax2.axis("off")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

ROWS_PER_PAGE = 5   # one attack per row → one image per page

def render_image_page(pdf, image_name, results_for_image):
    """Render one page: all 5 attacks for a single image."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(
        f"Image: {image_name}",
        fontsize=11, fontweight="bold", y=0.98
    )

    outer = gridspec.GridSpec(
        ROWS_PER_PAGE, 1, figure=fig,
        hspace=0.55, top=0.94, bottom=0.03, left=0.04, right=0.96
    )

    for row_idx, (attack_name, attack_params) in enumerate(ATTACKS):
        result = results_for_image.get(attack_name, {})
        render_row(fig, outer[row_idx], result, image_name,
                   attack_name, attack_params)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def render_code_pages(pdf, source_files):
    """Append source code listings as PDF pages."""
    for filepath in source_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Split into chunks that fit on one page (~80 lines per page)
        chunk_size = 80
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        filename = os.path.basename(filepath)

        for page_num, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.axis("off")

            header = f"{filename}  (page {page_num + 1}/{len(chunks)})"
            code_text = "".join(chunk)
            full_text = f"{header}\n{'─' * 80}\n{code_text}"

            ax.text(
                0.01, 0.99, full_text,
                transform=ax.transAxes,
                fontsize=5.5,
                verticalalignment="top",
                fontfamily="monospace",
                wrap=False,
            )

            pdf.savefig(fig, dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DS577 Assignment 3 - Step 5: Generate Visualizations & PDF")
    print("=" * 60)

    # Verify results exist
    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: '{RESULTS_DIR}/' not found. Run run_attacks.py first.")
        return

    # Load all results
    print("Loading results...")
    all_results = {}
    for image_name in IMAGE_NAMES:
        all_results[image_name] = {}
        for attack_name, _ in ATTACKS:
            all_results[image_name][attack_name] = load_result(image_name, attack_name)

    # Generate PDF
    print(f"Generating '{OUTPUT_PDF}'...")
    source_files = ["minimal_example.py", "load_images.py", "run_attacks.py", "visualize.py"]

    with PdfPages(OUTPUT_PDF) as pdf:

        # Page 1: cover / summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.65, "DS577 — Machine Learning & Cybersecurity",
                ha="center", va="center", fontsize=14, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.58, "Assignment 3: Adversarial Crafting",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.52, "Corrin Courville",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.text(0.5, 0.42,
                "Model: ResNet50 (ImageNet weights)\n"
                "Library: foolbox 2.4.0\n"
                "Images: 10 ImageNet samples\n"
                "Attacks: BlendedUniformNoiseAttack, ContrastReductionAttack,\n"
                "         FGSM, SinglePixelAttack, SaliencyMapAttack",
                ha="center", va="center", fontsize=10,
                transform=ax.transAxes, linespacing=1.8)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # One page per image (5 attack rows each)
        for image_name in IMAGE_NAMES:
            print(f"  Rendering page for: {image_name}")
            render_image_page(pdf, image_name, all_results[image_name])

        # Code listing pages
        print("  Appending code listings...")
        render_code_pages(pdf, source_files)

    print(f"\nPDF saved: {OUTPUT_PDF}")
    total = len(IMAGE_NAMES) * len(ATTACKS)
    succeeded = sum(
        1 for img in all_results.values()
        for atk in img.values()
        if atk.get("adversarial") is not None and
           atk.get("adversarial_class") != atk.get("original_class")
    )
    print(f"Visualized {total} attack rows | {succeeded} successful attacks")
    print("=" * 60)


if __name__ == "__main__":
    main()
