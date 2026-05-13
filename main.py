"""
Adversarial examples against ResNet50 using foolbox.

Usage:
  python main.py                # run full pipeline
  python main.py --download     # download images only
  python main.py --attack       # run attacks only (images must exist)
  python main.py --visualize    # generate PDF only (results must exist)
  python main.py --demo         # run FGSM sanity check on a random image
  python main.py --output results.pdf  # custom PDF output path
"""

import argparse
import os


def run_download():
    from load_images import download_images, load_images
    download_images()
    images = load_images()
    print(f"\nLoaded {len(images)} images.")
    return images


def run_attacks():
    import numpy as np
    from config import IMAGE_DIR
    from run_attacks import build_model, wrap_foolbox, run_all_attacks, save_results, print_summary

    images_path = os.path.join(IMAGE_DIR, "images.npy")
    names_path  = os.path.join(IMAGE_DIR, "names.npy")
    if not os.path.exists(images_path):
        print(f"ERROR: '{images_path}' not found. Run with --download first.")
        return

    images = np.load(images_path)
    names  = np.load(names_path).tolist()
    print(f"Loaded {len(images)} images.")

    print("\nLoading model...")
    keras_model = build_model()
    fmodel = wrap_foolbox(keras_model)

    from config import ATTACKS
    n_total = len(images) * len(ATTACKS)
    print(f"\nRunning attacks ({len(images)} images x {len(ATTACKS)} attacks = {n_total} total)...")
    results = run_all_attacks(images, names, keras_model, fmodel)

    save_results(results)
    print_summary(results)


def run_visualize(output_pdf):
    from visualize import main as visualize_main
    visualize_main(output_pdf=output_pdf)


def run_demo():
    from demo import main as demo_main
    demo_main()


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial example pipeline: ResNet50 + foolbox attacks on ImageNet images."
    )
    parser.add_argument("--download",  action="store_true", help="Download images")
    parser.add_argument("--attack",    action="store_true", help="Run adversarial attacks")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization PDF")
    parser.add_argument("--demo",      action="store_true", help="Run FGSM sanity check")
    parser.add_argument("--output",    default=None,        help="Output PDF path for --visualize")
    args = parser.parse_args()

    run_all = not any([args.download, args.attack, args.visualize, args.demo])

    if run_all or args.download:
        print("=== Downloading images ===")
        run_download()

    if run_all or args.attack:
        print("\n=== Running attacks ===")
        run_attacks()

    if run_all or args.visualize:
        print("\n=== Generating visualizations ===")
        from config import DEFAULT_OUTPUT_PDF
        run_visualize(output_pdf=args.output or DEFAULT_OUTPUT_PDF)

    if args.demo:
        print("=== Demo: FGSM on random image ===")
        run_demo()


if __name__ == "__main__":
    main()
