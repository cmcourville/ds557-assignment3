# Adversarial Examples on ImageNet

Generates adversarial examples against ResNet50 using [foolbox 2.4.0](https://github.com/bethgelab/foolbox). Five attack methods are applied to 10 diverse ImageNet images, with results visualized in a PDF report.

**Attacks:** BlendedUniformNoiseAttack, ContrastReductionAttack, FGSM, SinglePixelAttack, SaliencyMapAttack

## Setup

**Requirements:** macOS, Python 3.11 (TensorFlow requires ≥3.10; foolbox 2.4.0 requires TF1 graph mode)

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv ~/adversarial-env
source ~/adversarial-env/bin/activate
pip install -r requirements.txt
```

### Patch foolbox for Python 3.10+ compatibility

foolbox 2.4.0 uses `collections.Iterable` and `collections.Mapping`, removed in Python 3.10:

```bash
find ~/adversarial-env/lib/python3.11/site-packages/foolbox/ -name "*.py" \
  -exec sed -i '' 's/from collections import Iterable/from collections.abc import Iterable/g' {} +

sed -i '' 's/collections\.Mapping/collections.abc.Mapping/g' \
  ~/adversarial-env/lib/python3.11/site-packages/foolbox/models/base.py
```

## Usage

```bash
source ~/adversarial-env/bin/activate

# Full pipeline (download → attack → visualize)
python main.py

# Individual stages
python main.py --download    # fetch 10 ImageNet images
python main.py --attack      # run all 5 attacks (requires images)
python main.py --visualize   # generate PDF report (requires results)
python main.py --demo        # FGSM sanity check on a random image

# Custom output path
python main.py --visualize --output my_report.pdf
```

Output: `adversarial_results.pdf` — one page per image, showing original / adversarial / noise for each attack.

## Files

| File | Purpose |
|---|---|
| `main.py` | CLI entry point — orchestrates the full pipeline |
| `config.py` | Centralized configuration (images, attacks, paths) |
| `load_images.py` | Downloads 10 ImageNet images, saves to `images/` |
| `run_attacks.py` | Runs all 5 attacks on all images, saves results to `results/` |
| `visualize.py` | Loads results and generates the PDF report |
| `demo.py` | FGSM sanity check on a random image (environment verification) |
| `requirements.txt` | Pinned dependencies |

## Dependencies

Key version pins:

| Package | Version | Reason |
|---|---|---|
| `tensorflow-macos` | 2.13.0 | Latest TF with foolbox 2.x graph-mode compatibility |
| `numpy` | 1.24.3 | TF 2.13 requires `numpy<=1.24.3`; NumPy 2.x breaks compiled extensions |
| `foolbox` | 2.4.0 | v1 attack API; requires TF1 graph mode |
| `setuptools` | 70.3.0 | foolbox uses `pkg_resources` removed in setuptools 71+ |
