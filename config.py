import os

IMAGE_DIR = "images"
RESULTS_DIR = "results"
IMAGE_SIZE = (224, 224)
DEFAULT_OUTPUT_PDF = "adversarial_results.pdf"

BASE = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"
IMAGE_URLS = [
    ("tench",           f"{BASE}/n01440764_tench.JPEG"),
    ("goldfish",        f"{BASE}/n01443537_goldfish.JPEG"),
    ("snail",           f"{BASE}/n01944390_snail.JPEG"),
    ("tusker",          f"{BASE}/n02504458_African_elephant.JPEG"),
    ("tabby_cat",       f"{BASE}/n02123045_tabby.JPEG"),
    ("sports_car",      f"{BASE}/n04285008_sports_car.JPEG"),
    ("acoustic_guitar", f"{BASE}/n02676566_acoustic_guitar.JPEG"),
    ("banana",          f"{BASE}/n07753592_banana.JPEG"),
    ("mushroom",        f"{BASE}/n13037406_gyromitra.JPEG"),
    ("volcano",         f"{BASE}/n09472597_volcano.JPEG"),
]

IMAGE_NAMES = [name for name, _ in IMAGE_URLS]

ATTACKS = [
    ("BlendedUniformNoiseAttack", "epsilons=1000, max_directions=1000"),
    ("ContrastReductionAttack",   "epsilons=1000"),
    ("FGSM",                      "default epsilon"),
    ("SinglePixelAttack",         "max_pixels=1000"),
    ("SaliencyMapAttack",         "max_iter=2000, fast=True"),
]
