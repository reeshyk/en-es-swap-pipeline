import os
import torch
import spacy
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForMaskedLM

from config import (
    SENTENCE_TRANSFORMER_MODEL,
    MLM_MODEL,
    TRANSLATION_EN_ES,
    TRANSLATION_ES_EN,
    MUSE_EN_ES,
    MUSE_ES_EN,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Populated by setup_models()
nlp_en = None
nlp_es = None
st_model = None
en_es_tokenizer = None
en_es_model = None
es_en_tokenizer = None
es_en_model = None
mlm_tokenizer = None
mlm_model = None
en_to_es_dict: dict = {}
es_to_en_dict: dict = {}


def download_dependencies():
    import subprocess
    import urllib.request
    from config import MUSE_DIR

    subprocess.check_call([
        "pip", "install", "-q",
        "datasets<2.20", "wordfreq", "sentence-transformers",
        "transformers", "spacy", "scikit-learn",
    ])
    subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_md"])
    subprocess.check_call(["python", "-m", "spacy", "download", "es_core_news_md"])

    os.makedirs(MUSE_DIR, exist_ok=True)
    for fname, url in [
        ("en-es.txt", "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.txt"),
        ("es-en.txt", "https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.txt"),
    ]:
        dest = os.path.join(MUSE_DIR, fname)
        if not os.path.exists(dest):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, dest)


def load_muse_dictionary(path: str) -> dict:
    pairs: dict = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src, tgt = parts[0].lower(), parts[1].lower()
            if src != tgt and src.isalpha() and tgt.isalpha():
                pairs.setdefault(src, []).append(tgt)
    print(f"Loaded {len(pairs)} source words from {path}")
    return pairs


def setup_models():
    global nlp_en, nlp_es, st_model
    global en_es_tokenizer, en_es_model, es_en_tokenizer, es_en_model
    global mlm_tokenizer, mlm_model
    global en_to_es_dict, es_to_en_dict

    print(f"Using device: {device}")

    nlp_en = spacy.load("en_core_web_md")
    nlp_es = spacy.load("es_core_news_md")

    print("Loading MUSE dictionaries...")
    en_to_es_dict = load_muse_dictionary(MUSE_EN_ES)
    es_to_en_dict = load_muse_dictionary(MUSE_ES_EN)

    print("Loading sentence transformer...")
    st_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device)

    print("Loading translation models...")
    en_es_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_EN_ES)
    en_es_model = MarianMTModel.from_pretrained(TRANSLATION_EN_ES).to(device)

    es_en_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_ES_EN)
    es_en_model = MarianMTModel.from_pretrained(TRANSLATION_ES_EN).to(device)

    print("Loading MLM model...")
    mlm_tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL)
    mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL).to(device)

    print("All models loaded.")
