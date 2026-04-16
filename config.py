import os


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IS_COLAB = _is_colab()

# In Colab: data lives on Google Drive. Locally: use a "data/" folder in the repo root.
DRIVE_BASE = os.environ.get(
    "IW_DATA_DIR",
    "/content/drive/MyDrive" if IS_COLAB else "data",
)

MUSE_DIR = os.path.join(DRIVE_BASE, "dicts")
MUSE_EN_ES = os.path.join(MUSE_DIR, "en-es.txt")
MUSE_ES_EN = os.path.join(MUSE_DIR, "es-en.txt")

SENTIMENT_MODELS = {
    "mbert": "nlptown/bert-base-multilingual-uncased-sentiment",
    "xlm_roberta": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "distilbert": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
}

SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-mpnet-base-v2"
LABSE_MODEL = "sentence-transformers/LaBSE"
MLM_MODEL = "bert-base-multilingual-cased"
TRANSLATION_EN_ES = "Helsinki-NLP/opus-mt-en-es"
TRANSLATION_ES_EN = "Helsinki-NLP/opus-mt-es-en"

WORD_COUNT_MIN = 10
WORD_COUNT_MAX = 20
SAMPLE_SIZE_PER_LABEL = 3000
N_SAMPLES_MULTI = 400
MAX_SWAPS = 3

POS_TAGS = ["NOUN", "ADJ", "VERB"]
POSITIONS = ["beginning", "middle", "end"]
