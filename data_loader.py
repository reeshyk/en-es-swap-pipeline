import os
import pandas as pd
from config import IS_COLAB, DRIVE_BASE, WORD_COUNT_MIN, WORD_COUNT_MAX, SAMPLE_SIZE_PER_LABEL, N_SAMPLES_MULTI


def mount_drive():
    os.makedirs(DRIVE_BASE, exist_ok=True)
    if IS_COLAB:
        from google.colab import drive
        try:
            drive.mount("/content/drive")
        except ValueError:
            print("Drive already mounted.")
    else:
        print(f"Local mode: data directory is '{DRIVE_BASE}'")


def download_and_filter_dataset(hf_token: str | None = None) -> pd.DataFrame:
    from huggingface_hub import login
    from datasets import load_dataset

    if hf_token:
        login(token=hf_token)
    else:
        login()

    dataset = load_dataset("Brand24/mms", split="train")

    filtered = dataset.filter(
        lambda x: [
            lang in ["en", "es"] and dom == "reviews"
            for lang, dom in zip(x["language"], x["domain"])
        ],
        batched=True,
    )
    filtered = filtered.select_columns(
        ["text", "_id", "label", "language", "domain", "original_dataset"]
    )

    df = pd.DataFrame(filtered)
    df["word_count"] = df["text"].str.split().str.len()
    df = df[(df["word_count"] >= WORD_COUNT_MIN) & (df["word_count"] <= WORD_COUNT_MAX)]

    print(f"Filtered dataset: {len(df)} rows")
    return df


def save_full_dataset(df: pd.DataFrame):
    path = f"{DRIVE_BASE}/mms_filtered.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved full dataset to {path}")


def create_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_en = df[df["language"] == "en"]
    df_es = df[df["language"] == "es"]

    df_en_sample = df_en.groupby("label").sample(n=SAMPLE_SIZE_PER_LABEL, random_state=42)
    df_es_sample = df_es.groupby("label").sample(n=SAMPLE_SIZE_PER_LABEL, random_state=42)

    df_en_sample.to_parquet(f"{DRIVE_BASE}/mms_en_sample.parquet", index=False)
    df_es_sample.to_parquet(f"{DRIVE_BASE}/mms_es_sample.parquet", index=False)

    print(f"English sample: {len(df_en_sample)} rows")
    print(df_en_sample["label"].value_counts())
    print(f"\nSpanish sample: {len(df_es_sample)} rows")
    print(df_es_sample["label"].value_counts())

    return df_en_sample, df_es_sample


def load_samples_from_drive() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_en_sample = pd.read_parquet(f"{DRIVE_BASE}/mms_en_sample.parquet")
    df_es_sample = pd.read_parquet(f"{DRIVE_BASE}/mms_es_sample.parquet")
    print(f"English sample: {len(df_en_sample)} rows")
    print(f"Spanish sample: {len(df_es_sample)} rows")
    return df_en_sample, df_es_sample


def create_stratified_sample(
    df_en: pd.DataFrame,
    df_es: pd.DataFrame,
    n: int = N_SAMPLES_MULTI,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_label = n // 3

    def _sample(df):
        return pd.concat([
            df[df["label"] == 0].sample(n=per_label, random_state=42),
            df[df["label"] == 1].sample(n=per_label, random_state=42),
            df[df["label"] == 2].sample(n=n - 2 * per_label, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_en_sub = _sample(df_en)
    df_es_sub = _sample(df_es)

    print(f"English {n} sample: {len(df_en_sub)} rows")
    print(f"Spanish {n} sample: {len(df_es_sub)} rows")

    return df_en_sub, df_es_sub
