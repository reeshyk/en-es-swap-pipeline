import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

import model_setup as ms
from config import POS_TAGS, POSITIONS, SENTIMENT_MODELS


def load_sentiment_pipeline(model_key: str):
    model_name = SENTIMENT_MODELS[model_key]
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )


def parse_sentiment_label(label: str, model_key: str) -> int:
    if model_key == "mbert":
        return int(label.split()[0])
    elif model_key == "xlm_roberta":
        return {"negative": 1, "neutral": 3, "positive": 5}.get(label.lower(), 3)
    elif model_key == "distilbert":
        return {"negative": 1, "neutral": 3, "positive": 5}.get(label.lower(), 3)
    return 3


def batch_sentiment(sentences: list[str], model_pipeline, model_key: str) -> dict:
    unique = list(dict.fromkeys(sentences))
    results = model_pipeline(unique, batch_size=64, truncation=True, max_length=512)
    return {
        sent: (parse_sentiment_label(r["label"], model_key), r["score"])
        for sent, r in zip(unique, results)
    }


def analyze_dataset(df: pd.DataFrame, source_lang: str = "en") -> pd.DataFrame:
    from word_swapper import swap_word

    sentiment_pipeline = load_sentiment_pipeline("mbert")

    print("Generating swaps...")
    rows_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row["text"]
        row_data = {
            "original_text": sentence,
            "label": row["label"],
            "no_eligible_count": 0,
            "no_candidates_count": 0,
        }

        for pos in POS_TAGS:
            for position in POSITIONS:
                swap_result = swap_word(sentence, source_lang=source_lang, pos=pos, position=position)
                key = f"{pos}_{position}"
                if swap_result == "no_eligible":
                    row_data["no_eligible_count"] += 1
                    row_data[f"{key}_sentence"] = None
                elif swap_result == "no_candidates":
                    row_data["no_candidates_count"] += 1
                    row_data[f"{key}_sentence"] = None
                else:
                    row_data[f"{key}_sentence"] = swap_result["sentence"]
                    row_data[f"{key}_original_word"] = swap_result["original_word"]
                    row_data[f"{key}_replacement"] = swap_result["replacement"]
                    row_data[f"{key}_similarity"] = swap_result["similarity"]

        rows_data.append(row_data)

    all_sentences = []
    for row_data in rows_data:
        all_sentences.append(row_data["original_text"])
        for pos in POS_TAGS:
            for position in POSITIONS:
                s = row_data.get(f"{pos}_{position}_sentence")
                if s is not None:
                    all_sentences.append(s)

    print(f"Running sentiment on {len(all_sentences)} sentences...")
    sentiment_map = batch_sentiment(all_sentences, sentiment_pipeline, "mbert")

    for row_data in rows_data:
        orig_score, orig_conf = sentiment_map[row_data["original_text"]]
        row_data["original_score"] = orig_score
        row_data["original_confidence"] = orig_conf

        for pos in POS_TAGS:
            for position in POSITIONS:
                key = f"{pos}_{position}"
                swapped = row_data.get(f"{key}_sentence")
                if swapped is not None:
                    swap_score, swap_conf = sentiment_map[swapped]
                    row_data[f"{key}_score"] = swap_score
                    row_data[f"{key}_confidence"] = swap_conf
                    row_data[f"{key}_score_diff"] = swap_score - orig_score

    return pd.DataFrame(rows_data)


def analyze_pregenerated_swaps_with_sentiment(
    pregenerated_rows: list[dict],
    model_pipeline,
    model_key: str,
) -> pd.DataFrame:
    all_sentences = []
    for row_data in pregenerated_rows:
        all_sentences.append(row_data["original_text"])
        for pos in POS_TAGS:
            for position in POSITIONS:
                s = row_data.get(f"{pos}_{position}_sentence")
                if s is not None:
                    all_sentences.append(s)

    print(f"Running {model_key} sentiment on {len(set(all_sentences))} unique sentences...")
    sentiment_map = batch_sentiment(all_sentences, model_pipeline, model_key)

    final_results = []
    for row_data in pregenerated_rows:
        current = row_data.copy()
        orig_score, orig_conf = sentiment_map[current["original_text"]]
        current["original_score"] = orig_score
        current["original_confidence"] = orig_conf

        for pos in POS_TAGS:
            for position in POSITIONS:
                key = f"{pos}_{position}"
                swapped = current.get(f"{key}_sentence")
                if swapped is not None:
                    swap_score, swap_conf = sentiment_map[swapped]
                    current[f"{key}_score"] = swap_score
                    current[f"{key}_confidence"] = swap_conf
                    current[f"{key}_score_diff"] = swap_score - orig_score

        final_results.append(current)

    return pd.DataFrame(final_results)


def score_multi_swap_sentiment(
    rows_data: list[dict],
    model_pipeline,
    model_key: str,
    max_swaps: int = 3,
) -> pd.DataFrame:
    all_sentences = []
    for row_data in rows_data:
        all_sentences.append(row_data["original_text"])
        for n in range(1, max_swaps + 1):
            s = row_data.get(f"swap_{n}_sentence")
            if s is not None:
                all_sentences.append(s)

    print(f"  Running {model_key} on {len(set(all_sentences))} unique sentences...")
    sentiment_map = batch_sentiment(all_sentences, model_pipeline, model_key)

    final_results = []
    for row_data in rows_data:
        current = row_data.copy()
        orig_score, orig_conf = sentiment_map[row_data["original_text"]]
        current["original_score"] = orig_score
        current["original_confidence"] = orig_conf

        for n in range(1, max_swaps + 1):
            key = f"swap_{n}"
            swapped = current.get(f"{key}_sentence")
            if swapped is not None:
                swap_score, swap_conf = sentiment_map[swapped]
                current[f"{key}_score"] = swap_score
                current[f"{key}_confidence"] = swap_conf
                current[f"{key}_score_diff"] = swap_score - orig_score
                current[f"{key}_flipped"] = swap_score != orig_score

        final_results.append(current)

    return pd.DataFrame(final_results)


def print_swap_analysis(df_results: pd.DataFrame, lang: str = "en"):
    print(f"\n{'='*60}")
    print(f"Swap Analysis - {lang.upper()}")
    print(f"{'='*60}\n")

    print(f"{'':15}", end="")
    for position in POSITIONS:
        print(f"{position:>20}", end="")
    print()
    print("-" * 75)

    for pos in POS_TAGS:
        print(f"{pos:<15}", end="")
        for position in POSITIONS:
            key = f"{pos}_{position}"
            sentence_col = f"{key}_sentence"
            diff_col = f"{key}_score_diff"

            if sentence_col not in df_results.columns:
                print(f"{'N/A':>20}", end="")
                continue

            successful = df_results[sentence_col].notna()
            total = successful.sum()
            if total == 0:
                print(f"{'0/0':>20}", end="")
                continue

            shifted = (
                df_results.loc[successful, diff_col].notna()
                & (df_results.loc[successful, diff_col] != 0)
            ).sum()
            print(f"{f'{shifted}/{total}':>20}", end="")
        print()

    print("-" * 75)
    print(f"\nTotal skipped - no eligible token: {df_results['no_eligible_count'].sum()}")
    print(f"Total skipped - no candidates:     {df_results['no_candidates_count'].sum()}")


def show_sentiment_shifts(
    df_results: pd.DataFrame,
    n: int = 10,
    min_diff: int = 1,
    pos_filter: str | None = None,
    direction: str | None = None,
):
    examples = []

    for _, row in df_results.iterrows():
        for pos in POS_TAGS:
            for position in POSITIONS:
                key = f"{pos}_{position}"
                diff_col = f"{key}_score_diff"
                sentence_col = f"{key}_sentence"

                if diff_col not in df_results.columns:
                    continue
                if pd.isna(row.get(diff_col)) or pd.isna(row.get(sentence_col)):
                    continue
                if abs(row[diff_col]) < min_diff:
                    continue
                if direction == "negative" and row[diff_col] >= 0:
                    continue
                if direction == "positive" and row[diff_col] <= 0:
                    continue
                if pos_filter and pos != pos_filter:
                    continue

                examples.append({
                    "original": row["original_text"],
                    "swapped": row[sentence_col],
                    "original_word": row[f"{key}_original_word"],
                    "replacement": row[f"{key}_replacement"],
                    "original_score": row["original_score"],
                    "swapped_score": row[f"{key}_score"],
                    "score_diff": row[diff_col],
                    "pos": pos,
                    "position": position,
                })

    examples.sort(key=lambda x: abs(x["score_diff"]), reverse=True)

    for ex in examples[:n]:
        print(f"\n{'='*60}")
        print(f"POS: {ex['pos']} | Position: {ex['position']} | Diff: {int(ex['score_diff']):+d}")
        print(f"Original  ({int(ex['original_score'])} stars): {ex['original']}")
        print(f"Swapped   ({int(ex['swapped_score'])} stars): {ex['swapped']}")
        print(f"Swap: '{ex['original_word']}' → '{ex['replacement']}'")
