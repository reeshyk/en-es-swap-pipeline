import gc
import random
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

import model_setup as ms
from config import POS_TAGS, POSITIONS, SENTIMENT_MODELS, DRIVE_BASE, MAX_SWAPS
from sentiment_analysis import analyze_pregenerated_swaps_with_sentiment, score_multi_swap_sentiment
from word_swapper import swap_word_fast
from metrics import (
    compute_flip_rate, compute_semsim, compute_robustness_gap,
    compute_flip_rate_by_lmi, compute_asymmetry, print_all_metrics,
    compute_flip_rate_by_n, compute_msdf_from_multi, compute_semsim_by_n,
    compute_robustness_gap_by_n, compute_lmi_by_n, compute_asymmetry_by_n,
    print_all_metrics_by_n,
)


def preprocess_and_generate_swaps(df: pd.DataFrame, source_lang: str = "en") -> list[dict]:
    if source_lang == "en":
        tokenizer, trans_model = ms.en_es_tokenizer, ms.en_es_model
        source_nlp, target_nlp = ms.nlp_en, ms.nlp_es
    else:
        tokenizer, trans_model = ms.es_en_tokenizer, ms.es_en_model
        source_nlp, target_nlp = ms.nlp_es, ms.nlp_en

    texts = df["text"].tolist()
    translations = {}

    print(f"  Pre-translating {source_lang.upper()} sentences...")
    for i in tqdm(range(0, len(texts), 64), desc="Translating"):
        batch = texts[i:i + 64]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(ms.device)
        with torch.no_grad():
            translated = trans_model.generate(**inputs)
        for src, tgt in zip(batch, translated):
            translations[src] = tokenizer.decode(tgt, skip_special_tokens=True)

    print("  Pre-parsing sentences...")
    docs = {text: source_nlp(text) for text in tqdm(texts, desc="Parsing originals")}
    translated_docs = {t: target_nlp(t)
                       for t in tqdm(translations.values(), desc="Parsing translations")}

    print("  Generating swaps...")
    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating swaps"):
        sentence = row["text"]
        doc = docs[sentence]
        translated_sentence = translations.get(sentence)
        translated_doc = translated_docs.get(translated_sentence)

        row_data = {
            "original_text": sentence,
            "label": row["label"],
            "no_eligible_count": 0,
            "no_candidates_count": 0,
        }

        for pos in POS_TAGS:
            for position in POSITIONS:
                swap_result = swap_word_fast(
                    sentence, doc, translated_doc,
                    source_lang=source_lang, pos=pos, position=position,
                )
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

        all_rows.append(row_data)

    return all_rows


def generate_swaps_for_sentence(
    sentence: str, doc, translated_doc, source_lang: str = "en", max_swaps: int = MAX_SWAPS
) -> dict:
    nlp = ms.nlp_en if source_lang == "en" else ms.nlp_es
    results = {}
    excluded_words: set = set()
    current_sentence = sentence
    current_doc = doc
    all_swap_details = []

    for n in range(1, max_swaps + 1):
        combos = [
            (pos, position)
            for pos in POS_TAGS
            for position in POSITIONS
        ]
        random.shuffle(combos)

        swap_made = False
        for pos, position in combos:
            try:
                result = swap_word_fast(
                    current_sentence, current_doc, translated_doc,
                    source_lang=source_lang, pos=pos, position=position,
                    exclude_words=frozenset(excluded_words),
                )
            except Exception as e:
                print(f"Error in swap_word_fast: {e}")
                result = "no_eligible"

            if isinstance(result, dict):
                excluded_words.add(result["original_word"].lower())
                current_sentence = result["sentence"]
                current_doc = nlp(current_sentence)
                all_swap_details.append({
                    "swap_n": n,
                    "pos": pos,
                    "position": position,
                    "original_word": result["original_word"],
                    "replacement": result["replacement"],
                    "similarity": result["similarity"],
                })
                swap_made = True
                break

        if swap_made:
            results[n] = {
                "sentence": current_sentence,
                "swaps": list(all_swap_details),
                "n_swaps_made": n,
            }
        else:
            break

    return results


def preprocess_and_generate_multi_swaps(
    df: pd.DataFrame, source_lang: str = "en", max_swaps: int = MAX_SWAPS
) -> list[dict]:
    if source_lang == "en":
        tokenizer, trans_model = ms.en_es_tokenizer, ms.en_es_model
        source_nlp, target_nlp = ms.nlp_en, ms.nlp_es
    else:
        tokenizer, trans_model = ms.es_en_tokenizer, ms.es_en_model
        source_nlp, target_nlp = ms.nlp_es, ms.nlp_en

    texts = df["text"].tolist()
    translations = {}

    print("  Pre-translating...")
    for i in tqdm(range(0, len(texts), 64), desc="Translating"):
        batch = texts[i:i + 64]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(ms.device)
        with torch.no_grad():
            translated = trans_model.generate(**inputs)
        for src, tgt in zip(batch, translated):
            translations[src] = tokenizer.decode(tgt, skip_special_tokens=True)

    print("  Pre-parsing original sentences...")
    docs = {text: source_nlp(text) for text in tqdm(texts, desc="Parsing originals")}

    print("  Pre-parsing translated sentences...")
    translated_docs = {t: target_nlp(t)
                       for t in tqdm(set(translations.values()), desc="Parsing translations")}

    print("  Generating multi-swaps...")
    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating swaps"):
        sentence = row["text"]
        doc = docs[sentence]
        translated_sentence = translations.get(sentence)
        translated_doc = translated_docs.get(translated_sentence)

        swap_results = generate_swaps_for_sentence(
            sentence, doc, translated_doc,
            source_lang=source_lang, max_swaps=max_swaps,
        )

        row_data = {"original_text": sentence, "label": row["label"]}
        for n in range(1, max_swaps + 1):
            key = f"swap_{n}"
            if n in swap_results:
                row_data[f"{key}_sentence"] = swap_results[n]["sentence"]
                row_data[f"{key}_swaps"] = str(swap_results[n]["swaps"])
                row_data[f"{key}_n_made"] = swap_results[n]["n_swaps_made"]
            else:
                row_data[f"{key}_sentence"] = None
                row_data[f"{key}_swaps"] = None
                row_data[f"{key}_n_made"] = 0

        all_rows.append(row_data)

    return all_rows


def compare_all_models(all_model_results: dict, lang: str = "en"):
    print(f"\n{'='*60}")
    print(f"Model Comparison - {lang.upper()} - Flip Rate")
    print(f"{'='*60}\n")

    keys = [f"{pos}_{position}" for pos in POS_TAGS for position in POSITIONS]
    print(f"{'':20}", end="")
    for key in keys:
        print(f"{key:>18}", end="")
    print()
    print("-" * (20 + 18 * len(keys)))

    for model_key, results in all_model_results.items():
        df = results[lang]
        print(f"{model_key:<20}", end="")
        for key in keys:
            diff_col = f"{key}_score_diff"
            sentence_col = f"{key}_sentence"
            if sentence_col not in df.columns:
                print(f"{'N/A':>18}", end="")
                continue
            swapped = df[sentence_col].notna()
            total = swapped.sum()
            if total == 0:
                print(f"{'0/0':>18}", end="")
                continue
            flipped = (df.loc[swapped, diff_col] != 0).sum()
            print(f"{flipped / total * 100:>17.1f}%", end="")
        print()


def run_all_models_single_swap(
    en_swaps: list[dict],
    es_swaps: list[dict],
    models_to_test: dict | None = None,
) -> dict:
    if models_to_test is None:
        models_to_test = SENTIMENT_MODELS

    all_model_results = {}

    for model_key, model_name in models_to_test.items():
        print(f"\n{'='*60}\nLoading {model_key}: {model_name}\n{'='*60}")
        try:
            model_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
            )

            df_en = analyze_pregenerated_swaps_with_sentiment(en_swaps, model_pipeline, model_key)
            df_es = analyze_pregenerated_swaps_with_sentiment(es_swaps, model_pipeline, model_key)

            all_model_results[model_key] = {"en": df_en, "es": df_es}

            df_en.to_parquet(f"{DRIVE_BASE}/{model_key}_en_results.parquet", index=False)
            df_es.to_parquet(f"{DRIVE_BASE}/{model_key}_es_results.parquet", index=False)
            print(f"Saved {model_key} results to Drive")

        except Exception as e:
            print(f"Failed on {model_key}: {e}")
        finally:
            del model_pipeline
            torch.cuda.empty_cache()
            gc.collect()

    return all_model_results


def run_all_models_multi_swap(
    en_multi: list[dict],
    es_multi: list[dict],
    models_to_test: dict | None = None,
    max_swaps: int = MAX_SWAPS,
) -> dict:
    if models_to_test is None:
        models_to_test = SENTIMENT_MODELS

    all_multi_results = {}

    for model_key, model_name in models_to_test.items():
        print(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")
        try:
            model_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
            )

            print("\nScoring English...")
            df_en = score_multi_swap_sentiment(en_multi, model_pipeline, model_key, max_swaps)
            print("\nScoring Spanish...")
            df_es = score_multi_swap_sentiment(es_multi, model_pipeline, model_key, max_swaps)

            df_en = compute_msdf_from_multi(df_en, max_swaps)
            df_es = compute_msdf_from_multi(df_es, max_swaps)

            df_en.to_parquet(f"{DRIVE_BASE}/{model_key}_en_multi_{max_swaps}.parquet", index=False)
            df_es.to_parquet(f"{DRIVE_BASE}/{model_key}_es_multi_{max_swaps}.parquet", index=False)

            print("\nComputing metrics...")
            fr_en = compute_flip_rate_by_n(df_en, max_swaps)
            fr_es = compute_flip_rate_by_n(df_es, max_swaps)
            semsim_en = compute_semsim_by_n(df_en, max_swaps)
            semsim_es = compute_semsim_by_n(df_es, max_swaps)
            gap_en = compute_robustness_gap_by_n(df_en, max_swaps)
            gap_es = compute_robustness_gap_by_n(df_es, max_swaps)
            lmi_en = compute_lmi_by_n(df_en, source_lang="en", max_swaps=max_swaps)
            lmi_es = compute_lmi_by_n(df_es, source_lang="es", max_swaps=max_swaps)
            asymmetry = compute_asymmetry_by_n(df_en, df_es, max_swaps)

            all_multi_results[model_key] = {
                "en": df_en, "es": df_es,
                "fr_en": fr_en, "fr_es": fr_es,
                "semsim_en": semsim_en, "semsim_es": semsim_es,
                "gap_en": gap_en, "gap_es": gap_es,
                "lmi_en": lmi_en, "lmi_es": lmi_es,
                "asymmetry": asymmetry,
            }

            print_all_metrics_by_n(fr_en, semsim_en, gap_en, lmi_en, asymmetry,
                                   df_en, lang=f"{model_key} EN", max_swaps=max_swaps)
            print_all_metrics_by_n(fr_es, semsim_es, gap_es, lmi_es, asymmetry,
                                   df_es, lang=f"{model_key} ES", max_swaps=max_swaps)

        except Exception as e:
            print(f"Failed on {model_key}: {e}")
        finally:
            del model_pipeline
            torch.cuda.empty_cache()
            gc.collect()

    return all_multi_results


def load_model_results(models_to_test: dict | None = None) -> dict:
    if models_to_test is None:
        models_to_test = SENTIMENT_MODELS
    all_model_results = {}
    for model_key in models_to_test:
        try:
            df_en = pd.read_parquet(f"{DRIVE_BASE}/{model_key}_en_results.parquet")
            df_es = pd.read_parquet(f"{DRIVE_BASE}/{model_key}_es_results.parquet")
            all_model_results[model_key] = {"en": df_en, "es": df_es}
            print(f"Loaded {model_key}: EN={len(df_en)} rows, ES={len(df_es)} rows")
        except Exception as e:
            print(f"Could not load {model_key}: {e}")
    return all_model_results


def load_multi_results(models_to_test: dict | None = None, max_swaps: int = MAX_SWAPS) -> dict:
    if models_to_test is None:
        models_to_test = SENTIMENT_MODELS
    all_multi_results = {}
    for model_key in models_to_test:
        try:
            df_en = pd.read_parquet(f"{DRIVE_BASE}/{model_key}_en_multi_{max_swaps}.parquet")
            df_es = pd.read_parquet(f"{DRIVE_BASE}/{model_key}_es_multi_{max_swaps}.parquet")
            all_multi_results[model_key] = {"en": df_en, "es": df_es}
            print(f"Loaded {model_key}")
        except Exception as e:
            print(f"Could not load {model_key}: {e}")
    return all_multi_results


def save_swap_data(en_swaps, es_swaps, suffix: str = ""):
    tag = f"_{suffix}" if suffix else ""
    with open(f"{DRIVE_BASE}/en_swaps_data{tag}.pkl", "wb") as f:
        pickle.dump(en_swaps, f)
    with open(f"{DRIVE_BASE}/es_swaps_data{tag}.pkl", "wb") as f:
        pickle.dump(es_swaps, f)
    print(f"Swap data saved to Drive (suffix='{suffix}')")


def load_swap_data(suffix: str = "") -> tuple:
    tag = f"_{suffix}" if suffix else ""
    with open(f"{DRIVE_BASE}/en_swaps_data{tag}.pkl", "rb") as f:
        en_swaps = pickle.load(f)
    with open(f"{DRIVE_BASE}/es_swaps_data{tag}.pkl", "rb") as f:
        es_swaps = pickle.load(f)
    return en_swaps, es_swaps
