"""
Entry point for the multilingual sentiment robustness pipeline.

Stages (comment/uncomment as needed):
  1. Data setup         – download, filter, sample, save to Drive
  2. Single-swap eval   – generate swaps + score with each sentiment model
  3. Single-swap metrics
  4. Multi-swap eval    – cumulative swaps + score with each sentiment model
  5. Cross-model report
"""

import pandas as pd

import model_setup
from data_loader import (
    mount_drive,
    download_and_filter_dataset,
    save_full_dataset,
    create_samples,
    load_samples_from_drive,
    create_stratified_sample,
)
from sentiment_analysis import (
    analyze_dataset,
    print_swap_analysis,
    show_sentiment_shifts,
)
from metrics import (
    compute_flip_rate,
    compute_semsim,
    compute_robustness_gap,
    compute_flip_rate_by_lmi,
    compute_asymmetry,
    print_all_metrics,
)
from model_comparison import (
    preprocess_and_generate_swaps,
    preprocess_and_generate_multi_swaps,
    run_all_models_single_swap,
    run_all_models_multi_swap,
    compare_all_models,
    load_model_results,
    load_multi_results,
    save_swap_data,
    load_swap_data,
)
from config import DRIVE_BASE, MAX_SWAPS


def stage_1_data_setup():
    """Download, filter, and sample the MMS dataset. Run once."""
    mount_drive()
    df = download_and_filter_dataset()
    save_full_dataset(df)
    df_en_sample, df_es_sample = create_samples(df)
    return df_en_sample, df_es_sample


def stage_2_single_swap_eval(df_en_sample, df_es_sample):
    """Generate single-swap data and evaluate each sentiment model."""
    print("\n=== Generating Swaps for English ===")
    en_swaps = preprocess_and_generate_swaps(df_en_sample, source_lang="en")

    print("\n=== Generating Swaps for Spanish ===")
    es_swaps = preprocess_and_generate_swaps(df_es_sample, source_lang="es")

    save_swap_data(en_swaps, es_swaps)

    print("\n=== Running All Sentiment Models ===")
    all_model_results = run_all_models_single_swap(en_swaps, es_swaps)
    return all_model_results


def stage_3_single_swap_metrics(all_model_results):
    """Compute and print single-swap metrics for every model."""
    compare_all_models(all_model_results, lang="en")
    compare_all_models(all_model_results, lang="es")

    all_model_metrics = {}
    for model_key, results in all_model_results.items():
        df_en = results["en"]
        df_es = results["es"]
        print(f"\n{'='*60}\nMetrics for {model_key.upper()}\n{'='*60}")

        fr_en = compute_flip_rate(df_en, lang=f"{model_key} EN")
        fr_es = compute_flip_rate(df_es, lang=f"{model_key} ES")
        semsim_en = compute_semsim(df_en, lang=f"{model_key} EN")
        semsim_es = compute_semsim(df_es, lang=f"{model_key} ES")
        gap_en = compute_robustness_gap(df_en, lang=f"{model_key} EN")
        gap_es = compute_robustness_gap(df_es, lang=f"{model_key} ES")
        lmi_en = compute_flip_rate_by_lmi(df_en, source_lang="en")
        lmi_es = compute_flip_rate_by_lmi(df_es, source_lang="es")
        asymmetry = compute_asymmetry(df_en, df_es)

        all_model_metrics[model_key] = {
            "fr_en": fr_en, "fr_es": fr_es,
            "semsim_en": semsim_en, "semsim_es": semsim_es,
            "gap_en": gap_en, "gap_es": gap_es,
            "lmi_en": lmi_en, "lmi_es": lmi_es,
            "asymmetry": asymmetry,
        }

        print_all_metrics(fr_en, semsim_en, gap_en, lmi_en, asymmetry, lang=f"{model_key} EN")
        print_all_metrics(fr_es, semsim_es, gap_es, lmi_es, asymmetry, lang=f"{model_key} ES")

    return all_model_metrics


def stage_4_multi_swap_eval(df_en_sample, df_es_sample):
    """Generate cumulative multi-swap data and evaluate each model."""
    df_en_sub, df_es_sub = create_stratified_sample(df_en_sample, df_es_sample)

    print("\n=== Generating Multi-Swap Data ===")
    en_multi = preprocess_and_generate_multi_swaps(df_en_sub, source_lang="en", max_swaps=MAX_SWAPS)
    es_multi = preprocess_and_generate_multi_swaps(df_es_sub, source_lang="es", max_swaps=MAX_SWAPS)

    save_swap_data(en_multi, es_multi, suffix="multi")

    print("\n=== Running All Models on Multi-Swap Data ===")
    all_multi_results = run_all_models_multi_swap(en_multi, es_multi, max_swaps=MAX_SWAPS)
    return all_multi_results


def main():
    # ── Setup ─────────────────────────────────────────────────────────────────
    mount_drive()
    model_setup.setup_models()

    # ── Load pre-existing data (skip stage_1 if already done) ─────────────────
    df_en_sample, df_es_sample = load_samples_from_drive()

    # ── Single-swap pipeline ───────────────────────────────────────────────────
    # en_swaps, es_swaps = load_swap_data()           # reload if already generated
    # all_model_results = load_model_results()        # reload if already scored
    all_model_results = stage_2_single_swap_eval(df_en_sample, df_es_sample)
    stage_3_single_swap_metrics(all_model_results)

    # ── Multi-swap pipeline ────────────────────────────────────────────────────
    # all_multi_results = load_multi_results()        # reload if already generated
    stage_4_multi_swap_eval(df_en_sample, df_es_sample)

    print("\nAll done!")


if __name__ == "__main__":
    main()
