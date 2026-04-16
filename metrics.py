import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from wordfreq import word_frequency

from config import POS_TAGS, POSITIONS, LABSE_MODEL
import model_setup as ms

_labse: SentenceTransformer | None = None


def _get_labse() -> SentenceTransformer:
    global _labse
    if _labse is None:
        print("Loading LaBSE model...")
        _labse = SentenceTransformer(LABSE_MODEL, device=ms.device)
    return _labse


# ── Helpers ──────────────────────────────────────────────────────────────────

def _label_to_class(score: int) -> str:
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    return "positive"


def _is_correct(pred_score: int, true_label: int) -> bool:
    label_map = {0: 1, 1: 3, 2: 5}
    true_score = label_map.get(true_label)
    if true_score is None:
        return False
    return _label_to_class(pred_score) == _label_to_class(true_score)


def compute_lmi(sentence: str, lang: str = "en") -> float:
    words = sentence.lower().split()
    if not words:
        return 0.0
    other_lang = "es" if lang == "en" else "en"
    other_count = sum(
        1 for w in words
        if word_frequency(w, other_lang) > word_frequency(w, lang)
    )
    return other_count / len(words)


# ── Single-swap metrics ───────────────────────────────────────────────────────

def compute_flip_rate(df_results: pd.DataFrame, lang: str = "") -> dict:
    flip_rates = {}
    for pos in tqdm(POS_TAGS, desc=f"Flip Rate {lang}"):
        for position in POSITIONS:
            key = f"{pos}_{position}"
            diff_col = f"{key}_score_diff"
            sentence_col = f"{key}_sentence"
            if sentence_col not in df_results.columns:
                continue
            swapped = df_results[sentence_col].notna()
            total = swapped.sum()
            if total == 0:
                continue
            flipped = (df_results.loc[swapped, diff_col] != 0).sum()
            flip_rates[key] = {
                "flipped": int(flipped),
                "total": int(total),
                "rate": float(flipped / total),
            }
    return flip_rates


def compute_semsim(df_results: pd.DataFrame, lang: str = "") -> dict:
    labse = _get_labse()
    semsim_results = {}

    for pos in tqdm(POS_TAGS, desc=f"SemSim {lang}"):
        for position in POSITIONS:
            key = f"{pos}_{position}"
            sentence_col = f"{key}_sentence"
            diff_col = f"{key}_score_diff"
            if sentence_col not in df_results.columns:
                continue
            valid = df_results[df_results[sentence_col].notna()]
            if len(valid) == 0:
                continue

            orig_emb = labse.encode(valid["original_text"].tolist(), batch_size=64,
                                    convert_to_numpy=True, show_progress_bar=False)
            swap_emb = labse.encode(valid[sentence_col].tolist(), batch_size=64,
                                    convert_to_numpy=True, show_progress_bar=False)
            sims = [float(util.cos_sim(o, s)) for o, s in zip(orig_emb, swap_emb)]

            high_sim = np.array(sims) >= 0.8
            flipped = valid[diff_col].values != 0
            flip_at_high_sim = (high_sim & flipped).sum()
            total_high_sim = high_sim.sum()

            semsim_results[key] = {
                "mean_similarity": float(np.mean(sims)),
                "flip_rate_at_high_sim": float(flip_at_high_sim / total_high_sim) if total_high_sim > 0 else 0.0,
                "total_high_sim": int(total_high_sim),
            }
    return semsim_results


def compute_robustness_gap(df_results: pd.DataFrame, lang: str = "") -> dict:
    orig_correct = df_results.apply(
        lambda r: _is_correct(r["original_score"], r["label"]), axis=1
    ).mean()

    gaps = {}
    for pos in tqdm(POS_TAGS, desc=f"Robustness Gap {lang}"):
        for position in POSITIONS:
            key = f"{pos}_{position}"
            score_col = f"{key}_score"
            sentence_col = f"{key}_sentence"
            if score_col not in df_results.columns:
                continue
            valid = df_results[df_results[sentence_col].notna()].copy()
            if len(valid) == 0:
                continue
            swap_correct = valid.apply(
                lambda r: _is_correct(r[score_col], r["label"]), axis=1
            ).mean()
            gaps[key] = {
                "original_accuracy": float(orig_correct),
                "swapped_accuracy": float(swap_correct),
                "robustness_gap": float(orig_correct - swap_correct),
            }
    return gaps


def compute_flip_rate_by_lmi(
    df_results: pd.DataFrame, source_lang: str = "en", n_bins: int = 5
) -> pd.DataFrame | None:
    lmi_flip_data = []
    for pos in tqdm(POS_TAGS, desc=f"LMI {source_lang.upper()}"):
        for position in POSITIONS:
            key = f"{pos}_{position}"
            sentence_col = f"{key}_sentence"
            diff_col = f"{key}_score_diff"
            if sentence_col not in df_results.columns:
                continue
            valid = df_results[df_results[sentence_col].notna()].copy()
            if len(valid) == 0:
                continue
            valid["lmi"] = valid[sentence_col].apply(lambda s: compute_lmi(s, lang=source_lang))
            valid["flipped"] = valid[diff_col] != 0
            valid["swap_key"] = key
            lmi_flip_data.append(valid[["lmi", "flipped", "swap_key"]])

    if not lmi_flip_data:
        return None

    combined = pd.concat(lmi_flip_data)
    combined["lmi_bin"] = pd.cut(combined["lmi"], bins=n_bins)
    summary = combined.groupby("lmi_bin")["flipped"].agg(["mean", "count"])
    summary.columns = ["flip_rate", "count"]
    return summary


def compute_asymmetry(df_en: pd.DataFrame, df_es: pd.DataFrame) -> dict:
    asymmetry = {}
    for pos in tqdm(POS_TAGS, desc="Asymmetry EN↔ES"):
        for position in POSITIONS:
            key = f"{pos}_{position}"
            diff_col = f"{key}_score_diff"
            sentence_col = f"{key}_sentence"

            en_valid = df_en[df_en[sentence_col].notna()] if sentence_col in df_en.columns else pd.DataFrame()
            es_valid = df_es[df_es[sentence_col].notna()] if sentence_col in df_es.columns else pd.DataFrame()

            en_rate = float((en_valid[diff_col] != 0).sum() / len(en_valid)) if len(en_valid) > 0 else 0.0
            es_rate = float((es_valid[diff_col] != 0).sum() / len(es_valid)) if len(es_valid) > 0 else 0.0

            asymmetry[key] = {
                "en_to_es_flip_rate": en_rate,
                "es_to_en_flip_rate": es_rate,
                "asymmetry": en_rate - es_rate,
            }
    return asymmetry


def print_all_metrics(fr, semsim, gap, lmi, asymmetry, lang: str = "en"):
    print(f"\n{'='*60}")
    print(f"FULL METRICS REPORT - {lang.upper()}")
    print(f"{'='*60}")

    print("\n--- Flip Rate ---")
    for key, val in fr.items():
        print(f"{key:<20} {val['flipped']}/{val['total']} ({val['rate']*100:.1f}%)")

    print("\n--- Semantic Similarity & Flip Rate at High SemSim ---")
    for key, val in semsim.items():
        print(f"{key:<20} mean_sim={val['mean_similarity']:.3f} "
              f"flip@sim>=0.8={val['flip_rate_at_high_sim']*100:.1f}% "
              f"(n={val['total_high_sim']})")

    print("\n--- Robustness Gap ---")
    for key, val in gap.items():
        print(f"{key:<20} orig_acc={val['original_accuracy']*100:.1f}% "
              f"swap_acc={val['swapped_accuracy']*100:.1f}% "
              f"gap={val['robustness_gap']*100:.1f}%")

    print("\n--- LMI Flip Rate ---")
    print(lmi)

    print("\n--- Asymmetry (en->es vs es->en) ---")
    for key, val in asymmetry.items():
        print(f"{key:<20} en->es={val['en_to_es_flip_rate']*100:.1f}% "
              f"es->en={val['es_to_en_flip_rate']*100:.1f}% "
              f"diff={val['asymmetry']*100:+.1f}%")


# ── Multi-swap metrics ────────────────────────────────────────────────────────

def compute_flip_rate_by_n(df_results: pd.DataFrame, max_swaps: int = 3) -> dict:
    rates = {}
    for n in range(1, max_swaps + 1):
        key = f"swap_{n}"
        sentence_col = f"{key}_sentence"
        flipped_col = f"{key}_flipped"
        if sentence_col not in df_results.columns:
            continue
        valid = df_results[df_results[sentence_col].notna()]
        total = len(valid)
        if total == 0:
            continue
        flipped = valid[flipped_col].sum()
        rates[n] = {"flipped": int(flipped), "total": int(total), "rate": float(flipped / total)}
    return rates


def compute_msdf_from_multi(df_results: pd.DataFrame, max_swaps: int = 3) -> pd.DataFrame:
    msdf_values = []
    for _, row in df_results.iterrows():
        min_flips = None
        for n in range(1, max_swaps + 1):
            flipped_col = f"swap_{n}_flipped"
            if flipped_col in df_results.columns and row.get(flipped_col) is True:
                min_flips = n
                break
        msdf_values.append(min_flips)
    df_results = df_results.copy()
    df_results["msdf"] = msdf_values
    return df_results


def compute_semsim_by_n(df_results: pd.DataFrame, max_swaps: int = 3) -> dict:
    labse = _get_labse()
    semsim_results = {}
    for n in tqdm(range(1, max_swaps + 1), desc="SemSim by swap count"):
        key = f"swap_{n}"
        sentence_col = f"{key}_sentence"
        flipped_col = f"{key}_flipped"
        if sentence_col not in df_results.columns:
            continue
        valid = df_results[df_results[sentence_col].notna()]
        if len(valid) == 0:
            continue
        orig_emb = labse.encode(valid["original_text"].tolist(), batch_size=64,
                                convert_to_numpy=True, show_progress_bar=False)
        swap_emb = labse.encode(valid[sentence_col].tolist(), batch_size=64,
                                convert_to_numpy=True, show_progress_bar=False)
        sims = [float(util.cos_sim(o, s)) for o, s in zip(orig_emb, swap_emb)]
        flipped = valid[flipped_col].values
        high_sim = np.array(sims) >= 0.8
        flip_at_high_sim = (high_sim & flipped).sum()
        total_high_sim = high_sim.sum()
        semsim_results[n] = {
            "mean_similarity": float(np.mean(sims)),
            "flip_rate_at_high_sim": float(flip_at_high_sim / total_high_sim) if total_high_sim > 0 else 0.0,
            "total_high_sim": int(total_high_sim),
        }
    return semsim_results


def compute_robustness_gap_by_n(df_results: pd.DataFrame, max_swaps: int = 3) -> dict:
    orig_correct = df_results.apply(
        lambda r: _is_correct(r["original_score"], r["label"]), axis=1
    ).mean()
    gaps = {}
    for n in tqdm(range(1, max_swaps + 1), desc="Robustness Gap by swap count"):
        key = f"swap_{n}"
        score_col = f"{key}_score"
        sentence_col = f"{key}_sentence"
        if score_col not in df_results.columns:
            continue
        valid = df_results[df_results[sentence_col].notna()].copy()
        if len(valid) == 0:
            continue
        swap_correct = valid.apply(
            lambda r: _is_correct(r[score_col], r["label"]), axis=1
        ).mean()
        gaps[n] = {
            "original_accuracy": float(orig_correct),
            "swapped_accuracy": float(swap_correct),
            "robustness_gap": float(orig_correct - swap_correct),
        }
    return gaps


def compute_lmi_by_n(
    df_results: pd.DataFrame, source_lang: str = "en", max_swaps: int = 3, n_bins: int = 5
) -> dict:
    lmi_results = {}
    for n in tqdm(range(1, max_swaps + 1), desc="LMI by swap count"):
        key = f"swap_{n}"
        sentence_col = f"{key}_sentence"
        flipped_col = f"{key}_flipped"
        if sentence_col not in df_results.columns:
            continue
        valid = df_results[df_results[sentence_col].notna()].copy()
        if len(valid) == 0:
            continue
        valid["lmi"] = valid[sentence_col].apply(lambda s: compute_lmi(s, lang=source_lang))
        valid["lmi_bin"] = pd.cut(valid["lmi"], bins=n_bins)
        summary = valid.groupby("lmi_bin")[flipped_col].agg(["mean", "count"])
        summary.columns = ["flip_rate", "count"]
        lmi_results[n] = summary
    return lmi_results


def compute_asymmetry_by_n(
    df_en: pd.DataFrame, df_es: pd.DataFrame, max_swaps: int = 3
) -> dict:
    asymmetry = {}
    for n in range(1, max_swaps + 1):
        key = f"swap_{n}"
        flipped_col = f"{key}_flipped"
        sentence_col = f"{key}_sentence"
        en_valid = df_en[df_en[sentence_col].notna()] if sentence_col in df_en.columns else pd.DataFrame()
        es_valid = df_es[df_es[sentence_col].notna()] if sentence_col in df_es.columns else pd.DataFrame()
        en_rate = float(en_valid[flipped_col].mean()) if len(en_valid) > 0 else 0.0
        es_rate = float(es_valid[flipped_col].mean()) if len(es_valid) > 0 else 0.0
        asymmetry[n] = {
            "en_to_es_flip_rate": en_rate,
            "es_to_en_flip_rate": es_rate,
            "asymmetry": en_rate - es_rate,
        }
    return asymmetry


def print_all_metrics_by_n(fr, semsim, gap, lmi, asymmetry, msdf_df, lang: str = "en", max_swaps: int = 3):
    print(f"\n{'='*60}")
    print(f"METRICS REPORT — {lang.upper()}")
    print(f"{'='*60}")

    print("\n--- Flip Rate by Swap Count ---")
    print(f"{'N swaps':<12} {'Flipped':<10} {'Total':<10} {'Rate'}")
    print("-" * 42)
    for n, val in fr.items():
        print(f"{n:<12} {val['flipped']:<10} {val['total']:<10} {val['rate']*100:.1f}%")

    print("\n--- MSDF (Minimum Swaps to Flip) ---")
    msdf_counts = msdf_df["msdf"].value_counts().sort_index()
    never_flipped = msdf_df["msdf"].isna().sum()
    print(f"Mean MSDF: {msdf_df['msdf'].mean():.2f}")
    print(f"Never flipped: {never_flipped} / {len(msdf_df)} sentences")
    for n, count in msdf_counts.items():
        print(f"  Flipped at swap {int(n)}: {count} sentences")

    print("\n--- Semantic Similarity by Swap Count ---")
    for n, val in semsim.items():
        print(f"Swap {n}: mean_sim={val['mean_similarity']:.3f} "
              f"flip@sim>=0.8={val['flip_rate_at_high_sim']*100:.1f}% "
              f"(n={val['total_high_sim']})")

    print("\n--- Robustness Gap by Swap Count ---")
    for n, val in gap.items():
        print(f"Swap {n}: orig_acc={val['original_accuracy']*100:.1f}% "
              f"swap_acc={val['swapped_accuracy']*100:.1f}% "
              f"gap={val['robustness_gap']*100:.1f}%")

    print("\n--- LMI Flip Rate by Swap Count ---")
    for n, summary in lmi.items():
        print(f"\nSwap {n}:")
        print(summary.to_string())

    print("\n--- Asymmetry by Swap Count ---")
    for n, val in asymmetry.items():
        print(f"Swap {n}: en->es={val['en_to_es_flip_rate']*100:.1f}% "
              f"es->en={val['es_to_en_flip_rate']*100:.1f}% "
              f"diff={val['asymmetry']*100:+.1f}%")
