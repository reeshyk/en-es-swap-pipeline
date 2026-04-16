import torch
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import word_frequency

import model_setup as ms
from config import POS_TAGS, POSITIONS


@lru_cache(maxsize=10000)
def translate_sentence(sentence: str, source_lang: str = "en") -> str:
    if source_lang == "en":
        tokenizer, trans_model = ms.en_es_tokenizer, ms.en_es_model
    else:
        tokenizer, trans_model = ms.es_en_tokenizer, ms.es_en_model

    inputs = tokenizer(
        [sentence], return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(ms.device)
    with torch.no_grad():
        translated = trans_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def get_best_replacement(sentence: str, token, target_lemma: str, source_lang: str = "en") -> str:
    target_lang = "es" if source_lang == "en" else "en"
    target_nlp = ms.nlp_es if target_lang == "es" else ms.nlp_en

    if token.pos_ == "VERB":
        translated_sentence = translate_sentence(sentence, source_lang)
        target_doc = target_nlp(translated_sentence)
        for t in target_doc:
            if t.pos_ == "VERB" and t.lemma_.lower() == target_lemma:
                return t.text.lower()
    return target_lemma


def score_in_context(sentence: str, original_word: str, replacement: str) -> float:
    masked = sentence.replace(original_word, ms.mlm_tokenizer.mask_token, 1)
    inputs = ms.mlm_tokenizer(
        masked, return_tensors="pt", truncation=True, max_length=512
    ).to(ms.device)

    mask_idx = (inputs["input_ids"] == ms.mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_idx) == 0:
        return float("-inf")

    with torch.no_grad():
        outputs = ms.mlm_model(**inputs)

    logits = outputs.logits[0, mask_idx[0]]
    probs = torch.log_softmax(logits, dim=-1)

    token_id = ms.mlm_tokenizer.convert_tokens_to_ids(replacement)
    if token_id == ms.mlm_tokenizer.unk_token_id:
        return float("-inf")

    return probs[token_id].item()


def verify_semantic_similarity(source_word: str, replacement: str, threshold: float = 0.5):
    embeddings = ms.st_model.encode([source_word, replacement], convert_to_numpy=True)
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return similarity >= threshold, similarity


def get_position(token, doc) -> str:
    for sent in doc.sents:
        if sent.start <= token.i < sent.end:
            sent_length = sent.end - sent.start
            relative_pos = token.i - sent.start
            third = sent_length / 3
            if relative_pos < third:
                return "beginning"
            elif relative_pos < third * 2:
                return "middle"
            else:
                return "end"
    return "middle"


def _find_replacement(sentence, token, dictionary, source_lang):
    target_lang = "es" if source_lang == "en" else "en"
    lookup = token.lemma_.lower() if token.pos_ == "VERB" else token.text.lower()

    if lookup not in dictionary:
        return None

    for target_lemma in dictionary[lookup]:
        if target_lemma == lookup:
            continue
        if word_frequency(target_lemma, target_lang) < word_frequency(target_lemma, source_lang):
            continue

        replacement = get_best_replacement(sentence, token, target_lemma, source_lang)
        context_score = score_in_context(sentence, token.text, replacement)
        if context_score == float("-inf"):
            context_score = score_in_context(sentence, token.text, target_lemma)
            if context_score == float("-inf"):
                continue
            replacement = target_lemma

        is_similar, similarity = verify_semantic_similarity(token.text, replacement)
        if not is_similar:
            continue

        return replacement, similarity, context_score

    return None


def swap_word(sentence: str, source_lang: str = "en", pos: str | None = None, position: str | None = None):
    dictionary = ms.en_to_es_dict if source_lang == "en" else ms.es_to_en_dict
    nlp = ms.nlp_en if source_lang == "en" else ms.nlp_es

    doc = nlp(sentence)
    eligible = [t for t in doc if t.pos_ in ([pos.upper()] if pos else POS_TAGS)]
    if position:
        eligible = [t for t in eligible if get_position(t, doc) == position.lower()]

    if not eligible:
        return "no_eligible"

    result = _find_replacement(sentence, eligible[0], dictionary, source_lang)
    if result is None:
        return "no_candidates"

    replacement, similarity, context_score = result
    return {
        "sentence": sentence.replace(eligible[0].text, replacement, 1),
        "original_word": eligible[0].text,
        "replacement": replacement,
        "similarity": similarity,
        "context_score": context_score,
    }


def swap_word_fast(
    sentence: str,
    doc,
    translated_doc,
    source_lang: str = "en",
    pos: str | None = None,
    position: str | None = None,
    exclude_words: set | None = None,
):
    dictionary = ms.en_to_es_dict if source_lang == "en" else ms.es_to_en_dict
    exclude_words = exclude_words or set()

    eligible = [
        t for t in doc
        if t.pos_ in ([pos.upper()] if pos else POS_TAGS)
        and t.text.lower() not in exclude_words
    ]
    if position:
        eligible = [t for t in eligible if get_position(t, doc) == position.lower()]

    if not eligible:
        return "no_eligible"

    result = _find_replacement(sentence, eligible[0], dictionary, source_lang)
    if result is None:
        return "no_candidates"

    replacement, similarity, context_score = result
    return {
        "sentence": sentence.replace(eligible[0].text, replacement, 1),
        "original_word": eligible[0].text,
        "replacement": replacement,
        "similarity": similarity,
        "context_score": context_score,
    }
