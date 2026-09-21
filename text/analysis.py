"""Text analysis, topic modeling, name detection, and optional dating helpers."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from text.transcription import normalize_greek


VALID_ONE_LETTER_WORDS = {"Ο", "Η"}
GREEK_STOPWORDS = {
    "ΚΑΙ", "ΔΕ", "ΤΕ", "ΓΑΡ", "ΜΕΝ", "ΟΥΝ", "Ο", "Η", "ΤΟ", "ΤΟΥ",
    "ΤΗΣ", "ΤΩΝ", "ΤΟΝ", "ΤΗΝ", "ΤΟΙΣ", "ΤΑΙΣ", "ΤΟΥΣ", "ΤΑΣ", "ΕΝ",
    "ΕΙΣ", "ΕΚ", "ΕΞ", "ΕΠΙ", "ΠΡΟΣ", "ΑΠΟ", "ΥΠΕΡ", "ΚΑΤΑ", "ΔΙΑ",
    "ΜΕΤΑ", "ΠΑΡΑ",
}

NAME_LEXICON = [
    {"lemma": "ΑΛΕΞΑΝΔΡΟΣ", "stem": "ΑΛΕΞΑΝΔΡ", "gender": "male"},
    {"lemma": "ΔΙΟΝΥΣΙΟΣ", "stem": "ΔΙΟΝΥΣΙ", "gender": "male"},
    {"lemma": "ΔΗΜΗΤΡΙΟΣ", "stem": "ΔΗΜΗΤΡΙ", "gender": "male"},
    {"lemma": "ΘΕΟΔΩΡΟΣ", "stem": "ΘΕΟΔΩΡ", "gender": "male"},
    {"lemma": "ΑΠΟΛΛΩΝΙΟΣ", "stem": "ΑΠΟΛΛΩΝΙ", "gender": "male"},
    {"lemma": "ΗΡΑΚΛΕΙΔΗΣ", "stem": "ΗΡΑΚΛΕΙΔ", "gender": "male"},
    {"lemma": "ΦΙΛΙΠΠΟΣ", "stem": "ΦΙΛΙΠΠ", "gender": "male"},
    {"lemma": "ΝΙΚΟΛΑΟΣ", "stem": "ΝΙΚΟΛΑ", "gender": "male"},
    {"lemma": "ΠΤΟΛΕΜΑΙΟΣ", "stem": "ΠΤΟΛΕΜΑΙ", "gender": "male"},
    {"lemma": "ΣΩΚΡΑΤΗΣ", "stem": "ΣΩΚΡΑΤ", "gender": "male"},
    {"lemma": "ΜΕΝΑΝΔΡΟΣ", "stem": "ΜΕΝΑΝΔΡ", "gender": "male"},
    {"lemma": "ΑΠΟΛΛΟΔΩΡΟΣ", "stem": "ΑΠΟΛΛΟΔΩΡ", "gender": "male"},
    {"lemma": "ΑΡΙΣΤΩΝ", "stem": "ΑΡΙΣΤΩΝ", "gender": "male"},
    {"lemma": "ΘΕΩΝ", "stem": "ΘΕΩΝ", "gender": "male"},
    {"lemma": "ΑΜΜΩΝΙΟΣ", "stem": "ΑΜΜΩΝΙ", "gender": "male"},
    {"lemma": "ΣΑΡΑΠΙΩΝ", "stem": "ΣΑΡΑΠΙΩΝ", "gender": "male"},
    {"lemma": "ΗΡΩΔΗΣ", "stem": "ΗΡΩΔ", "gender": "male"},
    {"lemma": "ΣΩΤΗΡΙΧΟΣ", "stem": "ΣΩΤΗΡΙΧ", "gender": "male"},
    {"lemma": "ΖΩΣΙΜΟΣ", "stem": "ΖΩΣΙΜ", "gender": "male"},
    {"lemma": "ΑΓΑΘΟΚΛΗΣ", "stem": "ΑΓΑΘΟΚΛ", "gender": "male"},
    {"lemma": "ΑΓΑΘΩΝ", "stem": "ΑΓΑΘΩΝ", "gender": "male"},
    {"lemma": "ΑΙΣΧΙΝΗΣ", "stem": "ΑΙΣΧΙΝ", "gender": "male"},
    {"lemma": "ΑΝΔΡΟΝΙΚΟΣ", "stem": "ΑΝΔΡΟΝΙΚ", "gender": "male"},
    {"lemma": "ΑΝΤΙΓΟΝΟΣ", "stem": "ΑΝΤΙΓΟΝ", "gender": "male"},
    {"lemma": "ΑΝΤΙΠΑΤΡΟΣ", "stem": "ΑΝΤΙΠΑΤΡ", "gender": "male"},
    {"lemma": "ΑΤΤΙΚΟΣ", "stem": "ΑΤΤΙΚ", "gender": "male"},
    {"lemma": "ΔΗΜΟΣΘΕΝΗΣ", "stem": "ΔΗΜΟΣΘΕΝ", "gender": "male"},
    {"lemma": "ΔΙΟΔΩΡΟΣ", "stem": "ΔΙΟΔΩΡ", "gender": "male"},
    {"lemma": "ΔΙΟΓΕΝΗΣ", "stem": "ΔΙΟΓΕΝ", "gender": "male"},
    {"lemma": "ΕΥΚΛΕΙΔΗΣ", "stem": "ΕΥΚΛΕΙΔ", "gender": "male"},
    {"lemma": "ΗΡΑΚΛΗΣ", "stem": "ΗΡΑΚΛ", "gender": "male"},
    {"lemma": "ΘΕΟΦΙΛΟΣ", "stem": "ΘΕΟΦΙΛ", "gender": "male"},
    {"lemma": "ΙΣΟΚΡΑΤΗΣ", "stem": "ΙΣΟΚΡΑΤ", "gender": "male"},
    {"lemma": "ΚΑΛΛΙΚΡΑΤΗΣ", "stem": "ΚΑΛΛΙΚΡΑΤ", "gender": "male"},
    {"lemma": "ΛΥΣΑΝΔΡΟΣ", "stem": "ΛΥΣΑΝΔΡ", "gender": "male"},
    {"lemma": "ΝΙΚΙΑΣ", "stem": "ΝΙΚΙ", "gender": "male"},
    {"lemma": "ΞΕΝΟΦΩΝ", "stem": "ΞΕΝΟΦΩΝ", "gender": "male"},
    {"lemma": "ΠΕΡΙΚΛΗΣ", "stem": "ΠΕΡΙΚΛ", "gender": "male"},
    {"lemma": "ΠΛΑΤΩΝ", "stem": "ΠΛΑΤΩΝ", "gender": "male"},
    {"lemma": "ΣΤΡΑΤΩΝ", "stem": "ΣΤΡΑΤΩΝ", "gender": "male"},
    {"lemma": "ΤΙΜΟΘΕΟΣ", "stem": "ΤΙΜΟΘΕ", "gender": "male"},
    {"lemma": "ΧΑΙΡΕΦΩΝ", "stem": "ΧΑΙΡΕΦΩΝ", "gender": "male"},
    {"lemma": "ΕΥΤΥΧΟΣ", "stem": "ΕΥΤΥΧ", "gender": "male"},
    {"lemma": "ΕΡΜΙΑΣ", "stem": "ΕΡΜΙ", "gender": "male"},
    {"lemma": "ΚΛΑΥΔΙΟΣ", "stem": "ΚΛΑΥΔΙ", "gender": "male"},
    {"lemma": "ΕΥΟΔΟΣ", "stem": "ΕΥΟΔ", "gender": "male"},
    {"lemma": "ΕΥΚΑΡΠΟΣ", "stem": "ΕΥΚΑΡΠ", "gender": "male"},
    {"lemma": "ΚΛΕΟΠΑΤΡΑ", "stem": "ΚΛΕΟΠΑΤΡ", "gender": "female"},
    {"lemma": "ΑΡΣΙΝΟΗ", "stem": "ΑΡΣΙΝΟ", "gender": "female"},
    {"lemma": "ΒΕΡΕΝΙΚΗ", "stem": "ΒΕΡΕΝΙΚ", "gender": "female"},
    {"lemma": "ΕΙΡΗΝΗ", "stem": "ΕΙΡΗΝ", "gender": "female"},
    {"lemma": "ΣΩΤΗΡΙΑ", "stem": "ΣΩΤΗΡΙ", "gender": "female"},
    {"lemma": "ΑΓΑΘΗ", "stem": "ΑΓΑΘ", "gender": "female"},
    {"lemma": "ΑΛΕΞΑΝΔΡΑ", "stem": "ΑΛΕΞΑΝΔΡ", "gender": "female"},
    {"lemma": "ΑΝΤΙΓΟΝΗ", "stem": "ΑΝΤΙΓΟΝ", "gender": "female"},
    {"lemma": "ΑΡΤΕΜΙΣ", "stem": "ΑΡΤΕΜΙ", "gender": "female"},
    {"lemma": "ΑΘΗΝΑΙΣ", "stem": "ΑΘΗΝΑΙ", "gender": "female"},
    {"lemma": "ΔΗΜΗΤΡΑ", "stem": "ΔΗΜΗΤΡ", "gender": "female"},
    {"lemma": "ΔΙΟΝΥΣΙΑ", "stem": "ΔΙΟΝΥΣΙ", "gender": "female"},
    {"lemma": "ΕΥΤΥΧΙΑ", "stem": "ΕΥΤΥΧΙ", "gender": "female"},
    {"lemma": "ΘΕΟΔΩΡΑ", "stem": "ΘΕΟΔΩΡ", "gender": "female"},
    {"lemma": "ΙΟΥΛΙΑ", "stem": "ΙΟΥΛΙ", "gender": "female"},
    {"lemma": "ΚΛΑΥΔΙΑ", "stem": "ΚΛΑΥΔΙ", "gender": "female"},
    {"lemma": "ΖΩΣΙΜΗ", "stem": "ΖΩΣΙΜ", "gender": "female"},
]
NAME_LEXICON = sorted(NAME_LEXICON, key=lambda item: len(item["stem"]), reverse=True)

NAME_STOPWORDS = {
    "ΑΓΑΘΗΙ", "ΤΥΧΗΙ", "ΒΟΥΛΗ", "ΒΟΥΛΗΙ", "ΔΗΜΟΣ", "ΔΗΜΩΙ", "ΠΟΛΙΣ",
    "ΠΟΛΕΩΣ", "ΕΔΟΞΕΝ", "ΚΑΙ", "ΔΕ", "ΤΟΥ", "ΤΗΣ", "ΤΩΝ", "ΤΟΝ",
    "ΤΗΝ", "ΤΟΙΣ", "ΤΑΙΣ", "ΕΝ", "ΕΠΙ", "ΠΕΡΙ", "ΠΡΟΣ", "ΚΑΤΑ",
    "ΑΠΟ", "ΥΠΕΡ", "ΠΑΡΑ", "ΜΕΤΑ", "ΔΙΑ",
}
PLAUSIBLE_NAME_ENDINGS = {
    "", "Σ", "Ν", "ΟΣ", "ΟΝ", "ΟΥ", "ΩΙ", "Ω", "Ε", "ΗΣ", "ΗΝ", "Η",
    "ΑΙ", "ΑΙΣ", "ΑΣ", "ΑΝ", "Α", "ΙΟΣ", "ΙΟΝ", "ΙΟΥ", "ΙΩΙ", "ΙΩΝ",
    "ΕΙΟΣ", "ΕΙΟΥ",
}


def is_clean_greek_word(token: str) -> bool:
    return re.fullmatch(r"[Α-Ω]+", token) is not None


def keep_clean_greek_token(token: str) -> bool:
    return is_clean_greek_word(token) and (len(token) >= 2 or token in VALID_ONE_LETTER_WORDS)


def normalize_name_text(value: str) -> str:
    value = normalize_greek(str(value))
    return re.sub(r"[^Α-Ω]", "", value)


def match_proper_name(word: str):
    word = normalize_name_text(word)
    if word in NAME_STOPWORDS or len(word) < 4:
        return None
    for entry in NAME_LEXICON:
        stem = entry["stem"]
        if word.startswith(stem) and word[len(stem):] in PLAUSIBLE_NAME_ENDINGS:
            return entry
    return None


def analyze_vocabulary(unique_segmented: dict[str, list[str]]):
    import pandas as pd

    clean_words = {
        inscription_id: [token for token in tokens if keep_clean_greek_token(token)]
        for inscription_id, tokens in unique_segmented.items()
    }
    all_words = [word for words in clean_words.values() for word in words]
    word_counts = Counter(all_words)
    summary = {
        "total_tokens": sum(word_counts.values()),
        "vocab_size": len(word_counts),
        "type_token_ratio": len(word_counts) / max(sum(word_counts.values()), 1),
        "hapax_count": sum(1 for count in word_counts.values() if count == 1),
        "rare_count_le_2": sum(1 for count in word_counts.values() if count <= 2),
    }
    word_freq_df = pd.DataFrame(word_counts.most_common(), columns=["word", "frequency"])
    return clean_words, word_counts, summary, word_freq_df


def analyze_names(clean_words: dict[str, list[str]]):
    import pandas as pd

    rows = []
    for inscription_id, words in clean_words.items():
        for word in words:
            match = match_proper_name(word)
            if match is not None:
                rows.append(
                    {
                        "inscription_id": inscription_id,
                        "token": word,
                        "lemma": match["lemma"],
                        "gender": match["gender"],
                        "source": "manual_stem_lexicon",
                    }
                )

    names_df = pd.DataFrame(rows, columns=["inscription_id", "token", "lemma", "gender", "source"])
    if names_df.empty:
        name_freq_df = pd.DataFrame(
            columns=["lemma", "gender", "frequency", "different_forms", "inscriptions", "observed_forms"]
        )
        gender_summary_df = pd.DataFrame(columns=["gender", "token_count", "name_types", "inscriptions"])
        return names_df, name_freq_df, gender_summary_df

    name_freq_df = (
        names_df.groupby(["lemma", "gender"], as_index=False)
        .agg(
            frequency=("token", "count"),
            different_forms=("token", "nunique"),
            inscriptions=("inscription_id", "nunique"),
            observed_forms=("token", lambda values: ", ".join(sorted(set(values)))),
        )
        .sort_values("frequency", ascending=False)
    )
    gender_summary_df = (
        names_df.groupby("gender", as_index=False)
        .agg(
            token_count=("token", "count"),
            name_types=("lemma", "nunique"),
            inscriptions=("inscription_id", "nunique"),
        )
        .sort_values("token_count", ascending=False)
    )
    return names_df, name_freq_df, gender_summary_df


def clean_tokens_for_clustering(tokens: list[str]) -> list[str]:
    return [
        token
        for token in tokens
        if keep_clean_greek_token(token) and len(token) >= 3 and token not in GREEK_STOPWORDS
    ]


def average_token_length(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def analyze_text_clusters(unique_segmented, greek_transcripts):
    import numpy as np
    import pandas as pd
    from scipy.sparse import hstack
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    rows = []
    for inscription_id, tokens in unique_segmented.items():
        clean_tokens = clean_tokens_for_clustering(tokens)
        raw_text = greek_transcripts.get(inscription_id, "")
        raw_continuous = re.sub(r"[^Α-Ω]", "", normalize_greek(raw_text))
        if not clean_tokens:
            continue
        rows.append(
            {
                "inscription_id": inscription_id,
                "tokens": tokens,
                "text": " ".join(clean_tokens),
                "raw_greek_text": raw_text,
                "raw_continuous_text": raw_continuous,
                "char_length": len(raw_text),
                "line_count": raw_text.count("\n") + 1 if raw_text else 0,
                "token_count": len(clean_tokens),
                "avg_token_length": average_token_length(clean_tokens),
                "damaged_marker_count": raw_text.count("*"),
                "damaged_marker_ratio": raw_text.count("*") / max(len(raw_text), 1),
            }
        )

    docs_df = pd.DataFrame(rows)
    if len(docs_df) < 3:
        return docs_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    texts = docs_df["text"].tolist()
    raw_texts = docs_df["raw_continuous_text"].tolist()
    word_vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[Α-Ω]{3,}\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=False,
        analyzer="char",
        ngram_range=(4, 8),
        min_df=2,
        max_df=0.85,
    )
    word_tfidf = word_vectorizer.fit_transform(texts)
    char_tfidf = char_vectorizer.fit_transform(raw_texts)
    features = hstack([0.75 * normalize(word_tfidf), 0.25 * normalize(char_tfidf)])

    silhouette_rows = []
    for k in range(2, min(8, len(docs_df) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)
        silhouette_rows.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(features, labels, metric="cosine"),
                "cluster_sizes": dict(Counter(labels)),
            }
        )
    silhouette_df = pd.DataFrame(silhouette_rows)
    best_k = int(silhouette_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    docs_df["cluster"] = kmeans.fit_predict(features)

    n_svd = min(3, max(1, min(features.shape) - 1))
    coords = TruncatedSVD(n_components=n_svd, random_state=42).fit_transform(features)
    for idx, col in enumerate(["svd_x", "svd_y", "svd_z"][:n_svd]):
        docs_df[col] = coords[:, idx]

    phrase_vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[Α-Ω]{3,}\b",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.90,
    )
    phrase_tfidf = phrase_vectorizer.fit_transform(texts)
    phrase_terms = phrase_vectorizer.get_feature_names_out()
    phrase_rows = []
    for cluster_id in sorted(docs_df["cluster"].unique()):
        mask = docs_df["cluster"].values == cluster_id
        mean_scores = np.asarray(phrase_tfidf[mask].mean(axis=0)).ravel()
        for index in mean_scores.argsort()[::-1][:10]:
            if mean_scores[index] > 0:
                phrase_rows.append(
                    {"cluster": cluster_id, "phrase": phrase_terms[index], "mean_tfidf": mean_scores[index]}
                )
    phrases_df = pd.DataFrame(phrase_rows)

    topic_vectorizer = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[Α-Ω]{3,}\b",
        min_df=2,
        max_df=0.75,
    )
    topic_counts = topic_vectorizer.fit_transform(texts)
    topic_terms = topic_vectorizer.get_feature_names_out()
    lda = LatentDirichletAllocation(
        n_components=min(best_k, 8),
        random_state=42,
        learning_method="batch",
        max_iter=50,
        doc_topic_prior=0.1,
        topic_word_prior=0.1,
    )
    topic_matrix = lda.fit_transform(topic_counts)
    docs_df["dominant_topic"] = topic_matrix.argmax(axis=1)
    docs_df["dominant_topic_probability"] = topic_matrix.max(axis=1)

    topic_rows = []
    for topic_id, weights in enumerate(lda.components_):
        top_words = [topic_terms[i] for i in weights.argsort()[::-1][:25]]
        topic_rows.append({"topic": topic_id, "top_words": ", ".join(top_words)})
    topics_df = pd.DataFrame(topic_rows)
    return docs_df, silhouette_df, phrases_df, topics_df


def clean_model_text(text: str) -> str:
    text = normalize_greek(str(text)).replace("*", "#")
    text = re.sub(r"[^Α-Ω#\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_for_aeneas(text: str, max_chars: int = 740) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return trimmed if trimmed else text[:max_chars]


def prepare_aeneas_candidates(unique_segmented, greek_transcripts):
    import pandas as pd

    rows = []
    for inscription_id, tokens in unique_segmented.items():
        raw_greek_text = greek_transcripts.get(inscription_id, "")
        segmented_text = " ".join(tokens)
        text_for_model = clean_model_text(segmented_text)
        if len(text_for_model) < 50:
            raw_model_text = clean_model_text(raw_greek_text)
            if len(raw_model_text) > len(text_for_model):
                text_for_model = raw_model_text
        text_for_model = trim_for_aeneas(text_for_model)
        readable_chars = len(re.sub(r"[^Α-Ω]", "", text_for_model))
        damage_ratio = text_for_model.count("#") / max(len(text_for_model), 1)
        rows.append(
            {
                "inscription_id": inscription_id,
                "text_for_model": text_for_model,
                "model_char_count": len(text_for_model),
                "readable_char_count": readable_chars,
                "damage_ratio": damage_ratio,
                "aeneas_length_ok": len(text_for_model) >= 50,
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["selection_score"] = (
        candidates["aeneas_length_ok"].astype(int) * 10_000
        + candidates["readable_char_count"]
        - candidates["damage_ratio"] * 1_000
    )
    return candidates.sort_values(
        ["selection_score", "damage_ratio", "model_char_count"],
        ascending=[False, True, False],
    )


def year_label(year: int) -> str:
    year = int(year)
    if year < 0:
        return f"{abs(year)} BCE"
    if year == 0:
        return "around 1 BCE/CE"
    return f"{year} CE"


def decade_label(year: int) -> str:
    return f"{year_label(year)} to {year_label(year + 10)}"


def load_aeneas_attributor(aeneas_repo_dir: str | Path, aeneas_model_dir: str | Path):
    import pickle

    import jax
    import numpy as np

    repo_dir = Path(aeneas_repo_dir).expanduser().resolve()
    model_dir = Path(aeneas_model_dir).expanduser().resolve()
    checkpoint_path = model_dir / "ithaca_153143996_2.pkl"
    dataset_path = model_dir / "iphi.json"
    retrieval_path = model_dir / "iphi_emb_xid153143996.pkl"

    missing = [
        str(path)
        for path in (checkpoint_path, dataset_path, retrieval_path)
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError("Aeneas model files are missing. Expected:\n" + "\n".join(missing))

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from predictingthepast.eval import inference as aeneas_inference
    from predictingthepast.models.model import Model
    from predictingthepast.util import alphabet as util_alphabet

    with open(checkpoint_path, "rb") as handle:
        checkpoint = pickle.load(handle)

    params = jax.device_put(checkpoint["params"])
    model = Model(**checkpoint["model_config"])
    forward = model.apply
    alphabet = util_alphabet.GreekAlphabet()
    vocab_char_size = checkpoint["model_config"]["vocab_char_size"]

    def predict_date(input_text: str) -> dict:
        import numpy as np

        input_text = re.sub(r"\s+", " ", str(input_text)).strip()
        if len(input_text) < 25:
            raise ValueError(f"Input text too short for Aeneas: {len(input_text)} chars")
        if len(input_text) >= 768:
            input_text = input_text[:740].rsplit(" ", 1)[0] or input_text[:740]

        attribution = aeneas_inference.attribute(
            input_text,
            forward=forward,
            params=params,
            alphabet=alphabet,
            vocab_char_size=vocab_char_size,
        )
        year_scores = np.asarray(attribution.year_scores, dtype=float)
        if year_scores.size == 0 or np.isnan(year_scores).all():
            raise ValueError("Aeneas returned empty/NaN year_scores.")

        years = np.arange(-800, -800 + (len(year_scores) * 10), 10)
        top_indices = np.argsort(year_scores)[::-1][:3]
        best_index = int(top_indices[0])
        best_year = int(years[best_index])
        top_3 = [
            f"{decade_label(int(years[index]))} ({float(year_scores[index]):.3f})"
            for index in top_indices
        ]
        return {
            "aeneas_predicted_date": decade_label(best_year),
            "aeneas_confidence": float(year_scores[best_index]),
            "aeneas_raw_year": best_year,
            "aeneas_top_3_decades": "; ".join(top_3),
            "aeneas_status": "ok",
        }

    return predict_date


def predict_aeneas_dates(candidates_df, aeneas_repo_dir: str | Path, aeneas_model_dir: str | Path):
    import pandas as pd

    predict_date = load_aeneas_attributor(aeneas_repo_dir, aeneas_model_dir)
    rows = []
    selected = candidates_df[candidates_df["aeneas_length_ok"]].head(10)
    for _, row in selected.iterrows():
        try:
            prediction = predict_date(row["text_for_model"])
        except Exception as error:
            prediction = {
                "aeneas_predicted_date": "Aeneas prediction failed",
                "aeneas_confidence": None,
                "aeneas_raw_year": None,
                "aeneas_top_3_decades": None,
                "aeneas_status": repr(error),
            }
        rows.append({"inscription_id": row["inscription_id"], **prediction})
    return pd.DataFrame(rows)

