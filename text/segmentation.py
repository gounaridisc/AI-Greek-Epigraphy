"""Vocabulary building and scriptio continua word segmentation."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

from text.transcription import normalize_greek


ALPHABET = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
HF_VOCAB_DATASET = "Ericu950/Inscriptions_1"
VOCAB_TEXT_FIELD = "Edition_with_brackets"
CLEAN_GREEK_WORD_RE = re.compile(r"^[Α-Ω]+$")
EDGE_PUNCTUATION = ".,;:··!?\"'“”‘’"
EDITORIAL_BRACKETS = "[](){}⟦⟧⟨⟩<>"
DAMAGE_MARKERS = {"-", "–", "—", "*", "…", "̣", "?"}

TRUSTED_ONE_LETTER_WORDS = {"Ο", "Η"}
TRUSTED_SHORT_WORDS = {
    "ΤΟ", "ΔΕ", "ΕΝ", "ΕΙ", "ΕΚ", "ΕΞ", "ΟΙ", "ΑΙ", "ΤΗ", "ΤΩ",
    "ΩΣ", "ΜΗ", "ΟΥ", "ΑΝ", "ΓΕ", "ΤΑ", "ΟΝ", "ΩΝ",
}

SEED_WORDS = [
    "Ο", "Η", "ΤΟ", "ΤΟΥ", "ΤΗΝ", "ΤΗΣ", "ΤΩΝ", "ΚΑΙ", "ΔΕ", "ΕΝ",
    "ΕΙΣ", "ΕΠΙ", "ΠΡΟΣ", "ΑΠΟ", "ΥΠΕΡ", "ΘΕΟΣ", "ΔΗΜΟΣ", "ΔΗΜΟΥ",
    "ΒΟΥΛΗ", "ΑΝΔΡΑ", "ΠΟΛΕΩΣ", "ΑΓΑΘΗΙ", "ΑΘΗΝΑΙ", "ΠΡΟΣΟΔΟΣ", "ΩΣ",
]

EXTRA_WORDS = [
    "ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ", "ΒΟΥΛΗ", "ΔΗΜΩΙ", "ΔΗΜΟΣ", "ΔΗΜΟΥ",
    "ΓΡΑΜΜΑΤΕΥΣ", "ΓΡΑΜΜΑΤΕΙ", "ΓΡΑΜΜΑΤΕΑ", "ΓΡΑΜΜΑΤΕΩΣ", "ΕΙΠΕ",
    "ΕΠΑΙΝΕΣΑΙ", "ΣΤΕΦΑΝΩΣΑΙ", "ΠΟΛΙΝ", "ΠΟΛΕΩΣ", "ΠΟΛΕΙ", "ΠΟΛΗΙ",
    "ΠΟΛΙΤΗΣ", "ΠΟΛΙΤΗΝ", "ΠΟΛΙΤΑΙ", "ΠΟΛΙΤΑΣ", "ΠΟΛΙΤΩΝ", "ΠΟΛΙΤΑΙΣ",
    "ΑΡΧΟΝΤΟΣ", "ΑΓΑΘΗΙ", "ΤΥΧΗΙ", "ΨΗΦΙΣΜΑ", "ΚΑΙ", "ΔΕ", "ΤΟΥ",
    "ΤΗΣ", "ΤΩΝ", "ΤΟΝ", "ΤΗΝ", "ΤΟΙΣ", "ΤΑΙΣ", "ΕΝ", "ΕΠΙ", "ΠΕΡΙ",
    "ΠΡΟΣ", "ΚΑΤΑ", "ΑΘΗΝΑΙ", "ΘΥΣΙΑ", "ΠΡΟΣΟΔΟΣ", "ΩΣ", "ΔΕΔΟΧΘΑΙ",
    "ΕΚΚΛΗΣΙΑ", "ΕΚΚΛΗΣΙΑΙ", "ΠΡΥΤΑΝΕΙΣ", "ΠΡΥΤΑΝΕΩΝ", "ΠΡΥΤΑΝΕΙΑΣ",
    "ΑΡΧΩΝ", "ΑΡΧΟΝΤΙ", "ΑΡΧΟΝΤΕΣ", "ΝΟΜΟΘΕΤΑΙ", "ΝΟΜΟΘΕΤΩΝ",
    "ΝΟΜΟΘΕΤΑΙΣ", "ΣΤΡΑΤΗΓΟΣ", "ΣΤΡΑΤΗΓΟΙ", "ΙΕΡΕΥΣ", "ΙΕΡΕΙΑ",
    "ΙΕΡΟΝ", "ΙΕΡΟΥ", "ΘΕΟΙΣ", "ΘΕΩΙ", "ΘΕΟΥ", "ΑΝΕΘΗΚΕΝ",
    "ΑΝΕΘΗΚΑΝ", "ΤΙΜΗΣΑΙ", "ΤΙΜΗΣΕΝ", "ΤΙΜΗΝ", "ΣΤΕΦΑΝΩΙ", "ΧΡΥΣΩΙ",
    "ΑΡΕΤΗΣ", "ΕΥΝΟΙΑΣ", "ΔΙΚΑΙΟΣΥΝΗΣ", "ΜΕΝ", "ΓΑΡ", "ΟΥΝ", "ΠΑΡΑ",
    "ΜΕΤΑ", "ΥΠΟ", "ΥΠΕΡ", "ΔΙΑ", "ΑΝΤΙ", "ΜΕΧΡΙ", "ΑΧΡΙ", "ΤΟΙ",
    "ΤΑΙ", "ΤΑΣ", "ΤΟΥΣ", "ΤΑ", "ΟΝ", "ΩΝ", "ΟΙΣ", "ΑΙΣ",
]

EXTRA_WORD_WEIGHT = 5_000
OOV_BASE = 8.0
INSERTION_PENALTY = 4.0
MAX_UNKNOWN_WORD_LEN = 16
ONE_LETTER_UNKNOWN_PENALTY = 1_000_000
KNOWN_WORD_BONUS = 2.0
KNOWN_LENGTH_BONUS = 0.20
MANUAL_WORD_BONUS = 1.5
SHORT_WORD_MAX_LEN_FOR_NO_BONUS = 2
WILDCARD_PENALTY = 1.2
STAR_MATCH_LETTERS = set(ALPHABET)
MAX_WILDCARD_STARS_TO_EXPAND = 2


def keep_vocab_word(word: str) -> bool:
    if len(word) == 1:
        return word in TRUSTED_ONE_LETTER_WORDS
    if len(word) == 2:
        return word in TRUSTED_SHORT_WORDS
    return 3 <= len(word) <= 25


def clean_vocab_words(text: str):
    for raw_token in str(text).split():
        token = raw_token.strip(EDGE_PUNCTUATION)
        for bracket in EDITORIAL_BRACKETS:
            token = token.replace(bracket, "")
        if any(marker in token for marker in DAMAGE_MARKERS):
            continue
        normalized = normalize_greek(token)
        if CLEAN_GREEK_WORD_RE.fullmatch(normalized) and keep_vocab_word(normalized):
            yield normalized


def add_weighted_words(counter: Counter, words: list[str], weight: int) -> None:
    for word in words:
        normalized = normalize_greek(word)
        if keep_vocab_word(normalized):
            counter[normalized] += weight


def build_vocabulary(max_rows: int | None = None, skip_hf: bool = False) -> Counter:
    counter: Counter[str] = Counter()
    rows_read = 0

    if not skip_hf:
        try:
            from datasets import load_dataset

            dataset = load_dataset(HF_VOCAB_DATASET, split="train", streaming=True)
            for row in dataset:
                if max_rows is not None and rows_read >= max_rows:
                    break
                for word in clean_vocab_words(row.get(VOCAB_TEXT_FIELD) or ""):
                    counter[word] += 1
                rows_read += 1
                if rows_read % 25_000 == 0:
                    print(f"  ... {rows_read:,} rows, {len(counter):,} vocabulary types")
        except Exception as error:
            print(f"Warning: could not load {HF_VOCAB_DATASET}: {error}")

    add_weighted_words(counter, SEED_WORDS + EXTRA_WORDS, EXTRA_WORD_WEIGHT)
    print(f"Vocabulary rows read: {rows_read:,}; vocabulary size: {len(counter):,}")
    return counter


def make_segmenter(word_freq: Counter):
    n_tokens = max(sum(word_freq.values()), 1)
    max_word_len = min(max((len(word) for word in word_freq), default=1), 25)
    manual_words = {normalize_greek(word) for word in SEED_WORDS + EXTRA_WORDS}

    @lru_cache(maxsize=250_000)
    def best_wildcard_match(pattern: str):
        if "*" not in pattern or all(char == "*" for char in pattern):
            return None, 0
        star_positions = [index for index, char in enumerate(pattern) if char == "*"]
        if len(star_positions) > MAX_WILDCARD_STARS_TO_EXPAND:
            return None, 0

        best_word = None
        best_count = 0
        chars = list(pattern)
        for replacements in product(STAR_MATCH_LETTERS, repeat=len(star_positions)):
            for position, replacement in zip(star_positions, replacements):
                chars[position] = replacement
            candidate_word = "".join(chars)
            count = word_freq.get(candidate_word, 0)
            if count > best_count:
                best_word = candidate_word
                best_count = count
        return best_word, best_count

    @lru_cache(maxsize=250_000)
    def damaged_word_logprob(candidate: str) -> float:
        if all(char == "*" for char in candidate):
            return math.log(1.0 / (n_tokens * (20 ** len(candidate))))

        greek_letters = sum(char in ALPHABET for char in candidate)
        if len(candidate) < 3 or greek_letters < 2:
            return -math.inf

        matched_word, count = best_wildcard_match(candidate)
        if matched_word:
            score = math.log(count / n_tokens)
            if len(candidate) > SHORT_WORD_MAX_LEN_FOR_NO_BONUS:
                score += KNOWN_WORD_BONUS + KNOWN_LENGTH_BONUS * len(candidate)
                if matched_word in manual_words:
                    score += MANUAL_WORD_BONUS
            return score - WILDCARD_PENALTY * candidate.count("*")

        return math.log(5.0 / (n_tokens * (OOV_BASE ** min(len(candidate), MAX_UNKNOWN_WORD_LEN))))

    @lru_cache(maxsize=250_000)
    def word_logprob(word: str) -> float:
        if not word:
            return -math.inf
        if "*" in word:
            return damaged_word_logprob(word)

        count = word_freq.get(word, 0)
        if count:
            score = math.log(count / n_tokens)
            if len(word) > SHORT_WORD_MAX_LEN_FOR_NO_BONUS:
                score += KNOWN_WORD_BONUS + KNOWN_LENGTH_BONUS * len(word)
                if word in manual_words:
                    score += MANUAL_WORD_BONUS
            return score

        if len(word) == 1 and word not in TRUSTED_ONE_LETTER_WORDS:
            return math.log(1.0 / (n_tokens * ONE_LETTER_UNKNOWN_PENALTY))
        return math.log(10.0 / (n_tokens * (OOV_BASE ** min(len(word), MAX_UNKNOWN_WORD_LEN))))

    def segment_run(sequence: str) -> list[str]:
        sequence = normalize_greek(sequence).replace(" ", "")
        if not sequence:
            return []
        if all(char == "*" for char in sequence):
            return [sequence]

        n = len(sequence)
        scores = [-math.inf] * (n + 1)
        back_lengths = [0] * (n + 1)
        back_tokens = [""] * (n + 1)
        scores[0] = 0.0

        for end in range(1, n + 1):
            for length in range(1, min(max_word_len, end) + 1):
                start = end - length
                candidate = sequence[start:end]
                candidate_score = word_logprob(candidate)
                if candidate_score == -math.inf:
                    continue
                score = scores[start] + candidate_score - INSERTION_PENALTY
                if score > scores[end]:
                    scores[end] = score
                    back_lengths[end] = length
                    back_tokens[end] = candidate

        words = []
        index = n
        while index > 0:
            length = back_lengths[index] or min(index, max_word_len)
            words.append(back_tokens[index] or sequence[index - length:index])
            index -= length
        return list(reversed(words))

    def segment_text(text: str) -> list[str]:
        tokens = []
        current_run = []

        def flush_run() -> None:
            if current_run:
                tokens.extend(segment_run("".join(current_run)))
                current_run.clear()

        for char in str(text):
            normalized_char = normalize_greek(char)
            if normalized_char in ALPHABET or char == "*":
                current_run.append(normalized_char)
            else:
                flush_run()
                if char in {"&", ":", ")"}:
                    tokens.append(char)
        flush_run()
        return tokens

    return segment_text


def deduplicate_segmented(segmented: dict[str, list[str]]) -> dict[str, list[str]]:
    signature_to_ids = defaultdict(list)
    for inscription_id, tokens in segmented.items():
        signature_to_ids[tuple(tokens)].append(inscription_id)
    return {ids[0]: list(signature) for signature, ids in signature_to_ids.items()}

