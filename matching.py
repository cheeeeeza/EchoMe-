"""
Compares what the child said to the target phrase.

Phonetic: words are aligned (so we know which were missing/extra), then each
          substituted word is compared by sound using Metaphone codes.
          "wabbit" vs "rabbit" scores high-ish -> mispronunciation, not a wrong word.
Semantic: sentence embeddings if sentence-transformers is installed,
          otherwise a lightweight content-word overlap.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import jellyfish

import config

SAME_WORD = 0.9           # word similarity at/above this counts as the same word
MISPRONOUNCED_MIN = 0.5   # between this and SAME_WORD = same word, said differently; below = wrong/missing word
_VARIANTS = {"ok": "okay", "mom": "mum", "mommy": "mummy", "gonna": "going", "wanna": "want"}
_FILLERS = {"um", "uh", "erm", "hmm", "ah"}
_STOPWORDS = {"a", "an", "the", "to", "is", "am", "are", "i", "you", "it", "and", "of", "in", "on", "my"}


@dataclass
class MatchResult:
    verdict: str                     # correct | partial | incorrect | no_response
    phonetic: float                  # 0-1
    semantic: float                  # 0-1
    missing: list[str] = field(default_factory=list)
    mispronounced: list[tuple[str, str]] = field(default_factory=list)  # (target, heard)
    extra: list[str] = field(default_factory=list)


def normalise(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    words = re.findall(r"[a-z']+", text)
    return [_VARIANTS.get(w, w) for w in words if w not in _FILLERS]


def word_sound_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ma, mb = jellyfish.metaphone(a) or a, jellyfish.metaphone(b) or b
    sound = _lev_sim(ma, mb)
    spelling = _lev_sim(a, b)
    return 0.7 * sound + 0.3 * spelling


def _lev_sim(a: str, b: str) -> float:
    return 1 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1)


# --- semantic --------------------------------------------------------------
_embedder = None
_embedder_tried = False


def _get_embedder():
    global _embedder, _embedder_tried
    if not _embedder_tried:
        _embedder_tried = True
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None
    return _embedder


def semantic_similarity(target: str, heard: str) -> float:
    model = _get_embedder()
    if model is not None:
        from sentence_transformers import util
        emb = model.encode([target, heard], convert_to_tensor=True)
        return max(0.0, float(util.cos_sim(emb[0], emb[1])))
    t = {w for w in normalise(target) if w not in _STOPWORDS}
    h = {w for w in normalise(heard) if w not in _STOPWORDS}
    if not t:
        return 1.0 if not h else 0.0
    return len(t & h) / len(t | h)


# --- main comparison ------------------------------------------------------
def compare(target: str, heard: str, heard_words: list[tuple[str, float]] | None = None) -> MatchResult:
    tw = normalise(target)
    if heard_words:
        # keep each heard word lined up with Whisper's confidence in it
        hw, hp = [], []
        for word, prob in heard_words:
            for tok in normalise(word):
                hw.append(tok)
                hp.append(prob)
    else:
        hw = normalise(heard)
        hp = [1.0] * len(hw)
    if not hw:
        return MatchResult("no_response", 0.0, 0.0, missing=tw)

    missing, mispronounced, extra = [], [], []
    word_scores = []
    sm = difflib.SequenceMatcher(a=tw, b=hw, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                prob = hp[j1 + k]
                if prob < config.UNCLEAR_WORD_CONFIDENCE:
                    # right word on paper, but Whisper wasn't sure -> probably said differently
                    mispronounced.append((tw[i1 + k], f"{hw[j1 + k]}?"))
                    word_scores.append(0.5 + 0.5 * prob)
                else:
                    word_scores.append(1.0)
        elif op == "delete":
            missing += tw[i1:i2]
            word_scores += [0.0] * (i2 - i1)
        elif op == "insert":
            extra += hw[j1:j2]
        else:  # replace: pair words up by position, leftovers are missing/extra
            tseg, hseg = tw[i1:i2], hw[j1:j2]
            # Kids often split/merge words ("ice cream" / "icecream"): compare joined too
            if len(tseg) != len(hseg) and "".join(tseg) == "".join(hseg):
                word_scores += [1.0] * len(tseg)
                continue
            for k, t in enumerate(tseg):
                if k < len(hseg):
                    s = word_sound_similarity(t, hseg[k])
                    if s >= SAME_WORD:          # spelling variant (colour/color, mum/mom)
                        word_scores.append(1.0)
                        continue
                    word_scores.append(s)
                    if s >= MISPRONOUNCED_MIN:
                        mispronounced.append((t, hseg[k]))
                    else:
                        missing.append(t)   # a different word entirely
                else:
                    missing.append(t)
                    word_scores.append(0.0)
            extra += hseg[len(tseg):]

    phonetic = sum(word_scores) / len(word_scores) if word_scores else 0.0
    phonetic = max(0.0, phonetic - 0.05 * len(extra))   # small penalty for added words
    semantic = semantic_similarity(target, heard)

    if phonetic >= config.CORRECT_PHONETIC and not missing and not mispronounced:
        verdict = "correct"
    elif phonetic >= config.PARTIAL_PHONETIC or semantic >= config.PARTIAL_SEMANTIC:
        verdict = "partial"
    else:
        verdict = "incorrect"
    return MatchResult(verdict, round(phonetic, 2), round(semantic, 2), missing, mispronounced, extra)