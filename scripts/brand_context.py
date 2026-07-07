#!/usr/bin/env python3
"""
brand_context.py — Context-first brand relation extraction.

Pipeline
========
1. load_category_terms()         → set of product/category nouns (not brands)
2. is_brand_context_sentence()   → gate: only process brand-relevant sentences
3. extract_context_slots()       → structured relation slots (purchase / comparison …)
4. extract_noun_phrases()        → capitalized + whitelist candidates from slots
5. classify_candidate()          → role + decision
6. score_brand_candidate()       → 0-1 confidence
7. extract_brand_context_mentions() → full extraction for one post row

External knowledge: whitelist (xlsx) + taxonomy CSVs only.
No hard-coded retailer / platform / product-noun tables.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT               = Path(__file__).parent.parent
_TAXONOMY_CSV       = _ROOT / "configs" / "taxonomy" / "product_taxonomy_clean.csv"
_TARGET_CLUSTERS_CSV = _ROOT / "configs" / "taxonomy" / "target_clusters_171.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_PUNCT_RE  = re.compile(r"[^\w\s&']")
_SPACE_RE  = re.compile(r"\s+")
_AMP_RE    = re.compile(r"&")


def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = _AMP_RE.sub("and", s)
    s = _PUNCT_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def _norm_match(s: str) -> str:
    """Unicode NFKC + accent strip + apostrophe norm (same as build_dashboard Layer 0)."""
    s = unicodedata.normalize("NFKC", s)
    nfd = unicodedata.normalize("NFD", s)
    s = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    s = re.sub(r"[''`´]", "'", s)
    s = re.sub(r"[–—−]", "-", s)
    return s


def _strip_articles(s: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", s.strip(), flags=re.IGNORECASE).strip()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Category terms
# ──────────────────────────────────────────────────────────────────────────────

def _split_multi(raw: str) -> list[str]:
    """Split on comma, semicolon, pipe, slash, parentheses boundaries."""
    parts = re.split(r"[,;|/()]+", str(raw))
    return [p.strip() for p in parts if p.strip()]


def load_category_terms(
    taxonomy_csv: Optional[Path] = None,
    target_clusters_csv: Optional[Path] = None,
) -> set[str]:
    """Return a set of normalized product/category terms (not brand names).

    Sources:
      product_taxonomy_clean.csv  →  category columns + tag columns
      target_clusters_171.csv     →  cluster names, aliases, seed keywords
    """
    t_csv  = taxonomy_csv         or _TAXONOMY_CSV
    tc_csv = target_clusters_csv  or _TARGET_CLUSTERS_CSV
    terms: set[str] = set()

    def _add(raw):
        for part in _split_multi(raw):
            norm = _norm(part)
            # Keep phrases between 2 and 30 chars, skip pure numbers
            if 2 < len(norm) <= 30 and not all(c.isdigit() for c in norm.replace(" ", "")):
                terms.add(norm)
                # trivial plural / singular variants
                if norm.endswith("s") and len(norm) > 4:
                    terms.add(norm[:-1])
                elif not norm.endswith("s") and len(norm) > 2:
                    terms.add(norm + "s")

    if t_csv.exists():
        pt = pd.read_csv(t_csv)
        for col in [
            "first_category_name", "second_category_name", "third_category_name",
            "main_theme", "sub_theme_en",
            "scenarios_tags", "function_tags", "style_tags",
        ]:
            if col not in pt.columns:
                continue
            for val in pt[col].dropna():
                # Long descriptive sentences → extract only shorter sub-phrases
                for part in _split_multi(str(val)):
                    words = part.split()
                    if len(words) <= 6:          # keep short phrases only
                        _add(part)
                    else:
                        # Take first 4 words as a representative phrase
                        _add(" ".join(words[:4]))

    if tc_csv.exists():
        tc = pd.read_csv(tc_csv)
        for col in ["target_cluster", "parent_category", "aliases"]:
            if col not in tc.columns:
                continue
            for val in tc[col].dropna():
                _add(val)
        if "seed_keywords" in tc.columns:
            for val in tc["seed_keywords"].dropna():
                # seed_keywords are "|"-separated
                for kw in str(val).split("|"):
                    _add(kw.strip())

    # Remove noise tokens too generic to be category terms
    noise = {"nan", "none", "null", "other", "yes", "no", "new", "old",
             "good", "bad", "big", "small", "best", "top", "high", "low"}
    terms -= noise
    return terms


# ──────────────────────────────────────────────────────────────────────────────
# 2. Brand-context sentence detection
# ──────────────────────────────────────────────────────────────────────────────

_CTX_GROUPS: dict[str, re.Pattern] = {
    "purchase": re.compile(
        r"\b(?:bought|ordered|purchased|picked\s+up|grabbed|hauled|"
        r"arrived|shipped)\b",
        re.IGNORECASE,
    ),
    "ownership": re.compile(
        r"\bgot\b|\bmy\s+\w+",
        re.IGNORECASE,
    ),
    "usage": re.compile(
        r"\b(?:use[d]?|using|tried|testing|wearing|"
        r"switched?\s+to|currently\s+use|been\s+using)\b",
        re.IGNORECASE,
    ),
    "recommendation": re.compile(
        r"\b(?:recommend(?:ed)?|suggest(?:ed)?|lov(?:e[d]?|ing)|hat(?:e[d]?|ing)|"
        r"obsessed\s+with|holy\s+grail|worth\s+it|not\s+worth\s+it|"
        r"overrated|underrated|worked\s+for\s+me|helped|broke\s+me\s+out|"
        r"irritated|made\s+me\s+break\s+out|stopped\s+working|returned)\b",
        re.IGNORECASE,
    ),
    "comparison": re.compile(
        r"\b(?:vs\.?|versus|compared\s+to|better\s+than|worse\s+than|"
        r"switched?\s+from|instead\s+of|dupe\s+for|alternative\s+to|"
        r"similar\s+to|replaced?\s+with)\b",
        re.IGNORECASE,
    ),
    "trend": re.compile(
        r"\b(?:viral|tiktok\s+shop|haul|honest\s+review|first\s+impression|"
        r"restock(?:ed)?|sold\s+out|back\s+in\s+stock)\b|"
        r"\btiktok\s+made\s+me\s+buy\b",
        re.IGNORECASE,
    ),
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? and semicolons."""
    raw = re.split(r"(?<=[.!?])\s+|;\s*", text)
    return [s.strip() for s in raw if s.strip()]


def is_brand_context_sentence(
    sentence: str,
    whitelist: dict[str, str],
    category_terms: set[str],
) -> tuple[bool, list[str]]:
    """Return (is_brand_context, matched_signal_labels).

    Returns True if any context signal is present, or a whitelist brand appears.
    """
    matched: list[str] = []
    for label, pat in _CTX_GROUPS.items():
        if pat.search(sentence):
            matched.append(label)

    if not matched:
        # Check whitelist presence even without explicit signal verb
        s_norm = _norm(_norm_match(sentence))
        for wl_key in whitelist:
            if wl_key in s_norm:
                matched.append("whitelist_brand_present")
                break

    return bool(matched), matched


# ──────────────────────────────────────────────────────────────────────────────
# 3. Slot extraction
# ──────────────────────────────────────────────────────────────────────────────

_BOUNDARY_RE = re.compile(
    r"\b(?:because|but\b|so\b|although|though|while|after|before|from|at|on|"
    r"with|for|that|which)\b|and\s+then|[.!?;:()]",
    re.IGNORECASE,
)


def _trim_slot(text: str, max_tokens: int = 8) -> str:
    """Truncate at first boundary token; limit to max_tokens."""
    text = _ARTICLE_STRIP_RE.sub("", text.strip())
    m = _BOUNDARY_RE.search(text)
    if m:
        text = text[: m.start()].strip()
    return " ".join(text.split()[:max_tokens]).strip()


_ARTICLE_STRIP_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

# (compiled-pattern, relation_type, slot_shape)
# slot_shape: "fwd1" = group(1) is object after verb
#             "pair"  = group(1) and group(2) are two slots
#             "bwd1"  = group(1) is object before verb phrase
_SLOT_DEFS: list[tuple[re.Pattern, str, str]] = [

    # --- Forward-looking (verb + object) ------------------------------------

    # purchase: bought/ordered/purchased/grabbed/hauled/picked up X
    (re.compile(
        r"\b(?:bought|ordered|purchased|grabbed|hauled|picked\s+up)\s+"
        r"(?:the\s+|a\s+|an\s+|my\s+|some\s+)?(.{3,60})",
        re.IGNORECASE,
    ), "purchase", "fwd1"),

    # "got" + capitalized or article+noun (avoid "got it", "got that")
    (re.compile(
        r"\bgot\s+(?:the\s+|a\s+|an\s+)?([A-Z][A-Za-z0-9\s&'-]{2,50})",
    ), "purchase", "fwd1"),

    # usage: used/using/tried/testing/wearing X
    (re.compile(
        r"\b(?:use[d]?|using|tried|testing|wearing)\s+"
        r"(?:the\s+|a\s+|an\s+|my\s+)?(.{3,60})",
        re.IGNORECASE,
    ), "usage", "fwd1"),

    # switched to X
    (re.compile(
        r"\bswitched?\s+to\s+(?:the\s+|a\s+|an\s+)?(.{3,60})",
        re.IGNORECASE,
    ), "usage", "fwd1"),

    # recommend/suggest/love/hate/obsessed with X
    (re.compile(
        r"\b(?:recommend(?:ed)?|suggest(?:ed)?|lov(?:e[d]?|ing)|hat(?:e[d]?|ing)|"
        r"obsessed\s+with)\s+(?:the\s+|a\s+)?(.{3,60})",
        re.IGNORECASE,
    ), "recommendation", "fwd1"),

    # dupe for X / alternative to X / similar to X / instead of X
    (re.compile(
        r"\b(?:dupe\s+for|alternative\s+to|similar\s+to|"
        r"replacement\s+for|instead\s+of)\s+(.{3,50})",
        re.IGNORECASE,
    ), "dupe_alternative", "fwd1"),

    # --- Comparison pairs ---------------------------------------------------

    # X vs Y
    (re.compile(
        r"(.{3,40}?)\s+(?:vs\.?|versus)\s+(.{3,40})",
        re.IGNORECASE,
    ), "comparison", "pair"),

    # switched from X to Y
    (re.compile(
        r"\bswitched?\s+from\s+(.{3,40}?)\s+to\s+(.{3,40})",
        re.IGNORECASE,
    ), "comparison", "pair"),

    # X better/worse than Y
    (re.compile(
        r"(.{3,40}?)\s+(?:better|worse)\s+than\s+(.{3,40})",
        re.IGNORECASE,
    ), "comparison", "pair"),

    # --- Backward-looking (subject + verb) ----------------------------------

    # my X arrived/came/shipped
    (re.compile(
        r"\bmy\s+([A-Za-z][A-Za-z0-9\s&'-]{2,40}?)\s+"
        r"(?:arrived|came|shipped)\b",
        re.IGNORECASE,
    ), "purchase", "bwd1"),

    # X worked for me / broke me out / sold out / is worth it …
    (re.compile(
        r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\s&'-]{1,40}?)\s+"
        r"(?:worked\s+for\s+me|broke\s+me\s+out|irritated\s+my|"
        r"stopped\s+working|sold\s+out|restocked|back\s+in\s+stock|"
        r"is\s+worth\s+it|is\s+overrated|is\s+underrated|"
        r"is\s+amazing|is\s+terrible|helped\s+me|returned)\b",
    ), "evaluation", "bwd1"),
]


def extract_context_slots(sentence: str) -> list[dict]:
    """Extract structured relation slots from a brand-context sentence."""
    slots: list[dict] = []
    seen: set[str] = set()

    for pat, relation_type, shape in _SLOT_DEFS:
        for m in pat.finditer(sentence):
            if shape == "pair":
                for grp in (m.group(1), m.group(2)):
                    text = _trim_slot(grp)
                    if text and len(text) >= 2 and text not in seen:
                        seen.add(text)
                        slots.append({
                            "slot_text":    text,
                            "relation_type": relation_type,
                            "pattern_name":  shape,
                            "sentence":      sentence,
                        })
            else:
                text = _trim_slot(m.group(1))
                if text and len(text) >= 2 and text not in seen:
                    seen.add(text)
                    slots.append({
                        "slot_text":    text,
                        "relation_type": relation_type,
                        "pattern_name":  shape,
                        "sentence":      sentence,
                    })

    return slots


# ──────────────────────────────────────────────────────────────────────────────
# 4. Noun phrase extraction
# ──────────────────────────────────────────────────────────────────────────────

# Matches: CeraVe, Shark FlexStyle, Sol de Janeiro, ALLCAPS, Drunk Elephant
_CAP_PHRASE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&''-]{1,20}(?:\s+[A-Z][A-Za-z0-9&''-]{1,20}){0,4})\b"
)
_QUOTE_RE = re.compile(r'["""]([^"""]{3,40})["""]')

# Number of tokens to look either side of a candidate when checking
# proximity to a category term
_PROXIMITY_WINDOW = 6


def _find_whitelist_spans(text: str, whitelist: dict[str, str]) -> list[tuple[int, int, str]]:
    """Return (start, end, display_name) for all whitelist matches in text."""
    norm_text = _norm(_norm_match(text))
    tokens = norm_text.split()
    spans: list[tuple[int, int, str]] = []
    for n in range(min(5, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i: i + n])
            if gram in whitelist:
                # Approximate character position
                spans.append((i, i + n, whitelist[gram]))
    return spans


def _phrase_is_near_category(phrase: str, sentence: str, category_terms: set[str],
                              window: int = _PROXIMITY_WINDOW) -> bool:
    """True if a category term appears within `window` tokens of the phrase in sentence."""
    tokens = sentence.lower().split()
    phrase_tokens = phrase.lower().split()
    # Find phrase start in token list
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i: i + len(phrase_tokens)] == phrase_tokens:
            lo = max(0, i - window)
            hi = min(len(tokens), i + len(phrase_tokens) + window)
            context_tokens = tokens[lo:i] + tokens[i + len(phrase_tokens): hi]
            context_str = " ".join(context_tokens)
            # Check if any category term appears in the context window
            for term in category_terms:
                if term in context_str:
                    return True
    return False


def extract_noun_phrases(
    slot_text: str,
    sentence: str,
    whitelist: dict[str, str],
    category_terms: set[str],
) -> list[dict]:
    """Extract brand candidates from a slot text.

    Sources:
      A. Whitelist brand matches (exact / normalized)
      B. Capitalized phrases (1-5 tokens)
      C. Quoted phrases
      D. Category-adjacent capitalized phrases (within 6 tokens)
    """
    candidates: list[dict] = []
    seen_norms: set[str] = set()

    def _add(raw: str, source: str, near_cat: bool = False, wl_display: str = ""):
        norm = _norm(raw)
        if not norm or len(norm) < 2 or norm in seen_norms:
            return
        seen_norms.add(norm)
        candidates.append({
            "raw":        raw.strip(),
            "norm":       norm,
            "source":     source,
            "near_cat":   near_cat,
            "wl_display": wl_display,   # non-empty if whitelist match
        })

    # A. Whitelist matches in slot_text — longest n-gram first, no overlapping spans
    slot_norm   = _norm(_norm_match(slot_text))
    slot_tokens = slot_norm.split()
    consumed: set[int] = set()
    for n in range(min(5, len(slot_tokens)), 0, -1):
        for i in range(len(slot_tokens) - n + 1):
            if any(j in consumed for j in range(i, i + n)):
                continue
            gram = " ".join(slot_tokens[i: i + n])
            if gram in whitelist:
                _add(gram, "whitelist", wl_display=whitelist[gram])
                consumed.update(range(i, i + n))

    # Also scan sentence for whitelist brands (catches brands not in slot_text)
    sent_norm   = _norm(_norm_match(sentence))
    sent_tokens = sent_norm.split()
    s_consumed: set[int] = set()
    for n in range(min(5, len(sent_tokens)), 0, -1):
        for i in range(len(sent_tokens) - n + 1):
            if any(j in s_consumed for j in range(i, i + n)):
                continue
            gram = " ".join(sent_tokens[i: i + n])
            if gram in whitelist:
                _add(gram, "whitelist", wl_display=whitelist[gram])
                s_consumed.update(range(i, i + n))

    # B. Capitalized phrases in slot_text
    for m in _CAP_PHRASE_RE.finditer(slot_text):
        phrase = m.group(1).strip()
        near = _phrase_is_near_category(phrase, sentence, category_terms)
        _add(phrase, "cap_phrase", near_cat=near)

    # C. Quoted phrases in slot_text / sentence
    for text_src in (slot_text, sentence):
        for m in _QUOTE_RE.finditer(text_src):
            phrase = m.group(1).strip()
            _add(phrase, "quoted")

    # D. Sentence-level: capitalized phrases adjacent to category terms
    for m in _CAP_PHRASE_RE.finditer(sentence):
        phrase = m.group(1).strip()
        if _phrase_is_near_category(phrase, sentence, category_terms):
            _add(phrase, "cat_adjacent", near_cat=True)

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# 5 & 6. Classification + scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_brand_candidate(candidate_info: dict) -> float:
    """Score a candidate 0-1.

    Positive signals:
      +0.60  exact whitelist brand
      +0.50  candidate contains a whitelist brand (brand_product_combo)
      +0.35  capitalized phrase within 6 tokens of category/product term
      +0.30  extracted from purchase/usage/review slot
      +0.25  comparison pair candidate
      +0.20  recommendation / evaluation signal
      +0.15  trend / viral / haul / restock signal
      +0.10  appears in multiple contexts (multi_context)

    Negative signals:
      -0.70  exact category term
      -0.40  all lowercase, no whitelist brand
      -0.30  no brand-context sentence
      -0.25  too generic (one common token, < 3 chars)
      -0.20  source / subreddit / community label
      -0.20  context too short or single-char noise
    """
    score = 0.0
    reasons = candidate_info.get("signals", [])

    # Positive
    if "exact_whitelist"      in reasons: score += 0.60
    if "contains_whitelist"   in reasons: score += 0.50
    if "near_cat"             in reasons or candidate_info.get("near_cat"): score += 0.35
    if "purchase"             in reasons: score += 0.30
    if "usage"                in reasons: score += 0.30
    if "recommendation"       in reasons: score += 0.20
    if "comparison"           in reasons: score += 0.25
    if "dupe_alternative"     in reasons: score += 0.25
    if "evaluation"           in reasons: score += 0.20
    if "trend"                in reasons: score += 0.15
    if "multi_context"        in reasons: score += 0.10
    if "quoted"               in reasons: score += 0.10
    if "whitelist_brand_present" in reasons: score += 0.15

    # Negative
    if "exact_category"       in reasons: score -= 0.70
    if "all_lowercase_no_wl"  in reasons: score -= 0.40
    if "no_context_sentence"  in reasons: score -= 0.30
    if "too_generic"          in reasons: score -= 0.25
    if "community_label"      in reasons: score -= 0.20
    if "noise"                in reasons: score -= 0.20

    return round(min(max(score, 0.0), 1.0), 3)


_STRONG_SLOT_SIGNALS = frozenset({
    "purchase", "usage", "comparison", "recommendation",
    "dupe_alternative", "evaluation", "trend",
})


def _is_product_noun(brand_norm: str, category_terms: set[str]) -> bool:
    """True if brand_norm looks like a product/category noun rather than a brand.

    Checks:
      - exact match in category_terms
      - any individual word (>3 chars) in brand_norm is a category term
    """
    if brand_norm in category_terms:
        return True
    stripped = _strip_articles(brand_norm)
    if stripped in category_terms:
        return True
    for word in brand_norm.split():
        if len(word) > 3 and word in category_terms:
            return True
    return False


def classify_candidate(
    candidate: dict,
    context_signals: list[str],
    whitelist: dict[str, str],
    category_terms: set[str],
) -> dict:
    """Assign role + decision to a candidate dict.

    Priority order:
    1. Exact whitelist brand            → display_brand / true_brand
    2. Contains whitelist brand         → display_brand / brand_product_combo
    3. Exact category term              → product_context
    4. All-lowercase + no whitelist     → product_context or reject
    5. Capitalized near category term   → score-dependent
    """
    raw   = candidate.get("raw", "")
    norm  = candidate.get("norm", "")
    src   = candidate.get("source", "")
    signals: list[str] = list(context_signals)  # copy

    wl_display = candidate.get("wl_display", "")
    near_cat   = candidate.get("near_cat", False)

    has_strong_slot = bool(_STRONG_SLOT_SIGNALS & set(signals))

    # --- Rule 1: exact whitelist match
    if src == "whitelist" or wl_display:
        # Even whitelist brands are downgraded to product_context when they are
        # also a product/category noun AND no strong relation slot is present.
        # This blocks "Desk", "My Skin", "Cream" from appearing as brands.
        if _is_product_noun(norm, category_terms) and not has_strong_slot:
            signals.append("exact_category")
            score = score_brand_candidate({"signals": signals})
            return {
                "brand": wl_display or whitelist.get(norm, raw),
                "brand_norm": norm,
                "product_context": raw,
                "role": "product_context",
                "decision": "product_context",
                "score": score, "signals": signals,
            }

        signals.append("exact_whitelist")
        if near_cat:
            signals.append("near_cat")
        score = score_brand_candidate({"signals": signals, "near_cat": near_cat})
        return {
            "brand":         wl_display or whitelist.get(norm, raw),
            "brand_norm":    norm,
            "product_context": "",
            "role":          "true_brand",
            "decision":      "display_brand",
            "score":         max(score, 0.60),
            "signals":       signals,
        }

    # --- Rule 2: candidate contains a whitelist brand as a sub-phrase
    #   Use n-gram lookup against the candidate's own tokens (O(n²) hash lookup)
    #   instead of scanning all 94k whitelist keys — which would be O(94k) per call.
    #   Word-boundary semantics come naturally: each token is its own unit.
    norm_match  = _norm(_norm_match(raw))
    nm_tokens   = norm_match.split()
    contained_brand   = ""
    contained_display = ""
    # Try longest sub-phrase first, skip exact match (already handled by Rule 1)
    for n in range(min(5, len(nm_tokens)), 0, -1):
        for i in range(len(nm_tokens) - n + 1):
            gram = " ".join(nm_tokens[i: i + n])
            if gram == norm_match:
                continue
            if gram in whitelist:
                remaining = " ".join(nm_tokens[:i] + nm_tokens[i + n:]).strip()
                contained_brand   = gram
                contained_display = whitelist[gram]
                break
        if contained_brand:
            break

    if contained_brand:
        remaining = norm_match.replace(contained_brand, "").strip()
        signals.append("contains_whitelist")
        if near_cat:
            signals.append("near_cat")
        score = score_brand_candidate({"signals": signals, "near_cat": near_cat})
        return {
            "brand":         contained_display,
            "brand_norm":    contained_brand,
            "product_context": remaining,
            "role":          "brand_product_combo",
            "decision":      "display_brand",
            "score":         max(score, 0.50),
            "signals":       signals,
        }

    # --- Rule 3: exact category term
    if norm in category_terms or _strip_articles(norm) in category_terms:
        signals.append("exact_category")
        score = score_brand_candidate({"signals": signals})
        return {
            "brand":         raw,
            "brand_norm":    norm,
            "product_context": raw,
            "role":          "product_context",
            "decision":      "product_context",
            "score":         score,
            "signals":       signals,
        }

    # --- Rule 4: all-lowercase + no whitelist
    if raw == raw.lower() and not contained_brand:
        # Check category overlap
        if any(t in norm for t in category_terms if len(t) > 3):
            signals.append("exact_category")
            score = score_brand_candidate({"signals": signals})
            return {
                "brand": raw, "brand_norm": norm,
                "product_context": raw,
                "role": "product_context", "decision": "product_context",
                "score": score, "signals": signals,
            }
        signals.append("all_lowercase_no_wl")
        score = score_brand_candidate({"signals": signals})
        return {
            "brand": raw, "brand_norm": norm,
            "product_context": "",
            "role": "ambiguous_candidate",
            "decision": "reject" if score < 0.30 else "brand_candidate_audit",
            "score": score, "signals": signals,
        }

    # --- Rule 5: capitalized phrase, possibly near category term
    if near_cat:
        signals.append("near_cat")

    # Too generic: single common English word
    if len(norm.split()) == 1 and len(norm) < 4:
        signals.append("too_generic")

    score = score_brand_candidate({"signals": signals, "near_cat": near_cat})

    if score >= 0.60:
        decision = "display_brand"
        role     = "ambiguous_candidate" if not near_cat else "ambiguous_candidate"
    elif score >= 0.55:
        decision = "brand_candidate_audit"
        role     = "ambiguous_candidate"
    else:
        decision = "product_context" if norm in category_terms else "reject"
        role     = "product_context" if norm in category_terms else "irrelevant"

    return {
        "brand":         raw,
        "brand_norm":    norm,
        "product_context": "",
        "role":          role,
        "decision":      decision,
        "score":         score,
        "signals":       signals,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main extraction function
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BrandContextMention:
    mention_id:       str
    candidate:        str
    brand:            str
    brand_norm:       str
    product_context:  str
    role:             str
    decision:         str
    score:            float
    relation_type:    str
    evidence_sentence: str
    context_window:   str
    matched_signals:  list[str]
    candidate_source: str
    category:         str
    cluster_id:       str
    community:        str
    published_at:     str
    engagement_score: float
    url:              str


def extract_brand_context_mentions(
    row,
    whitelist: dict[str, str],
    category_terms: set[str],
) -> list[BrandContextMention]:
    """Full extraction for one post row.

    `row` must have: mention_id, title, text (or ner), category,
    community, published_at, sentiment_compound, engagement_score, url.
    Reads ner (title+text[:400]) if available, else constructs it.
    """
    ner: str = getattr(row, "ner", None) or (
        str(getattr(row, "title", "") or "") + " " +
        str(getattr(row, "text",  "") or "")[:400]
    ).strip()

    mention_id       = str(getattr(row, "mention_id",       ""))
    category         = str(getattr(row, "category",         ""))
    cluster_id       = str(getattr(row, "target_cluster_id",""))
    community        = str(getattr(row, "community",        ""))
    published_at     = str(getattr(row, "published_at",     ""))
    engagement_score = float(getattr(row, "engagement_score", 0) or 0)
    url              = str(getattr(row, "url",              ""))

    sentences    = _split_sentences(ner)
    mentions:    list[BrandContextMention] = []
    seen_brands: set[str]                  = set()  # deduplicate by article-stripped norm

    for sentence in sentences:
        is_ctx, ctx_signals = is_brand_context_sentence(sentence, whitelist, category_terms)
        if not is_ctx:
            continue

        # Gather slots from this sentence
        slots = extract_context_slots(sentence)
        slot_relation: dict[str, str] = {}  # norm → relation_type from first slot hit

        # Collect candidates: from slots + sentence-level scan
        all_candidates: list[dict] = []
        slot_texts: set[str] = set()

        for slot in slots:
            st = slot["slot_text"]
            slot_texts.add(st)
            nps = extract_noun_phrases(st, sentence, whitelist, category_terms)
            for np in nps:
                np["_relation"] = slot["relation_type"]
                all_candidates.append(np)

        # Also extract directly from sentence (catches whitelist brands not in slots)
        nps_sentence = extract_noun_phrases("", sentence, whitelist, category_terms)
        for np in nps_sentence:
            np.setdefault("_relation", "")
            all_candidates.append(np)

        # Track multi-context: seen in >1 slot or sentence scan
        norm_counts: dict[str, int] = {}
        for cand in all_candidates:
            norm_counts[cand["norm"]] = norm_counts.get(cand["norm"], 0) + 1

        # Classify and score each unique candidate
        seen_in_sentence: set[str] = set()
        for cand in all_candidates:
            n = cand["norm"]
            if n in seen_in_sentence:
                continue
            seen_in_sentence.add(n)

            extra_signals = list(ctx_signals)
            rel = cand.get("_relation", "")
            if rel:
                extra_signals.append(rel)
            if norm_counts.get(n, 0) > 1:
                extra_signals.append("multi_context")
            if cand.get("near_cat"):
                extra_signals.append("near_cat")
            if cand.get("source") == "quoted":
                extra_signals.append("quoted")

            result = classify_candidate(cand, extra_signals, whitelist, category_terms)

            # Skip if not a useful brand output
            if result["decision"] not in ("display_brand", "brand_candidate_audit"):
                continue

            display = result.get("brand") or cand["raw"]
            brand_n = result.get("brand_norm") or cand["norm"]

            # De-duplicate per post: compare article-stripped lowercase keys
            # so "The Shark" and "Shark" collapse to the same brand
            dedup_key = _strip_articles(display.lower())
            if dedup_key in seen_brands:
                continue
            seen_brands.add(dedup_key)

            ctx_window = ner[max(0, ner.find(cand["raw"]) - 80):
                             ner.find(cand["raw"]) + len(cand["raw"]) + 80]

            mentions.append(BrandContextMention(
                mention_id        = mention_id,
                candidate         = cand["raw"],
                brand             = display,
                brand_norm        = brand_n,
                product_context   = result.get("product_context", ""),
                role              = result.get("role", ""),
                decision          = result["decision"],
                score             = result["score"],
                relation_type     = rel or (ctx_signals[0] if ctx_signals else ""),
                evidence_sentence = sentence[:200],
                context_window    = ctx_window[:300],
                matched_signals   = result.get("signals", []),
                candidate_source  = cand.get("source", ""),
                category          = category,
                cluster_id        = cluster_id,
                community         = community,
                published_at      = published_at,
                engagement_score  = engagement_score,
                url               = url,
            ))

    return mentions


# ──────────────────────────────────────────────────────────────────────────────
# 8. Dashboard-compatible aggregation
# ──────────────────────────────────────────────────────────────────────────────

from collections import Counter, defaultdict


def build_brand_tables_ctx(
    df_cur: "pd.DataFrame",
    df_prev: "pd.DataFrame",
    whitelist: dict[str, str],
    category_terms: set[str],
    min_mentions: int = 3,
    max_cat_pct: float = 0.40,
) -> dict[str, "pd.DataFrame"]:
    """Context-aware brand aggregation — drop-in replacement for build_brand_tables.

    Returns cat_brand_data in the same format as the old function:
      brand, cur_mentions, prev_mentions, brand_spike,
      avg_sentiment, confidence, source, evidence
    """
    brand_cur_cnt:   dict[str, Counter]         = defaultdict(Counter)
    brand_prev_cnt:  dict[str, Counter]         = defaultdict(Counter)
    brand_sentiment: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    brand_meta:      dict[str, dict]            = defaultdict(dict)

    for row in df_cur.itertuples():
        mentions = extract_brand_context_mentions(row, whitelist, category_terms)
        for m in mentions:
            if m.decision != "display_brand":
                continue
            cat = m.category
            brand_cur_cnt[cat][m.brand] += 1
            brand_sentiment[cat][m.brand].append(
                float(getattr(row, "sentiment_compound", 0) or 0)
            )
            if m.brand not in brand_meta[cat]:
                brand_meta[cat][m.brand] = {
                    "confidence": m.score,
                    "source":     m.candidate_source,
                    "evidence":   m.relation_type,
                }

    for row in df_prev.itertuples():
        mentions = extract_brand_context_mentions(row, whitelist, category_terms)
        for m in mentions:
            if m.decision == "display_brand":
                brand_prev_cnt[row.category][m.brand] += 1

    # Cross-category frequency filter
    global_brand_cats: Counter = Counter()
    for cat, cnt in brand_cur_cnt.items():
        for b in cnt:
            global_brand_cats[b] += 1
    total_cats = max(len(brand_cur_cnt), 1)

    cat_brand_data: dict[str, "pd.DataFrame"] = {}
    for cat in df_cur["category"].unique():
        cur_c  = brand_cur_cnt.get(cat,  Counter())
        prev_c = brand_prev_cnt.get(cat, Counter())
        meta   = brand_meta.get(cat, {})
        rows   = []
        for b, cc in cur_c.items():
            if cc < min_mentions:
                continue
            if global_brand_cats[b] / total_cats > max_cat_pct:
                continue
            pc       = prev_c.get(b, 0)
            sents    = brand_sentiment[cat].get(b, [])
            avg_sent = round(sum(sents) / len(sents), 3) if sents else 0.0
            m_       = meta.get(b, {"confidence": 0.5, "source": "unknown", "evidence": ""})
            rows.append({
                "brand":         b,
                "cur_mentions":  cc,
                "prev_mentions": pc,
                "brand_spike":   round(cc / max(pc, 1), 2),
                "avg_sentiment": avg_sent,
                "confidence":    m_["confidence"],
                "source":        m_["source"],
                "evidence":      m_["evidence"],
            })
        if rows:
            cat_brand_data[cat] = (
                pd.DataFrame(rows)
                .sort_values(
                    ["cur_mentions", "brand_spike", "avg_sentiment"],
                    ascending=[False, False, False],
                )
                .head(30)
                .reset_index(drop=True)
            )

    return cat_brand_data


# ──────────────────────────────────────────────────────────────────────────────
# 9. Validation / quick test
# ──────────────────────────────────────────────────────────────────────────────

_TEST_CASES = [
    ("I got the Shark FlexStyle after comparing it with the Dyson Airwrap.",
     {"Shark FlexStyle": "display_brand", "Dyson": "display_brand"}),
    ("I need a walking pad that fits under my desk.",
     {}),
    ("CeraVe cleanser destroyed my skin, switched to Vanicream and it's much better.",
     {"CeraVe": "display_brand", "Vanicream": "display_brand"}),
    ("Any good air fryer recommendations?",
     {}),
    ("Stanley tumbler is overrated but Owala is actually worth it.",
     {"Stanley": "display_brand", "Owala": "display_brand"}),
]


def _make_dummy_row(text: str, idx: int = 0):
    """Create a minimal row-like object for testing."""
    from types import SimpleNamespace
    return SimpleNamespace(
        mention_id=str(idx), ner=text, title="", text="",
        category="test", target_cluster_id="", community="test",
        published_at="2026-01-01", engagement_score=1.0, url="",
        sentiment_compound=0.0,
    )


def run_validation(whitelist: dict[str, str], category_terms: set[str]) -> None:
    print("\n" + "=" * 70)
    print("BRAND CONTEXT VALIDATION")
    print("=" * 70)

    for text, expected in _TEST_CASES:
        print(f"\nText: {text!r}")
        row = _make_dummy_row(text)
        mentions = extract_brand_context_mentions(row, whitelist, category_terms)
        display_brands = {m.brand for m in mentions if m.decision == "display_brand"}
        audit_brands   = {m.brand for m in mentions if m.decision == "brand_candidate_audit"}
        products       = {m.brand for m in mentions if m.decision == "product_context"}

        print(f"  display_brand  : {sorted(display_brands)}")
        print(f"  audit          : {sorted(audit_brands)}")
        print(f"  product_context: {sorted(products)}")

        for m in mentions:
            print(f"    [{m.decision:22s}] {m.brand!r:30s} score={m.score:.2f}  "
                  f"rel={m.relation_type!r}  signals={m.matched_signals}")

        if expected:
            ok = all(
                any(m.brand == b and m.decision in (d, "brand_candidate_audit")
                    for m in mentions)
                for b, d in expected.items()
            )
            print(f"  → {'PASS ✓' if ok else 'FAIL ✗'}")
        else:
            ok = len(display_brands) == 0
            print(f"  → {'PASS ✓' if ok else 'FAIL ✗'} (expect no display brands)")

    print("\n" + "=" * 70)
    print("Top category terms (sample):")
    sample = sorted(category_terms)[:30]
    for i in range(0, len(sample), 5):
        print("  " + "  ".join(f"{t!r:25s}" for t in sample[i:i+5]))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_ROOT))
    # Lazy import to avoid circular deps when used as module
    from scripts.build_dashboard_500k import load_whitelist  # type: ignore[import]

    print("Loading whitelist …")
    wl = load_whitelist()
    print(f"  {len(wl)} brands")

    print("Loading category terms …")
    ct = load_category_terms()
    print(f"  {len(ct)} terms")

    run_validation(wl, ct)
