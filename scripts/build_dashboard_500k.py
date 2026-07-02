#!/usr/bin/env python3
"""Build dashboard_data_500k.pkl — dual-layer brand extraction pipeline.

Architecture
============
Layer 1  Whitelist scan
    Single canonical source: Brand List Available全域.xlsx
    N-gram lookup → confidence 1.0, source "whitelist"

Layer 2  Candidate discovery
    Regex patterns detect brand-like tokens (ALL_CAPS, CamelCase, TitleCase, quoted)
    KNOWN_BRANDS supplements regex for words that look generic but are real brands
    Each candidate is scored; score >= CANDIDATE_THRESHOLD → accepted

HARD_REJECT
    NLTK English stopwords  (function words, systematic)
    wordfreq top-1 000      (ultra-common English, e.g. "great", "best")
    Manual domain additions (product-category descriptors, Reddit slang)

Disambiguation
    Tokens that match brand patterns but are also common English words
    (wordfreq > AMBIGUITY_FREQ_THRESHOLD) trigger a context window check
    before being accepted.

Output columns per brand
    brand, cur_mentions, prev_mentions, brand_spike,
    avg_sentiment, confidence, source, evidence
"""
from __future__ import annotations
import logging
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_dashboard")

ROOT    = Path(__file__).parent.parent
PARQUET = ROOT / "data" / "processed" / "nlp_clustered_500k.parquet"
OUT_PKL = ROOT / "data" / "processed" / "dashboard_data_500k.pkl"
BRAND_XLS = Path.home() / "Downloads" / "Brand List Available全域.xlsx"

# ── NLP package setup ─────────────────────────────────────────────────────────
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _nltk_sw
    _NLTK_STOPWORDS: set[str] = set(_nltk_sw.words("english"))
except Exception:
    _NLTK_STOPWORDS = set()
    log.warning("NLTK stopwords unavailable — falling back to empty set")

try:
    from wordfreq import word_frequency, top_n_list
    _WF_TOP1K: set[str] = set(top_n_list("en", 1_000))   # ultra-common words
    _WF_TOP5K: set[str] = set(top_n_list("en", 5_000))   # common words
    def _wfreq(w: str) -> float:
        return word_frequency(w, "en")
    _WORDFREQ_OK = True
except Exception:
    _WF_TOP1K = _WF_TOP5K = set()
    def _wfreq(w: str) -> float:
        return 0.0
    _WORDFREQ_OK = False
    log.warning("wordfreq unavailable — frequency-based scoring disabled")

# English system dictionary (for multi-word phrase checks)
with open("/usr/share/dict/words") as _f:
    _ENG_DICT: set[str] = {w.strip().lower() for w in _f}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  HARD_REJECT
# ─────────────────────────────────────────────────────────────────────────────
# Built from: NLTK stopwords + wordfreq top-1000 + manual domain additions.
# A token in HARD_REJECT is refused immediately — no scoring, no context check.

# Absolute minimum rejects — only things that can NEVER be a brand:
# pure function words (from NLTK) + single-char tokens + bare units.
# Domain noise is intentionally removed so the whitelist can speak for itself.
_ABSOLUTE_NOISE: set[str] = {
    "a", "i",                                          # single chars
    "mm", "cm", "kg", "lb", "oz", "ml", "mg",         # bare units (no brand uses these alone)
    "usd", "gbp", "eur", "cad", "aud",                 # currency codes
    "lol", "lmao", "omg", "wtf", "tbh", "ngl", "imo", # pure internet slang
}

HARD_REJECT: set[str] = _NLTK_STOPWORDS | _ABSOLUTE_NOISE

# ─────────────────────────────────────────────────────────────────────────────
# Product / category terms — rejected even when present in the whitelist.
# These are descriptors, ingredients, product types, or shopping jargon,
# not brand names.
# ─────────────────────────────────────────────────────────────────────────────
PRODUCT_TERMS: set[str] = {
    # ── skincare ingredients & routines ──────────────────────────────────────
    "serum", "sunscreen", "cleanser", "moisturizer", "retinol", "toner",
    "foundation", "mascara", "tretinoin", "niacinamide", "hyaluronic",
    "ceramide", "salicylic", "glycolic", "spf", "aha", "bha", "vitamin c",
    "peptide", "peptides", "exfoliant", "primer", "concealer", "bronzer",
    "blush", "eyeliner", "lipstick", "highlighter", "contour", "setting spray",
    "micellar", "essence", "ampoule", "sheet mask", "eye cream", "lip balm",
    "tinted", "bb cream", "cc cream", "powder", "loose powder",
    "retinoid", "benzoyl", "azelaic", "tranexamic", "kojic",
    "spf 30", "spf 50", "broad spectrum", "mineral sunscreen",
    # ── baby / parenting (common Reddit nouns, NOT brands) ───────────────────
    "nap", "nursing", "feeding", "pumping", "breastfeeding", "breastfed",
    "swaddle", "latch", "diaper", "diapers", "wipe", "wipes",
    "stroller", "carrier", "crib", "bassinet", "pacifier", "formula",
    "breast milk", "breast pump", "bottle", "nipple", "nipples",
    "bedtime", "naptime", "sleep training", "sleep regression",
    "wake window", "wake windows", "tummy time", "teething",
    "milestone", "milestones", "feeding schedule", "growth spurt",
    "postpartum", "maternity", "paternity", "pregnancy", "pregnant",
    "newborn", "infant", "toddler", "toddlers",
    "moms", "dads", "dad", "mom", "mommy", "daddy", "parent", "parents",
    "nanny", "babysitter", "doula", "midwife", "pediatrician",
    "awake", "drowsy", "soothing", "fussy", "colic", "colicky",
    "little one", "little ones", "baby girl", "baby boy",
    "boob", "boobs", "breast", "latching",
    "toys", "toy", "gear", "nursery", "carseat", "car seat",
    # ── person / role words ──────────────────────────────────────────────────
    "kids", "children", "child", "teen", "teenager", "adult", "woman", "man",
    "girl", "boy", "people", "person", "friend", "family", "partner",
    "husband", "wife", "sister", "brother", "grandma", "grandpa",
    # ── body / anatomy ───────────────────────────────────────────────────────
    "skin", "hair", "face", "body", "eye", "eyes", "lips", "lip",
    "nail", "nails", "scalp", "pores", "pore", "acne", "redness",
    "aging", "wrinkle", "wrinkles", "dark spot", "dark spots", "scar",
    # ── activity / lifestyle ─────────────────────────────────────────────────
    "workout", "exercise", "running", "walking", "hiking", "cycling",
    "yoga", "pilates", "gym", "training", "recovery", "stretch",
    "sleep", "sleeping", "napping", "rest", "relaxation", "meditation",
    "cooking", "baking", "meal prep", "meal planning", "recipe",
    "travel", "commute", "commuting", "outdoor", "outdoors", "camping",
    # ── generic descriptors (capitalized versions get caught by regex) ────────
    "natural", "organic", "clean", "pure", "fresh", "gentle", "sensitive",
    "hypoallergenic", "fragrance free", "unscented", "lightweight",
    "hydrating", "nourishing", "brightening", "firming", "smoothing",
    "daily", "morning", "evening", "night", "weekly", "routine",
    "affordable", "budget", "luxury", "high end", "drugstore", "premium",
    "travel size", "full size", "mini", "starter",
    "new", "original", "classic", "limited", "exclusive", "special",
    # ── generic product / object nouns ──────────────────────────────────────
    "air fryer", "walking pad", "smart ring", "headphones", "earbuds",
    "backpack", "tumbler", "candle", "leggings", "protein powder",
    "supplement", "collagen", "creatine", "pre workout", "whey", "bcaa",
    "skincare", "makeup", "haircare", "fragrance", "perfume", "cologne",
    "deodorant", "shampoo", "conditioner", "body wash", "face wash",
    "hair mask", "hair oil", "dry shampoo", "hair serum",
    "nail polish", "nail art", "nail gel", "nail lamp",
    "lip gloss", "lip liner", "brow gel", "brow pencil",
    "setting powder", "face mask", "pore strip",
    "camera", "lens", "tripod", "flash", "sensor",
    "keyboard", "mouse", "monitor", "speaker", "microphone",
    "gaming chair", "desk mat", "cable", "charger", "power bank",
    "knife", "blade", "sheath", "handle", "steel",
    "pen", "ink", "nib", "notebook", "journal", "planner",
    "sneaker", "shoe", "boot", "sandal", "slipper",
    "jacket", "coat", "shirt", "pants", "dress", "skirt", "shorts",
    "necklace", "bracelet", "earring",
    "bag", "purse", "wallet", "tote", "clutch",
    "protein", "vitamin", "omega", "probiotic", "prebiotic",
    "bread", "cake", "cookie", "cookies", "pasta", "pizza",
    "chocolate", "vanilla", "caramel", "cinnamon", "flour",
    # ── generic shopping / review discourse ─────────────────────────────────
    "review", "dupe", "haul", "restock", "worth it", "recommendation",
    "product", "item", "purchase", "order", "shipping", "delivery",
    "return", "refund", "discount", "sale", "deal", "coupon", "code",
    "holy grail", "repurchase", "empties", "destash", "pan",
    # ── generic ingredient / material words ──────────────────────────────────
    "acid", "oil", "butter", "extract", "cream", "lotion", "gel",
    "spray", "mist", "drops", "solution", "formula", "treatment",
    "active", "actives", "ingredient", "ingredients",
    # ── tech generic ─────────────────────────────────────────────────────────
    "wireless", "bluetooth", "usb", "hdmi", "oled", "amoled", "ssd",
    "android", "ios", "windows", "linux", "macos", "app",
    # ── Reddit / social discourse ─────────────────────────────────────────────
    "reddit", "post", "thread", "comment", "upvote", "downvote",
    "subreddit", "mod", "ama", "tldr", "faq", "wiki",
    "update", "psa", "rant", "vent", "unpopular opinion",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  Brand-like regex patterns  (candidate detection)
# ─────────────────────────────────────────────────────────────────────────────
# These patterns find tokens that LOOK like brands.
# Matching one of these makes a token a "candidate"; it still needs scoring.

# ALL_CAPS abbreviation-style brands: OPI, NYX, LAMY, TWSBI, CIVIVI
_RE_ALLCAPS   = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")

# CamelCase / InternalCaps: CeraVe, YouTube, DeLonghi, TikTok
_RE_CAMEL     = re.compile(r"\b([A-Z][a-z]+[A-Z][a-zA-Z]{1,15})\b")

# TitleCase single word mid-sentence (not after . or sentence start)
_RE_TITLE     = re.compile(r"(?<=[a-z,;:!?]\s)([A-Z][a-z]{2,15})\b")

# Multi-word TitleCase phrase (2-3 words, each capitalized)
_RE_PHRASE    = re.compile(r"\b([A-Z][a-z]{1,15}(?:\s[A-Z][a-z]{1,15}){1,2})\b")

# Quoted strings — "The Ordinary", "Drunk Elephant"
_RE_QUOTED    = re.compile(r'["""]([^"""]{3,35})["""]')

# Patterns combined (applied in priority order)
_CANDIDATE_PATTERNS = [_RE_QUOTED, _RE_CAMEL, _RE_ALLCAPS, _RE_PHRASE, _RE_TITLE]

# Threshold: tokens with wordfreq ABOVE this are "common words" and need
# context disambiguation before being accepted as brands
AMBIGUITY_FREQ_THRESHOLD = 5e-5   # e.g. coach=9e-5, apple=6e-5 → ambiguous
                                   # nikon=2e-6, cerave=4e-8 → uncommon → accept

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  KNOWN_BRANDS — manual supplement to regex
# ─────────────────────────────────────────────────────────────────────────────
# Brands that might not be caught by the above regex (e.g. common English words
# that happen to be real brands) — or that need to be recognised as brands
# even when written in lowercase.  These are always treated as brand candidates
# and subjected to context disambiguation when their wordfreq is high.

KNOWN_BRANDS: set[str] = {
    # tech / electronics
    "canon", "apple", "amazon", "sony", "samsung", "huawei", "oppo", "xiaomi",
    "oneplus", "motorola", "nokia", "asus", "acer", "lenovo", "intel", "nvidia",
    "amd", "qualcomm", "bosch", "philips", "sharp", "pioneer", "onkyo",
    "marantz", "denon", "bose", "klipsch", "anker", "belkin", "logitech",
    "corsair", "razer", "steelseries", "hyperx", "jabra", "sennheiser",
    "shure", "rode", "zoom", "focusrite", "behringer", "yamaha",
    "nikon", "sigma", "fujifilm", "olympus", "pentax", "leica", "hasselblad",
    "kindle", "kobo", "boox", "dyson", "shark", "roomba", "eufy", "ecovacs",
    "nest", "ring", "arlo", "blink", "hue", "echo", "ikea", "muji",
    # beauty / personal care
    "nyx", "opi", "essie", "orly", "zoya", "revlon", "covergirl", "maybelline",
    "loreal", "lancome", "mac", "elf", "ulta", "sephora", "glossier",
    "fenty", "nars", "tarte", "stila", "benefit", "urban", "tatcha",
    "cerave", "neutrogena", "olay", "cetaphil", "aveeno", "eucerin",
    "clinique", "estee", "kiehls", "cosrx", "innisfree", "laneige",
    "listerine", "colgate", "crest", "sensodyne", "oral",
    # fashion / apparel
    "nike", "adidas", "puma", "reebok", "fila", "vans", "converse",
    "asics", "hoka", "brooks", "saucony", "mizuno", "balenciaga", "yeezy",
    "gucci", "prada", "dior", "chanel", "hermes", "versace", "armani",
    "burberry", "lacoste", "polo", "ralph", "tommy", "calvin", "gap",
    "supreme", "stussy", "carhartt", "patagonia", "columbia", "arcteryx",
    "salomon", "merrell", "keen", "teva", "birkenstock", "ugg", "clarks",
    "timberland", "coach", "target",
    # food / beverage appliances
    "nespresso", "keurig", "vitamix", "ninja", "cuisinart", "breville",
    "kitchenaid", "zojirushi", "instant",
    # tools / outdoor / knives
    "dewalt", "milwaukee", "makita", "ryobi", "craftsman", "stanley",
    "spyderco", "benchmade", "gerber", "kershaw", "victorinox", "leatherman",
    "civivi", "kizer", "vosteed", "protech", "microtech",
    # stationery
    "lamy", "parker", "kaweco", "twsbi", "jinhao", "pelikan", "montblanc",
    "pentel", "staedtler", "pilot", "rhodia", "leuchtturm",
    # fragrance
    "dior", "chanel", "versace", "guerlain", "valentino", "givenchy",
    "ysl", "bvlgari", "creed", "lattafa", "rasasi",
    # gaming / software brands
    "nintendo", "sega", "atari", "ubisoft", "blizzard", "valve",
    # audio / hi-fi
    "truthear", "moondrop", "fiio", "hifiman", "audeze", "focal",
    "beyerdynamic", "grado", "wiim", "naim",
    # misc
    "lego", "hasbro", "fossil", "casio", "seiko", "omega", "rolex",
    "boss", "hugo", "on",   # "on" = On Running shoes
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  Brand-context signals
# ─────────────────────────────────────────────────────────────────────────────

_BRAND_CTX_RE = re.compile(
    r"\b(bought|buy|purchase[d]?|order[ed]?|tried|testing|review[ed]?|dupe|"
    r"worth it|recommend|restock[ed]?|sold out|shop|sephora|ulta|amazon|"
    r"tiktok shop|nordstrom|ssense|mrporter|from|brand|product|collection|"
    r"line|series|collab|loving|obsessed|discovered|found|picked up|gifted)\b",
    re.IGNORECASE,
)

# Anti-brand context: signals that an ambiguous word is NOT used as a brand
_ANTI_CTX_RE = re.compile(
    r"\b(coach said|my coach|fitness coach|basketball coach|soccer coach|"
    r"apple pie|apple tree|apple juice|apple cider|apple orchard|"
    r"ring road|wedding ring|engagement ring|diamond ring|gold ring|"
    r"gap year|mind the gap|bridge the gap|target practice|hit the target|"
    r"boss level|final boss|polo match|water polo|"
    r"bare skin|skin type|skin care routine|"
    r"on the|on a|on my|on your|on his|on her|on its|on our|on their|"
    r"on top|on sale|on time|on track|on board)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"['\-\.\&,!?/\\()\[\]_*@#%^~`|]+")
_SPACE_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = _PUNCT_RE.sub(" ", s.lower())
    return _SPACE_RE.sub(" ", s).strip()


def normalize_display(raw: str) -> str:
    """Fix capitalization for display: SAMSUNg→Samsung, oPI→OPI, lenovo→Lenovo."""
    b = _PUNCT_RE.sub("", raw).strip()
    if not b:
        return ""
    if re.match(r"^[A-Z0-9]{2,6}$", b):   # short ALL-CAPS codes → keep
        return b
    has_weird = any(c.isupper() for c in b[1:]) and any(c.islower() for c in b)
    if has_weird or b.isupper():
        return b.title()
    if b.islower():
        return b.title()
    return b


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  Whitelist loading  (single source)
# ─────────────────────────────────────────────────────────────────────────────

def load_whitelist() -> dict[str, str]:
    """Load Brand List Available全域.xlsx → {norm: display_name}.

    Filtering:
    - Hard reject if in HARD_REJECT.
    - Single-word entries that are ultra-common (top-1000) AND not in KNOWN_BRANDS → skip.
    - Multi-word phrases: skip only if every word is a hard reject.
    """
    if not BRAND_XLS.exists():
        log.warning("Whitelist not found: %s", BRAND_XLS)
        return {}

    all_raw: list[str] = []
    xl = pd.ExcelFile(BRAND_XLS)
    for sheet in xl.sheet_names:
        df_raw = pd.read_excel(BRAND_XLS, sheet_name=sheet, header=None)
        all_raw.extend(
            str(v).strip() for v in df_raw.values.flatten()
            if pd.notna(v) and str(v).strip() not in ("", "nan", "brand_name")
        )
    log.info("Whitelist raw entries: %d", len(all_raw))

    whitelist: dict[str, str] = {}
    for b in all_raw:
        norm = _norm(b)
        if not norm or len(norm) < 2:
            continue
        words = norm.split()

        # Only reject pure NLTK stopwords / absolute noise — trust the whitelist otherwise
        if len(words) == 1 and norm in HARD_REJECT:
            continue
        if len(words) > 1 and all(w in HARD_REJECT for w in words):
            continue
        # Skip product category / ingredient terms masquerading as brands
        if norm in PRODUCT_TERMS:
            continue
        # Reject single common English words not in KNOWN_BRANDS:
        # wordfreq > 1e-4 means the word appears in everyday language too often
        # to be reliably treated as a brand (e.g. "nap", "carrier", "toys").
        # KNOWN_BRANDS entries (apple, ring, on…) are exempt — they're real brands.
        if len(words) == 1 and norm not in KNOWN_BRANDS and _WORDFREQ_OK:
            if _wfreq(norm) > 1e-4:
                continue

        display = normalize_display(b)
        if display and norm not in whitelist:
            whitelist[norm] = display

    log.info("Whitelist after filter: %d brands", len(whitelist))
    return whitelist


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  Candidate extraction  (regex-based discovery)
# ─────────────────────────────────────────────────────────────────────────────

def extract_candidate_tokens(text: str) -> list[tuple[str, str]]:
    """Return (raw_token, pattern_type) pairs from brand-like patterns in text."""
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    for pat, label in [
        (_RE_QUOTED,  "quoted"),
        (_RE_CAMEL,   "camelcase"),
        (_RE_ALLCAPS, "allcaps"),
        (_RE_PHRASE,  "title_phrase"),
        (_RE_TITLE,   "title_word"),
    ]:
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            if raw and raw not in seen:
                seen.add(raw)
                results.append((raw, label))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATE_THRESHOLD = 0.65   # minimum score to accept a discovered candidate

def score_candidate(
    norm: str,
    raw: str,
    pattern_type: str,
    context: str,
    global_freq: int = 1,
    comm_count: int = 1,
) -> tuple[float, str]:
    """Score a candidate. Returns (score 0–1, evidence string).

    Signals
    -------
    +0.30  ALL_CAPS token (very strong brand signal)
    +0.20  CamelCase / InternalCaps
    +0.20  Uncommon word (wordfreq < 1e-5, not in top-5k)
    +0.30  Near brand-context keyword
    +0.10  Appears in 2+ communities (less likely to be noise)
    −0.25  Common word (wordfreq > AMBIGUITY_FREQ_THRESHOLD)
    −0.40  Anti-brand context detected
    Reject  In HARD_REJECT
    """
    if norm in HARD_REJECT:
        return -1.0, "hard_reject"

    score    = 0.0
    evidence = []

    # Pattern-type bonus
    if pattern_type == "allcaps":
        score    += 0.30;  evidence.append("allcaps")
    elif pattern_type == "camelcase":
        score    += 0.20;  evidence.append("camelcase")
    elif pattern_type == "quoted":
        score    += 0.20;  evidence.append("quoted")

    # Word frequency signal
    freq = _wfreq(norm.split()[0]) if norm else 0.0
    if freq < 1e-5 or not _WORDFREQ_OK:
        score    += 0.20;  evidence.append("uncommon")
    elif freq > AMBIGUITY_FREQ_THRESHOLD:
        score    -= 0.25;  evidence.append("-common")

    # Brand-context keyword nearby
    if _BRAND_CTX_RE.search(context):
        score    += 0.30;  evidence.append("brand_ctx")

    # Anti-context (e.g. "basketball coach", "apple pie")
    if _ANTI_CTX_RE.search(context):
        score    -= 0.40;  evidence.append("-anti_ctx")

    # KNOWN_BRANDS override: if it's a known brand, give it a floor
    if norm in KNOWN_BRANDS:
        score     = max(score, 0.45)
        evidence.append("known_brand")

    # Community spread boost
    if comm_count >= 3:
        score    += 0.10;  evidence.append(f"+{comm_count}comm")

    return round(min(score, 0.99), 3), ",".join(evidence) or "no_signal"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  Per-post extraction  (both layers)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BrandMatch:
    brand:      str
    confidence: float
    source:     str    # "whitelist" | "candidate" | "known_brand"
    evidence:   str


def extract_brands_from_post(
    text: str,
    whitelist: dict[str, str],
    cand_freq: dict[str, int] | None   = None,
    cand_comms: dict[str, int] | None  = None,
) -> list[BrandMatch]:
    """Extract brand matches from one post. Returns one BrandMatch per brand."""
    # Pre-normalize text so "La-Mer" → "La Mer" (2 tokens) before splitting,
    # enabling hyphenated brand names to form correct n-grams.
    norm_text = _PUNCT_RE.sub(" ", text)
    norm_text = _SPACE_RE.sub(" ", norm_text).strip()
    tokens  = norm_text.split()
    matched : dict[str, BrandMatch] = {}

    # ── Layer 1: whitelist n-gram scan ────────────────────────────────────
    for n in range(1, min(5, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            key  = _norm(gram)
            if key not in whitelist:
                continue
            # Context window for disambiguation
            pos = text.find(gram)
            ctx = text[max(0, pos - 60) : pos + len(gram) + 60]
            # Reject if clearly in anti-brand context
            if _ANTI_CTX_RE.search(ctx):
                continue
            # Reject product-category terms even if they appear in whitelist
            if key in PRODUCT_TERMS:
                continue
            display = whitelist[key]
            if key not in matched:
                matched[key] = BrandMatch(display, 1.0, "whitelist", "whitelist_match")

    # ── Layer 2: candidate discovery (regex + KNOWN_BRANDS) ───────────────
    # Also check KNOWN_BRANDS that appear in text (lowercase match)
    text_lower = text.lower()
    for brand_norm in KNOWN_BRANDS:
        if brand_norm in matched:
            continue
        if brand_norm not in text_lower:
            continue
        # Find the actual occurrence and check context
        idx = text_lower.find(brand_norm)
        ctx = text[max(0, idx - 60) : idx + len(brand_norm) + 60]
        if _ANTI_CTX_RE.search(ctx):
            continue
        # Only accept known brands if they appear near a brand signal
        # (avoids "apple pie" being accepted just because "apple" is in KNOWN_BRANDS)
        freq = _wfreq(brand_norm)
        if freq > AMBIGUITY_FREQ_THRESHOLD and not _BRAND_CTX_RE.search(ctx):
            continue
        display = normalize_display(brand_norm)
        matched[brand_norm] = BrandMatch(display, 0.85, "known_brand", "known_brand+ctx")

    # Regex-based candidates
    candidates = extract_candidate_tokens(text)
    for raw, pat_type in candidates:
        cand_norm = _norm(raw)
        if not cand_norm or len(cand_norm) < 3 or cand_norm in matched:
            continue
        if cand_norm in HARD_REJECT:
            continue
        if cand_norm in PRODUCT_TERMS:
            continue
        pos = text.find(raw)
        ctx = text[max(0, pos - 60) : pos + len(raw) + 60] if pos >= 0 else ""
        freq_  = cand_freq.get(cand_norm, 1)  if cand_freq  else 1
        comms_ = cand_comms.get(cand_norm, 1) if cand_comms else 1
        score, evidence = score_candidate(cand_norm, raw, pat_type, ctx, freq_, comms_)
        if score >= CANDIDATE_THRESHOLD:
            display = normalize_display(raw)
            if display:
                matched[cand_norm] = BrandMatch(display, score, "candidate", evidence)

    return list(matched.values())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  Category-level stats aggregation
# ─────────────────────────────────────────────────────────────────────────────

def agg_period(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("category").agg(
        mentions        = ("mention_id",         "count"),
        communities     = ("community",          "nunique"),
        mean_sentiment  = ("sentiment_compound",  "mean"),
        mean_engagement = ("engagement_score",    "mean"),
    ).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  Per-window brand table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_brand_tables(
    df_cur:    pd.DataFrame,
    df_prev:   pd.DataFrame,
    whitelist: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Two-pass brand extraction.

    Pass 1: scan all cur posts to build global candidate frequencies.
    Pass 2: full extraction with scoring, using freq context.
    """
    # Pass 1 — global candidate frequency
    log.info("    Pass 1: global candidate freq scan …")
    cand_freq : Counter           = Counter()
    cand_comms: dict[str, set]   = defaultdict(set)

    for row in df_cur.itertuples():
        seen: set[str] = set()
        for raw, _ in extract_candidate_tokens(row.ner):
            cn = _norm(raw)
            if cn and cn not in HARD_REJECT and cn not in seen:
                cand_freq[cn] += 1
                cand_comms[cn].add(row.community)
                seen.add(cn)
        for bk in KNOWN_BRANDS:
            if bk in row.ner.lower() and bk not in seen:
                cand_freq[bk] += 1
                cand_comms[bk].add(row.community)
                seen.add(bk)

    cand_comm_count = {cn: len(s) for cn, s in cand_comms.items()}

    # Pass 2 — extraction with scoring
    log.info("    Pass 2: brand extraction + sentiment …")
    brand_cur_cnt:   dict[str, Counter]               = defaultdict(Counter)
    brand_prev_cnt:  dict[str, Counter]               = defaultdict(Counter)
    brand_sentiment: dict[str, dict[str, list]]       = defaultdict(lambda: defaultdict(list))
    brand_meta:      dict[str, dict[str, BrandMatch]] = defaultdict(dict)

    for row in df_cur.itertuples():
        matches = extract_brands_from_post(row.ner, whitelist, cand_freq, cand_comm_count)
        for m in matches:
            key = _norm(m.brand)
            brand_cur_cnt[row.category][m.brand]          += 1
            brand_sentiment[row.category][m.brand].append(row.sentiment_compound)
            if m.brand not in brand_meta[row.category]:
                brand_meta[row.category][m.brand] = m

    for row in df_prev.itertuples():
        matches = extract_brands_from_post(row.ner, whitelist)
        for m in matches:
            brand_prev_cnt[row.category][m.brand] += 1

    # Cross-category frequency filter: brand appearing in >40% of categories = generic
    global_brand_cats: Counter = Counter()
    for cat, cnt in brand_cur_cnt.items():
        for b in cnt:
            global_brand_cats[b] += 1
    total_cats = max(len(brand_cur_cnt), 1)

    MIN_MENTIONS = 3
    MAX_CAT_PCT  = 0.40

    cat_brand_data: dict[str, pd.DataFrame] = {}
    for cat in df_cur["category"].unique():
        cur_c  = brand_cur_cnt.get(cat,  Counter())
        prev_c = brand_prev_cnt.get(cat, Counter())
        meta   = brand_meta.get(cat, {})
        rows   = []
        for b, cc in cur_c.items():
            if cc < MIN_MENTIONS:
                continue
            if global_brand_cats[b] / total_cats > MAX_CAT_PCT:
                continue
            pc       = prev_c.get(b, 0)
            sents    = brand_sentiment[cat].get(b, [])
            avg_sent = round(sum(sents) / len(sents), 3) if sents else 0.0
            m        = meta.get(b, BrandMatch(b, 0.5, "unknown", ""))
            rows.append({
                "brand":         b,
                "cur_mentions":  cc,
                "prev_mentions": pc,
                "brand_spike":   round(cc / max(pc, 1), 2),
                "avg_sentiment": avg_sent,
                "confidence":    m.confidence,
                "source":        m.source,
                "evidence":      m.evidence,
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading parquet …")
    df = pd.read_parquet(PARQUET)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["category"]     = df["category"].fillna("unknown")
    df["title"]        = df["title"].fillna("")
    df["text"]         = df["text"].fillna("").str[:400]
    df["ner"]          = (df["title"] + " " + df["text"]).str.strip()
    log.info("Rows: %d", len(df))

    latest  = df["published_at"].max()
    windows : dict[str, dict] = {}
    for i in range(4):
        end        = latest - pd.Timedelta(days=14 * i)
        start      = end    - pd.Timedelta(days=14)
        prev_end   = start
        prev_start = start  - pd.Timedelta(days=14)
        label = f"{start.strftime('%m/%d')}–{end.strftime('%m/%d')}"
        windows[label] = dict(
            cur_start=start, cur_end=end,
            prev_start=prev_start, prev_end=prev_end,
        )

    whitelist = load_whitelist()

    all_window_data: dict[str, dict] = {}

    for win_label, win in windows.items():
        log.info("=== Window: %s ===", win_label)
        df_cur  = df[(df["published_at"] >= win["cur_start"])  & (df["published_at"] < win["cur_end"])]
        df_prev = df[(df["published_at"] >= win["prev_start"]) & (df["published_at"] < win["prev_end"])]
        log.info("  cur=%d  prev=%d", len(df_cur), len(df_prev))

        # Category-level stats
        cur_s  = agg_period(df_cur).add_suffix("_c").rename(columns={"category_c":  "category"})
        prev_s = agg_period(df_prev).add_suffix("_p").rename(columns={"category_p": "category"})
        all_cats = pd.DataFrame({"category": df["category"].unique()})
        stats = (all_cats
                 .merge(cur_s,  on="category", how="left")
                 .merge(prev_s, on="category", how="left")
                 .fillna(0))

        stats["spike_ratio"]       = (stats["mentions_c"] / stats["mentions_p"].replace(0, 1)).round(3)
        max_spike                  = stats["spike_ratio"].replace([np.inf, -np.inf], 0).quantile(0.97)
        stats["normalized_spike"]  = (stats["spike_ratio"].clip(upper=max_spike) / max(max_spike, 1)).round(4)
        max_comm                   = stats["communities_c"].max()
        stats["cross_community"]   = (stats["communities_c"] / max(max_comm, 1)).round(4)
        stats["sentiment_score"]   = ((stats["mean_sentiment_c"] + 1) / 2).round(4)
        med_eng                    = stats["mean_engagement_c"].median()
        stats["eng_momentum"]      = (
            stats["mean_engagement_c"] / stats["mean_engagement_p"].replace(0, med_eng)
        ).clip(upper=4).fillna(1).round(4)
        stats["eng_momentum_norm"] = (
            stats["eng_momentum"] / max(stats["eng_momentum"].quantile(0.97), 1)
        ).round(4)

        stats["trend_score"] = (
            0.25 * stats["normalized_spike"]  +
            0.25 * stats["cross_community"]   +
            0.25 * stats["sentiment_score"]   +
            0.25 * stats["eng_momentum_norm"]
        ).round(4)

        stats["mentions_delta"] = (stats["mentions_c"] - stats["mentions_p"]).astype(int)

        def _classify_direction(row: pd.Series) -> str:
            if (row["mentions_c"]    >= 50 and
                row["mentions_delta"] >= 50 and
                row["spike_ratio"]   >= 2.3 and
                row["communities_c"] >= 3):
                return "rising"
            if (row["mentions_p"]    >= 50 and
                row["mentions_delta"] <= -25 and
                row["spike_ratio"]   <= 0.5):
                return "declining"
            return "stable"

        stats["trend_direction"] = stats.apply(_classify_direction, axis=1)
        stats = stats.rename(columns={
            "mentions_c":       "current_mentions",
            "mentions_p":       "previous_mentions",
            "communities_c":    "current_communities",
            "mean_sentiment_c": "mean_sentiment",
            "mean_engagement_c":"mean_engagement",
        })
        stats = stats.sort_values("trend_score", ascending=False).reset_index(drop=True)

        # Brand extraction
        cat_brand_data = build_brand_tables(df_cur, df_prev, whitelist)
        stats["top_brands"] = stats["category"].map(
            lambda c: cat_brand_data.get(c, pd.DataFrame()).head(20)["brand"].tolist()
            if c in cat_brand_data else []
        )

        # Weekly breakdown
        df_win = df[
            (df["published_at"] >= win["prev_start"]) &
            (df["published_at"] <  win["cur_end"])
        ].copy()
        with pd.option_context("mode.chained_assignment", None):
            df_win["week"] = df_win["published_at"].dt.to_period("W").dt.start_time
        weekly = (
            df_win.groupby(["category", "week"])
            .agg(mentions=("mention_id", "count"),
                 mean_sentiment=("sentiment_compound", "mean"))
            .reset_index()
        )
        weekly["week"] = weekly["week"].dt.strftime("%Y-%m-%d")

        # Sample posts
        top_posts: dict[str, list[dict]] = {}
        for cat, grp in df_cur.groupby("category"):
            top_posts[cat] = (
                grp.nlargest(5, "engagement_score")
                [["title", "community", "engagement_score", "sentiment_label", "url"]]
                .to_dict("records")
            )
        stats["sample_posts"] = stats["category"].map(lambda c: top_posts.get(c, []))

        all_window_data[win_label] = {
            "stats":          stats,
            "weekly":         weekly,
            "cat_brand_data": cat_brand_data,
            "window":         win,
        }

    payload = {"windows": all_window_data, "window_labels": list(windows.keys())}
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f)
    log.info("Saved → %s", OUT_PKL)


if __name__ == "__main__":
    main()
