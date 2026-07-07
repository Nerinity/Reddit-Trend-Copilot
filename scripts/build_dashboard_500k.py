#!/usr/bin/env python3
"""Build dashboard_data_500k.pkl — 8-layer brand extraction pipeline.

Architecture
============
Layer 0   Normalization
    Unicode NFKC, accent stripping, apostrophe/dash normalization.
    Applied to raw text before any matching so accented variants
    (L'Oréal → LOreal) resolve correctly against the whitelist.

Layer 1   Whitelist exact + normalized exact
    Single canonical source: Brand List Available全域.xlsx.
    N-gram lookup after Layer 0 normalization → confidence 1.0.

Layer 1.5 English aliases / spelling variants
    BRAND_ALIASES dict maps lowercase aliases and common misspellings
    to canonical display names (e.g. "loreal" → "L'Oréal",
    "la roche posay" → "La Roche-Posay").

Layer 2   Product-line-to-brand mapping
    PRODUCT_LINE_MAP maps specific product lines to the parent brand
    (e.g. "airpods" → "Apple", "galaxy buds" → "Samsung").
    Adds the parent brand in addition to any other matches.

Layer 3   Regex candidate discovery
    Patterns detect brand-like tokens: ALL_CAPS, CamelCase, TitleCase,
    multi-word TitleCase, and quoted strings.
    KNOWN_BRANDS supplements regex for words that look generic.

Layer 4   Contextual scoring
    Each regex candidate is scored with score_candidate():
    pattern bonus, wordfreq signal, brand-context keywords.
    score >= CANDIDATE_THRESHOLD (0.65) → accepted.

Layer 5   Ambiguous brand resolver
    _BRAND_CTX_RE / _ANTI_CTX_RE applied to context windows:
    common words (apple, coach, ring …) accepted only when brand
    signals are present and anti-brand signals are absent.

Layer 6   Cross-community evidence
    cand_comms tracking in build_brand_tables Pass 1 gives a +0.10
    boost in score_candidate for brands seen in 3+ communities.

Layer 7   Hard reject
    HARD_REJECT (NLTK stopwords + absolute noise) and PRODUCT_TERMS
    (ingredients, product types, descriptors) applied at every layer.

Layer 8   Review queue
    Candidates with score in [0.45, 0.65) are flagged but not accepted.
    Written to brand_review_queue.csv after each full run.

Output columns per brand
    brand, cur_mentions, prev_mentions, brand_spike,
    avg_sentiment, confidence, source, evidence
"""
from __future__ import annotations
import logging
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_dashboard")

# Context-first brand extraction (new pipeline)
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent))
try:
    from scripts.brand_context import (
        load_category_terms,
        build_brand_tables_ctx,
    )
    _CTX_BRAND_OK = True
except Exception as _e:
    log.warning("brand_context import failed (%s) — falling back to legacy pipeline", _e)
    _CTX_BRAND_OK = False

ROOT    = Path(__file__).parent.parent
PARQUET = ROOT / "data" / "processed" / "nlp_clustered_500k.parquet"
OUT_PKL = ROOT / "data" / "processed" / "dashboard_data_500k.pkl"
BRAND_XLS = Path.home() / "Downloads" / "Brand List Available全域.xlsx"
ARCHIVE_DIR = ROOT / "data" / "processed" / "archive"
ARCHIVE_PKL = ARCHIVE_DIR / "dashboard_weekly_archive.pkl"
ARCHIVE_CSV = ARCHIVE_DIR / "dashboard_weekly_archive.csv"

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
# SECTION 3.5  BRAND_ALIASES  (Layer 1.5)
# ─────────────────────────────────────────────────────────────────────────────
# Maps lowercase alias / common misspelling → canonical display name.
# Applied AFTER Layer 1 whitelist misses.  Keys are searched case-insensitively
# in the full post text (not just token boundaries) so multi-word aliases work.

BRAND_ALIASES: dict[str, str] = {
    # L'Oréal variants
    "loreal":                   "L'Oréal",
    "l oreal":                  "L'Oréal",
    "l'oreal":                  "L'Oréal",
    # La Roche-Posay
    "la roche posay":           "La Roche-Posay",
    "laroche posay":            "La Roche-Posay",
    "la roche-posay":           "La Roche-Posay",
    # SK-II
    "skii":                     "SK-II",
    "sk ii":                    "SK-II",
    "sk2":                      "SK-II",
    # e.l.f. Cosmetics
    "elf cosmetics":            "e.l.f. Cosmetics",
    # Pat McGrath
    "pat mcgrath":              "Pat McGrath Labs",
    "pmg":                      "Pat McGrath Labs",
    # Charlotte Tilbury
    "charlotte tilbury":        "Charlotte Tilbury",
    "ct beauty":                "Charlotte Tilbury",
    # Too Faced
    "too faced":                "Too Faced",
    # Urban Decay
    "urban decay":              "Urban Decay",
    # The Ordinary
    "the ordinary":             "The Ordinary",
    # The Inkey List
    "the inkey list":           "The Inkey List",
    "inkey list":               "The Inkey List",
    # Paula's Choice
    "paulas choice":            "Paula's Choice",
    "paula's choice":           "Paula's Choice",
    # SkinCeuticals
    "skinceuticals":            "SkinCeuticals",
    # Glow Recipe
    "glow recipe":              "Glow Recipe",
    # Sunday Riley
    "sunday riley":             "Sunday Riley",
    # Dear Klairs
    "dear klairs":              "Dear Klairs",
    "klairs":                   "Dear Klairs",
    # Some By Mi
    "some by mi":               "Some By Mi",
    # Dr. Jart+
    "dr jart":                  "Dr. Jart+",
    "dr. jart":                 "Dr. Jart+",
    "drjart":                   "Dr. Jart+",
    # Mario Badescu
    "mario badescu":            "Mario Badescu",
    # Peter Thomas Roth
    "peter thomas roth":        "Peter Thomas Roth",
    # Nature Republic
    "nature republic":          "Nature Republic",
    # Etude House
    "etude house":              "Etude House",
    # Drunk Elephant
    "drunk elephant":           "Drunk Elephant",
    # Good Molecules
    "good molecules":           "Good Molecules",
    # On Running
    "on running":               "On Running",
    # New Balance
    "new balance":              "New Balance",
    # Hoka One One
    "hoka one one":             "HOKA",
    # Arc'teryx
    "arcteryx":                 "Arc'teryx",
    "arc teryx":                "Arc'teryx",
    # Le Creuset
    "le creuset":               "Le Creuset",
    # De'Longhi
    "delonghi":                 "De'Longhi",
    "de longhi":                "De'Longhi",
    # KitchenAid
    "kitchenaid":               "KitchenAid",
    # Vitamix
    "vitamix":                  "Vitamix",
    # Lodge Cast Iron
    "lodge cast iron":          "Lodge",
    # Staub
    "staub cookware":           "Staub",
    # Zojirushi
    "zojirushi":                "Zojirushi",
    # Stanley (drinkware, not tools)
    "stanley cup":              "Stanley",
    "stanley tumbler":          "Stanley",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3.6  PRODUCT_LINE_MAP  (Layer 2)
# ─────────────────────────────────────────────────────────────────────────────
# Maps specific product lines (lowercase) → parent brand display name.
# When a product line is found in text, the parent brand is ADDED to matches
# (the product line itself may also remain as its own match).
# Keys are searched case-insensitively as substrings in the post text.

PRODUCT_LINE_MAP: dict[str, str] = {
    # Apple
    "airpods":          "Apple",
    "iphone":           "Apple",
    "ipad":             "Apple",
    "macbook":          "Apple",
    "apple watch":      "Apple",
    "apple tv":         "Apple",
    "apple card":       "Apple",
    # Samsung
    "galaxy s":         "Samsung",
    "galaxy z":         "Samsung",
    "galaxy a":         "Samsung",
    "galaxy tab":       "Samsung",
    "galaxy buds":      "Samsung",
    "galaxy watch":     "Samsung",
    # Google
    "pixel phone":      "Google",
    "pixel buds":       "Google",
    "pixel watch":      "Google",
    "chromecast":       "Google",
    "google home":      "Google",
    "nest mini":        "Google",
    "nest hub":         "Google",
    # Microsoft
    "surface pro":      "Microsoft",
    "surface laptop":   "Microsoft",
    "surface go":       "Microsoft",
    "xbox series":      "Microsoft",
    "xbox one":         "Microsoft",
    # Sony
    "playstation 5":    "Sony",
    "playstation 4":    "Sony",
    "ps5":              "Sony",
    "ps4":              "Sony",
    "wh-1000":          "Sony",   # WH-1000XM noise-cancelling headphone series
    "linkbuds":         "Sony",
    # Nintendo
    "nintendo switch":  "Nintendo",
    "switch oled":      "Nintendo",
    "switch lite":      "Nintendo",
    # Amazon
    "echo dot":         "Amazon",
    "echo show":        "Amazon",
    "echo pop":         "Amazon",
    "fire tv":          "Amazon",
    "fire tablet":      "Amazon",
    # Dyson
    "dyson airwrap":    "Dyson",
    "dyson supersonic": "Dyson",
    "dyson v":          "Dyson",
    "dyson tp":         "Dyson",
    # iRobot
    "roomba":           "iRobot",
    # GoPro
    "gopro hero":       "GoPro",
    "gopro max":        "GoPro",
    # Instant Brands
    "instant pot":      "Instant Brands",
    # Nespresso
    "nespresso vertuo": "Nespresso",
    "nespresso original":"Nespresso",
    # Keurig
    "keurig k-":        "Keurig",
    # Vitamix blenders
    "vitamix a":        "Vitamix",
    "vitamix e":        "Vitamix",
    # Shark vacuums
    "shark navigator":  "Shark",
    "shark iz":         "Shark",
    "shark stratos":    "Shark",
    # LAMY pens
    "lamy safari":      "LAMY",
    "lamy 2000":        "LAMY",
    # Pilot pens
    "pilot metropolitan": "Pilot",
    "pilot kakuno":     "Pilot",
    # Twsbi
    "twsbi eco":        "TWSBI",
    "twsbi diamond":    "TWSBI",
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


def normalize_for_match(s: str) -> str:
    """Layer 0: unicode NFKC + accent strip + apostrophe/dash normalization.

    Makes L'Oréal → L'Oreal, naïve → naive, etc., so whitelist keys built
    from accent-free names still match accented text (and vice-versa).
    """
    s = unicodedata.normalize("NFKC", s)
    # Strip combining diacritics (accents)
    nfd = unicodedata.normalize("NFD", s)
    s = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    # Normalize curly/fancy apostrophes → straight '
    s = re.sub(r"[‘’`´]", "'", s)
    # Normalize em/en dashes → hyphen
    s = re.sub(r"[–—−]", "-", s)
    return s


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
        # wordfreq > 5e-5 means the word appears in everyday language too often
        # to be reliably treated as a brand (e.g. "camp", "mountain", "ring", "snow").
        # Lowered from 1e-4: common activity/geography/descriptor words at 5e-6–1e-4
        # were entering the whitelist and generating noise (Runner, Camp, Mountain, Miles).
        # KNOWN_BRANDS entries (apple, ring, on…) are exempt — they're real brands.
        if len(words) == 1 and norm not in KNOWN_BRANDS and _WORDFREQ_OK:
            if _wfreq(norm) > 5e-5:
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
# SECTION 9  Per-post extraction  (8-layer pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BrandMatch:
    brand:      str
    confidence: float
    source:     str    # "whitelist"|"alias"|"product_line"|"known_brand"|"candidate"
    evidence:   str


# Layer 8: review queue — borderline candidates (score in [0.45, CANDIDATE_THRESHOLD))
# Reset and exported to CSV by main() after each full run.
_review_queue: list[dict] = []


def extract_brands_from_post(
    text: str,
    whitelist: dict[str, str],
    cand_freq:  dict[str, int] | None  = None,
    cand_comms: dict[str, int] | None  = None,
    community:  str                    = "",
) -> list[BrandMatch]:
    """Extract brand matches — 8-layer pipeline. Returns one BrandMatch per brand."""
    global _review_queue

    # ── Layer 0: Normalize ────────────────────────────────────────────────
    # Unicode NFKC + accent stripping + apostrophe/dash normalization so that
    # "L'Oréal" in text resolves against "loreal" whitelist key, etc.
    norm_text = normalize_for_match(text)
    # Pre-normalize for n-gram tokenization (hyphens → spaces, collapse whitespace)
    gram_text = _PUNCT_RE.sub(" ", norm_text)
    gram_text = _SPACE_RE.sub(" ", gram_text).strip()
    tokens    = gram_text.split()
    matched: dict[str, BrandMatch] = {}

    # ── Layer 1: Whitelist exact + normalized exact ───────────────────────
    for n in range(1, min(5, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            key  = _norm(gram)
            # Also try compact key: handles "Arc'teryx" → norm "arc teryx" ≠ whitelist "arcteryx"
            key_compact = key.replace(" ", "")
            if key in whitelist:
                actual_key = key
            elif key_compact != key and key_compact in whitelist:
                actual_key = key_compact
            else:
                continue
            # Layer 7: hard reject even within whitelist
            if actual_key in PRODUCT_TERMS:
                continue
            # Capitalization guard: single-word whitelist entries that appear lowercase in the
            # original text are common English words (kitchen, destroyed, burn, desk, shower).
            # Real brands appear capitalized mid-sentence (Salomon, CeraVe) or ALL_CAPS (BURN).
            gram_words = gram.split()
            if len(gram_words) == 1 and gram[0].islower() and actual_key not in KNOWN_BRANDS:
                continue
            # For multi-word phrases starting with a possessive pronoun (my, your, our, his, her,
            # their), reject if the phrase appears at a sentence boundary — that's grammatical
            # capitalization, not a brand name ("My Skin has been glowing" ≠ brand "My Skin").
            # The same phrase mid-sentence with capital first letter IS a brand signal.
            _POSSESSIVE = {"my", "your", "our", "his", "her", "their", "its"}
            if len(gram_words) > 1 and gram_words[0].lower() in _POSSESSIVE:
                pos = norm_text.find(gram)
                if pos >= 0:
                    pre = norm_text[:pos].rstrip()
                    if not pre or pre[-1] in ".!?\n":
                        continue
            # Layer 5: context window — anti-brand signal
            pos = norm_text.find(gram)
            ctx = text[max(0, pos - 60) : pos + len(gram) + 60] if pos >= 0 else ""
            if _ANTI_CTX_RE.search(ctx):
                continue
            display = whitelist[actual_key]
            if actual_key not in matched:
                matched[actual_key] = BrandMatch(display, 1.0, "whitelist", "whitelist_match")

    # ── Layer 1.5: BRAND_ALIASES — spelling variants / lowercase aliases ──
    text_lower = norm_text.lower()
    for alias, canonical in BRAND_ALIASES.items():
        if alias not in text_lower:
            continue
        alias_key = _norm(alias)
        if alias_key in matched:
            continue
        idx = text_lower.find(alias)
        ctx = text[max(0, idx - 60) : idx + len(alias) + 60]
        # Layer 7 + Layer 5
        if _ANTI_CTX_RE.search(ctx):
            continue
        matched[alias_key] = BrandMatch(canonical, 1.0, "alias", "brand_alias")

    # ── Layer 2: PRODUCT_LINE_MAP — product line → parent brand ──────────
    for product_line, parent_brand in PRODUCT_LINE_MAP.items():
        if product_line not in text_lower:
            continue
        parent_key = _norm(parent_brand)
        if parent_key in matched:
            continue
        idx = text_lower.find(product_line)
        ctx = text[max(0, idx - 60) : idx + len(product_line) + 60]
        if _ANTI_CTX_RE.search(ctx):
            continue
        matched[parent_key] = BrandMatch(
            parent_brand, 0.95, "product_line", f"product_line:{product_line}"
        )

    # ── Layer 3: Regex candidate discovery + KNOWN_BRANDS ────────────────
    # KNOWN_BRANDS — words that look generic but are real brands
    for brand_norm in KNOWN_BRANDS:
        if brand_norm in matched:
            continue
        if brand_norm not in text_lower:
            continue
        idx = text_lower.find(brand_norm)
        ctx = text[max(0, idx - 60) : idx + len(brand_norm) + 60]
        # Layer 5: Ambiguous brand resolver
        if _ANTI_CTX_RE.search(ctx):
            continue
        freq = _wfreq(brand_norm)
        if freq > AMBIGUITY_FREQ_THRESHOLD and not _BRAND_CTX_RE.search(ctx):
            continue
        display = normalize_display(brand_norm)
        matched[brand_norm] = BrandMatch(display, 0.85, "known_brand", "known_brand+ctx")

    # Regex patterns (ALL_CAPS, CamelCase, TitleCase, quoted)
    candidates = extract_candidate_tokens(text)
    for raw, pat_type in candidates:
        cand_norm = _norm(raw)
        if not cand_norm or len(cand_norm) < 3 or cand_norm in matched:
            continue
        # Layer 7: Hard reject
        if cand_norm in HARD_REJECT or cand_norm in PRODUCT_TERMS:
            continue
        pos = text.find(raw)
        ctx = text[max(0, pos - 60) : pos + len(raw) + 60] if pos >= 0 else ""
        freq_  = cand_freq.get(cand_norm, 1)  if cand_freq  else 1
        comms_ = cand_comms.get(cand_norm, 1) if cand_comms else 1
        # Layer 4: contextual scoring (includes Layer 5 anti-ctx, Layer 6 community boost)
        score, evidence = score_candidate(cand_norm, raw, pat_type, ctx, freq_, comms_)
        if score >= CANDIDATE_THRESHOLD:
            display = normalize_display(raw)
            if display:
                matched[cand_norm] = BrandMatch(display, score, "candidate", evidence)
        elif 0.45 <= score < CANDIDATE_THRESHOLD:
            # Layer 8: Review queue — borderline candidates flagged for inspection
            _review_queue.append({
                "brand_norm": cand_norm,
                "brand_raw":  raw,
                "score":      score,
                "evidence":   evidence,
                "context":    ctx[:120],
                "community":  community,
            })

    # Longest-match deduplication — two rules:
    # 1. Prefix rule: "arc" is prefix of "arcteryx" → remove "arc" (tokenisation artifact).
    # 2. Word-containment rule: if multi-gram "the ordinary" matched, remove 1-gram "ordinary"
    #    because it is a content word inside the longer match.  Skip short function words
    #    (len < 4) so "the", "of", "by" don't incorrectly suppress 1-gram brands.
    if len(matched) > 1:
        to_remove: set[str] = set()
        matched_keys = list(matched.keys())
        # Rule 1: prefix
        for k1 in matched_keys:
            for k2 in matched_keys:
                if k1 != k2 and k2.startswith(k1) and len(k2) > len(k1):
                    to_remove.add(k1)
        # Rule 2: word containment in multi-gram
        multi_keys = [k for k in matched_keys if " " in k]
        if multi_keys:
            multi_content_words: set[str] = set()
            for mk in multi_keys:
                for w in mk.split():
                    if len(w) >= 4:   # skip "the", "of", "by", "and" etc.
                        multi_content_words.add(w)
            for k1 in matched_keys:
                if k1 not in to_remove and " " not in k1 and k1 in multi_content_words:
                    to_remove.add(k1)
        for k in to_remove:
            del matched[k]

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
        matches = extract_brands_from_post(
            row.ner, whitelist, cand_freq, cand_comm_count, community=row.community
        )
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
    global _review_queue
    _review_queue = []   # reset Layer 8 queue for this run

    log.info("Loading parquet …")
    df = pd.read_parquet(PARQUET)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["title"]        = df["title"].fillna("")
    df["text"]         = df["text"].fillna("").str[:400]
    df["ner"]          = (df["title"] + " " + df["text"]).str.strip()
    log.info("Rows: %d", len(df))

    # Use target_cluster (171 TikTok Shop taxonomy) as primary dashboard dimension.
    # Posts scoring >= TC_MIN_SCORE get their matched cluster; the rest → "Other".
    TC_MIN_SCORE = 0.10
    if "target_cluster" in df.columns:
        score_col = df["target_cluster_score"] if "target_cluster_score" in df.columns else pd.Series(1.0, index=df.index)
        valid_mask = (
            df["target_cluster"].notna() &
            ~df["target_cluster"].isin(["unassigned", "unknown", ""]) &
            (score_col >= TC_MIN_SCORE)
        )
        df["category"] = "Other"
        df.loc[valid_mask, "category"] = df.loc[valid_mask, "target_cluster"]
        log.info("target_cluster taxonomy applied to %d / %d posts  (%.1f%%)",
                 valid_mask.sum(), len(df), 100 * valid_mask.sum() / len(df))
        log.info("  Posts in 'Other' (score < %.2f): %d", TC_MIN_SCORE, (~valid_mask).sum())
    else:
        df["category"] = df["category"].fillna("unknown")
        log.info("target_cluster column not found — using legacy category labels. "
                 "Run scripts/assign_target_clusters.py to enable 171-cluster taxonomy.")

    latest  = df["published_at"].max()
    archive_cutoff = latest - pd.Timedelta(days=28)

    # Archive older-than-4-week history separately. The main dashboard payload stays
    # lightweight and product-facing, while historical weekly facts remain available.
    archive_df = df[df["published_at"] < archive_cutoff].copy()
    if not archive_df.empty:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_df["week"] = archive_df["published_at"].dt.tz_convert(None).dt.to_period("W").dt.start_time
        archive_weekly = (
            archive_df.groupby(["category", "week"])
            .agg(
                mentions=("mention_id", "count"),
                communities=("community", "nunique"),
                mean_sentiment=("sentiment_compound", "mean"),
                mean_engagement=("engagement_score", "mean"),
            )
            .reset_index()
        )
        archive_weekly["week"] = archive_weekly["week"].dt.strftime("%Y-%m-%d")
        with open(ARCHIVE_PKL, "wb") as f:
            pickle.dump(
                {
                    "archive_cutoff": archive_cutoff,
                    "rows": len(archive_df),
                    "weekly": archive_weekly,
                },
                f,
            )
        archive_weekly.to_csv(ARCHIVE_CSV, index=False)
        log.info(
            "Archived older history (< %s): %d rows, %d category-weeks → %s",
            archive_cutoff.strftime("%Y-%m-%d"),
            len(archive_df),
            len(archive_weekly),
            ARCHIVE_PKL,
        )

    windows : dict[str, dict] = {}
    for i in range(4):
        end        = latest - pd.Timedelta(days=7 * i)
        start      = end    - pd.Timedelta(days=7)
        prev_end   = start
        prev_start = start  - pd.Timedelta(days=7)
        label = f"{start.strftime('%m/%d')}–{end.strftime('%m/%d')}"
        windows[label] = dict(
            cur_start=start, cur_end=end,
            prev_start=prev_start, prev_end=prev_end,
            cadence="weekly",
        )

    whitelist = load_whitelist()

    # Load category terms for context-first pipeline
    if _CTX_BRAND_OK:
        category_terms = load_category_terms()
        log.info("Category terms loaded: %d", len(category_terms))
    else:
        category_terms = set()

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

        cur_total = max(len(df_cur), 1)
        prev_total = max(len(df_prev), 1)
        n_cats = max(len(stats), 1)

        stats["spike_ratio"]       = (stats["mentions_c"] / stats["mentions_p"].replace(0, 1)).round(3)
        # Smoothed share ratio corrects for weeks where total Reddit volume is
        # unusually high/low, so a category only rises if it gains share.
        stats["current_share"]     = ((stats["mentions_c"] + 1) / (cur_total + n_cats)).round(6)
        stats["previous_share"]    = ((stats["mentions_p"] + 1) / (prev_total + n_cats)).round(6)
        stats["share_ratio"]       = (stats["current_share"] / stats["previous_share"].replace(0, 1 / (prev_total + n_cats))).round(3)
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
        stats["community_delta"] = (stats["communities_c"] - stats["communities_p"]).astype(int)
        stats["log_growth"] = (
            np.log1p(stats["mentions_c"]) - np.log1p(stats["mentions_p"])
        ).round(4)
        stats["growth_z"] = (
            stats["mentions_delta"] / np.sqrt(stats["mentions_c"] + stats["mentions_p"] + 1)
        ).round(4)

        direction_thresholds = {
            "rising": {
                "min_current_mentions": 25,
                "min_delta": "max(10, 20% of previous mentions)",
                "min_share_ratio": 1.35,
                "min_log_growth": round(float(np.log(1.30)), 4),
                "min_growth_z": 2.0,
                "min_current_communities": 3,
            },
            "declining": {
                "min_previous_mentions": 25,
                "max_delta": "-max(10, 20% of previous mentions)",
                "max_share_ratio": 0.75,
                "max_log_growth": round(float(np.log(0.80)), 4),
                "max_growth_z": -2.0,
            },
        }

        def _classify_direction(row: pd.Series) -> str:
            rising_delta_floor = max(10, 0.20 * row["mentions_p"])
            declining_delta_floor = max(10, 0.20 * row["mentions_p"])

            if (row["mentions_c"]     >= 25 and
                row["mentions_delta"] >= rising_delta_floor and
                row["share_ratio"]    >= 1.35 and
                row["log_growth"]     >= np.log(1.30) and
                row["growth_z"]       >= 2.0 and
                row["communities_c"]  >= 3):
                return "rising"
            if (row["mentions_p"]     >= 25 and
                row["mentions_delta"] <= -declining_delta_floor and
                row["share_ratio"]    <= 0.75 and
                row["log_growth"]     <= np.log(0.80) and
                row["growth_z"]       <= -2.0):
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

        # Brand extraction — use context-first pipeline when available
        if _CTX_BRAND_OK and category_terms:
            cat_brand_data = build_brand_tables_ctx(
                df_cur, df_prev, whitelist, category_terms
            )
        else:
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
            df_win["week"] = df_win["published_at"].dt.tz_convert(None).dt.to_period("W").dt.start_time
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
            "direction_thresholds": direction_thresholds,
        }

    payload = {"windows": all_window_data, "window_labels": list(windows.keys())}
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f)
    log.info("Saved → %s", OUT_PKL)

    # ── Brand posts index (for Streamlit Cloud where full parquet is unavailable) ──
    # For each (category, brand) in the latest window, store top 10 posts by engagement.
    log.info("Building brand posts index …")
    latest_win   = list(windows.keys())[0]
    latest_df    = df[df["published_at"] >= windows[latest_win]["cur_start"]].copy()
    POST_COLS    = ["title", "community", "engagement_score", "sentiment_label",
                    "url", "published_at"]
    brand_posts_index: dict[str, dict[str, list[dict]]] = {}

    latest_cat_brand = all_window_data[latest_win]["cat_brand_data"]
    for cat, bdf in latest_cat_brand.items():
        if bdf.empty:
            continue
        brand_posts_index[cat] = {}
        cat_df = latest_df[latest_df["category"] == cat]
        for brand in bdf["brand"].tolist():
            pat = re.escape(brand)
            mask = (
                cat_df["ner_input"].str.contains(pat, case=False, na=False) |
                cat_df["title"].str.contains(pat, case=False, na=False)
            )
            hits = (
                cat_df[mask]
                .nlargest(10, "engagement_score")[POST_COLS]
                .copy()
            )
            hits["published_at"] = hits["published_at"].astype(str)
            brand_posts_index[cat][brand] = hits.to_dict("records")

    brand_posts_path = ROOT / "data" / "processed" / "brand_posts_index.pkl"
    with open(brand_posts_path, "wb") as f:
        pickle.dump(brand_posts_index, f)
    total_entries = sum(len(v) for v in brand_posts_index.values())
    log.info("Brand posts index: %d categories, %d brands → %s (%.0f KB)",
             len(brand_posts_index), total_entries, brand_posts_path,
             brand_posts_path.stat().st_size / 1024)

    # ── Layer 8: Export review queue ──────────────────────────────────────
    if _review_queue:
        rq_df = (
            pd.DataFrame(_review_queue)
            .sort_values("score", ascending=False)
            .drop_duplicates("brand_norm")
            .reset_index(drop=True)
        )
        rq_path = ROOT / "data" / "processed" / "brand_review_queue.csv"
        rq_df.to_csv(rq_path, index=False)
        log.info("Review queue: %d borderline candidates → %s", len(rq_df), rq_path)


if __name__ == "__main__":
    main()
