#!/usr/bin/env python3
"""Build dashboard_data_500k.pkl with per-brand sentiment & equal-weight trend score."""
from __future__ import annotations
import logging, pickle, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_dashboard")

ROOT    = Path(__file__).parent.parent
PARQUET = ROOT / "data" / "processed" / "nlp_clustered_500k.parquet"
OUT_PKL = ROOT / "data" / "processed" / "dashboard_data_500k.pkl"
BRAND_FILES = [
    Path.home() / "Downloads" / "Brand List Available全域.xlsx",
    Path.home() / "Downloads" / "US Available Brand Name.xlsx",
    Path.home() / "Documents" / "品牌打标.xlsx",
]

# ── English dictionary ────────────────────────────────────────────────────────
with open("/usr/share/dict/words") as _f:
    _ENG_DICT = {w.strip().lower() for w in _f}

# Brands that ARE real but also common English words — explicit keep list
KNOWN_BRANDS = {
    "canon","apple","amazon","ring","nest","echo","dove","tide","arc","blink",
    "arlo","hue","vera","boss","hugo","polo","coach","gucci","prada","fossil",
    "casio","seiko","omega","rolex","bulova","tissot","citizen","seiko",
    "nike","adidas","puma","reebok","fila","vans","converse","saucony","brooks",
    "samsung","huawei","oppo","oneplus","motorola","ikea","muji","lego",
    "hasbro","mattel","fisher","kitchenaid","vitamix","ninja","instant",
    "cuisinart","breville","dewalt","milwaukee","makita","ryobi","bosch",
    "craftsman","stanley","spyderco","benchmade","gerber","kershaw","mora",
    "victorinox","nikon","sigma","fujifilm","olympus","pentax","leica",
    "hasselblad","godox","profoto","invisalign","colgate","crest","sensodyne",
    "listerine","lamy","pilot","parker","rhodia","leuchtturm","staedtler",
    "pentel","essie","nyx","mac","elf","covergirl","olay","cetaphil",
    "vanicream","sony","sharp","pioneer","onkyo","marantz","denon","bose",
    "klipsch","target","gap","next","shell","zojirushi","nespresso","keurig",
    "kindle","kobo","boox","shark","dyson","roomba","irobot","ecovacs",
    "eufy","anker","belkin","logitech","corsair","razer","steelseries",
    "hyperx","jabra","plantronics","sennheiser","shure","audio","rode",
    "zoom","focusrite","behringer","yamaha","roland","korg","fender",
    "gibson","epiphone","ibanez","schecter","peavey","marshall","orange",
    "vox","mesa","blackstar","revlon","loreal","maybelline","neutrogena",
    "cerave","aveeno","eucerin","clinique","lancome","dior","chanel",
    "hermes","burberry","versace","armani","calvin","tommy","ralph",
    "lacoste","supreme","palace","stussy","carhartt","patagonia","columbia",
    "northface","arcteryx","salomon","merrell","keen","teva","birkenstock",
    "clarks","timberland","ugg","dr","crocs","vans","new","asics","hoka",
    "on","altra","brooks","mizuno","saucony","balenciaga","yeezy",
    # phones / tech brands often in English dict
    "xiaomi","redmi","realme","oneplus","oppo","nokia","asus","acer",
    "lenovo","intel","nvidia","amd","qualcomm","mediatek",
    # gaming brands
    "sega","atari","bandai","namco","konami","capcom","ubisoft","activision",
    "blizzard","nintendo","valve","epic","bethesda","bungie","respawn",
    "rockstar","obsidian","cdpr","riot","mojang",
    # fragrance brands
    "lattafa","rasasi","creed","aventus","guerlain","valentino","givenchy",
    "thierry","mugler","ysl","dolce","gabbana","hermès","bvlgari","davidoff",
    # audio / AV
    "truthear","moondrop","fiio","shanling","topping","smsl","schiit",
    "hifiman","audeze","beyerdynamic","akg","grado","focal","meze",
    "wiim","naim","bluesound","lyngdorf","rega","cambridge","arcam",
    # stationery brands
    "twsbi","kaweco","jinhao","platinum","sailor","pelikan","montblanc",
    "waterman","cross","monteverde","noodlers","diamine","iroshizuku",
    "herbin","washi","midori","rhodia","leuchtturm","clairefontaine",
    "majohn","kakuno","uniball","staedtler","faber",
    # knife brands
    "civivi","spyderco","kershaw","benchmade","gerber","buck","victorinox",
    "leatherman","sog","cold","esee","ontario","kabar","tops","zero",
    "tolerance","microtech","protech","kizer","vosteed","sencut","ganzo",
    "lansky","sharpal","worksharp","wicked","edge",
    # beauty brands
    "nyx","essie","opi","orly","zoya","sally","revlon","covergirl","maybelline",
    "loreal","lancome","mac","elf","ulta","sephora","glossier","rare",
    "fenty","charlotte","tilbury","nars","too","faced","urban","decay",
    "benefit","becca","tarte","stila","caudalie","drunk","elephant",
    "tatcha","paula","choice","cosrx","anua","beauty","some","by","mi",
    "innisfree","etude","laneige","sulwhasoo","sk2","skii","missha",
    "klairs","dear","neogen","bioderma","la","roche","posay","avene",
    "vichy","uriage","svr","nuxe","embryolisse","filorga","medik8",
}

# Never a brand — product/category words, common phrases, stop words
HARD_NOISE = {
    # stop words & grammar
    "a","an","the","i","you","he","she","it","we","they","me","him","her",
    "us","them","my","your","his","its","our","their","who","what","which",
    "when","where","why","how","and","but","or","if","in","on","at","to",
    "of","with","by","from","up","out","as","into","about","after","before",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","can",
    # ultra-common verbs
    "get","got","go","make","take","know","think","want","need","find",
    "feel","look","come","use","buy","see","keep","run","give","help",
    "work","love","like","try","ask","tell","say","show","start","stop",
    # ultra-common adjectives/adverbs
    "good","bad","new","old","big","small","large","long","short","high",
    "low","best","better","worse","great","nice","cool","hot","cold","hard",
    "easy","first","last","same","right","left","only","just","more","most",
    "less","very","too","also","even","still","already","now","here","there",
    "then","than","so","really","quite","much","many","some","any","all",
    "both","each","every","few","other","another","lot","lots","bit",
    # product-category words (should never be brand names in display)
    "camera","lens","teeth","tooth","dentist","dental","crown","root","gum",
    "system","home","house","room","wall","desk","table","chair","light",
    "lighting","led","switch","plug","solar","power","energy","grid","game",
    "games","player","players","dog","cat","puppy","kitten","bird","budgie",
    "fish","pet","vet","ring","diamond","cut","wedding","custom","stone",
    "hair","skin","face","eye","lip","nail","nails","body","hand","feet",
    "sound","speaker","audio","video","screen","display","monitor","battery",
    "cake","bread","cookie","vanilla","chocolate","lemon","frosting","icing",
    "hat","cap","head","brim","ink","nib","fountain","pen","pens","refill",
    "coat","coats","polish","acrylic","gel","base","top","finish","color",
    "school","college","work","travel","indoor","outdoor","living","moving",
    # common Reddit/discourse words
    "thanks","thank","please","hey","hi","hello","anyone","however","also",
    "looking","there","now","since","then","here","today","recently","check",
    "got","finally","thoughts","im","let","size","gen","usd","desired",
    "weight","form","intended","country","condition","shot","type","original",
    "baby","fly","apartment","layout","bedroom","kitchen","birthday",
    "strawberry","lemon","chocolate","frosting","buttercream","baking",
    "cupcakes","party","happy","cheese","filling","fondant","loaf","bread",
    "salt","bulk","rise","rest","bake","mixed","cold","pan","sourdough",
    "dutch","stretch","cats","dogs","german","potty","training","kittens",
    "engagement","oval","cad","moissanite","lab","sapphire","ttrpg","dm",
    "gm","gameplay","gaming","combat","fun","rpg","lot","find","found",
    "braces","panel","panels","kwh","dc","ac","mppt","pv","inverter",
    "cozy","design","grow","replace","lamp","lights","recessed","fit","hats",
    "style","baseball","sports","running","parrot","journal","journals",
    "pages","bullet","started","finished","well","art","health","eating",
    "diet","gas","bacteria","gut","gi","sibo","bloating","probiotics",
    "options","maybe","ram","ssd","upgrade","used","nice","first time",
    "off grid","on grid","the cat","edc","deal","mini","pocket","carbon",
    "steel","blade","lock","bar","tv","pc","sub","setup","subwoofer","amp",
    "avr","dolby","speakers","soundbar","charge","mix","crate","weeks",
    "days","years","months","shooting","ef","af","rf","parfum","le","homme",
    "fragrances","intense","perfume","cologne","fresh","colours","purple",
    "orange","yellow","grey","pretty","feeling","dark","glasses","vision",
    "online","sunglasses","trip","park","canyon","lake","parks","camp",
    "hiking","weekend","miles","mountain","loop","national","trail","gran",
    "reading","book","books","device","basic","library","eat","stomach",
    "chronic","journaling","lacquer","strong","flour","dough","starter",
    "baked","machine","ap","arthur","npd","id","ebay","shipping","sale",
    "handle","folding","clip","fixed","gf","knife","knives","smart","world",
    "dog","cat","sometimes","introducing","resident","indoor","outdoor",
    "camera","lens","shooting","upgrade","pro","core","screen",
    "nail","pens","pen","ink","nib","fountain","refill","lacquer",
    # beauty / personal care category words
    "skincare","makeup","haircare","curls","toning","eyeliner","mascara",
    "blush","bronzer","concealer","foundation","serum","moisturizer",
    "cleanser","toner","retinol","tretinoin","niacinamide","sunscreen",
    "primer","highlighter","spf","spf30","spf50","lips","lashes","brow",
    "brows","liner","exfoliant","moisturize","exfoliate","exfoliating",
    "ingredient","ingredients","routine","routines","hydration","hydrating",
    "oily","dry","sensitive","combination","pores","pore","acne","breakout",
    "redness","rosacea","hyperpigmentation","pigmentation","scars","scar",
    "wrinkles","wrinkle","aging","anti","peptide","peptides","vitamin",
    "glycolic","salicylic","hyaluronic","ceramide","ceramides","snail",
    "fragrance","scent","scented","unscented","spf","spf50","spf30",
    # baking / food category words
    "cookies","cakes","macarons","brownies","croissant","vegan","tastes",
    "bakes","sourdough","muffins","donuts","pastries","pastry","brownie",
    "cookie","cupcake","macaron","waffle","waffles","pancake","pancakes",
    "pizza","sandwich","salad","soup","stew","sauce","pasta","noodles",
    "chocolate","vanilla","caramel","matcha","earl","lavender","honey",
    "almond","walnut","pecan","pistachio","hazelnut","cinnamon","ginger",
    "sour","sweet","savory","spicy","salty","tangy","bitter","umami",
    # solar / electrical terms
    "watts","arrays","wires","cycles","volts","amps","ampere","kwh","wh",
    "inverter","mppt","charge","controller","lithium","lifepo4","lead",
    "agm","gel","battery","batteries","panels","modules","cells","cell",
    "efficiency","output","input","load","grid","offgrid","backup","ups",
    # tech generic terms
    "oled","amoled","dslr","hdmi","cassette","hifi","icons","distro","ubuntu",
    "android","ios","windows","linux","macos","chrome","firefox","safari",
    "gddr","ddr","nvme","sata","ssd","hdd","usb","bluetooth","wifi","ethernet",
    "fps","ghz","mhz","mhz","watt","wh","ah","mah","resolution","refresh",
    "latency","bandwidth","throughput","protocol","driver","firmware","bios",
    # gaming generic terms
    "gameboy","handheld","indie","pixels","sprites","dlc","rpg","fps","pvp",
    "pve","meta","buff","nerf","loot","grind","grinding","farming","speedrun",
    "modding","mods","patch","update","patch","hotfix","release","launch",
    # fashion generic terms
    "laces","sneaker","streetwear","hypebeast","resell","deadstock","cop",
    "retail","grail","grails","collab","collaboration","limited","exclusive",
    "sold","out","waitlist","raffle","draw","lottery","restock","drop",
    "sizing","tts","fit","slim","regular","relaxed","wide","narrow",
    # misc noise words / Reddit slang / geography / generic terms
    "pakistan","inked","fixie","kadet","distro","bleu","hawas",
    "calories","enduro","versa","glitch","whatsapp","spotify","roblox",
    "tapes","hifi","icons","ultramax","films","cassette","camo",
    "decky","buds","imac","ryzen","btech",
    # console/product model codes (not brand names)
    "ps4","ps5","ps3","ps2","wii","switch","gameboy","3ds","2ds",
    "ipad","iphone","macbook","imac","airpod","airpods","watch",
}

# ── Normalization ─────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"['\-\.\&,!?/\\()\[\]_*@#%^~`|]+")
_SPACE_RE = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = _PUNCT_RE.sub(" ", s.lower())
    return _SPACE_RE.sub(" ", s).strip()

def normalize_display(b: str) -> str:
    """Clean up weird capitalizations for display: SAMSUNg→Samsung, LeNoVo→Lenovo."""
    b = _PUNCT_RE.sub("", b).strip()
    if not b:
        return ""
    # Short all-caps brand codes (OPI, NYX, ILNP, BKL ≤ 6 chars) → keep
    if re.match(r'^[A-Z0-9]{2,6}$', b):
        return b
    # Long weird-case → title case
    has_weird = any(c.isupper() for c in b[1:]) and any(c.islower() for c in b)
    if has_weird or b.isupper():
        # Check if it's a known brand with standard casing
        low = b.lower()
        # Try to find canonical form
        for known in ["Samsung","Lenovo","Fujifilm","CeraVe","L'Oréal"]:
            if known.lower() == low:
                return known
        return b.title()
    if b.islower():
        return b.title()
    return b

# ── Load whitelist ────────────────────────────────────────────────────────────
def load_whitelist() -> dict[str, str]:
    """Returns norm→display_name. Aggressively filters non-brands."""
    all_raw: list[str] = []
    for fp in BRAND_FILES:
        if not fp.exists():
            log.warning("Missing: %s", fp); continue
        xl = pd.ExcelFile(fp)
        for sheet in xl.sheet_names:
            raw_df = pd.read_excel(fp, sheet_name=sheet, header=None)
            flat = raw_df.values.flatten()
            all_raw.extend(str(v).strip() for v in flat
                           if pd.notna(v) and str(v).strip() not in ("","nan","brand_name"))
    log.info("Raw entries: %d", len(all_raw))

    whitelist: dict[str, str] = {}
    for b in all_raw:
        norm = _norm(b)
        if not norm or len(norm) < 3:
            continue
        words = norm.split()

        if len(words) == 1:
            # Hard noise → always skip
            if norm in HARD_NOISE:
                continue
            # Common English word → skip UNLESS explicitly in KNOWN_BRANDS
            if norm in _ENG_DICT and norm not in KNOWN_BRANDS:
                continue
            # Still a common word? Secondary check with HARD_NOISE broader set
            if norm.isalpha() and len(norm) < 4 and norm not in KNOWN_BRANDS:
                continue
        else:
            # Multi-word: require at least one non-dict non-noise word OR be in KNOWN_BRANDS
            has_uncommon = any(
                w not in _ENG_DICT and w not in HARD_NOISE
                for w in words
            )
            all_noise = all(w in HARD_NOISE for w in words)
            if all_noise:
                continue
            # All-common multi-word phrases (like "off grid", "the cat") → skip
            if not has_uncommon and all(w in _ENG_DICT for w in words):
                continue

        display = normalize_display(b)
        if not display:
            continue
        if norm not in whitelist:
            whitelist[norm] = display

    log.info("Whitelist after filter: %d brands", len(whitelist))
    return whitelist

# ── Brand extraction ──────────────────────────────────────────────────────────
def extract_brands(text: str, whitelist: dict[str, str], max_n: int = 4) -> list[str]:
    tokens = text.split()
    found: list[str] = []
    for n in range(1, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i+n])
            key  = _norm(gram)
            if key in whitelist:
                found.append(whitelist[key])
    return found

# ── Category-level aggregation ────────────────────────────────────────────────
def agg_period(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("category").agg(
        mentions        = ("mention_id",       "count"),
        communities     = ("community",        "nunique"),
        mean_sentiment  = ("sentiment_compound","mean"),
        mean_engagement = ("engagement_score",  "mean"),
    ).reset_index()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading parquet …")
    df = pd.read_parquet(PARQUET)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["category"]     = df["category"].fillna("unknown")
    df["title"]        = df["title"].fillna("")
    df["text"]         = df["text"].fillna("").str[:400]
    df["ner"]          = (df["title"] + " " + df["text"]).str.strip()
    log.info("Rows: %d", len(df))

    latest = df["published_at"].max()

    windows: dict[str, dict] = {}
    for i in range(4):
        end        = latest - pd.Timedelta(days=14*i)
        start      = end    - pd.Timedelta(days=14)
        prev_end   = start
        prev_start = start  - pd.Timedelta(days=14)
        label = f"{start.strftime('%m/%d')}–{end.strftime('%m/%d')}"
        windows[label] = dict(cur_start=start, cur_end=end,
                              prev_start=prev_start, prev_end=prev_end)

    whitelist = load_whitelist()

    all_window_data: dict[str, dict] = {}

    for win_label, win in windows.items():
        log.info("=== Window: %s ===", win_label)
        df_cur  = df[(df["published_at"] >= win["cur_start"])  & (df["published_at"] < win["cur_end"])]
        df_prev = df[(df["published_at"] >= win["prev_start"]) & (df["published_at"] < win["prev_end"])]
        log.info("  cur=%d  prev=%d", len(df_cur), len(df_prev))

        # ── Category stats ────────────────────────────────────────────────
        cur_s  = agg_period(df_cur).add_suffix("_c").rename(columns={"category_c":"category"})
        prev_s = agg_period(df_prev).add_suffix("_p").rename(columns={"category_p":"category"})
        all_cats = pd.DataFrame({"category": df["category"].unique()})
        stats = all_cats.merge(cur_s, on="category", how="left").merge(prev_s, on="category", how="left").fillna(0)

        stats["spike_ratio"]         = (stats["mentions_c"] / stats["mentions_p"].replace(0,1)).round(3)
        max_spike                    = stats["spike_ratio"].replace([np.inf,-np.inf],0).quantile(0.97)
        stats["normalized_spike"]    = (stats["spike_ratio"].clip(upper=max_spike) / max(max_spike,1)).round(4)
        max_comm                     = stats["communities_c"].max()
        stats["cross_community"]     = (stats["communities_c"] / max(max_comm,1)).round(4)
        stats["sentiment_score"]     = ((stats["mean_sentiment_c"] + 1) / 2).round(4)
        med_eng                      = stats["mean_engagement_c"].median()
        stats["eng_momentum"]        = (stats["mean_engagement_c"] / stats["mean_engagement_p"].replace(0, med_eng)).clip(upper=4).fillna(1).round(4)
        stats["eng_momentum_norm"]   = (stats["eng_momentum"] / max(stats["eng_momentum"].quantile(0.97),1)).round(4)

        # Equal weights 25% each
        stats["trend_score"] = (
            0.25 * stats["normalized_spike"] +
            0.25 * stats["cross_community"] +
            0.25 * stats["sentiment_score"] +
            0.25 * stats["eng_momentum_norm"]
        ).round(4)

        stats["trend_direction"] = stats["spike_ratio"].apply(
            lambda x: "rising" if x>=1.4 else ("declining" if x<=0.6 else "stable"))
        stats["mentions_delta"]  = (stats["mentions_c"] - stats["mentions_p"]).astype(int)
        stats = stats.rename(columns={
            "mentions_c":"current_mentions","mentions_p":"previous_mentions",
            "communities_c":"current_communities",
            "mean_sentiment_c":"mean_sentiment","mean_engagement_c":"mean_engagement",
        })
        stats = stats.sort_values("trend_score", ascending=False).reset_index(drop=True)

        # ── Per-brand extraction with sentiment ───────────────────────────
        log.info("  Extracting brands + sentiment …")
        # Accumulators: brand_cat → {cur_mentions, prev_mentions, sentiments}
        brand_cur_cnt:  dict[str, Counter] = defaultdict(Counter)
        brand_prev_cnt: dict[str, Counter] = defaultdict(Counter)
        brand_sentiment: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for row in df_cur.itertuples():
            brands = list(set(extract_brands(row.ner, whitelist)))
            cat    = row.category
            sent   = row.sentiment_compound
            for b in brands:
                brand_cur_cnt[cat][b]        += 1
                brand_sentiment[cat][b].append(sent)

        for row in df_prev.itertuples():
            brands = list(set(extract_brands(row.ner, whitelist)))
            for b in brands:
                brand_prev_cnt[row.category][b] += 1

        # Global brand frequency across categories (to remove generic terms)
        global_brand_cats: Counter = Counter()
        for cat, cnt in brand_cur_cnt.items():
            for b in cnt:
                global_brand_cats[b] += 1
        total_cats = len(brand_cur_cnt)

        MIN_MENTIONS   = 2       # at least 2 posts in current window
        MAX_CAT_PCT    = 0.40    # brand in >40% of categories = generic word, skip

        cat_brand_data: dict[str, pd.DataFrame] = {}
        for cat in df["category"].unique():
            cur_c  = brand_cur_cnt.get(cat,  Counter())
            prev_c = brand_prev_cnt.get(cat, Counter())
            rows   = []
            for b, cc in cur_c.items():
                if cc < MIN_MENTIONS:
                    continue
                if global_brand_cats[b] / max(total_cats, 1) > MAX_CAT_PCT:
                    continue  # appears in too many categories = generic
                pc   = prev_c.get(b, 0)
                bspk = round(cc / max(pc, 1), 2)
                sents= brand_sentiment[cat].get(b, [])
                avg_sent = round(sum(sents)/len(sents), 3) if sents else 0.0
                rows.append({"brand": b, "cur_mentions": cc, "prev_mentions": pc,
                             "brand_spike": bspk, "avg_sentiment": avg_sent})
            if rows:
                bdf = pd.DataFrame(rows).sort_values("cur_mentions", ascending=False).head(25)
                cat_brand_data[cat] = bdf

        stats["top_brands"] = stats["category"].map(
            lambda c: cat_brand_data.get(c, pd.DataFrame()).head(20)["brand"].tolist()
            if c in cat_brand_data else [])

        # ── Weekly breakdown ──────────────────────────────────────────────
        df_win = df[(df["published_at"] >= win["prev_start"]) & (df["published_at"] < win["cur_end"])].copy()
        with pd.option_context("mode.chained_assignment", None):
            df_win["week"] = df_win["published_at"].dt.to_period("W").dt.start_time
        weekly = (df_win.groupby(["category","week"])
                  .agg(mentions=("mention_id","count"), mean_sentiment=("sentiment_compound","mean"))
                  .reset_index())
        weekly["week"] = weekly["week"].dt.strftime("%Y-%m-%d")

        # ── Sample posts ──────────────────────────────────────────────────
        top_posts: dict[str, list[dict]] = {}
        for cat, grp in df_cur.groupby("category"):
            top_posts[cat] = (grp.nlargest(5,"engagement_score")
                              [["title","community","engagement_score","sentiment_label","url"]]
                              .to_dict("records"))
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
