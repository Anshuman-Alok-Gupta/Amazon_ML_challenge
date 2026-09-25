"""Text normalisation and light parsing of business names / addresses.

Designed for ~12M records per split: one lean pass per record (precompiled regexes, dict
lookups), parallelised over processes, results cached as parquet.

Everything here is hand-written, country-agnostic vocabulary (no external lookups). French
terms are included because the test set contains France, which never appears in training.
Native-script names (Devanagari, Tamil, ...) are romanised with unidecode; doubled letters are
then collapsed everywhere ("limittedd" -> "limited", "street" -> "stret") so transliterations and
typos land on the same spelling as the reference records.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from unidecode import unidecode

# ---------------------------------------------------------------- vocabularies
# Keys are in *collapsed* form (doubled letters removed), values are canonical legal tokens.
LEGAL_CANON = {
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "limited": "ltd", "ltd": "ltd", "limitet": "ltd", "ltda": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "praivet": "pvt", "piraivet": "pvt", "prayvet": "pvt",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "lc": "llc", "lp": "llp", "elelpi": "llp", "plc": "plc", "pc": "pc",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "sca": "sca", "scop": "scop", "selarl": "sarl", "opc": "opc",
}
LEGAL_TOKENS = frozenset(LEGAL_CANON.values())
LEGAL_BITS = {t: 1 << i for i, t in enumerate(sorted(LEGAL_TOKENS))}
NAME_STOP = frozenset({"the", "and", "of", "le", "la", "les", "de", "des", "du", "et", "l", "d", "a", "an",
                       "dba", "aka", "ta", "et"})

ADDR_CANON = {
    # English
    "rd": "road", "st": "stret", "str": "stret", "street": "stret", "saint": "stret",
    "ave": "avenue", "av": "avenue", "avn": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "boul": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "sq": "square", "ter": "terace", "terrace": "terace", "hwy": "highway",
    "pkwy": "parkway", "fwy": "freway", "cir": "circle", "trl": "trail", "ste": "suite",
    "apt": "apartment", "apartments": "apartment", "apts": "apartment", "apartement": "apartment",
    "fl": "flor", "flr": "flor", "floor": "flor", "bldg": "building", "blk": "block",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount", "ft": "fort", "ctr": "center", "centre": "center", "unit": "unit",
    # Indian
    "nr": "near", "opp": "oposite", "oposite": "oposite", "sec": "sector", "ph": "phase",
    "chk": "chowk", "mkt": "market", "extn": "extension", "ext": "extension", "indl": "industrial",
    "estt": "estate", "cplx": "complex", "stn": "station", "rly": "railway", "hsg": "housing",
    "soc": "society", "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk",
    "gali": "gali", "mg": "mahatma gandhi",
    # common city renames (both spellings occur)
    "bangalore": "bengaluru", "bengaluru": "bengaluru", "madras": "chenai", "chenai": "chenai",
    "bombay": "mumbai", "calcuta": "kolkata", "kolkata": "kolkata", "baroda": "vadodara",
    "vadodra": "vadodara", "gurgaon": "gurugram", "poona": "pune", "trivandrum": "thiruvananthapuram",
    "mysore": "mysuru", "mangalore": "mangaluru", "benares": "varanasi", "pondicherry": "puducherry",
    # French
    "r": "rue", "ch": "chemin", "chem": "chemin", "rte": "route", "imm": "immeuble",
    "fbg": "faubourg", "fg": "faubourg", "qu": "quai", "al": "ale", "all": "ale", "allee": "ale",
    "imp": "impase", "res": "residence", "zi": "zone industriele", "za": "zone artisanale",
    "cedex": "", "bp": "", "cs": "", "bis": "",
    # junk
    "null": "", "none": "", "na": "",
}
STATE_CODES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "conecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawai": "hi", "idaho": "id", "ilinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "masachusets": "ma",
    "michigan": "mi", "minesota": "mn", "misisipi": "ms", "misouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pensylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tenese": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    # India (incl. romanised native-script spellings seen in the data)
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "asam": "as", "bihar": "br",
    "chhatisgarh": "cg", "chatisgarh": "cg", "goa": "goa", "gujarat": "gj", "gujrat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "karnattk": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp", "mdhy prdesh": "mp",
    "maharashtra": "mh", "maharashttr": "mh", "mhaaraashttr": "mh", "maharastra": "mh",
    "manipur": "mn_in", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orisa": "od", "punjab": "pb", "rajasthan": "rj", "sikim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "tmilnadu": "tn", "tamilnatu": "tn", "telangana": "ts", "tripura": "tr",
    "utar pradesh": "up", "utarakhand": "uk", "utaranchal": "uk", "west bengal": "wb",
    "pashchimbng": "wb", "pshchimbng": "wb", "delhi": "dl", "new delhi": "dl", "dili": "dl",
    "jamu and kashmir": "jk", "chandigarh": "ch_in", "puducherry": "py",
    # romanised native-script state names, as they appear in the training data
    "mharastr": "mh", "tmilnatu": "tn", "utr prdesh": "up", "krnatk": "ka", "pshcimbng": "wb", "telngan": "ts",
    "hriyana": "hr", "rajsthan": "rj", "kerln": "kl", "andhrprdesh": "ap", "pnjab": "pb",
    "od isha": "od",
    # France (metropolitan regions + overseas), collapsed spellings
    "ile de france": "fidf", "auvergne rhone alpes": "fara", "nouvele aquitaine": "fnaq",
    "ocitanie": "focc", "hauts de france": "fhdf", "provence alpes cote d azur": "fpac",
    "provence alpes cote dazur": "fpac", "paca": "fpac", "grand est": "fges", "bretagne": "fbre",
    "normandie": "fnor", "pays de la loire": "fpdl", "centre val de loire": "fcvl",
    "bourgogne franche comte": "fbfc", "corse": "fcor", "guadeloupe": "fgua", "martinique": "fmtq",
    "guyane": "fguy", "la reunion": "flre", "reunion": "flre", "mayote": "fmay",
}
STATE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(STATE_CODES, key=len, reverse=True)) + r")\b"
)
# A comma segment that is exactly one of these is a state/region ("..., FL" / "..., Tamil Nadu").
REGION_ALIAS = {**{v: v for v in STATE_CODES.values()}, **STATE_CODES, "tg": "ts"}
# Telangana was split from Andhra Pradesh in 2014 and the sources mix the two: one block.
REGION_MERGE = {"ts": "ap"}

# ---------------------------------------------------------------- regexes
DOUBLE = re.compile(r"([a-z])\1+")
DOMAIN = re.compile(r"(?:https?\W+)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|co|in|fr|biz|info|us|io)\b(?:\.[a-z]{2})?")
DOTTED = re.compile(r"(?<![a-z0-9])([a-z])\.(?=[a-z]\b)")    # s.a.r.l. -> sarl, p.l.c -> plc
NON_ALNUM = re.compile(r"[^a-z0-9]+")
ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th|er|eme|e)\b")
# PO boxes, <NULL> placeholders and number prefixes ("No.", "S No." survey number, "Door No", "#").
JUNK = re.compile(r"<null>|\bp\.?\s?o\.?\s+box\s*\d*|\bpost box\s*\d*|"
                  r"\b(?:s|sy|survey|door|plot|flat|shop|house|h)?\s?\.?\s?no\b\.?|#")
LANDMARK = re.compile(
    r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|close to|"
    r"pres de|en face de|a cote de|derriere)\b[^,]*"
)
NUM = re.compile(r"\d+")
POSTAL = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")


def _latin(s: str) -> tuple[str, bool]:
    """Romanise; also report whether the original used a non-Latin script."""
    if s.isascii():
        return s.lower(), False
    nonlatin = any(ord(c) > 0x24F for c in s)
    return unidecode(s).lower(), nonlatin


def name_fields(raw: str) -> tuple:
    s, nonlatin = _latin(raw or "")
    s = DOMAIN.sub(r" \1 ", s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "").replace("`", "")
    s = DOTTED.sub(r"\1", s)
    s = DOUBLE.sub(r"\1", NON_ALNUM.sub(" ", s))
    toks = [LEGAL_CANON.get(t, t) for t in s.split()]
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in NAME_STOP]
    if not core:
        core = [t for t in toks if t not in NAME_STOP] or toks
    legal = 0
    for t in toks:
        legal |= LEGAL_BITS.get(t, 0)
    return (" ".join(toks), " ".join(core), "".join(core), legal, nonlatin)


def _num(t: str) -> str:
    return t.lstrip("0") or "0"


def address_fields(raw: str) -> tuple:
    """(clean, alpha, numbers, postal, landmark, nonlatin, region).

    Segment-aware: a comma segment that is just a state / region ("..., FL", "TX, AUSTIN, ...",
    "Pune, Maharashtra 411001") gives the record's region and is kept as a code, so state codes
    are never mistaken for abbreviations (FL is not "floor", CT is not "court").
    """
    s, nonlatin = _latin(raw or "")
    s = JUNK.sub(" ", s)
    s = DOTTED.sub(r"\1", ORDINAL.sub(r"\1", s))
    toks, landmarks = [], []
    strong = medium = ""
    for seg in re.split(r"[,;\n]", s):
        m = LANDMARK.search(seg)
        if m:
            landmarks.append(m.group(0))
            seg = seg[: m.start()]
        words = DOUBLE.sub(r"\1", NON_ALNUM.sub(" ", seg)).split()
        if not words:
            continue
        alpha = " ".join(w for w in words if not w.isdigit())
        if alpha in REGION_ALIAS:
            strong = REGION_ALIAS[alpha]
            toks += [_num(w) if w.isdigit() else "" for w in words] + [strong]
            continue
        alpha_words = [w for w in words if not w.isdigit()]
        last = alpha_words[-1] if len(alpha_words) >= 2 else ""
        if last in REGION_ALIAS and len(last) <= 3:   # "denver co 80202", "pune mh"
            medium = REGION_ALIAS[last]
        for w in words:
            if w.isdigit():
                toks.append(_num(w))
            elif w == last and medium and REGION_ALIAS.get(w) == medium:
                toks.append(medium)
            else:
                toks.append(ADDR_CANON.get(w, w))
    out = " ".join(t for t in toks if t)
    weak = ""
    for m in STATE_RE.finditer(out):
        weak = STATE_CODES[m.group(1)]
    out = STATE_RE.sub(lambda m: STATE_CODES[m.group(1)], out)
    region = strong or medium or weak
    region = REGION_MERGE.get(region, region)
    landmark = DOUBLE.sub(r"\1", NON_ALNUM.sub(" ", " ".join(landmarks))).strip()
    nums = NUM.findall(out)
    pm = POSTAL.findall(out)
    alpha = " ".join(t for t in out.split() if not t.isdigit())
    return (out, alpha, " ".join(dict.fromkeys(nums)), pm[-1] if pm else "", landmark, nonlatin, region)


NAME_COLS = ["name_clean", "name_core", "name_compact", "name_legal", "name_nonlatin"]
ADDR_COLS = ["addr_clean", "addr_alpha", "addr_numbers", "postal", "addr_landmark", "addr_nonlatin", "region"]


def _chunk(args):
    names, addrs = args
    return [name_fields(x) for x in names], [address_fields(x) for x in addrs]


def default_workers() -> int:
    """Normalisation processes: ER_WORKERS if set, else 3.

    Each worker costs ~350 MB (pandas import): 3 suits a 16 GB laptop; on a large cloud
    machine set ER_WORKERS to the vCPU count.
    """
    env = os.environ.get("ER_WORKERS")
    if env:
        return max(1, int(env))
    return min(3, max(1, (os.cpu_count() or 2) - 1))


def prepare(df: pd.DataFrame, workers: int | None = None, executor=None) -> pd.DataFrame:
    """Return df (index reset) with all normalised columns added."""
    df = df.reset_index(drop=True)
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    workers = workers or default_workers()
    step = 100_000
    jobs = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(df), step)]
    del names, addrs

    def to_frame(part):
        # Convert each chunk to Arrow-backed columns as it arrives: holding every chunk's
        # Python tuples (~35M strings for a 5M-row source) would cost several GB.
        nf = pd.DataFrame(part[0], columns=NAME_COLS)
        af = pd.DataFrame(part[1], columns=ADDR_COLS)
        f = pd.concat([nf, af], axis=1)
        for c in f.columns:
            if f[c].dtype == object:
                f[c] = f[c].astype("string[pyarrow]")
        return f

    if executor is not None:
        frames = [to_frame(part) for part in executor.map(_chunk, jobs)]
    elif workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            frames = [to_frame(part) for part in ex.map(_chunk, jobs)]
    else:
        frames = [to_frame(_chunk(j)) for j in jobs]
    del jobs
    out = pd.concat([df, pd.concat(frames, ignore_index=True)], axis=1)
    out["name_legal"] = out["name_legal"].astype("int64")
    out["country_norm"] = out["country"].str.strip().str.lower()
    return out


SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def normalize_tsv(tsv: Path, out: Path, chunk_rows: int = 500_000, workers: int | None = None) -> int:
    """Stream a source TSV through prepare() into a parquet file, one chunk at a time.

    Memory stays at one chunk (a 5M-row source read whole, plus its Python strings, needs
    several GB). Row order is preserved. Returns the number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    writer, schema, n = None, None, 0
    reader = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_rows)
    try:
        with ProcessPoolExecutor(workers or default_workers()) as ex:
            for chunk in reader:
                chunk = chunk[SOURCE_COLS].fillna("")
                for c in SOURCE_COLS:
                    chunk[c] = chunk[c].astype(str).str.strip()
                table = pa.Table.from_pandas(prepare(chunk, executor=ex), preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp, schema)
                writer.write_table(table.cast(schema))
                n += len(chunk)
                del chunk, table
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return n


def prepare_cached(df: pd.DataFrame, cache: Path | None) -> pd.DataFrame:
    """prepare() with a parquet cache keyed by the caller-chosen path."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    out = prepare(df)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out
