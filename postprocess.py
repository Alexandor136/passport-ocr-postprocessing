#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative post-processing / acceptance policy for Russian passport OCR.

Usage:
    python postprocess.py data/test_predictions.jsonl answers.jsonl

The script never accesses the network.  It uses only the bundled reference
data and the lexicon/name artifacts built from the public training labels.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import gzip
import json
import math
import os
import re
import statistics
import sys

FIELDS = [
    "Last_name_ru",
    "First_name_ru",
    "Middle_name_ru",
    "Sex_ru",
    "Birth_date",
    "Birth_place_ru",
    "Issue_date",
    "Issue_organisation_code",
    "Licence_number",
    "Issue_organization_ru",
]
NAME_FIELDS = ("Last_name_ru", "First_name_ru", "Middle_name_ru")

REF_DATE = _dt.date(2026, 10, 1)

TRANSLIT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X",
    "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8",
}
REVERSE_TRANSLIT = {v: k for k, v in TRANSLIT.items()}

TOKEN_RE = re.compile(r"[А-ЯЁA-Z0-9-]+")
NAME_RE = re.compile(r"[А-ЯЁA-Z][А-ЯЁA-Z .-]*")
DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")
CODE_RE = re.compile(r"\d{3}-\d{3}")
LICENCE_RE = re.compile(r"\d{2} \d{2} \d{6}")

# Abbreviations written with a trailing dot in the "birth place" field.
BP_DOT_ENDINGS = {"ОБЛ", "КР", "РЕСП", "Г", "С", "Д", "П", "ПОС", "ДЕР", "СТ"}


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def norm(s):
    return " ".join(str(s or "").split()).upper()


def join_lines(value):
    if not value or not value.get("lines"):
        return ""
    return " ".join(line.get("text", "") for line in value["lines"])


def extract(doc):
    viz = doc.get("viz") or {}
    return {field: norm(join_lines(viz.get(field))) for field in FIELDS}


def min_conf(value):
    if not value or not value.get("lines"):
        return None
    values = [
        c
        for line in value["lines"]
        for c in (line.get("char_conf") or [])
    ]
    return min(values) if values else None


def geometry(value):
    if not value or not value.get("lines"):
        return None
    box_h = value["box"][3]
    heights = [line["box"][3] for line in value["lines"]]
    median_h = statistics.median(heights) if heights else 1
    expected = max(1, round(box_h / median_h)) if median_h else 1
    return {"box_h": box_h, "n": len(value["lines"]), "expected": expected,
            "missing_lines": max(0, expected - len(value["lines"]))}


def parse_date(s):
    m = DATE_RE.fullmatch(s or "")
    if not m:
        return None
    day, month, year = map(int, m.groups())
    try:
        return _dt.date(year, month, day)
    except ValueError:
        return None


def ymd(d):
    return f"{d.year % 100:02d}{d.month:02d}{d.day:02d}"


def digits(s):
    return "".join(ch for ch in (s or "") if ch.isdigit())


def age_on(birth, date):
    return date.year - birth.year - ((date.month,
    date.day) < (birth.month,
    birth.day))


def add_years(date, years):
    try:
        return date.replace(year=date.year + years)
    except ValueError:
        return date.replace(year=date.year + years, day=28)


def icao_check(s):
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            value = int(ch)
        elif "A" <= ch <= "Z":
            value = ord(ch) - 55
        else:
            value = 0
        total += value * weights[i % 3]
    return total % 10


def translit(s):
    out = []
    for ch in norm(s):
        if ch in TRANSLIT:
            out.append(TRANSLIT[ch])
        elif ch in " -<":
            out.append("<")
        else:
            out.append("?")
    return "".join(out)


def translit_letters(s):
    return translit(s).replace("<", "")


def reverse_translit(letters):
    return "".join(REVERSE_TRANSLIT.get(ch, "?") for ch in letters)


# --------------------------------------------------------------------------- #
# MRZ
# --------------------------------------------------------------------------- #
class MRZ:
    __slots__ = ("raw_name", "name_letters", "doc9", "doc_chk_ok", "dob6",
                 "dob_chk_ok", "sex", "issue6", "code6", "line2_len")

    def __init__(self):
        self.raw_name = ""
        self.name_letters = ""
        self.doc9 = ""
        self.doc_chk_ok = False
        self.dob6 = ""
        self.dob_chk_ok = False
        self.sex = ""
        self.issue6 = ""
        self.code6 = ""
        self.line2_len = 0


def parse_mrz(doc):
    mrz = doc.get("mrz")
    if not mrz or not mrz.get("lines"):
        return None
    lines = mrz["lines"]
    line1 = (lines[0].get("text") or "").strip()
    if not line1.startswith("PNRUS") or len(line1) < 6:
        return None
    out = MRZ()
    out.raw_name = line1[5:]
    out.name_letters = "".join(ch for ch in out.raw_name if ch != "<")
    if not out.name_letters or any(
        ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for ch in out.name_letters
    ):
        return None
    if len(lines) < 2:
        return None
    line2 = (lines[1].get("text") or "").strip()
    if len(line2) < 41 or line2[10:13] != "RUS" or line2[20] not in ("M", "F"):
        return None
    out.line2_len = len(line2)
    out.doc9 = line2[0:9]
    out.dob6 = line2[13:19]
    out.sex = line2[20]
    out.issue6 = line2[29:35]
    out.code6 = line2[35:41]
    if not (
        out.doc9.isdigit()
        and out.dob6.isdigit()
        and out.issue6.isdigit()
        and out.code6.isdigit()
    ):
        return None
    out.doc_chk_ok = line2[9].isdigit() and icao_check(
        out.doc9
    ) == int(line2[9])
    out.dob_chk_ok = line2[19].isdigit() and icao_check(
        out.dob6
    ) == int(line2[19])
    return out


def mrz_birth_date(dob6, issue_date=None):
    """Resolve the century of a 6-digit MRZ date."""
    yy, mm, dd = int(dob6[0:2]), int(dob6[2:4]), int(dob6[4:6])
    candidates = []
    for century in (1900, 2000):
        try:
            candidates.append(_dt.date(century + yy, mm, dd))
        except ValueError:
            pass
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if issue_date:
        valid = [d for d in candidates if d <= issue_date and 14 <= age_on(d,
        issue_date) <= 110]
        if len(valid) == 1:
            return valid[0]
        if valid:
            return valid[0]
    return candidates[0] if yy >= 30 else candidates[1]


def mrz_issue_date(issue6):
    yy, mm, dd = int(issue6[0:2]), int(issue6[2:4]), int(issue6[4:6])
    year = 1900 + yy if yy >= 90 else 2000 + yy
    try:
        return _dt.date(year, mm, dd)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# resources / character LM for names
# --------------------------------------------------------------------------- #
class CharLM:
    def __init__(self, n=4, alpha=0.05):
        self.n = n
        self.alpha = alpha
        self.counts = [collections.Counter() for _ in range(n + 1)]
        self.totals = [collections.Counter() for _ in range(n + 1)]
        self.vocab = set()

    def train(self, strings):
        for s in strings:
            s = "^" + norm(s) + "$"
            self.vocab.update(s)
            for i, ch in enumerate(s):
                for k in range(1, min(self.n, i + 1) + 1):
                    context = s[i - k + 1:i]
                    self.counts[k][(context, ch)] += 1
                    self.totals[k][context] += 1

    def score(self, s):
        s = "^" + norm(s) + "$"
        vocab_size = len(self.vocab) + 1
        total = 0.0
        for i, ch in enumerate(s):
            prob = None
            for k in range(min(self.n, i + 1), 0, -1):
                context = s[i - k + 1:i]
                denominator = self.totals[k].get(context, 0)
                if denominator:
                    prob = (self.counts[k].get((context,
                    ch),
                    0) + self.alpha) / (
                        denominator + self.alpha * vocab_size)
                    break
            if prob is None:
                prob = 1.0 / vocab_size
            total += math.log(prob)
        return total / max(1, len(s))


class Resources:
    def __init__(self,
    tokens,
    names,
    ref_codes,
    support=None,
    name_lm_delta=1.0):
        self.tokens = set(tokens)
        self.support = collections.Counter(support or {})
        self.ref_codes = set(ref_codes)
        self.name_lm_delta = name_lm_delta
        self.name_counts = collections.Counter()
        self.name_lms = {}
        for field in NAME_FIELDS:
            self.name_lms[field] = CharLM(4, 0.05)
        for field, value in names:
            if not value:
                continue
            for token in TOKEN_RE.findall(value):
                self.name_counts[token] += 1
            self.name_lms[field].train([value])

    def name_bonus(self, value):
        bonus = 0.0
        for token in TOKEN_RE.findall(value):
            if self.name_counts.get(token):
                bonus += 0.35
        return bonus


def load_resources(artifacts_dir, reference_path):
    tokens = set()
    names = []
    support = collections.Counter()
    lex_path = os.path.join(artifacts_dir, "lexicon.txt.gz")
    with gzip.open(lex_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if token:
                tokens.add(token)
    support_path = os.path.join(artifacts_dir, "support.txt.gz")
    with gzip.open(support_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                support[parts[0]] = int(parts[1])
    names_path = os.path.join(artifacts_dir, "names.txt.gz")
    with gzip.open(names_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1]:
                names.append((parts[0], parts[1]))
    ref_codes = set()
    with open(reference_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ref_codes.add(row["code"])
    return Resources(tokens, names, ref_codes, support)


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class Config:
    conf_min = 0.90
    require_mrz = True
    fill_missing = True
    repair_names = True
    repair_dob = True
    repair_code = True
    repair_issue = True
    name_lm_delta = 1.0
    confirm_empty_middle = True
    token_check = True
    token_oov_support_max = 4
    punct_check = True
    geometry_check = True
    max_name_block = 41  # a full MRZ name block may be truncated


def passport_valid(birth, issue):
    if not birth or not issue:
        return False
    if issue > REF_DATE or issue < birth:
        return False
    age = age_on(birth, issue)
    if age < 14 or age > 110:
        return False
    if age < 20:
        return add_years(birth, 20) > REF_DATE
    if age < 45:
        return add_years(birth, 45) > REF_DATE
    return True


def series_ok(fields, issue_date=None):
    issue_date = issue_date or parse_date(fields.get("Issue_date"))
    licence = digits(fields.get("Licence_number"))
    if not issue_date or len(licence) != 10:
        return False
    series_year = int(licence[2:4])
    return (issue_date.year % 100 - series_year) % 100 in (0, 1, 2, 3)


def format_ok(fields, middle_may_be_empty=False):
    for field in ("Last_name_ru", "First_name_ru"):
        if not NAME_RE.fullmatch(fields.get(field) or ""):
            return False
    middle = fields.get("Middle_name_ru") or ""
    if middle:
        if not NAME_RE.fullmatch(middle):
            return False
    elif not middle_may_be_empty:
        return False
    if fields.get("Sex_ru") not in ("МУЖ.", "ЖЕН."):
        return False
    if not parse_date(fields.get("Birth_date")):
        return False
    if not parse_date(fields.get("Issue_date")):
        return False
    if not CODE_RE.fullmatch(fields.get("Issue_organisation_code") or ""):
        return False
    if not LICENCE_RE.fullmatch(fields.get("Licence_number") or ""):
        return False
    if not fields.get("Birth_place_ru") or not fields.get(
        "Issue_organization_ru"
    ):
        return False
    return True


ALPHABET = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-"


def variants1(word):
    """All single-edit (delete / substitute / insert / swap) neighbours."""
    out = set()
    for i in range(len(word)):
        out.add(word[:i] + word[i + 1:])
        for ch in ALPHABET:
            if ch != word[i]:
                out.add(word[:i] + ch + word[i + 1:])
    for i in range(len(word) + 1):
        for ch in ALPHABET:
            out.add(word[:i] + ch + word[i:])
    for i in range(len(word) - 1):
        out.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])
    return out


def max_support(word, support):
    best = 0
    for variant in variants1(word):
        count = support.get(variant, 0)
        if count > best:
            best = count
    return best


def token_ok(token, res, cfg):
    if token in res.tokens:
        return True
    if token.isdigit() or len(token) == 1:
        return True
    if "-" in token:
        return all(token_ok(part,
        res,
        cfg) for part in token.split("-") if part)
    if not token.isalpha():
        # a mix of letters and digits inside a word is a typical OCR artefact
        return False
    return max_support(token, res.support) <= cfg.token_oov_support_max


def tokens_ok(value, res, cfg):
    return all(token_ok(token, res, cfg) for token in TOKEN_RE.findall(value))


def punctuation_ok(value):
    parts = value.split()
    if not parts:
        return False
    last = parts[-1]
    bare = last.rstrip(".")
    if bare in BP_DOT_ENDINGS:
        return last.endswith(".")
    return True


def mrz_no_middle(mrz, fields, cfg):
    """True when the MRZ proves that the passport has no patronymic."""
    raw = mrz.raw_name.rstrip("<")
    # Drop an isolated trailing single character (a known OCR artefact).
    m = re.search(r"(?:<{2,})([A-Z0-9])$", raw)
    if m:
        raw = raw[:m.start()]
    if len(raw) >= cfg.max_name_block:
        return False
    letters = raw.replace("<", "")
    expected = (
        translit_letters(fields["Last_name_ru"])
        + translit_letters(fields["First_name_ru"])
    )
    return letters == expected


def fill_from_mrz(fields, mrz, res, cfg, reasons):
    filled = set()
    missing_names = [field for field in NAME_FIELDS if not fields[field]]
    if len(missing_names) == 1:
        field = missing_names[0]
        if (
            field == "Middle_name_ru"
            and cfg.confirm_empty_middle
            and mrz_no_middle(mrz, fields, cfg)
        ):
            fields[field] = ""
            filled.add(field)
        else:
            index = NAME_FIELDS.index(field)
            prefix = "".join(
                translit_letters(fields[f]) for f in NAME_FIELDS[:index]
            )
            suffix = "".join(
                translit_letters(fields[f]) for f in NAME_FIELDS[index + 1:]
            )
            body = mrz.name_letters
            if body.startswith(prefix) and body.endswith(suffix) and len(
                body
            ) > len(prefix) + len(suffix):
                middle = body[len(prefix):len(body) - len(suffix)]
                candidate = reverse_translit(middle)
                if len(candidate) >= 2 and "?" not in candidate:
                    fields[field] = candidate
                    filled.add(field)

    if not fields["Sex_ru"] and mrz.sex in ("M", "F"):
        fields["Sex_ru"] = "МУЖ." if mrz.sex == "M" else "ЖЕН."
        filled.add("Sex_ru")

    if not fields["Birth_date"] and mrz.dob_chk_ok:
        birth = mrz_birth_date(mrz.dob6)
        if birth:
            fields["Birth_date"] = (
                f"{birth.day:02d}.{birth.month:02d}.{birth.year:04d}"
            )
            filled.add("Birth_date")

    if not fields["Issue_date"]:
        issue = mrz_issue_date(mrz.issue6)
        if issue:
            fields["Issue_date"] = (
                f"{issue.day:02d}.{issue.month:02d}.{issue.year:04d}"
            )
            filled.add("Issue_date")

    if not fields["Issue_organisation_code"]:
        code = mrz.code6[:3] + "-" + mrz.code6[3:]
        if code in res.ref_codes:
            fields["Issue_organisation_code"] = code
            filled.add("Issue_organisation_code")

    return filled


def repair_dob(fields, mrz, cfg, reasons):
    if not cfg.repair_dob or not mrz.dob_chk_ok:
        return False
    current = parse_date(fields["Birth_date"])
    if not current:
        return False
    expected = ymd(current)
    if mrz.dob6 == expected:
        return False
    issue = parse_date(fields["Issue_date"])
    birth = mrz_birth_date(mrz.dob6, issue)
    if not birth:
        return False
    fields["Birth_date"] = (
        f"{birth.day:02d}.{birth.month:02d}.{birth.year:04d}"
    )
    return True


def repair_code(fields, mrz, res, cfg, reasons):
    """Returns True (MRZ used), "keep_viz" (MRZ ignored) or None (conflict)."""
    if not cfg.repair_code:
        return False
    current = (fields["Issue_organisation_code"] or "").replace("-", "")
    if current == mrz.code6:
        return False
    mrz_code = mrz.code6[:3] + "-" + mrz.code6[3:]
    current_code = fields["Issue_organisation_code"]
    current_ok = current_code in res.ref_codes
    mrz_ok = mrz_code in res.ref_codes
    if mrz_ok and not current_ok:
        fields["Issue_organisation_code"] = mrz_code
        return True
    if current_ok and not mrz_ok:
        return "keep_viz"
    return None  # both plausible or both implausible -> unresolved


def repair_issue(fields, mrz, res, cfg, reasons):
    """Returns True (MRZ used), "keep_viz" (MRZ ignored) or None (conflict)."""
    if not cfg.repair_issue:
        return False
    current = parse_date(fields["Issue_date"])
    mrz_date = mrz_issue_date(mrz.issue6)
    if not current or not mrz_date or ymd(current) == mrz.issue6:
        return False
    birth = parse_date(fields["Birth_date"])
    if not birth:
        return False

    def valid(date):
        return (1997 <= date.year <= 2026 and series_ok(fields, date)
                and passport_valid(birth, date))

    current_valid = valid(current)
    mrz_valid = valid(mrz_date)
    if mrz_valid and not current_valid:
        fields["Issue_date"] = (
            f"{mrz_date.day:02d}.{mrz_date.month:02d}.{mrz_date.year:04d}"
        )
        return True
    if current_valid and not mrz_valid:
        return "keep_viz"
    return None


def repair_names(fields, mrz, res, cfg, reasons):
    """Reconstruct one corrupted name field from the MRZ
    pick the better variant.

    Returns True when the conflict was resolved (either value kept), False when
    there was nothing to do, and None when the conflict remains unresolved.
    """
    if not cfg.repair_names:
        return False
    body = mrz.name_letters
    parts = [translit_letters(fields[f]) for f in NAME_FIELDS]
    if "".join(parts) == body:
        return False
    # Only repair when exactly one part does not fit the MRZ, and the other two
    # can be used as exact anchors.
    candidates = []
    for index in range(len(NAME_FIELDS)):
        prefix = "".join(parts[:index])
        suffix = "".join(parts[index + 1:])
        if body.startswith(prefix) and body.endswith(suffix) and len(
            body
        ) > len(prefix) + len(suffix):
            middle = body[len(prefix):len(body) - len(suffix)]
            if len(middle) >= 2:
                candidate = reverse_translit(middle)
                if "?" not in candidate:
                    candidates.append((index, candidate))
    if len(candidates) != 1:
        return None
    index, candidate = candidates[0]
    field = NAME_FIELDS[index]
    lm = res.name_lms[field]
    viz_score = lm.score(fields[field]) + res.name_bonus(fields[field])
    rec_score = lm.score(candidate) + res.name_bonus(candidate)
    if rec_score - viz_score > res.name_lm_delta:
        fields[field] = candidate
        return True
    if viz_score - rec_score > res.name_lm_delta:
        return True  # keep the visual value,
        # but accept the conflict as resolved
    return None


def check_mrz(fields, mrz, res, cfg, skip):
    reasons = []
    if mrz is None:
        if cfg.require_mrz:
            reasons.append("no_mrz")
        return reasons
    if "names" not in skip:
        current = "".join(translit_letters(fields[f]) for f in NAME_FIELDS)
        if current != mrz.name_letters:
            reasons.append("mrz_names")
    if "doc" not in skip:
        licence = digits(fields["Licence_number"])
        if len(licence) != 10 or licence[:3] + licence[4:] != mrz.doc9:
            reasons.append("mrz_doc")
    if "dob" not in skip:
        birth = parse_date(fields["Birth_date"])
        if not birth or ymd(birth) != mrz.dob6:
            reasons.append("mrz_dob")
    if "sex" not in skip:
        sex = "M" if fields["Sex_ru"].startswith("МУЖ") else "F"
        if sex != mrz.sex:
            reasons.append("mrz_sex")
    if "issue" not in skip:
        issue = parse_date(fields["Issue_date"])
        if not issue or ymd(issue) != mrz.issue6:
            reasons.append("mrz_issue")
    if "code" not in skip:
        code = (fields["Issue_organisation_code"] or "").replace("-", "")
        if code != mrz.code6:
            reasons.append("mrz_code")
    return reasons


def assess(doc, res, cfg=None):
    """Return (accepted, fields, reasons)."""
    cfg = cfg or Config()
    fields = extract(doc)
    mrz = parse_mrz(doc)
    filled = set()
    skip = set()
    reasons = []

    if cfg.fill_missing and mrz:
        filled |= fill_from_mrz(fields, mrz, res, cfg, reasons)

    if mrz:
        if repair_dob(fields, mrz, cfg, reasons):
            filled.add("Birth_date")
        code_result = repair_code(fields, mrz, res, cfg, reasons)
        if code_result is True:
            filled.add("Issue_organisation_code")
        elif code_result == "keep_viz":
            skip.add("code")
        elif code_result is None:
            reasons.append("mrz_code_conflict")
        issue_result = repair_issue(fields, mrz, res, cfg, reasons)
        if issue_result is True:
            filled.add("Issue_date")
        elif issue_result == "keep_viz":
            skip.add("issue")
        elif issue_result is None:
            reasons.append("mrz_issue_conflict")
        name_result = repair_names(fields, mrz, res, cfg, reasons)
        if name_result is True:
            skip.add("names")
        elif name_result is None:
            skip.add("names_conflict")

    middle_empty = bool(
        cfg.confirm_empty_middle
        and mrz
        and not fields["Middle_name_ru"]
        and mrz_no_middle(mrz, fields, cfg)
    )

    if any(
        not fields[f]
        for f in FIELDS
        if not (f == "Middle_name_ru" and middle_empty)
    ):
        reasons.append("missing")
    if not format_ok(fields, middle_may_be_empty=middle_empty):
        reasons.append("format")

    birth = parse_date(fields["Birth_date"])
    issue = parse_date(fields["Issue_date"])
    if not passport_valid(birth, issue):
        reasons.append("invalid_passport")
    if not series_ok(fields, issue):
        reasons.append("licence_series")

    if cfg.geometry_check:
        for field in ("Birth_place_ru", "Issue_organization_ru"):
            g = geometry((doc.get("viz") or {}).get(field))
            if g and g["missing_lines"] > 0:
                reasons.append("truncated:" + field)

    if cfg.conf_min:
        for field in FIELDS:
            if field in filled or field == "Middle_name_ru" and middle_empty:
                continue
            value = (doc.get("viz") or {}).get(field)
            confidence = min_conf(value)
            if confidence is None or confidence < cfg.conf_min:
                reasons.append("low_conf:" + field)

    if cfg.token_check:
        if not tokens_ok(fields["Birth_place_ru"], res, cfg):
            reasons.append("tokens:birth_place")
        if not tokens_ok(fields["Issue_organization_ru"], res, cfg):
            reasons.append("tokens:organization")

    if cfg.punct_check and not punctuation_ok(fields["Birth_place_ru"]):
        reasons.append("punctuation:birth_place")

    if "names_conflict" in skip:
        reasons.append("mrz_names_conflict")
    else:
        reasons.extend(check_mrz(fields, mrz, res, cfg, skip))

    accepted = not reasons
    return accepted, fields, sorted(set(reasons))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("output")
    parser.add_argument(
        "--artifacts",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "artifacts"
        ),
    )
    parser.add_argument(
        "--reference",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "reference",
            "subdivisions.csv",
        ),
    )
    args = parser.parse_args(argv)

    res = load_resources(args.artifacts, args.reference)
    cfg = Config()
    accepted = 0
    total = 0
    with open(args.predictions, encoding="utf-8") as src, open(
        args.output, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            doc = json.loads(line)
            total += 1
            ok, fields, reasons = assess(doc, res, cfg)
            record = {"doc_id": doc["doc_id"],
                      "decision": "accept" if ok else "reject",
                      "fields": {f: fields[f] for f in FIELDS} if ok else {}}
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            accepted += int(ok)
    print(
        f"processed {total} documents, accepted {accepted} "
        f"({accepted / max(1, total):.1%})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
