#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the static artifacts used by postprocess.py.

The artifacts are derived only from the data shipped with the task:

  * ``lexicon.txt.gz``  - every token seen in the reference subdivision names,
    in the public GeoNames dump for Russia (optional, ``--geo``) and in the
    training labels for the two free-text fields;
  * ``support.txt.gz``  - token frequencies from the reference names and the
    training labels (used to spot OCR corruptions);
  * ``names.txt.gz``    - ``field<TAB>value`` lines with every labelled name.

Usage:
    python build_artifacts.py --train-labels data/train_labels.jsonl \
        --reference reference/subdivisions.csv --out artifacts [--geo RU.txt]
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import re

TOKEN_RE = re.compile(r"[А-ЯЁA-Z0-9-]+")
NAME_FIELDS = ("Last_name_ru", "First_name_ru", "Middle_name_ru")
FREE_TEXT_FIELDS = ("Birth_place_ru", "Issue_organization_ru")


def norm(value):
    return " ".join(str(value or "").split()).upper()


def read_geo_tokens(path):
    """Tokenise the GeoNames RU dump (tab-separated: id, name, ascii, alt)."""
    cyrillic = re.compile(r"[А-ЯЁа-яё-]+")
    tokens = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            for name in [parts[1], parts[2]] + parts[3].split(","):
                if name and cyrillic.fullmatch(name):
                    tokens.update(TOKEN_RE.findall(name.upper()))
    return tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--geo", default=None)
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    tokens = set()
    support = collections.Counter()

    with open(args.reference, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            found = TOKEN_RE.findall(norm(row["name"]))
            tokens.update(found)
            support.update(found)

    if args.geo:
        geo = read_geo_tokens(args.geo)
        tokens.update(geo)

    names = []
    with open(args.train_labels, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = json.loads(line)["fields"]
            for field in FREE_TEXT_FIELDS:
                found = TOKEN_RE.findall(norm(fields.get(field)))
                tokens.update(found)
                support.update(found)
            for field in NAME_FIELDS:
                value = norm(fields.get(field))
                if value:
                    names.append((field, value))

    with gzip.open(
        os.path.join(args.out, "lexicon.txt.gz"), "wt", encoding="utf-8"
    ) as fh:
        for token in sorted(tokens):
            fh.write(token + "\n")
    with gzip.open(
        os.path.join(args.out, "support.txt.gz"), "wt", encoding="utf-8"
    ) as fh:
        for token, count in sorted(support.items()):
            fh.write(f"{token}\t{count}\n")
    with gzip.open(
        os.path.join(args.out, "names.txt.gz"), "wt", encoding="utf-8"
    ) as fh:
        for field, value in names:
            fh.write(f"{field}\t{value}\n")

    print(f"lexicon: {len(tokens)} tokens")
    print(f"support: {len(support)} tokens")
    print(f"names:   {len(names)} values")


if __name__ == "__main__":
    main()
