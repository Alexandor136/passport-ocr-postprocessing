# Russian passport OCR post-processing

Accept / reject policy for the Russian passport OCR task. CPU only, offline,
standard library only.

## Usage

```
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

Output: one JSON object per input line,

```json
{"doc_id": "...", "decision": "accept", "fields": {"Last_name_ru": "...", "...": "..."}}
```

`fields` is filled only for accepted documents (all 10 fields).

Optional flags:

```
--artifacts DIR    default: ./artifacts
--reference PATH   default: ./reference/subdivisions.csv
```

## Layout

```
postprocess.py        policy + CLI
build_artifacts.py    rebuilds artifacts/ from train labels + reference
validate_answers.py   checks the output format against the predictions file
artifacts/
  lexicon.txt.gz      tokens of reference names, GeoNames RU and train labels
  support.txt.gz      token frequencies (reference + train labels)
  names.txt.gz        field<TAB>value, every labelled name
reference/
  subdivisions.csv    code -> name reference (DaData, CC BY-SA 4.0)
answers.jsonl         predictions for data/test_predictions.jsonl
REPORT.md             results and rationale
```

## Policy in one paragraph

The machine-readable zone is treated as an independent source of truth: it is
parsed, used to restore missing fields, and the accepted document must agree
with it on names, document number, birth date, sex, issue date and department
code. Conflicts are rejected unless one side can be proven independently
(check digit, subdivision reference, character-level name model). Additional
gates: per-character OCR confidence, field geometry (detected line breaks),
token plausibility for the two free-text fields, field formats, and passport
validity on 2026-10-01. Documents without a readable MRZ are always rejected.

## Rebuilding artifacts

```
python build_artifacts.py \
    --train-labels data/train_labels.jsonl \
    --reference reference/subdivisions.csv \
    --out artifacts \
    --geo RU.txt        # optional GeoNames RU dump
```

Without `--geo` the lexicon contains only reference + train tokens; recall drops
by ~4 pp at the same error level.

## Tests

```
python -m unittest discover -s tests
```

The tests cover the MRZ parser, date/validity rules, transliteration round-trip
and the answer format.
