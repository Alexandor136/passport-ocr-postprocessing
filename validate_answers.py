import json
import sys
import collections

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

test_path = sys.argv[1]
ans_path = sys.argv[2]
test = [json.loads(l) for l in open(test_path, encoding="utf-8") if l.strip()]
ans = [json.loads(l) for l in open(ans_path, encoding="utf-8") if l.strip()]

problems = []
if len(ans) != len(test):
    problems.append(f"line count {len(ans)} != {len(test)}")
seen = set()
acc = rej = 0
for a in ans:
    did = a.get("doc_id")
    if did in seen:
        problems.append(f"duplicate {did}")
    seen.add(did)
    if a.get("decision") == "accept":
        acc += 1
        f = a.get("fields") or {}
        missing = [k for k in FIELDS if k not in f]
        if missing:
            problems.append(f"{did} missing fields {missing}")
        if set(f) - set(FIELDS):
            problems.append(f"{did} extra fields")
    elif a.get("decision") == "reject":
        rej += 1
    else:
        problems.append(f"{did} bad decision {a.get('decision')!r}")
test_ids = {d["doc_id"] for d in test}
if seen != test_ids:
    problems.append(
        f"doc_id mismatch: {len(test_ids - seen)} missing, "
        f"{len(seen - test_ids)} extra"
    )

print(f"documents {len(ans)}  accept {acc}  reject {rej}")
print("problems:", problems[:20] if problems else "none")
