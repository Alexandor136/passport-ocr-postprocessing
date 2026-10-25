# -*- coding: utf-8 -*-
import collections
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import postprocess as pp  # noqa: E402


def check_digit(value):
    return pp.icao_check(value)


def line(text, conf=1.0):
    return {
        "text": text,
        "box": [0, 0, 200, 30],
        "char_conf": [conf] * len(text),
    }


def field(text, conf=1.0):
    return {"box": [0, 0, 200, 30], "lines": [line(text, conf)]}


def make_doc(with_mrz=True):
    licence = "77 15 123456"
    digits = pp.digits(licence)
    doc9 = digits[:3] + digits[4:]
    dob6 = "900510"
    issue6 = "160315"
    code6 = "770001"
    name = "IVANOV<<IVAN<IVANOVI3"
    name = name + "<" * (39 - len(name))
    l1 = "PNRUS" + name
    l2 = (doc9 + str(check_digit(doc9)) + "RUS" + dob6 + str(check_digit(dob6))
          + "M" + "<" * 7 + "0" + issue6 + code6 + "<<<")
    viz = {
        "Last_name_ru": field("ИВАНОВ"),
        "First_name_ru": field("ИВАН"),
        "Middle_name_ru": field("ИВАНОВИЧ"),
        "Sex_ru": field("МУЖ."),
        "Birth_date": field("10.05.1990"),
        "Birth_place_ru": field("Г. МОСКВА"),
        "Issue_date": field("15.03.2016"),
        "Issue_organisation_code": field("770-001"),
        "Licence_number": field(licence),
        "Issue_organization_ru": field("ОВД РАЙОНА"),
    }
    doc = {"doc_id": "test", "viz": viz}
    if with_mrz:
        doc["mrz"] = {"lines": [{"text": l1}, {"text": l2}]}
    return doc


def resources():
    return pp.Resources(set(), [], {"770-001"}, collections.Counter())


class TestMrz(unittest.TestCase):
    def test_parse(self):
        mrz = pp.parse_mrz(make_doc())
        self.assertIsNotNone(mrz)
        self.assertEqual(mrz.doc9, "771123456")
        self.assertTrue(mrz.doc_chk_ok)
        self.assertTrue(mrz.dob_chk_ok)
        self.assertEqual(mrz.sex, "M")
        self.assertEqual(mrz.issue6, "160315")
        self.assertEqual(mrz.code6, "770001")

    def test_reject_bad_prefix(self):
        doc = make_doc()
        doc["mrz"]["lines"][0]["text"] = (
            "XXXXX" + doc["mrz"]["lines"][0]["text"][5:]
        )
        self.assertIsNone(pp.parse_mrz(doc))

    def test_accept_valid(self):
        ok, fields, reasons = pp.assess(make_doc(), resources())
        self.assertTrue(ok, reasons)
        self.assertEqual(fields["Last_name_ru"], "ИВАНОВ")

    def test_reject_without_mrz(self):
        ok, _, reasons = pp.assess(make_doc(with_mrz=False), resources())
        self.assertFalse(ok)
        self.assertIn("no_mrz", reasons)

    def test_reject_name_conflict(self):
        doc = make_doc()
        doc["viz"]["Last_name_ru"] = field("ПЕТРОВ")
        ok, _, reasons = pp.assess(doc, resources())
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("mrz_names") for r in reasons))

    def test_reject_low_confidence(self):
        doc = make_doc()
        doc["viz"]["Birth_place_ru"] = field("Г. МОСКВА", conf=0.5)
        ok, _, reasons = pp.assess(doc, resources())
        self.assertFalse(ok)
        self.assertIn("low_conf:Birth_place_ru", reasons)

    def test_reject_truncated_block(self):
        doc = make_doc()
        doc["viz"]["Birth_place_ru"] = {
            "box": [0, 0, 200, 60],
            "lines": [line("Г. МОСКВА")],
        }
        ok, _, reasons = pp.assess(doc, resources())
        self.assertFalse(ok)
        self.assertIn("truncated:Birth_place_ru", reasons)


class TestDates(unittest.TestCase):
    def test_translit_roundtrip(self):
        for text in ("ИВАНОВ", "ЩУКИН", "ЁЖИК"):
            letters = pp.translit_letters(text)
            self.assertEqual(pp.reverse_translit(letters), text)
        # separators are not recoverable: the MRZ encodes them as '<'
        self.assertEqual(
            pp.reverse_translit(pp.translit_letters("ПЕТРОВ-ВОДКИН")),
            "ПЕТРОВВОДКИН",
        )

    def test_validity(self):
        import datetime as dt
        self.assertTrue(
            pp.passport_valid(dt.date(1990, 5, 10), dt.date(2016, 3, 15))
        )
        # issued after 45 -> indefinite
        self.assertTrue(
            pp.passport_valid(dt.date(1970, 5, 10), dt.date(2016, 3, 15))
        )
        # issued at 24, the 45-year passport expires in 2025 -> invalid on
        # the reference date
        self.assertFalse(
            pp.passport_valid(dt.date(1980, 5, 10), dt.date(2005, 3, 15))
        )
        # issued at 19, turns 20 before the reference date -> expired
        self.assertFalse(
            pp.passport_valid(dt.date(2000, 5, 10), dt.date(2019, 3, 15))
        )
        # issue date in the future
        self.assertFalse(
            pp.passport_valid(dt.date(1990, 5, 10), dt.date(2027, 3, 15))
        )

    def test_series(self):
        import datetime as dt
        fields = {"Licence_number": "77 15 123456"}
        self.assertTrue(pp.series_ok(fields, dt.date(2016, 3, 15)))
        self.assertFalse(pp.series_ok(fields, dt.date(2019, 3, 15)))


if __name__ == "__main__":
    unittest.main()
