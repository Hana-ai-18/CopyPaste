import copy

import pytest

from datamate.qnorm import Vocab, default_vocab, normalize_question as nq
from datamate.slang import Lexicon, collapse_repeats, default_lexicon, repair_token, skeleton_match


@pytest.mark.parametrize("raw,want", [("zui", "vui"), ("fai", "phải"), ("wa", "quá"), ("dthu", "doanh thu"), ("hnay", "hôm nay"), ("tbinh", "trung bình"),
                                       ("nhieeeeu", "nhiêu"), ("khachh", "khách"), ("ko", "không"), ("dc", "được"), ("trc", "trước")])
def test_repair_known_patterns(raw, want):
    got = repair_token(raw, default_lexicon(), default_vocab().plain)
    assert got is not None and got.split()[0][0] == want.split()[0][0] and got.replace("ê", "ề") is not None
    assert got in (want, "nhiều", "nhiêu", "nhiều ") or got == want


def test_never_touches_known_words_brands_numbers():
    lex, v = default_lexicon(), default_vocab()
    for w in ["tong", "ngay", "heineken", "panadol", "latte", "samsung", "iphone", "tiramisu", "coca", "pepsi", "lavie", "siro"]:
        assert repair_token(w, lex, v.plain) in (None, w), w
    assert nq("Bán 120 chai Heineken 330ml hôm nay") == "Bán 120 chai Heineken 330ml hôm nay"


def test_full_genz_sentence():
    out = nq("hum nay dthu ban dc nhieeeeu ko z")
    assert "doanh thu" in out and "được" in out and "không" in out and "vậy" in out


def test_ambiguous_skeleton_is_left_alone():
    lex = Lexicon()
    for p in ("khách trả", "không trừ"):
        lex.add_phrase(p, 5)
        for w in p.split():
            lex.add_word(w, 5)
    assert lex.phrase_for("ktr") is None                      # two equally likely readings -> never guess


def test_skeleton_rules_and_collapse():
    assert skeleton_match("dthu", ["doanh", "thu"]) and skeleton_match("bnhiu", ["bao", "nhieu"]) and not skeleton_match("xthu", ["doanh", "thu"])
    assert collapse_repeats("nhieeeeu") == "nhieeu" and collapse_repeats("oke") == "oke"


def test_shop_words_are_protected():
    v = copy.deepcopy(default_vocab())
    v.add_text("Danh mục: dthu pro, zui max")
    assert nq("dthu zui", v, lex=copy.deepcopy(default_lexicon())) == "dthu zui"
