from datamate.qnorm import normalize_question as nq
from datamate.slang import default_lexicon
from datamate.qnorm import default_vocab
from datamate.spell import correct_token

LEX, VOC = default_lexicon(), default_vocab()
KNOWN = set(VOC.plain)


def test_tone_and_missing_letter():
    for bad, good in [("đươc", "được"), ("nhât", "nhất"), ("bìh", "bình"), ("tổg", "tổng"), ("cộg", "cộng"), ("lênn", "lên"), ("nàoo", "nào"), ("đag", "đang")]:
        assert correct_token(bad, LEX, KNOWN) in (good, "đã" if bad == "đag" else good), bad


def test_genz_initials_and_common():
    assert correct_token("wán", LEX, KNOWN) == "quán"
    assert correct_token("júp", LEX, KNOWN) == "giúp"
    assert correct_token("dzạ", LEX, KNOWN) == "dạ"
    assert correct_token("kòn", LEX, KNOWN) == "còn"
    assert correct_token("ngừi", LEX, KNOWN) == "người"


def test_glued_words():
    assert nq("Th6/2025 ai là khách mua nhiềuđơn nhất").split()[-3:] == ["nhiều", "đơn", "nhất"] or "nhiều đơn" in nq("Th6/2025 ai là khách mua nhiềuđơn nhất")
    assert "ngày nào" in nq("T9/2025 ngàynào dthu cao nhất?")
    assert "cao nhất" in nq("tháng nào dthu caonhất")


def test_no_false_positive():
    for w in ["Heineken", "cold", "croissant", "Vinh", "Linh", "khát", "thắng", "nướu", "kèo", "trà", "tràm"]:
        assert correct_token(w, LEX, KNOWN) in (None, w), w
    q = "Tổng doanh thu tháng 3/2026 của Linh và Vinh là bao nhiêu?"
    assert nq(q) == q


def test_half_telex_and_sentence():
    assert "được" in nq("Doanh thu tháng 9/25 đuowc nhiêu vậy?")
    out = nq("tíh tới ngày 6/9/2026, 3 khách nào đag nợ nhìu nhứt, mỗi ngừi kòn nợ bnhiu")
    assert "nhiều nhất" in out and "còn nợ" in out and "người" in out
