import pytest

from datamate.qnorm import Vocab, default_vocab, normalize_question as nq, phonotactic_decode, place_tone, split_tone, telex_decode

TELEX = {"tieenf": "tiền", "dduwowcj": "được", "ddown": "đơn", "toongr": "tổng", "vaayj": "vậy", "thangs": "tháng", "bans": "bán", "gias": "giá",
         "khachs": "khách", "nhoms": "nhóm", "maays": "mấy", "hown": "hơn", "thuowngr": "thưởng", "nguwowif": "người", "khuyeens": "khuyến", "hoanf": "hoàn"}


@pytest.mark.parametrize("raw,want", list(TELEX.items()))
def test_telex_words_decoded(raw, want):
    sentence = nq(f"{raw} dthu")                       # shorthand present -> the sentence is evidently typed this way
    assert split_tone(sentence.split(" doanh")[0]) == split_tone(want)


def test_shorthand_and_month_forms():
    assert nq("dthu hnay vs hqua bnhiu z?") == "doanh thu hôm nay với hôm qua bao nhiêu vậy?"
    assert nq("Th2/2026 và T11/2025 và q3") == "tháng 2/2026 và tháng 11/2025 và quý 3"
    assert "T2" in nq("Thứ T2 bán được bnhiu")           # T2 alone = thứ Hai, never a month


def test_amounts_and_units_untouched():
    assert nq("50k 3tr 10ml 2l 5kg") == "50k 3tr 10ml 2l 5kg"
    assert nq("giá 50k, có k?") == "giá 50k, có không?"


def test_brands_english_and_plain_words_untouched():
    for s in ["Doanh thu của Bia Heineken max trong tháng 7", "coffee cheese boost good food latte", "Cà phê sữa (M) tháng 5/2026 là bao nhiêu"]:
        assert nq(s) == s


def test_shop_vocabulary_protects_its_own_words():
    v = Vocab.load()
    v.add_text("Sản phẩm: bans gias")                    # a shop whose catalogue really contains these tokens
    assert nq("bans dthu", v) == "bans doanh thu" or nq("bans dthu", v).startswith("bans")


def test_place_tone_rules():
    assert place_tone("toan", "s") == "toán" and place_tone("mua", "f") == "mùa" and place_tone("hoa", "f") in ("hòa", "hoà")
    assert place_tone("duoc", "j").startswith("d") is True and place_tone("đươc", "j") == "được"
    assert phonotactic_decode("xyzzy") is None


def test_amounts_note():
    from datamate.qnorm import amounts_note

    assert amounts_note("đơn trên 500k và dưới 1tr5, hơn 2 triệu, 1.250.000") == "500k = 500000; 1tr5 = 1500000; 2 triệu = 2000000"
    assert amounts_note("bán 5kg và 10ml, 3 ly") == ""
    assert amounts_note("giá 12,5k") == "12,5k = 12500"


def test_agent_prompt_gets_reading_and_amounts(demo_split, sandbox):
    from datamate.agent import BusinessContext, DataMateAgent
    from datamate.config import load_config
    from datamate.llm import MockBackend
    from datamate.retrieval import ExemplarBank

    cfg = load_config()
    info = sandbox.load(demo_split / "data" / "biz_demo1", "biz_demo1", demo_split / "rulebook.md", demo_split / "calendar.csv")
    seen = {}

    def resp(msgs):
        seen["u"] = msgs[1]["content"]
        return "```python\nanswer_type='number'\nanswer=1\n```"

    ag = DataMateAgent(MockBackend(resp), sandbox, cfg, ExemplarBank([]))
    ag.answer(BusinessContext("b", info), "", "toongr ddown tren 500k bnhiu z?", "2026-03-04")
    assert "tổng đơn tren 500k bao nhiêu vậy" in seen["u"] and "500k = 500000" in seen["u"]
