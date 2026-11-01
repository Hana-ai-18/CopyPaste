"""Things learned from the real data: month-split sheets are merged, rulebook sections are selected, shards are balanced."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from datamate.data_loader import load_business
from datamate.rulebook import select_rulebook, split_rulebook
from datamate.run import shard_questions

ROOT = Path(__file__).resolve().parents[1]
RB = """# Rulebook
## 1. Don hang
rule one
## 6. Theo ngành
intro
### Cà phê
cafe rule
### Nhà thuốc
pharmacy rule
## 7. Khong tra loi
tail
"""


def test_month_sheets_are_merged(tmp_path):
    cols = ["Mã đơn", "Ngày", "Tiền"]
    with pd.ExcelWriter(tmp_path / "DonHang_ChiTiet.xlsx") as w:
        for m in ("T01_2025", "T02_2025", "T03_2025"):
            pd.DataFrame([["A1", "01/02", 5], ["A2", "02/02", 6]], columns=cols).to_excel(w, sheet_name=m, index=False)
    pd.DataFrame({"Tên": ["x"], "Đơn vị": ["thùng"], "Hệ số": [24]}).to_excel(tmp_path / "QuyDoi.xlsx", sheet_name="Sheet1", index=False)
    (tmp_path / "README.md").write_text("# shop\n")
    b = load_business(tmp_path)
    assert set(b.tables) == {"donhang_chitiet", "quydoi"}          # file names, like the README
    df = b.tables["donhang_chitiet"]
    assert len(df) == 6 and set(df["_sheet"]) == {"t01_2025", "t02_2025", "t03_2025"}
    assert "donhang_chitiet" in b.readme and "_sheet" in b.readme    # the LLM is told about the merge


def test_split_files_merged_under_common_name(tmp_path):
    for n in ("SoBanHang_1.csv", "SoBanHang_2026.csv"):
        pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(tmp_path / n, index=False)
    assert list(load_business(tmp_path).tables) == ["sobanhang"]


def test_rulebook_selection():
    b, ind, a = split_rulebook(RB)
    assert set(ind) == {"Cà phê", "Nhà thuốc"}
    assert select_rulebook(RB, "x" * 10, 10_000) == RB                       # fits -> untouched
    long_rb = RB + "filler\n" * 50
    out = select_rulebook(long_rb, "Nhà thuốc bán thuốc kê đơn, hoạt chất", len(long_rb) - 10)
    assert "pharmacy rule" in out and "cafe rule" not in out
    unsure = select_rulebook(long_rb, "không rõ ngành", len(long_rb) - 5)
    assert "pharmacy rule" in unsure and "cafe rule" in unsure                  # never guesses an industry


def test_shards_balanced_and_complete():
    qs = [{"id": f"q{i}", "business_id": f"b{i % 5}"} for i in range(100)] + [{"id": f"z{i}", "business_id": "big"} for i in range(60)]
    s = [shard_questions(qs, i, 2) for i in range(2)]
    assert sorted(q["id"] for part in s for q in part) == sorted(q["id"] for q in qs)
    assert abs(len(s[0]) - len(s[1])) <= 20
    assert all(len({q["business_id"] for q in part} & {q["business_id"] for q in s[1 - i]}) == 0 for i, part in enumerate(s))


def test_merge_shards_script(tmp_path):
    (tmp_path / "q.jsonl").write_text("\n".join(json.dumps({"id": i}) for i in "abc"))
    (tmp_path / "s0.jsonl").write_text(json.dumps({"id": "a", "type": "number", "answer": 1}) + "\n")
    (tmp_path / "s1.jsonl").write_text(json.dumps({"id": "c", "type": "boolean", "answer": True}) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/merge_shards.py"), "--questions", str(tmp_path / "q.jsonl"), "--out", str(tmp_path / "o.jsonl"),
                    str(tmp_path / "s0.jsonl"), str(tmp_path / "s1.jsonl")], check=True)
    out = [json.loads(l) for l in (tmp_path / "o.jsonl").read_text().splitlines()]
    assert [r["id"] for r in out] == ["a", "b", "c"] and out[1] == {"id": "b", "type": "abstain"}


def test_to_num_and_profile_range():
    import math

    from datamate.helpers import to_num
    from datamate.profile import profile_tables

    got = to_num(pd.Series(["38,000", "20000", "8.500", "1.250.000đ", "12,5", "-3,200", "abc", None, "1 bao 25kg"])).tolist()
    assert got[:6] == [38000, 20000, 8500, 1250000, 12.5, -3200] and all(math.isnan(x) for x in got[6:])
    days = [f"{d:02d}/01/2025" for d in range(1, 29)] * 1000 + ["15/06/2026"]          # last value beyond the first 20000 rows
    assert "2026-06-15" in profile_tables({"t": pd.DataFrame({"ngay": days, "x": range(len(days))})}, 5000)


def test_summary_rows_and_messy_dates(tmp_path):
    import numpy as np

    from datamate.helpers import parse_dates

    df = pd.DataFrame({"Ngày": ["01/07/2025", "02/07/2025", "Tổng cộng", "Ghi chú: số liệu ghi tay", "Ngày"],
                       "Khách": ["A", "B", None, None, "Khách"], "Tiền": ["1.000.000đ", "500000", "1.500.000đ", None, "Tiền"]})
    df.to_csv(tmp_path / "SoBanHang.csv", index=False)
    b = load_business(tmp_path)
    t = b.tables["sobanhang"]
    assert list(t["Ngày"]) == ["01/07/2025", "02/07/2025"] and "non-record rows" in b.readme
    got = parse_dates(pd.Series(["1 thg 1 2025", "45658", "03/02/2026", "2026-03-02", "15 tháng 3 năm 2025", None]))
    assert [str(x)[:10] for x in got[:5]] == ["2025-01-01", "2025-01-01", "2026-02-03", "2026-03-02", "2025-03-15"] and pd.isna(got[5])
    assert np.isnan(np.nan)


def test_readme_units_become_a_table(tmp_path):
    pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(tmp_path / "BanHang.csv", index=False)
    (tmp_path / "README.md").write_text("# x\n\n## Quy ước\n- **Quy đổi đơn vị**:\n  - Coca Cola 330ml: 1 lốc = 6 lon\n- Gạo ST25: 1 bao 25kg = 25 kg\n- Sữa: 1 hộp = 0,5 lít\n")
    b = load_business(tmp_path)
    u = b.tables["quy_doi_readme"]
    assert u.to_dict("records")[:2] == [{"ten_hang": "Coca Cola 330ml", "don_vi_lon": "lốc", "he_so": 6, "don_vi_co_ban": "lon"},
                                         {"ten_hang": "Gạo ST25", "don_vi_lon": "bao 25kg", "he_so": 25, "don_vi_co_ban": "kg"}]
    assert u.iloc[2]["he_so"] == 0.5 and "1 lốc = 6 lon" not in b.readme


def test_period_helpers_and_folds():
    from datamate.helpers import data_span, rows_in_period

    d = pd.Series(["01/03/2025", "15/03/2025", "02/04/2025"])
    assert data_span(d)[0].isoformat() == "2025-03-01" and rows_in_period(d, "2025-03-01", "2025-03-31") == 2
    assert rows_in_period(d, "2025-01-01", "2025-02-28") == 0


def test_business_folds_partition_questions(demo_split, tmp_path):
    from datamate.run import main

    ids = {}
    for name, flag in (("a", ["--biz_fold", "0/2"]), ("b", ["--exclude_fold", "0/2"]), ("all", [])):
        out = tmp_path / f"{name}.jsonl"
        main(["--data_dir", str(demo_split), "--out", str(out), "--logs", str(tmp_path / f"l{name}"), "--config", "configs/default.json",
              "--set", 'backend="mock"', "--set", f'mock_responses="{demo_split / "mock_responses.json"}"', *flag])
        ids[name] = {json.loads(l)["id"] for l in out.read_text().splitlines()}
    assert ids["a"] and ids["b"] and not (ids["a"] & ids["b"]) and ids["a"] | ids["b"] == ids["all"]


def test_norm_status_all_real_spellings():
    from datamate.helpers import norm_status

    want = {
        "hoan_thanh": ["Đã hoàn thành", "Hoàn thành", "Thành công", "đã xong", "xong", "Xong", "ok", "Đã giao", "đã lấy", "completed", "Đã nhập hàng"],
        "huy": ["Đã hủy", "hủy", "hủy đơn", "Đã hủy đơn", "khách hủy", "Huỷ đơn", "Đã huỷ", "cancel", "cancelled", "Hủy", "Huỷ"],
        "tra_hang": ["Đã trả hàng", "trả hàng", "Trả hàng", "Đã hoàn hàng", "hàng trả về", "Đổi trả", "Hoàn trả", "trả lại", "returned",
                     "trả lại hàng", "khách mang trả", "khách trả lại", "Khách trả"],
        "dang_giao": ["Đang giao dịch", "Đang giao hàng", "Đang vận chuyển", "Đang xử lý", "Đang giao", "đang gửi", "shipping", "Chờ giao", "đang ship", "chưa giao"],
        "khac": ["Không giao được", "nhập đầu kỳ", "pin yếu", None],
    }
    for label, vals in want.items():
        got = norm_status(pd.Series(vals)).tolist()
        assert got == [label] * len(vals), (label, list(zip(vals, got)))


def test_loader_survives_format_perturbations(tmp_path):
    """Unseen shops may export differently: other separators / encodings / a title above the header / blank lines."""
    import unicodedata

    base = pd.DataFrame({"Ngày": ["01/02/2026", "02/02/2026"], "Tên hàng": ["Cà phê sữa", "Trà đào"], "Số lượng": [2, 3], "Thành tiền": ["50,000", "75,000"]})
    cases = {
        "semicolon.csv": ("utf-8-sig", base.to_csv(sep=";", index=False)),
        "tab.csv": ("utf-8", base.to_csv(sep="\t", index=False)),
        "utf16.csv": ("utf-16", base.to_csv(index=False)),
        "blank_lines.csv": ("utf-8", "\n" + base.to_csv(index=False).replace("\n", "\n\n")),
    }
    for name, (enc, text) in cases.items():
        d = tmp_path / name.split(".")[0]
        d.mkdir()
        (d / name).write_bytes(text.encode(enc))
    d = tmp_path / "title_row"
    d.mkdir()
    with pd.ExcelWriter(d / "title.xlsx") as w:
        pd.DataFrame([["BÁO CÁO BÁN HÀNG", None, None, None], [None] * 4, list(base.columns)] + base.values.tolist()).to_excel(w, header=False, index=False)
    for sub in sorted(p for p in tmp_path.iterdir() if p.is_dir()):
        (t, df), = load_business(sub).tables.items()
        cols = [unicodedata.normalize("NFC", str(c)) for c in df.columns]
        assert cols[:2] == ["Ngày", "Tên hàng"] and len(df) == 2, (sub.name, cols, len(df))


def test_budget_exhaustion_is_resumable(demo_split, tmp_path):
    from datamate.run import main

    common = ["--data_dir", str(demo_split), "--out", str(tmp_path / "s.jsonl"), "--logs", str(tmp_path / "l"), "--config", "configs/default.json",
              "--set", 'backend="mock"', "--set", f'mock_responses="{demo_split / "mock_responses.json"}"']
    main([*common, "--set", "time_budget_s=-1"])                       # session ended before anything was answered
    first = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert first and all(r["type"] == "abstain" for r in first)
    assert len(json.loads((tmp_path / "l" / "unfinished.json").read_text())) == len(first)
    main([*common, "--resume"])                                        # next session: they are answered, not kept as abstain
    second = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert len(second) == len(first) and sum(r["type"] != "abstain" for r in second) >= len(first) - 2
    assert json.loads((tmp_path / "l" / "unfinished.json").read_text()) == []


def test_session_helpers(tmp_path):
    from datamate.session import import_previous_outputs, remaining_budget_s

    assert remaining_budget_s(11, 0, reserve_s=1800, now=3600) == 11 * 3600 - 3600 - 1800
    assert remaining_budget_s(1, 0, now=10_000) == 600                                   # never below the floor
    prev = tmp_path / "input" / "my-notebook" / "public_test_out" / "shard0"
    prev.mkdir(parents=True)
    (prev / "submission.jsonl").write_text("x")
    work = tmp_path / "work"
    work.mkdir()
    assert len(import_previous_outputs(tmp_path / "input", work)) == 1
    assert (work / "public_test_out" / "shard0" / "submission.jsonl").read_text() == "x"
    assert import_previous_outputs(tmp_path / "input", work) == []                         # never overwrites


def test_preflight_recommendations():
    from datamate.preflight import recommend

    rows = [{"decision": "answer", "elapsed_s": 30.0} for _ in range(5)]
    info = {"model_load_s": 300, "peak_vram_gb": 12.0, "gpu_total_gb": 15.0}
    fast = recommend(rows, info, n_questions=200, procs=2, budget_s=10 * 3600)
    assert fast["set"] == [] and fast["projected_hours"] < 10                                  # fits: keep full voting
    slow = recommend([{"decision": "answer", "elapsed_s": 90.0}] * 5, info, n_questions=1000, procs=2, budget_s=10 * 3600)
    assert "agent.stages=[3]" in slow["set"] and slow["projected_hours"] <= 10                # 12.6 h -> reduce voting
    hopeless = recommend([{"decision": "answer", "elapsed_s": 400.0}] * 5, info, n_questions=1000, procs=2, budget_s=10 * 3600)
    assert any("8B" in a for a in hopeless["advice"])
    tight = recommend(rows, {"model_load_s": 1, "peak_vram_gb": 14.5, "gpu_total_gb": 15.0}, 100, 1, 36000)
    assert "hf.batch_size=1" in tight["set"] and "hf.max_input_tokens=9000" in tight["set"]  # VRAM within 7 % of the limit
    oom = recommend([{"decision": "abstain", "elapsed_s": 5, "error": "CUDA out of memory"}] + rows, info, 100, 1, 36000)
    assert "hf.max_input_tokens=9000" in oom["set"]
    assert recommend([], info, 10, 1, 3600)["sec_per_question"] is None
