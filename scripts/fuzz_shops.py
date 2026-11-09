"""
Thử độ bền trên SHOP GIẢ biến đổi từ shop train (phòng dữ liệu private khác train/public). Không cần đáp án.

    python scripts/fuzz_shops.py --train /path/train [--biz biz_101,biz_106,biz_111] [--out /tmp/fuzz]

Mỗi biến đổi tạo một bản sao shop rồi đọc lại bằng pipeline, kiểm:
  1. không lỗi, vẫn dựng được V['don'];
  2. với biến đổi KHÔNG đổi nghĩa (đảo cột, đổi tên cột kiểu viết, tách file, thêm dòng tổng, thêm khách họ Tống,
     bỏ mục cột trong README) thì doanh thu từng tháng, số đơn tính doanh thu GIỮ NGUYÊN so với bản gốc.
Biến đổi có đổi nghĩa (xoá trạng thái 5% đơn) chỉ kiểm không lỗi + có cảnh báo.
"""
import argparse
import glob
import os
import random
import re
import shutil
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402

import datamate_pipeline as d  # noqa: E402

RNG = random.Random(7)


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


def _hdr(raw):
    """Dòng header thật = dòng đầu có ≥ 60% ô không trống (bỏ dòng tiêu đề kiểu 'DANH SÁCH HOÁ ĐƠN' phía trên)."""
    for i in range(min(10, len(raw))):
        if (raw.iloc[i].astype(str).str.strip() != "").mean() >= 0.6:
            df = raw.iloc[i + 1:].reset_index(drop=True)
            df.columns = [str(c) for c in raw.iloc[i]]
            return df
    return raw


def load(path):
    if path.endswith(".csv"):
        return {"": _hdr(pd.read_csv(path, dtype=str, keep_default_na=False, header=None))}
    return {k: _hdr(v) for k, v in pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, header=None).items()}


def save(path, sheets):
    if path.endswith(".csv"):
        sheets[""].to_csv(path, index=False)
    else:
        with pd.ExcelWriter(path) as w:
            for k, v in sheets.items():
                v.to_excel(w, sheet_name=k or "Sheet1", index=False)


def files(dirp):
    return [f for f in glob.glob(os.path.join(dirp, "*")) if f.endswith((".csv", ".xlsx"))]


def readme_rename(dirp, mp):
    rp = os.path.join(dirp, "README.md")
    if os.path.exists(rp) and mp:
        t = open(rp, encoding="utf-8").read()
        for a, b in mp.items():
            t = t.replace(f"**{a}**", f"**{b}**").replace(f"`{a}`", f"`{b}`")
        open(rp, "w", encoding="utf-8").write(t)


# ---------------------------------------------------------------- biến đổi (giữ nghĩa)
def t_shuffle_cols(dirp):
    for f in files(dirp):
        sh = load(f)
        save(f, {k: v[RNG.sample(list(v.columns), len(v.columns))] for k, v in sh.items()})


def t_rename_style(dirp):
    """Đổi KIỂU viết tên cột (bỏ dấu / HOA / gạch dưới) — README đổi theo để mô tả cột vẫn khớp."""
    mp = {}
    for f in files(dirp):
        sh = load(f)
        out = {}
        for k, v in sh.items():
            cols = {}
            for c in v.columns:
                style = RNG.choice(["noacc", "upper", "snake", "same"])
                n = strip_acc(c) if style == "noacc" else c.upper() if style == "upper" else re.sub(r"\s+", "_", strip_acc(c).lower()) if style == "snake" else c
                cols[c] = n
                if n != c:
                    mp[c] = n
            out[k] = v.rename(columns=cols)
        save(f, out)
    readme_rename(dirp, mp)


def t_split_file(dirp):
    """Tách bảng lớn nhất thành 2 file _1, _2 cùng cột (như 'danh sách dài tách nhiều file')."""
    fs = sorted(files(dirp), key=os.path.getsize, reverse=True)
    if not fs or not fs[0].endswith(".csv"):
        return
    f = fs[0]
    v = load(f)[""]
    h = len(v) // 2
    base = f[:-4]
    v.iloc[:h].to_csv(base + "_1.csv", index=False)
    v.iloc[h:].to_csv(base + "_2.csv", index=False)
    os.remove(f)
    rp = os.path.join(dirp, "README.md")
    if os.path.exists(rp):
        t = open(rp, encoding="utf-8").read().replace(f"`{os.path.basename(f)}`", f"`{os.path.basename(base)}_1.csv`")
        open(rp, "w", encoding="utf-8").write(t)


def t_add_total_rows(dirp):
    """Chèn dòng 'Tổng cộng' / 'Cộng ngày …' (chỉ ô chữ + số) vào bảng lớn nhất."""
    fs = sorted(files(dirp), key=os.path.getsize, reverse=True)
    if not fs:
        return
    f = fs[0]
    sh = load(f)
    out = {}
    for k, v in sh.items():
        rows = []
        for i, (_, r) in enumerate(v.iterrows()):
            rows.append(r.to_dict())
            if i % 97 == 96:
                tr = {c: "" for c in v.columns}
                tr[v.columns[RNG.randrange(len(v.columns))]] = RNG.choice(["Tổng cộng", "Cộng ngày 05/03", "TỔNG:", "Total"])
                tr[v.columns[-1]] = "1234567"
                rows.append(tr)
        out[k] = pd.DataFrame(rows, columns=v.columns)
    save(f, out)


def t_tong_customer(dirp):
    """Đổi tên vài khách thành họ 'Tống' viết không dấu ('Tong Van A') — không được bị coi là dòng tổng."""
    for f in files(dirp):
        sh = load(f)
        out = {}
        for k, v in sh.items():
            cc = [c for c in v.columns if re.search(r"khach|ben mua|doi tac", d.key(c)) and not re.search(r"ma|so|id|nhom|tra", d.key(c))]
            if cc and len(v):
                names = [x for x in v[cc[0]].unique() if x and not re.search(r"khach le|vang lai", d.key(x))][:3]
                v.loc[v[cc[0]].isin(names), cc[0]] = "Tong Van An"     # đổi TOÀN BỘ dòng của vài khách (không tạo khác biệt giả)
            out[k] = v
        save(f, out)


def t_drop_readme_cols(dirp):
    """Bỏ mục '## Cột trong …' của README (shop không mô tả cột) — phải đọc được bằng tên cột."""
    rp = os.path.join(dirp, "README.md")
    if os.path.exists(rp):
        t = open(rp, encoding="utf-8").read()
        t = re.sub(r"## Cột trong[^\n]*\n(?:- \*\*[^\n]*\n)+", "", t)
        open(rp, "w", encoding="utf-8").write(t)


# ---------------------------------------------------------------- biến đổi (đổi nghĩa)
def t_blank_status(dirp):
    for f in files(dirp):
        sh = load(f)
        out = {}
        for k, v in sh.items():
            cc = [c for c in v.columns if re.search(r"trang thai|tinh trang|status|xu ly|ket qua|ghi chu tt", d.key(c))]
            if cc:
                idx = v.sample(frac=0.05, random_state=3).index
                v.loc[idx, cc[0]] = ""
            out[k] = v
        save(f, out)


SAFE = [t_shuffle_cols, t_rename_style, t_split_file, t_add_total_rows, t_tong_customer, t_drop_readme_cols]
RISKY = [t_blank_status]


def summary(B):
    o = B.V.get("don")
    if o is None:
        return None
    ng = pd.to_datetime(o["ngay"], errors="coerce").dt.to_period("M")
    x = o[o["tinh_doanh_thu"]]
    return (x.groupby(ng[x.index])["doanh_thu"].sum().round(0).to_dict(), int(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--biz", default="biz_101,biz_106,biz_111,biz_125")
    ap.add_argument("--out", default="/tmp/fuzz_shops")
    a = ap.parse_args()
    fails = 0
    for bid in a.biz.split(","):
        src = os.path.join(a.train, "data", bid)
        base = summary(d.Business(src))
        for tf in SAFE + RISKY:
            dst = os.path.join(a.out, f"{bid}__{tf.__name__}")
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            try:
                tf(dst)
                B = d.Business(dst)
                sm = summary(B)
                if sm is None:
                    status = "LỖI: không dựng được V['don']"
                elif tf in SAFE:
                    same_n = sm[1] == base[1]
                    diff = {k: (base[0].get(k), v) for k, v in sm[0].items() if abs((base[0].get(k) or 0) - v) > 1}
                    status = "OK" if same_n and not diff else f"LỆCH: số đơn {base[1]}→{sm[1]}; tháng lệch {list(diff.items())[:2]}"
                else:
                    warn = [n for n in B.V_notes if "TRỐNG" in n or "KIỂM" in n]
                    status = "OK (có cảnh báo)" if warn else "THIẾU CẢNH BÁO"
            except Exception as e:
                status = f"LỖI: {type(e).__name__}: {e}"
            fails += not status.startswith("OK")
            print(f"{bid:<8} {tf.__name__:<20} {status}")
    print("Không đạt:", fails)
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
