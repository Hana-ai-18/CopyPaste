"""
Rule-Delta Guard — coi rulebook là mã nguồn có phiên bản.

BTC nói rõ rulebook KHÁC theo split (train → public đã thêm 3 ngành, ~10 định nghĩa, BỎ 3 luật, sửa vài luật).
Đường máy tính (compiler + ledger) được kiểm theo rulebook TRAIN. Nếu rulebook của split đang chạy đổi nghĩa một luật
mà intent dựa vào, máy tính sẽ TỰ TIN trả lời sai (mỗi câu −1). Module này:

  1. so rulebook đang chạy với VÂN TAY rulebook train (lexicon/rulebook_train_fingerprint.json — chỉ là túi từ của từng
     luật, không chứa văn bản gốc), gán mỗi luật: same / extended (chỉ THÊM chữ) / changed (có chữ bị bỏ/đổi) / new; và
     danh sách luật bị BỎ;
  2. intent của máy tính phụ thuộc luật nào (INTENT_DEPS, viết tay một lần) -> luật đó changed/removed thì KHÔNG cho máy
     tính trả lời intent đó (câu chuyển sang đường code, luật mới được nhắc trong prompt);
  3. câu hỏi chứa THUẬT NGỮ của một luật MỚI (vd 'khách sộp nhất', 'giờ cao điểm', 'nhà cung cấp chính') -> không cho
     máy tính đoán theo intent cũ.

Không có logic theo mã câu / mã shop. Dựng vân tay: python rule_delta.py --build /path/train/rulebook.md
"""
import json
import os
import re
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FP_PATH = os.path.join(HERE, "lexicon", "rulebook_train_fingerprint.json")


def _k(s):
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[*`_]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def rule_key(cell):
    """Khoá ổn định của một luật: ô đầu của dòng bảng, thường, bỏ dấu, bỏ ký tự đặc biệt."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9<> ]+", " ", _k(cell))).strip()


def tokens(text):
    return Counter(re.findall(r"[a-z0-9]+", _k(text)))


def parse_rules(text):
    """-> list dict(key, cell, body, sec, sub). Mỗi dòng bảng '| luật | mặc định |' là một luật (gạch đầu dòng: khoá = 60 ký tự đầu)."""
    out, sec, sub = [], "", ""
    for line in text.splitlines():
        h = re.match(r"^(#{2,4})\s*(.+)", line)
        if h:
            if len(h.group(1)) == 2:
                sec, sub = h.group(2).strip(), ""
            else:
                sub = h.group(2).strip()
            continue
        t = line.strip()
        if t.startswith("|"):
            cells = [c.strip() for c in t.strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= set("-: ") or rule_key(cells[0]) in ("luat", "cach noi", "cach ghi", "tu", "type", "truong hop", "muc"):
                continue
            cell, body = cells[0], " | ".join(cells[1:])
        elif re.match(r"^[-*]\s+", t):
            body = re.sub(r"^[-*]\s+", "", t)
            cell = body[:60]
        else:
            continue
        key = rule_key(cell)
        if sub:
            key = rule_key(sub) + " :: " + key        # luật ngành: tách theo ngành (cùng tên 'Bảo hành' ở hai ngành khác nhau)
        out.append({"key": key, "cell": cell, "body": body, "sec": sec, "sub": sub})
    return out


def build_fingerprint(text):
    fp = {}
    for r in parse_rules(text):
        fp[r["key"]] = {"tok": dict(tokens(r["body"])), "sec": _k(r["sec"])[:40]}
    return fp


# ---------------------------------------------------------------- phụ thuộc intent -> luật (khoá theo rule_key của rulebook TRAIN)
_DT = ["don tinh vao doanh thu", "ban duoc bao nhieu tien thu ve bao nhieu tien", "doanh thu mot don", "don thuoc ky nao"]
_NO = ["don duoc tinh vao no", "no mot don tai ngay x", "no theo khach", "don cod don ban qua san tmdt", "den het ngay x tinh den hom nay"]
INTENT_DEPS = {
    "doanh_thu": _DT,
    "doanh_thu_doi": _DT + ["phan tram thay doi tang truong"],
    "doanh_thu_binh_quan_don": _DT + ["so don", "gia tri trung binh mot don"],
    "so_don": ["so don", "don tinh vao doanh thu", "don thuoc ky nao"],
    "so_don_huy": ["don bi huy", "ti le don huy trong ky", "don thuoc ky nao"],
    "ngay_doanh_thu_cao_nhat": _DT,
    "thang_doanh_thu_cao_nhat": _DT,
    "doanh_thu_tang_lien_tuc": _DT,
    "ban_chay": ["ban chay", "mat hang", "don tinh vao doanh thu", "don thuoc ky nao"],
    "so_luong_ban": ["ban duoc bao nhieu <don vi>", "don tinh vao doanh thu", "don thuoc ky nao", "don vi co ban don vi lon"],
    "khach_mua_nhieu_tien_nhat": _DT + ["xac dinh khach"],
    "tien_nhap": ["tien nhap hang trong ky"],
    "ncc_nhap_nhieu_nhat": ["tien nhap hang trong ky"],
    "gia_nhap_binh_quan": ["gia nhap binh quan mot mat hang trong thang"],
    "nhap_tru_ban": ["nhap tru ban chenh nhap va ban", "don tinh vao doanh thu"],
    "tong_no": _NO,
    "so_khach_no": _NO + ["xac dinh khach"],
    "top_khach_no": _NO + ["xac dinh khach"],
    "tuoi_no": _NO + ["tuoi no"],
    "no_qua_n_ngay": _NO + ["tuoi no", "qua n hon n tren n duoi n"],
    "no_them_tra_no": ["no them tra no thu no trong ky", "don duoc tinh vao no"],
    "thu_no": ["no them tra no thu no trong ky", "don duoc tinh vao no"],
    "ton": ["den het ngay x tinh den hom nay", "don vi co ban don vi lon"],
    "ton_don_vi_lon": ["den het ngay x tinh den hom nay", "don vi co ban don vi lon", "ton theo don vi lon"],
    "so_mat_hang_het_hang": ["mat hang het hang cuoi ngay"],
    "gia_tri_ton_kho": ["gia tri hang ton kho"],
    "lo_gan_het_han_nhat": ["nha thuoc :: lo gan het han nhat cua mot thuoc"],
    "lo_het_han_trong": ["nha thuoc :: hang het han trong n ngay toi"],
    "lai_gop": _DT + ["lai gop"],
    "lai_rong": _DT + ["lai gop", "lai rong mot thang"],
}
# phụ thuộc theo BỘ LỌC / cờ trong đặc tả
SPEC_DEPS = [
    (lambda sp: (sp.get("loc") or {}).get("nhom") is not None, ["nhom hang"]),
    (lambda sp: (sp.get("loc") or {}).get("ke_don") is not None, ["nha thuoc :: thuoc ke don"]),
    (lambda sp: (sp.get("loc") or {}).get("hoat_chat") is not None, ["nha thuoc :: doanh thu theo hoat chat"]),
    (lambda sp: (sp.get("loc") or {}).get("size") is not None, ["ca phe :: size m size l"]),
    (lambda sp: bool(sp.get("theo_hang")), ["hang loi nhat"]),
]
# phụ thuộc theo CÁCH NÓI KỲ trong câu (chuỗi đã bỏ dấu)
PERIOD_DEPS = [
    (r"\btuan\b", "tuan"),
    (r"thang nay", "thang nay"),
    (r"\d+ ngay (gan nhat|qua|vua qua)", "n ngay gan nhat"),
    (r"\btet\b", "tet nam y dip tet mua tet"),
    (r"dau nam", "tu dau nam tu dau nam toi nay"),
    (r"\b(quy|q) ?[1-4]\b", "thang m quy q khong ghi nam"),
    (r"(thang|th) ?\d{1,2}(?! ?[/\-] ?\d)", "thang m quy q khong ghi nam"),
    (r"\bhom (nay|qua)\b", "ngay x hom nay hom qua"),
    (r"\b(tu|den|toi) (ngay )?\d{1,2}/\d{1,2}/\d{4}", "tu x den y nhom x y"),
]


class RuleDelta:
    """Kết quả so rulebook đang chạy với vân tay train + hàm chặn máy tính khi luật nền bị đổi."""

    def __init__(self, text, fp_path=FP_PATH):
        self.ok = os.path.exists(fp_path)
        self.fp = json.load(open(fp_path, encoding="utf-8")) if self.ok else {}
        self.rules = parse_rules(text)
        self.status, self.detail = {}, {}
        cur = {r["key"]: r for r in self.rules}
        for k, r in cur.items():
            if not self.ok:
                self.status[k] = "same"
                continue
            if k not in self.fp:
                self.status[k] = "new"
                continue
            old = Counter(self.fp[k]["tok"])
            new = tokens(r["body"])
            lost = old - new
            if not lost:
                self.status[k] = "same" if old == new else "extended"
            else:
                self.status[k] = "changed"
                self.detail[k] = sorted(lost)[:12]
        self.removed = sorted(k for k in self.fp if k not in cur) if self.ok else []
        # thuật ngữ của luật MỚI: đoạn ≥ 2 chữ của ô đầu (cắt ở chỗ giữ chỗ X/N/Y/D/M/Q và dấu phẩy)
        self.new_terms = []
        old_keys = " | ".join(self.fp)
        for k, st in self.status.items():
            if st != "new":
                continue
            cell = cur[k]["cell"]
            for seg in re.split(r",|/|\(|\)|<[^>]*>|\b[XNYDMQ]\b", cell):
                kk = rule_key(seg)
                # bỏ đoạn là khái niệm CŨ (có trong tên luật train, vd 'nhân viên') — chỉ giữ thuật ngữ thật sự mới
                if len(kk.split()) >= 2 and len(kk) >= 7 and kk not in old_keys:
                    self.new_terms.append((kk, k))

    # ---------------------------------------------------------------- báo cáo
    def summary(self):
        c = Counter(self.status.values())
        return {"fingerprint": self.ok, "dem": dict(c), "bo": self.removed,
                "doi": {k: v for k, v in self.detail.items()}, "moi": [k for k, s in self.status.items() if s == "new"]}

    def flagged(self, key):
        """'changed' / 'removed' / None cho một khoá luật train."""
        if not self.ok:
            return None
        if key in self.removed:
            return "removed"
        st = self.status.get(key)
        return "changed" if st == "changed" else None

    def new_terms_in(self, question_key):
        q = " " + rule_key(question_key) + " "
        return sorted({k for term, k in self.new_terms if " " + term + " " in q})

    def block_reason(self, spec, question_key):
        """Lý do KHÔNG được dùng máy tính cho đặc tả này trên rulebook đang chạy (None = được dùng)."""
        if not spec or not self.ok:
            return None
        it = spec.get("intent")
        hits = self.new_terms_in(question_key)
        if hits:
            return "câu dùng thuật ngữ của luật MỚI: " + "; ".join(hits[:3])
        if it in (None, "unanswerable"):
            return None
        deps = list(INTENT_DEPS.get(it, []))
        for cond, ks in SPEC_DEPS:
            try:
                if cond(spec):
                    deps += ks
            except Exception:
                pass
        q = rule_key(question_key)
        for pat, k in PERIOD_DEPS:
            if re.search(pat, q):
                deps.append(k)
        bad = [(k, self.flagged(k)) for k in dict.fromkeys(deps)]
        bad = [f"{k} ({f})" for k, f in bad if f]
        return ("luật nền đã đổi/bỏ so với train: " + "; ".join(bad)) if bad else None

    def mark(self, rule_key_):
        """Nhãn ngắn để chèn vào prompt cạnh luật: luật MỚI / ĐỔI so với bản train."""
        st = self.status.get(rule_key_)
        return {"new": "LUẬT MỚI", "changed": "LUẬT ĐÃ ĐỔI"}.get(st, "")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="rulebook TRAIN -> ghi vân tay vào lexicon/")
    ap.add_argument("--compare", help="rulebook của split khác -> in bảng so sánh")
    a = ap.parse_args()
    if a.build:
        fp = build_fingerprint(open(a.build, encoding="utf-8").read())
        json.dump(fp, open(FP_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=0, sort_keys=True)
        print(f"Đã ghi {len(fp)} luật vào {FP_PATH}")
        missing = sorted(({k for ks in INTENT_DEPS.values() for k in ks} | {k for _, ks in SPEC_DEPS for k in ks} | {k for _, k in PERIOD_DEPS}) - set(fp))
        print("Khoá phụ thuộc KHÔNG có trong rulebook train (sửa INTENT_DEPS):", missing)
    if a.compare:
        rd = RuleDelta(open(a.compare, encoding="utf-8").read())
        print(json.dumps(rd.summary(), ensure_ascii=False, indent=1))
        print("Thuật ngữ luật mới:", rd.new_terms)
