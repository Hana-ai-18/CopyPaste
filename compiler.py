"""
Bước 3 + 4 (đường 1): ĐẶC TẢ câu hỏi -> MÁY TÍNH THEO CÔNG THỨC (không dùng LLM để tính).

  spec = spec_rules(question, as_of, P, biz)       # đặc tả bằng luật chữ (không theo mã câu / mã shop)
  spec = parse_llm_spec(text, ...)                  # hoặc đặc tả do LLM viết (JSON theo danh mục INTENTS)
  ans  = compile_spec(spec, biz, as_of)             # gọi hàm ledger đã đối chiếu train -> đáp án đúng format §9

Đặc tả gồm: intent, kỳ (tu/den hoặc danh sách tháng, ngày X), bộ lọc (nhóm, hoạt chất, kê đơn, mặt hàng…), N, đơn vị,
kiểu đáp án, và `doc_khac` (các cách hiểu khác có thể có — dùng cho kiểm toán khi hai đường lệch nhau).
Máy tính trả None khi đặc tả không đủ / không chắc — khi đó chỉ còn đường model tự viết code.
"""
import math
import re

import numpy as np
import pandas as pd

import ledger as L

# --------------------------------------------------------------------------- danh mục intent (cũng là "từ điển" cho LLM)
INTENTS = {
    "doanh_thu": "doanh thu (§1) trong kỳ; lọc tuỳ chọn: nhom, hoat_chat, ke_don, size, mat_hang; ti_trong=true -> % trên doanh thu cả shop",
    "doanh_thu_doi": "% thay đổi doanh thu kỳ hỏi so với kỳ gốc (mặc định tháng liền trước) — §1",
    "doanh_thu_binh_quan_don": "doanh thu bình quân mỗi đơn trong kỳ (làm tròn về đồng)",
    "so_don": "số đơn (không huỷ, không trả) trong kỳ",
    "so_don_huy": "số đơn huỷ trong kỳ; ti_le=true -> % đơn huỷ trên mọi đơn đặt trong kỳ",
    "ngay_doanh_thu_cao_nhat": "ngày có doanh thu cao nhất trong kỳ -> date YYYY-MM-DD",
    "thang_doanh_thu_cao_nhat": "tháng có doanh thu cao nhất trong các tháng nêu -> date YYYY-MM",
    "doanh_thu_tang_lien_tuc": "doanh thu các tháng nêu có tăng liên tục (tháng sau > tháng trước) -> boolean",
    "ban_chay": "mặt hàng bán chạy nhất (theo số lượng cơ bản); n>1 -> list top n",
    "so_luong_ban": "số lượng bán của mat_hang trong kỳ, theo don_vi hỏi",
    "khach_mua_nhieu_tien_nhat": "khách có doanh thu cao nhất trong kỳ -> string",
    "tien_nhap": "tổng tiền các phiếu nhập (không huỷ) trong kỳ",
    "ncc_nhap_nhieu_nhat": "nhà cung cấp có tổng tiền nhập lớn nhất trong kỳ -> string",
    "gia_nhap_binh_quan": "giá nhập bình quân một don_vi của mat_hang trong kỳ (làm tròn về đồng)",
    "nhap_tru_ban": "số lượng nhập − số lượng bán của mat_hang trong kỳ (đơn vị cơ bản)",
    "tong_no": "tổng còn nợ lúc hết ngày X",
    "so_khach_no": "số khách còn nợ lúc hết ngày X",
    "top_khach_no": "bảng n khách nợ nhiều nhất lúc hết ngày X (khach_hang, so_tien_no)",
    "tuoi_no": "[nợ 0–7 ngày, 8–15 ngày, trên 15 ngày] lúc hết ngày X",
    "no_qua_n_ngay": "tổng nợ của các đơn có tuổi nợ > n ngày lúc hết ngày X",
    "no_them_tra_no": "[nợ thêm, trả nợ] của khách trong kỳ",
    "thu_no": "tiền thu nợ (trả sau ngày mua) trong kỳ",
    "ton": "tồn (đơn vị cơ bản) của mat_hang lúc hết ngày X",
    "ton_don_vi_lon": "[số don_vi lớn nguyên, số đơn vị cơ bản lẻ] của mat_hang lúc hết ngày X",
    "so_mat_hang_het_hang": "số mặt hàng mua về để bán có tồn 0 lúc hết ngày X",
    "gia_tri_ton_kho": "tồn × giá vốn danh mục lúc hết ngày X",
    "lo_gan_het_han_nhat": "hạn dùng của lô còn tồn gần hết hạn nhất của mat_hang -> date",
    "lo_het_han_trong": "lô còn tồn hết hạn từ hôm nay đến hết ngày X / trong n ngày: dem='lo'|'mat_hang'|'ds'|'tien_von'",
    "lai_gop": "lãi gộp trong kỳ; theo_hang=true -> mặt hàng lời nhất",
    "lai_rong": "lãi ròng trong kỳ",
}

ANSWER_TYPE = {
    "doanh_thu": "number", "doanh_thu_doi": "percent", "doanh_thu_binh_quan_don": "number", "so_don": "number", "so_don_huy": "number",
    "ngay_doanh_thu_cao_nhat": "date", "thang_doanh_thu_cao_nhat": "date", "doanh_thu_tang_lien_tuc": "boolean", "ban_chay": "string",
    "so_luong_ban": "number", "khach_mua_nhieu_tien_nhat": "string", "tien_nhap": "number", "ncc_nhap_nhieu_nhat": "string",
    "gia_nhap_binh_quan": "number", "nhap_tru_ban": "number", "tong_no": "number", "so_khach_no": "number", "top_khach_no": "table",
    "tuoi_no": "list", "no_qua_n_ngay": "number", "no_them_tra_no": "list", "thu_no": "number", "ton": "number", "ton_don_vi_lon": "list",
    "so_mat_hang_het_hang": "number", "gia_tri_ton_kho": "number", "lo_gan_het_han_nhat": "date", "lo_het_han_trong": "number",
    "lai_gop": "number", "lai_rong": "number",
}


def key(s):
    import unicodedata
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(ch for ch in s if unicodedata.category(ch) != "Mn")).strip()


def rhu(x, nd=0):
    """Làm tròn nửa lên (rulebook §9)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    q = 10 ** nd
    v = math.floor(abs(x) * q + 0.5 + 1e-9) / q
    v = v if x >= 0 else -v
    return int(v) if nd == 0 else round(v, nd)


# --------------------------------------------------------------------------- kỳ
# Ngày đầu tuần (0 = thứ Hai). Pipeline đặt lại từ rulebook của split đang chạy (extract_rule_params) — không gõ cứng.
WEEK_START = 0

_SKIP_AFTER_MONTH = re.compile(r"^\s*(lan|lần|don|bill|ngay|tuan|cai|chiec|hop|thung|lon|kg|goi|chai|%|phan tram|trieu|tr\b|k\b|nghin|ngan)")


def _yy(y):
    """'26' -> 2026, '2026' -> 2026; năm ngoài 2010–2039 -> None (không phải năm)."""
    y = int(y)
    if y < 100:
        y += 2000
    return y if 2010 <= y <= 2039 else None


def _resolve_year(mo, asof):
    """Tháng M không ghi năm (rulebook §2): của năm nay nếu không sau hôm nay; sau hôm nay thì của năm trước."""
    asof = pd.Timestamp(asof).normalize()
    y = asof.year
    return y - 1 if pd.Timestamp(y, mo, 1) > asof else y


def canon_period_text(t, asof=None):
    """Viết lại MỌI cách ghi kỳ về dạng chuẩn 'thang M/YYYY' và 'quy Q/YYYY' (trên chuỗi đã key(): thường, bỏ dấu).

    Bắt: 'T5/2026', 'th5/26', 'thg 5-2026', 'Th 11/25', '05/2026', '5-2026', '2/26' (M/YY, YY 20–39),
    'tháng 5' / 'th5' không ghi năm (năm theo rulebook §2), 'tháng 5 năm ngoái', 'Q2/25', 'quý 2' không ghi năm.
    KHÔNG bắt: 'T5' trơn (có thể là thứ Năm), 'ngày 5 tháng 6' (ngày), 'tháng 2 lần', ngày đủ dd/mm/yyyy.
    Chỉ dùng để giải KỲ; tìm tên hàng / khách vẫn dùng câu gốc."""
    s = " " + t + " "

    def my(m):
        mo, y = int(m.group(1)), _yy(m.group(2))
        if not (1 <= mo <= 12) or y is None:
            return m.group(0)
        return f" thang {mo}/{y} "
    # 1) có tiền tố tháng + năm: thang/thg/th/t + M + / - . + YYYY|YY
    s = re.sub(r"(?<![a-z0-9/])(?:thang|thg|th|t)\s*\.?\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})(?![\d/])", my, s)
    # 'tháng 5 năm 2025' / 'tháng 5 năm 25'
    s = re.sub(r"(?<![a-z0-9/])thang\s+(\d{1,2})\s+nam\s+(\d{4}|\d{2})(?![\d/])", my, s)

    # 2) không tiền tố: M/YYYY, M-YYYY, MM/YYYY, M/YY (YY 20–39) — bỏ qua nếu là phần cuối của ngày dd/mm/yyyy hoặc đã có 'thang '
    def bare(m):
        before = s[max(0, m.start() - 8):m.start()]
        # 'quý 2/2026', 'quys 3/2025', 'q 1/2026', 'qúy': số đứng sau chữ quý là QUÝ, không phải tháng
        if re.search(r"(thang|ngay|quy\w*|q)\s*$", before):
            return m.group(0)
        mo, ys = int(m.group(1)), m.group(2)
        y = _yy(ys)
        if not (1 <= mo <= 12) or y is None:
            return m.group(0)
        if len(ys) == 2 and not (20 <= int(ys) <= 39):
            return m.group(0)
        return f" thang {mo}/{y} "
    s = re.sub(r"(?<![\d/.:a-z])(\d{1,2})\s*[/\-]\s*(\d{4}|\d{2})(?![\d/.:])", bare, s)

    if asof is not None:
        A = pd.Timestamp(asof).normalize()

        # 3) tháng không ghi năm: 'thang 5', 'th5', 'thg 5' (không 't5' trơn)
        def nm(m):
            before = s2[max(0, m.start() - 12):m.start()]
            after = s2[m.end():m.end() + 14]
            mo = int(m.group(1))
            if not 1 <= mo <= 12 or re.search(r"ngay\s*\d{1,2}\s*$|mung\s*\d{1,2}\s*$", before) or re.match(r"\s*[/\-]\s*\d", after):
                return m.group(0)
            ma = re.match(r"\s*nam\s+(ngoai|truoc|roi|nay)\b", after)
            if ma:
                return m.group(0)               # xử lý ở bước 3b
            if _SKIP_AFTER_MONTH.match(after):
                return m.group(0)
            return f" thang {mo}/{_resolve_year(mo, A)} "
        s2 = s
        s = re.sub(r"(?<![a-z0-9/])(?:thang|thg|th)\s*\.?\s*(\d{1,2})(?![\d/])", nm, s2)

        # 3b) 'tháng 5 năm ngoái/trước/rồi' -> năm trước; 'năm nay' -> năm nay
        def nm2(m):
            mo = int(m.group(1))
            if not 1 <= mo <= 12:
                return m.group(0)
            y = A.year - 1 if m.group(2) in ("ngoai", "truoc", "roi") else A.year
            return f" thang {mo}/{y} "
        s = re.sub(r"(?<![a-z0-9/])(?:thang|thg|th)\s*(\d{1,2})\s+nam\s+(ngoai|truoc|roi|nay)\b", nm2, s)

        # 4) quý: 'q2/25', 'quy 2-2025', 'quy 2 nam 2025', 'quy 2' (không năm: năm nay nếu không sau hôm nay)
        def qy(m):
            q = int(m.group(1))
            ys = m.group(2)
            if ys:
                y = _yy(ys)
                if y is None:
                    return m.group(0)
            else:
                y = A.year if pd.Timestamp(A.year, (q - 1) * 3 + 1, 1) <= A else A.year - 1
            return f" quy {q}/{y} "
        s = re.sub(r"(?<![a-z0-9])(?:quy[sfrxj]?|q)\s*([1-4])(?:\s*(?:/|-|nam)\s*(\d{4}|\d{2}))?(?![\d/])", qy, s)
    return re.sub(r"\s+", " ", s).strip()


def _quarters(t):
    out = []
    for m in re.finditer(r"quy ([1-4])/(\d{4})", t):
        x = (int(m.group(2)), int(m.group(1)))
        if x not in out:
            out.append(x)
    return out


def _months(t):
    """Mọi 'tháng M/YYYY' (cả 'tháng M năm YYYY', danh sách '11/2025, 12/2025 và 1/2026', khoảng 'từ tháng A tới tháng B')."""
    out = []
    for m in re.finditer(r"thang (\d{1,2})(?:/| nam )(\d{4})", t):
        out.append((int(m.group(2)), int(m.group(1))))
    for m in re.finditer(r"thang \d{1,2}/\d{4}((?:\s*(?:,|va|;|&|->|-)\s*\d{1,2}/\d{4})+)", t):
        out += [(int(y), int(mo)) for mo, y in re.findall(r"(\d{1,2})/(\d{4})", m.group(1))]
    for m in re.finditer(r"(?:ba|3|cac) thang (\d{1,2}/\d{4}(?:\s*(?:,|va)\s*\d{1,2}/\d{4})+)", t):
        out += [(int(y), int(mo)) for mo, y in re.findall(r"(\d{1,2})/(\d{4})", m.group(1))]
    m = re.search(r"tu thang (\d{1,2})/(\d{4}) (?:den|toi|qua|->) (?:het )?thang (\d{1,2})/(\d{4})", t)
    if m:
        a, b = pd.Timestamp(int(m.group(2)), int(m.group(1)), 1), pd.Timestamp(int(m.group(4)), int(m.group(3)), 1)
        cur = a
        while cur <= b and len(out) < 30:
            out.append((cur.year, cur.month))
            cur += pd.offsets.MonthBegin(1)
    seen, res = set(), []
    for x in out:
        if x not in seen and 1 <= x[1] <= 12:
            seen.add(x)
            res.append(x)
    return sorted(res)


def _month_span(y, mo, asof):
    a = pd.Timestamp(y, mo, 1)
    return a, min(a + pd.offsets.MonthEnd(0), pd.Timestamp(asof).normalize())


def _period(t, asof):
    """Kỳ chính của câu: một tháng, tháng này, tuần trước/này, N ngày gần nhất, khoảng ngày. None nếu không rõ."""
    asof = pd.Timestamp(asof).normalize()
    ms = _months(t)
    m = re.search(r"tu (?:ngay )?(\d{1,2})/(\d{1,2})/(\d{4}) (?:den|toi) (?:het )?(?:ngay )?(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        a = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        b = pd.Timestamp(int(m.group(6)), int(m.group(5)), int(m.group(4)))
        return a, min(b, asof)
    if len(ms) == 1:
        return _month_span(ms[0][0], ms[0][1], asof)
    if len(ms) > 1:
        return None
    qs = _quarters(t)
    if len(qs) == 1:
        y, q = qs[0]
        a = pd.Timestamp(y, (q - 1) * 3 + 1, 1)
        return a, min(a + pd.offsets.QuarterEnd(0), asof)
    if len(qs) > 1:
        return None
    if re.search(r"tu dau nam", t) and not re.search(r"nam (ngoai|truoc|roi)|\bnam 20\d\d", t):
        return pd.Timestamp(asof.year, 1, 1), asof              # rulebook §2: 01/01 năm nay đến hết hôm nay
    if re.search(r"thang nay", t):
        return asof.replace(day=1), asof
    wd = (asof.weekday() - WEEK_START) % 7                       # số ngày từ đầu tuần (theo rulebook) đến hôm nay
    if re.search(r"tuan (truoc|trc|roi)", t):
        s0 = asof - pd.Timedelta(days=wd + 7)
        return s0, s0 + pd.Timedelta(days=6)
    if re.search(r"tuan nay", t):
        return asof - pd.Timedelta(days=wd), asof
    m = re.search(r"(\d{1,3}) ngay (gan nhat|qua|vua qua)", t)
    if m:
        return asof - pd.Timedelta(days=int(m.group(1)) - 1), asof
    if re.search(r"hom nay", t) and not re.search(r"het hom nay|cuoi ngay|den hom nay", t):
        return asof, asof
    if re.search(r"hom qua", t):
        return asof - pd.Timedelta(days=1), asof - pd.Timedelta(days=1)
    return None


def _date_X(t, asof):
    """Ngày 'đến hết ngày X' / 'cuối ngày X' / 'hôm nay' cho câu hỏi số CÒN LẠI."""
    m = re.search(r"(?:het|cuoi|den|toi|tinh den|ngay) (?:ngay )?(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        try:
            return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    if re.search(r"hom nay|cuoi ngay|hien (tai|gio|nay)|bay gio|luc nay|dang (con|ton)", t):
        return pd.Timestamp(asof).normalize()
    return None


def _data_range(biz):
    V = biz.V
    o = V.get("don")
    start = pd.to_datetime(o["ngay"], errors="coerce").min() if o is not None and "ngay" in o.columns else None
    end = getattr(biz, "data_end", None)
    if end is None and o is not None and "ngay" in o.columns:
        end = pd.to_datetime(o["ngay"], errors="coerce").max()
    return (start.normalize() if start is not None and not pd.isna(start) else None), (pd.Timestamp(end).normalize() if end is not None else None)


def _outside(biz, a, b):
    """Kỳ nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu (§7) -> True."""
    s, e = _data_range(biz)
    if a is None or b is None:
        return False
    if b < a:
        return True
    return (e is not None and a > e) or (s is not None and b < s)


# --------------------------------------------------------------------------- thực thể trong câu
def _names_in(t, values, minlen=4):
    hits = []
    for v in values:
        if isinstance(v, str):
            kv = key(v)
            if len(kv) >= minlen and re.search(r"(^|\s)" + re.escape(kv) + r"(\s|$)", t):
                hits.append(v)
    hits.sort(key=lambda v: -len(key(v)))
    return hits


def _product_in(biz, t, t_raw=None):
    ds = L.danh_sach_hang(biz.V)
    if not len(ds):
        return None
    vals = ds["ten_hang"].dropna().tolist() if "ten_hang" in ds.columns else []
    for tt in [t] + ([t_raw] if t_raw and t_raw != t else []):
        hit = _names_in(tt, vals)
        if hit:
            return hit[0]
    vals = ds["ma_hang"].dropna().tolist() if "ma_hang" in ds.columns else []
    hit = _names_in(t, vals, 3)
    return hit[0] if hit else None


def _attr_in(biz, t, col):
    h = biz.V.get("hang")
    if h is None or col not in h.columns:
        return None, False
    hit = _names_in(t, sorted(set(h[col].dropna().astype(str))), 3)
    return (hit[0] if hit else None), True


# thông tin hay bị hỏi mà nhiều shop KHÔNG ghi (§7: không suy từ cột khác)
NOT_RECORDED = [
    (r"\b(khach|nguoi) (nu|nam)\b|gioi tinh", r"gioi tinh|\bnam nu\b|\bphai\b|gender"),
    # địa chỉ: 'khách ở đường X', 'khách đường X', 'nhà ở' — KHÔNG bắt 'đường đen' trong tên món
    (r"(khach|nguoi|nha)( hang)? (o |tai |nha o |song o )?duong\b|dia chi|\b(o|tai) (quan|phuong)\b|\bnha o\b", r"dia chi|address|quan huyen|phuong xa"),
    (r"tan tam|le phep|chu dao|lanh nghe|tay nghe|quy men|hay khen|than thien|uy tin|chat luong phuc vu", None),
    (r"ly do huy|vi giao tre|huy vi|tra vi (loi|mua nham)|tra hang vi", r"ly do|nguyen nhan|reason"),
    (r"(uong|dung) moi lan|lieu dung|cach dung", r"lieu|cach dung|huong dan"),
]


def not_recorded(biz, t):
    """Câu hỏi thông tin shop không ghi (giới tính, địa chỉ, đánh giá chủ quan, lý do huỷ/trả, liều dùng) -> lý do; không thì None."""
    cols = " ".join(key(c) for df in biz.T.values() for c in df.columns) + " " + key(getattr(biz, "readme", "") or "")[:20000]
    for qpat, cpat in NOT_RECORDED:
        if re.search(qpat, t):
            if cpat is None:
                return "đánh giá chủ quan: dữ liệu không ghi"
            if not re.search(cpat, cols):
                return "shop không ghi loại thông tin này"
    return None


# --------------------------------------------------------------------------- đặc tả bằng luật chữ
# 'ngày nào' và các cách nói khác của nó: hôm nào, bữa nào, ngày mô/hôm mô/bữa mô (miền Trung), bữa nao
DAY_WHICH = r"\b(ngay|hom|bua)\w* (nao|mo)\b"
# số đếm bằng chữ sau 'top' ('top năm món' = 5 món, KHÔNG phải 'năm' = year)
NUM_WORDS = {"hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


def _top_n(t, noun):
    """Số k của 'top k …' / 'k món …'. Trả (k, chắc chắn?)."""
    m = re.search(r"\btop ?(\d{1,2})\b", t) or re.search(r"(?<![/\d])(\d{1,2}) " + noun, t)
    if m:
        return int(m.group(1)), True
    m = re.search(r"\btop (" + "|".join(NUM_WORDS) + r")\b", t)
    if m:
        return NUM_WORDS[m.group(1)], True
    m = re.search(r"\b(" + "|".join(NUM_WORDS) + r") " + noun, t)
    if m:
        return NUM_WORDS[m.group(1)], False          # 'năm món…' có thể là 'trong năm, món…' -> không chắc
    if re.search(r"\btop\b|nhung (mon|mat hang|san pham|sp|khach)\b|cac (mon|mat hang|san pham|sp|khach)\b", t):
        return None, False                             # hỏi NHIỀU mục mà không rõ bao nhiêu
    return 1, True


def spec_rules(question, as_of, biz, q_norm=None):
    """Nhận dạng intent + tham số bằng luật chữ trên câu đã chuẩn hoá. Trả dict hoặc None (không nhận ra chắc chắn)."""
    # intent đọc trên câu ĐÃ SỬA CHỮ (câu gốc bẩn có thể chứa 'nớ' -> 'no' làm lạc sang công nợ);
    # tên hàng / khách tìm trên cả hai (sửa chữ có thể làm hỏng tên riêng)
    t = key(q_norm or question)
    t_raw = key(question)
    t = re.sub(r"\bcancel(l?ed)?\b", "huy", t)
    t = re.sub(r"\bmonth\b", "thang", t)
    asof = pd.Timestamp(as_of).normalize()
    tp = canon_period_text(t, asof)          # kỳ viết tắt (T5/2026, 2/26, 5-2026, quý 2…) -> dạng chuẩn; chỉ dùng để giải kỳ
    P = _period(tp, asof)
    X = _date_X(t, asof)
    ms = _months(tp)
    S = {"q": t}
    # câu hỏi HAI đại lượng một lúc ('phí giao hàng và số đơn…', 'doanh thu và số đơn…'): máy tính chỉ trả một số -> nhường đường code
    if re.search(r"\b(va|cung voi|kem|lan) (so (don|khach|luong)|doanh thu|dthu|tien|ton kho|phi|lai|gia)\b", t) and \
            not re.search(r"no them va tra no|tang hay giam", t):
        return None
    nr = not_recorded(biz, t)
    if nr:
        return {"intent": "unanswerable", "ly_do": nr, "q": t}

    def mk(intent, **kw):
        d = dict(S, intent=intent, **kw)
        # độ tin: 'thap' = dạng câu mà máy tính còn lệch đáp án train (xếp hạng top-k, hàng lời nhất) -> chỉ dùng khi đường kia đồng ý
        d["tin_cay"] = "thap" if (intent == "ban_chay" and kw.get("n", 1) > 1) or (intent == "lai_gop" and kw.get("theo_hang")) else "cao"
        return d

    pct = bool(re.search(r"phan tram|%|ti le|ty le|ti trong|ty trong|chiem", t))
    # ---- lô / hạn dùng
    if re.search(r"gan het han nhat", t):
        return mk("lo_gan_het_han_nhat", X=X or asof, mat_hang=_product_in(biz, t, t_raw))
    if re.search(r"het han", t) and re.search(r"con (ton|hang)|dang con|con lai", t):
        mN = re.search(r"(\d+) ngay toi", t)
        mD = re.search(r"(?:den|toi) (?:het )?(?:ngay )?(\d{1,2})/(\d{1,2})/(\d{4})", t)
        N = int(mN.group(1)) if mN else ((pd.Timestamp(int(mD.group(3)), int(mD.group(2)), int(mD.group(1))) - asof).days if mD else None)
        if N is None:
            return None
        dem = "tien_von" if "tien von" in t else "lo" if re.search(r"(may|bao nhieu|bn|so) (lo|no)\b", t) else \
            "mat_hang" if re.search(r"(may|bao nhieu|bn|so) (mat hang|san pham|sp|loai)", t) else "ds"
        return mk("lo_het_han_trong", N=N, dem=dem)
    # ---- công nợ
    if re.search(r"\bno\b|cong no|mua chiu|ghi so", t):
        mN = re.search(r"(qua|hon|tren) (\d+) ngay", t)
        # 'các khoản nợ có tuổi nợ QUÁ 7 ngày còn bao nhiêu' hỏi MỘT số (nợ quá N ngày), không phải bảng các nhóm tuổi nợ
        buckets = re.search(r"0\W*7\b|8\W*15\b|(tung|moi|cac) nhom|nhom tuoi|chia (theo )?(nhom|tuoi)", t)
        if X is not None and mN and not buckets:
            return mk("no_qua_n_ngay", X=X, n=int(mN.group(2)))
        if X is not None and re.search(r"tuoi no|0.7|8.15", t):
            return mk("tuoi_no", X=X)
        if X is not None and re.search(r"(\d+) khach .*no (nhieu|nhiu)|khach no (nhieu|nhiu) nhat|top (\d+)|(\d+) khach .*(con )?no", t) \
                and re.search(r"bang|liet ke|kem|top|nhung|nao|moi nguoi|tung nguoi", t):
            n = re.search(r"(\d+) khach|top (\d+)", t)
            return mk("top_khach_no", X=X, n=int(n.group(1) or n.group(2)) if n else 3)
        if X is not None and re.search(r"(bao nhi\w*|bn|may|so) khach", t):
            return mk("so_khach_no", X=X)
        if X is not None and re.search(r"\bai\b|khach nao|(tung|tug|moi) (nguoi|ngui|nguoi)|nhung ai|top|liet ke", t):
            return None              # hỏi AI nợ / từng người -> bảng, không phải tổng nợ
        if X is not None and re.search(r"(tong|bao nhi\w*|bn|con no|chua thu|con thieu|con lai)", t):
            return mk("tong_no", X=X)
        if P is not None and re.search(r"no them|mua chiu them|ghi no them", t) and re.search(r"tra( no)?|thanh toan", t):
            names = set()
            for c in ("khach_ten", "khach_ten_dm"):
                if "don" in biz.V and c in biz.V["don"].columns:
                    names |= set(biz.V["don"][c].dropna().astype(str))
            hit = _names_in(t, names, 6)
            return mk("no_them_tra_no", tu=P[0], den=P[1], khach=hit[0] if hit else None) if hit else None
        if P is not None and re.search(r"thu (hoi|ve|duoc)?.*no|tra no|thu no", t):
            return mk("thu_no", tu=P[0], den=P[1])
        return None
    # ---- kho
    if re.search(r"het hang", t) and X is not None and re.search(r"(bao nhieu|may|bn|so) (mat hang|san pham|sp|mon|loai)", t):
        return mk("so_mat_hang_het_hang", X=X)
    if re.search(r"gia von", t) and re.search(r"kho|ton", t) and re.search(r"gia tri|tri gia|bao nhieu tien|tong", t):
        return mk("gia_tri_ton_kho", X=X or asof)
    if re.search(r"ton kho|kho con|con ton|ton cuoi|trong kho|con lai .*kho|con bao nhieu", t) and X is not None \
            and not re.search(r"het han|\blo\b|gia tri|tien", t):
        mh = _product_in(biz, t, t_raw)
        if mh is None:
            return None
        if re.search(r"chan|nguyen", t) and re.search(r"\ble\b|cai le|le ra|con le", t):
            u = L.don_vi_lon_trong_cau(biz.V, mh, t)
            return mk("ton_don_vi_lon", X=X, mat_hang=mh, don_vi=u) if u else None
        return mk("ton", X=X, mat_hang=mh)
    # ---- nhập
    if re.search(r"nhap tru (so )?ban|chenh (lech )?(giua )?nhap va ban", t) and P is not None:
        mh = _product_in(biz, t, t_raw)
        return mk("nhap_tru_ban", tu=P[0], den=P[1], mat_hang=mh) if mh else None
    if "binh quan" in t and "nhap" in t and P is not None:
        mh = _product_in(biz, t, t_raw)
        if mh is None:
            return None
        u, _f = L.don_vi_trong_cau(biz.V, mh, t)
        if u is None:
            mu = re.search(r"(?:mot|1|moi) (\w+)\s*(?:\?|$| la| vay| trong| nhi| ha| thang)", t)
            u = mu.group(1) if mu else None
        return mk("gia_nhap_binh_quan", tu=P[0], den=P[1], mat_hang=mh, don_vi=u)
    if re.search(r"nha cung cap|\bncc\b", t) and re.search(r"nhieu (tien )?nhat|lon nhat", t) and P is not None:
        return mk("ncc_nhap_nhieu_nhat", tu=P[0], den=P[1])
    if re.search(r"tien nhap|nhap hang|phieu nhap|bo ra .*nhap", t) and P is not None and re.search(r"tong|bao nhi|bn|het", t) \
            and not re.search(r"mat hang|binh quan|trung binh|ncc|nha cung cap|so luong", t):
        return mk("tien_nhap", tu=P[0], den=P[1])
    # ---- lãi
    if re.search(r"lai rong", t) and P is not None:
        return mk("lai_rong", tu=P[0], den=P[1])
    if re.search(r"lai gop|hang loi nhat|loi nhat", t) and P is not None:
        if re.search(r"(mat hang|mon|hang|san pham) (nao|gi)|hang loi nhat|loi nhat", t):
            return mk("lai_gop", tu=P[0], den=P[1], theo_hang=True)
        return mk("lai_gop", tu=P[0], den=P[1])
    # ---- đơn huỷ
    if re.search(r"\bhuy\b", t) and P is not None and not re.search(r"\bvi\b|ly do|tien|doanh thu|dthu|khach|mat hang", t):
        return mk("so_don_huy", tu=P[0], den=P[1], ti_le=pct)
    # ---- ngày / tháng cao nhất, tăng liên tục
    if re.search(DAY_WHICH, t) and re.search(r"cao nhat|nhieu (tien )?nhat|nhiu tien nhat|dat nhat|lon nhat", t) and P is not None:
        return mk("ngay_doanh_thu_cao_nhat", tu=P[0], den=P[1])
    if re.search(r"thang nao", t) and re.search(r"cao nhat|nhieu (tien )?nhat|nhiu tien nhat|lon nhat", t) and len(ms) >= 2:
        return mk("thang_doanh_thu_cao_nhat", thang=ms)
    if re.search(r"tang lien tuc|luon cao hon thang (truoc|trc)", t) and len(ms) >= 2:
        return mk("doanh_thu_tang_lien_tuc", thang=ms)
    # ---- bán chạy, số lượng bán, khách
    if re.search(r"ban chay|hot nhat|ban (duoc |dc )?nhieu nhat", t) and re.search(r"mat hang|mon|san pham|hang nao|\bsp\b|hang gi", t) and P is not None:
        n, sure = _top_n(t, r"(mat hang|mon|san pham|sp)\b")
        if n is None or not sure:
            return None
        return mk("ban_chay", tu=P[0], den=P[1], n=n)
    if re.search(r"(ai|khach nao) .*(nhieu tien nhat|chi nhieu nhat)|khach (mua|chi) nhieu tien nhat", t) and P is not None:
        if re.search(r"\bnhom\b|\bca\b|kenh|so don|nhieu (don|lan)", t):
            return None
        n, sure = _top_n(t, r"khach\b")
        if n is None or not sure:
            return None
        return mk("khach_mua_nhieu_tien_nhat", tu=P[0], den=P[1], n=n)
    if re.search(r"binh quan moi (don|bill|hoa don)|trung binh moi (don|bill|hoa don)|gia tri (trung binh|binh quan) (mot|moi) don", t) and P is not None \
            and not re.search(r"\bnhom\b|khach|kenh|\bca\b|nhan vien|tai quay|dien thoai|online", t):
        return mk("doanh_thu_binh_quan_don", tu=P[0], den=P[1])
    if P is not None and re.search(r"(ban (ra|duoc|dc)|ban) .*(bao nhi\w*|bn|may)\b|(bao nhi\w*|bn|may) (\w+ )?.*ban (ra|duoc|dc)", t) \
            and not re.search(r"tien|doanh thu|dthu|\bdon\b|bill|khach|%|phan tram|nhap", t):
        mh = _product_in(biz, t, t_raw)
        if mh is None:
            return None
        u, _f = L.don_vi_trong_cau(biz.V, mh, t)
        return mk("so_luong_ban", tu=P[0], den=P[1], mat_hang=mh, don_vi=u)
    # ---- doanh thu
    if re.search(r"doanh thu|dthu|revenue|ban (duoc|dc|đc) .*tien|tien ban|thu ve|ban duoc tong cong|tong cong .*tien", t):
        if re.search(r"so (voi|vs)|thay doi|tang (bao nhieu|bn)|giam (bao nhieu|bn)|tang truong", t) and pct and P is not None \
                and re.search(r"thang (lien )?(truoc|trc)|thang lien truoc", t):
            return mk("doanh_thu_doi", tu=P[0], den=P[1])
        if P is None or re.search(r"so (voi|vs)|chenh|hon|kem|tang|giam|trung binh|binh quan|moi (ngay|khach|thang|don|bill)|ca (sang|chieu)|kenh|tien mat|"
                                  r"chuyen khoan|\bcod\b|vi dien tu|zalo|online|nhan vien|khach|ngay le|cuoi tuan|tet|\bgio\b|tai quay|dien thoai|"
                                  r"qua (app|san|web|fb|facebook)|lien nhau|(may|bao nhieu|bn|nhieu|dem) (don|ngay|bill|lan)|so (don|ngay)|ngay nao|thang nao|"
                                  r"tro len|tro xuong|it nhat|toi thieu|≥|>=|lan luot|\bva doanh thu\b|tung (thang|ngay)|gia ban|ban ra voi|"
                                  r"cao nhat|nhieu nhat|lon nhat|thap nhat|it nhat|\bai\b|nguoi ban|tra bang|\bnhom\w*|binh qu|trung bi|ngay\w* nao|(hom|bua)\w* (nao|mo)|ngay mo|"
                                  r"thang\w* nao|\bmoi\w* (don|bill)", t):
            return None
        if re.search(r"(bao nhi\w*|bn\w*|may|nhieu|nhiu|nhieu|tong cong \w+) (kg|binh|chai|lon|hop|thung|cai|goi|vien|bao|cay|met|m2|m3|tui|ly|coc|phan|suat|cuon|bo|doi|chiec)\b", t):
            return None      # hỏi SỐ LƯỢNG theo đơn vị, không phải tiền
        loc = {}
        if re.search(r"ke don", t):
            loc["ke_don"] = not re.search(r"(khong|ko|k) (phai )?(thuoc )?ke don|ngoai thuoc ke don|tru thuoc ke don|khong can ke don|khong ke don", t)
        hc, has = _attr_in(biz, t, "hoat_chat")
        if re.search(r"hoat chat", t):
            if not hc:
                return None
            loc["hoat_chat"] = hc
        if re.search(r"\bnhom\b", t):
            nh, has = _attr_in(biz, t, "nhom")
            if not nh:
                return None
            loc["nhom"] = nh
        if not loc:
            mh = _product_in(biz, t, t_raw)
            if mh is not None:
                loc["ten"] = mh          # tên mặt hàng có trong câu -> doanh thu của mặt hàng đó
            elif re.search(r"(doanh thu|dthu|tien) (cua )?(mon|mat hang|hang|san pham|sp)\b|(doanh thu|dthu) cua (?!(cua hang|shop|tiem|quan|tiệm))", t):
                return None              # câu nói về MỘT mặt hàng mà không tìm ra tên -> không được trả doanh thu cả shop
        msz = re.search(r"\bsize ([sml])\b", t)
        if msz and "ten" not in loc:
            loc["size"] = msz.group(1).upper()
        if loc.get("ke_don") is False and not re.search(r"tru|ngoai|khong", t):
            return None
        return mk("doanh_thu", tu=P[0], den=P[1], loc=loc, ti_trong=pct)
    if re.search(r"(bao nhieu|may|bn|so) (don|bill|hoa don|order)", t) and P is not None and not re.search(
            r"huy|tra|khach|mat hang|tren|duoi|hon|it nhat|tro len|%|tien|\bca\b|kenh|nhan vien|tai quay|dien thoai|online|zalo|cod|chuyen khoan|tien mat|gio|ngay nao|moi ngay|trung binh|co (mon|hang|mat hang|san pham)|"
            r"\bnhom\b|mua (hang|mon|sp|san pham)|\bco (mua|ban|goi)\b|(hom|bua)\w* (nao|mo)", t):
        return mk("so_don", tu=P[0], den=P[1])
    return None


# --------------------------------------------------------------------------- chặn chữ lạ (vocab guard)
# Máy tính chỉ được TỰ QUYẾT khi mọi chữ trong câu đều đã được hiểu: từ vựng câu hỏi train (≥ 2 shop) + tên thực thể đã
# gắn vào đặc tả. Còn chữ lạ (tên món/nhân viên/nhóm không tìm thấy, điều kiện lọc mới…) -> có thể là bộ lọc bị bỏ sót
# -> trả TỔNG cả shop là sai. Khi đó đặc tả hạ xuống tin cậy thấp (chỉ dùng khi đường code đồng ý).
_TOK = re.compile(r"[a-z0-9]+")
_SKIP_SPEC_KEYS = {"q", "intent", "_nguon", "tin_cay", "tu", "den", "X", "n", "N", "thang", "ti_le", "ti_trong", "dem", "theo_hang"}


def spec_entity_tokens(spec):
    out = set()

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k not in _SKIP_SPEC_KEYS:
                    walk(x)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            out.update(_TOK.findall(key(v)))
    walk({k: v for k, v in (spec or {}).items() if k not in _SKIP_SPEC_KEYS})
    return out


def question_tokens(t):
    """Chữ của câu (đã key) trừ số, ngày, kỳ viết tắt (t5, th10, q2…)."""
    out = []
    for w in _TOK.findall(t):
        if w.isdigit() or re.fullmatch(r"(t|th|thg|q|quy|thang|nam|ngay|tuan)\d+|\d+(h|g|k|tr|m|d|ngay|thang)?", w):
            continue
        out.append(w)
    return out


ENTITY_COLS = ("ten_hang", "ma_hang", "nhom", "hoat_chat", "khach_ten", "khach_ma", "nhan_vien", "ho_ten", "ncc", "ncc_ten",
               "dong_xe", "nhan_vien_goi")


def shop_entity_tokens(biz, max_unique=5000):
    """Chữ xuất hiện trong TÊN thực thể của shop (món, mã hàng, nhóm, khách, nhân viên, NCC, dòng xe) — không lấy đơn vị,
    kênh, hình thức thanh toán (không phải bộ lọc 'ẩn')."""
    if getattr(biz, "_ent_tok", None) is not None:
        return biz._ent_tok
    out = set()
    for df in (biz.V or {}).values():
        for c in ENTITY_COLS:
            if c in df.columns:
                u = df[c].dropna().unique()
                if len(u) <= max_unique:
                    for v in u:
                        out.update(_TOK.findall(key(str(v))))
    biz._ent_tok = {w for w in out if not w.isdigit()}
    return biz._ent_tok


def residual_tokens(spec, t, vocab, shop_tokens=None):
    """Chữ trong câu KHÔNG thuộc từ vựng câu hỏi train, KHÔNG thuộc thực thể đã gắn vào đặc tả, nhưng LẠI có trong tên
    thực thể của shop (shop_tokens) = câu nhắc tới một món/nhân viên/nhóm… mà máy tính không gắn được (vd 'Đậu phụ rán
    giòn' khi shop ghi 'Đậu hũ chiên giòn', 'NV001', 'nhóm bia') -> danh sách (rỗng = không thấy bộ lọc bị bỏ sót).
    shop_tokens=None: mọi chữ ngoài từ vựng đều tính (chặt hơn)."""
    if not vocab:
        return []
    ent = spec_entity_tokens(spec)
    toks = [w for w in question_tokens(t) if len(w) > 1]
    unk = [w not in vocab and w not in ent for w in toks]
    out = [w for w, u in zip(toks, unk) if u and (shop_tokens is None or w in shop_tokens)]
    # DOANH THU CẢ SHOP (không lọc): ≥ 2 chữ lạ LIỀN NHAU = cụm tên món chưa gắn được (vd 'phụ rán' trong 'Đậu phụ rán
    # giòn' khi shop ghi 'Đậu hũ chiên giòn') dù không trùng tên nào của shop -> trả doanh thu cả shop là sai
    if (spec or {}).get("intent") == "doanh_thu" and not (spec or {}).get("loc"):
        for i in range(len(toks) - 1):
            if unk[i] and unk[i + 1]:
                out += [toks[i], toks[i + 1]]
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- máy tính
def _num(v, nd=0):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and math.isnan(v):
        return None
    return rhu(float(v), nd)


def _tie_at(vals, n):
    """Hoà ở biên k (hoặc hạng 1 khi n=1): BTC đã LOẠI mọi câu xếp hạng có hoà ở biên khỏi bộ test,
    nên nếu mình tính ra hoà ở đó thì gần như chắc mình đọc/hiểu sai -> đánh dấu để bỏ trống."""
    v = [float(x) for x in vals]
    return len(v) > n and abs(v[n - 1] - v[n]) < 1e-6


def _mark_tie(ans, tie):
    if ans is not None and tie:
        ans = dict(ans, _tie=True)
    return ans


def _sort_items(g, val_col):
    """Xếp giảm dần theo số đo, hoà thì theo MÃ tăng dần (không có mã theo tên) — §9."""
    k2 = "ma_hang" if "ma_hang" in g.columns and g["ma_hang"].notna().all() else "mat_hang"
    return g.sort_values([val_col, k2], ascending=[False, True]).reset_index(drop=True)


def compile_spec(spec, biz, as_of):
    """Đặc tả -> {"type", "answer"} đúng format §9; {"type": "unanswerable"} khi §7; None khi không tính được chắc chắn."""
    if not spec:
        return None
    it = spec.get("intent")
    V = biz.V
    if it == "unanswerable":
        return {"type": "unanswerable", "reason": spec.get("ly_do", "")}
    if not V or it not in ANSWER_TYPE:
        return None
    de = getattr(biz, "data_end", None)
    a, b = spec.get("tu"), spec.get("den")
    X = spec.get("X")
    asof = pd.Timestamp(as_of).normalize()
    unans = {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    if a is not None and _outside(biz, pd.Timestamp(a), pd.Timestamp(b)):
        return unans
    if it in ("thang_doanh_thu_cao_nhat", "doanh_thu_tang_lien_tuc"):
        spans = [_month_span(y, mo, asof) for y, mo in spec["thang"]]
        if any(_outside(biz, x, y) for x, y in spans):
            return unans if all(_outside(biz, x, y) for x, y in spans) else None
        vals = [L.doanh_thu_ky(V, x, y) for x, y in spans]
        if any(v is None for v in vals):
            return None
        if it == "thang_doanh_thu_cao_nhat":
            i = int(np.argmax(vals))
            return _mark_tie({"type": "date", "answer": f"{spec['thang'][i][0]:04d}-{spec['thang'][i][1]:02d}"},
                             _tie_at(sorted(vals, reverse=True), 1))
        return {"type": "boolean", "answer": bool(all(vals[i + 1] > vals[i] for i in range(len(vals) - 1)))}
    r = None
    alt = spec.get("_alt")
    if alt == "all_status" and it == "doanh_thu" and "don" in V:
        o = V["don"]
        ng = pd.to_datetime(o["ngay"], errors="coerce")
        return {"type": "number", "answer": _num(o.loc[(ng >= pd.Timestamp(a)) & (ng <= L._end(b)), "doanh_thu"].sum())}
    if alt == "denominator_valid" and it == "so_don_huy" and "don" in V:
        o = V["don"]
        ng = pd.to_datetime(o["ngay"], errors="coerce")
        x = o[(ng >= pd.Timestamp(a)) & (ng <= L._end(b))]
        nv = int(x["tinh_doanh_thu"].sum())
        return None if not nv else {"type": "percent", "answer": _num((x["nhom_tt"] == "huy").sum() / nv * 100, 2)}
    if alt:
        return None
    if it == "doanh_thu":
        loc = {k: v for k, v in (spec.get("loc") or {}).items() if v is not None}
        v = L.doanh_thu_loc(V, a, b, **loc) if loc else L.doanh_thu_ky(V, a, b)
        if v is None:
            return {"type": "unanswerable", "reason": "shop không ghi thông tin để lọc"} if loc else None
        if spec.get("ti_trong"):
            tot = L.doanh_thu_ky(V, a, b)
            if not loc or not tot:
                return None
            return {"type": "percent", "answer": _num(v / tot * 100, 2)}
        return {"type": "number", "answer": _num(v)}
    if it == "doanh_thu_doi":
        a0, b0 = pd.Timestamp(a), pd.Timestamp(b)
        pa = (a0 - pd.Timedelta(days=1)).replace(day=1)
        pb = min(pa + pd.offsets.MonthEnd(0), pa + pd.Timedelta(days=(b0 - a0).days)) if b0 < a0 + pd.offsets.MonthEnd(0) else pa + pd.offsets.MonthEnd(0)
        if _outside(biz, pa, pb):
            return unans
        cur, prev = L.doanh_thu_ky(V, a0, b0), L.doanh_thu_ky(V, pa, pb)
        if cur is None or not prev:
            return None
        return {"type": "percent", "answer": _num((cur - prev) / prev * 100, 2)}
    if it == "doanh_thu_binh_quan_don":
        return {"type": "number", "answer": _num(L.doanh_thu_binh_quan_don(V, a, b))} if L.doanh_thu_binh_quan_don(V, a, b) is not None else None
    if it == "so_don":
        v = L.so_don_ky(V, a, b)
        return None if v is None else {"type": "number", "answer": int(v)}
    if it == "so_don_huy":
        o = V.get("don")
        if o is None:
            return None
        ng = pd.to_datetime(o["ngay"], errors="coerce")
        x = o[(ng >= pd.Timestamp(a)) & (ng <= L._end(b))]
        nh = int((x["nhom_tt"] == "huy").sum())
        if spec.get("ti_le"):
            return None if not len(x) else {"type": "percent", "answer": _num(nh / len(x) * 100, 2)}
        return {"type": "number", "answer": nh}
    if it == "ngay_doanh_thu_cao_nhat":
        s = L.doanh_thu_theo_ngay(V, a, b)
        if s is None or not len(s):
            return None
        return _mark_tie({"type": "date", "answer": str(s.idxmax().date())}, _tie_at(s.sort_values(ascending=False).values, 1))
    if it == "ban_chay":
        g = L.ban_hang_ky(V, a, b)
        if g is None or not len(g):
            return None
        g = _sort_items(g, "so_luong_co_ban")
        n = int(spec.get("n") or 1)
        tie = _tie_at(g["so_luong_co_ban"].values, n)
        if n == 1:
            return _mark_tie({"type": "string", "answer": str(g.iloc[0]["mat_hang"])}, tie)
        return _mark_tie({"type": "list", "answer": [str(x) for x in g["mat_hang"].head(n)]}, tie)
    if it == "so_luong_ban":
        v = L.so_luong_ban(V, spec["mat_hang"], a, b, spec.get("don_vi"))
        return None if v is None else {"type": "number", "answer": _num(v, 2) if abs(v - round(v)) > 1e-9 else int(round(v))}
    if it == "khach_mua_nhieu_tien_nhat":
        g = L.khach_ky(V, a, b)
        if g is None:
            return None
        if not len(g):
            return {"type": "unanswerable", "reason": "không có khách (chỉ khách lẻ)"}
        n = int(spec.get("n") or 1)
        tie = _tie_at(g["doanh_thu"].values, n)
        if n > 1:
            return _mark_tie({"type": "list", "answer": [str(x) for x in g["khach_hang"].head(n)]}, tie)
        return _mark_tie({"type": "string", "answer": str(g.iloc[0]["khach_hang"])}, tie)
    if it == "tien_nhap":
        v = L.tien_nhap_ky(V, a, b)
        if v is None:
            return {"type": "unanswerable", "reason": "file không ghi tiền nhập"} if "nhap" in V or "phieu_nhap" in V else None
        return {"type": "number", "answer": _num(v)}
    if it == "ncc_nhap_nhieu_nhat":
        pn = V.get("phieu_nhap")
        if pn is None or "tong_tien" not in pn.columns:
            return None
        ng = pd.to_datetime(pn["ngay"], errors="coerce")
        sel = pn[(ng >= pd.Timestamp(a)) & (ng <= L._end(b)) & ~pn["huy"]]
        col = "ncc_ten" if "ncc_ten" in sel.columns and sel["ncc_ten"].notna().any() else "ncc"
        if col not in sel.columns or not len(sel):
            return None
        g = sel.groupby(col)["tong_tien"].sum().reset_index().sort_values(["tong_tien", col], ascending=[False, True])
        return _mark_tie({"type": "string", "answer": str(g.iloc[0][col])}, _tie_at(g["tong_tien"].values, 1))
    if it == "gia_nhap_binh_quan":
        v = L.gia_nhap_binh_quan(V, spec["mat_hang"], a, b, spec.get("don_vi"))
        if v is None:
            nd = V.get("nhap")
            if nd is not None and not ({"thanh_tien"} & set(nd.columns) or {"don_gia"} & set(nd.columns)):
                return {"type": "unanswerable", "reason": "file không ghi giá nhập"}
            return None
        return {"type": "number", "answer": _num(v)}
    if it == "nhap_tru_ban":
        v = L.nhap_tru_ban(V, spec["mat_hang"], a, b)
        return None if v is None else {"type": "number", "answer": _num(v)}
    if it in ("tong_no", "so_khach_no", "top_khach_no", "tuoi_no", "no_qua_n_ngay"):
        if X is None:
            return None
        if _outside(biz, pd.Timestamp("1900-01-01"), pd.Timestamp(X)):
            return unans
        if it == "tong_no":
            v = L.tong_no(V, X)
            return None if v is None else {"type": "number", "answer": _num(v)}
        if it == "so_khach_no":
            v = L.so_khach_con_no(V, X)
            return None if v is None else {"type": "number", "answer": int(v)}
        if it == "top_khach_no":
            tb = L.no_theo_khach(V, X)
            if tb is None:
                return None
            n = int(spec.get("n") or 3)
            return _mark_tie({"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                                          "rows": [[str(r.khach_hang), _num(r.so_tien_no)] for r in tb.head(n).itertuples()]}},
                             _tie_at(tb["so_tien_no"].values, n))
        if it == "tuoi_no":
            v = L.tuoi_no_nhom(V, X)
            return None if v is None else {"type": "list", "answer": [_num(x) for x in v]}
        v = L.no_qua_n_ngay(V, X, int(spec["n"]))
        return None if v is None else {"type": "number", "answer": _num(v)}
    if it == "no_them_tra_no":
        v = L.no_them_tra_no(V, a, b, spec.get("khach"))
        return None if v is None else {"type": "list", "answer": [_num(x) for x in v]}
    if it == "thu_no":
        v = L.thu_no_ky(V, a, b)
        return None if v is None else {"type": "number", "answer": _num(v)}
    if it in ("ton", "ton_don_vi_lon"):
        if X is None or not spec.get("mat_hang"):
            return None
        if it == "ton":
            v = L.ton_tai_ngay(V, spec["mat_hang"], X, de)
            if v is None:
                return {"type": "unanswerable", "reason": "shop không ghi tồn / nhập"} if not any(k in V for k in ("ton", "the_kho", "nhap", "lo_nhap")) else None
            return {"type": "number", "answer": _num(v)}
        v = L.ton_don_vi_lon(V, spec["mat_hang"], spec["don_vi"], X, de)
        if v is None:
            return {"type": "unanswerable", "reason": "shop không ghi tồn / nhập"} if not any(k in V for k in ("ton", "the_kho", "nhap", "lo_nhap")) else None
        return {"type": "list", "answer": v}
    if it == "so_mat_hang_het_hang":
        v = L.so_mat_hang_het_hang(V, X, de)
        return None if v is None else {"type": "number", "answer": int(v)}
    if it == "gia_tri_ton_kho":
        v = L.gia_tri_ton_kho(V, X, de)
        if v is None:
            return {"type": "unanswerable", "reason": "shop không ghi tồn hoặc giá vốn"} if not ("hang" in V and "gia_von" in V["hang"].columns and any(k in V for k in ("ton", "the_kho", "nhap"))) else None
        return {"type": "number", "answer": _num(v)}
    if it == "lo_gan_het_han_nhat":
        if not spec.get("mat_hang"):
            return None
        r = L.lo_gan_het_han_nhat(V, spec["mat_hang"], X or asof)
        return None if not r else {"type": "date", "answer": str(pd.Timestamp(r["han_dung"]).date())}
    if it == "lo_het_han_trong":
        x = L.lo_het_han_trong(V, asof, int(spec["N"]))
        if x is None:
            return {"type": "unanswerable", "reason": "shop không ghi lô / hạn dùng"} if "lo_nhap" not in V else None
        dem = spec.get("dem")
        if dem == "tien_von":
            return {"type": "number", "answer": _num(x["tien_von"].sum())} if "tien_von" in x.columns else None
        if dem == "lo":
            return {"type": "number", "answer": int(len(x))}
        if dem == "mat_hang":
            return {"type": "number", "answer": int(x["ma_hang"].nunique())} if "ma_hang" in x.columns else None
        # liệt kê hàng theo hạn dùng: hạn sớm nhất của mặt hàng trước, cùng hạn thì theo mã (§9)
        if "ma_hang" in x.columns:
            g = x.groupby("ma_hang").agg(han=("han_dung", "min"), ten=("ten_hang", "first")).reset_index().sort_values(["han", "ma_hang"])
            return {"type": "list", "answer": [str(v) for v in g["ten"]]}
        return None
    if it == "lai_gop":
        if spec.get("theo_hang"):
            g = L.lai_gop_ky(V, a, b, True)
            if g is None or not len(g):
                return None
            return _mark_tie({"type": "string", "answer": str(g.iloc[0]["mat_hang"])}, _tie_at(g["lai_gop"].values, 1))
        v = L.lai_gop_ky(V, a, b)
        return {"type": "unanswerable", "reason": "file không ghi giá vốn"} if v is None and "dong" in V else (None if v is None else {"type": "number", "answer": _num(v)})
    if it == "lai_rong":
        v = L.lai_rong_ky(V, a, b)
        return None if v is None else {"type": "number", "answer": _num(v)}
    return r


# --------------------------------------------------------------------------- các cách hiểu khác (kiểm toán khi hai đường lệch)
def alternative_readings(spec):
    """Sinh các đặc tả 'hiểu sai hay gặp' (xem lexicon/error_catalog.yaml) để giải thích con số của đường kia."""
    if not spec or spec.get("intent") in (None, "unanswerable"):
        return []
    it = spec["intent"]
    alts = []
    if it in ("doanh_thu", "so_don", "ban_chay", "so_luong_ban", "doanh_thu_binh_quan_don"):
        alts.append(("tinh_ca_don_tra_huy", dict(spec, _alt="all_status")))
    if it == "so_don_huy" and spec.get("ti_le"):
        alts.append(("mau_so_chi_don_tinh_doanh_thu", dict(spec, _alt="denominator_valid")))
    if it == "so_luong_ban" and spec.get("don_vi"):
        alts.append(("khong_quy_doi_don_vi", dict(spec, don_vi=None)))
    if it in ("no_qua_n_ngay",):
        alts.append(("gom_ca_N", dict(spec, n=int(spec["n"]) - 1)))
    if it in ("doanh_thu",) and (spec.get("loc") or {}).get("ke_don") is not None:
        alts.append(("dao_ke_don", dict(spec, loc=dict(spec["loc"], ke_don=not spec["loc"]["ke_don"]))))
    return alts


# --------------------------------------------------------------------------- đặc tả do LLM viết
SPEC_PROMPT = """Bạn phân tích câu hỏi kinh doanh thành ĐẶC TẢ JSON (không tính toán).
Chọn đúng MỘT intent trong danh mục; không có intent phù hợp thì "intent": "khac".
Danh mục intent:
{intents}
Trường: intent, tu, den (YYYY-MM-DD, kỳ phát sinh), X (YYYY-MM-DD, cho số còn lại 'đến hết ngày X'), thang (danh sách "YYYY-MM"),
mat_hang (đúng tên trong dữ liệu nếu câu nêu), don_vi, n, khach, loc (nhom / hoat_chat / ke_don / size), ti_trong, ti_le, theo_hang, dem,
doc_khac (danh sách các CÁCH HIỂU KHÁC có thể có của câu, mỗi cách một câu ngắn; câu rõ nghĩa thì []).
Hôm nay: {as_of}. Kỳ đã giải sẵn: {periods}
{examples}Câu hỏi: {q}
Câu đã sửa chữ: {q_norm}
Chỉ trả JSON một dòng."""


def spec_messages(q, q_norm, as_of, periods_txt, examples=""):
    """examples: vài câu TRAIN gần nhất + đặc tả đã kiểm (retriever.few_shot_text) — ví dụ mẫu, không phải đáp án."""
    intents = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    ex = f"Ví dụ câu train tương tự (đặc tả đúng, chỉ để tham khảo cách viết; ngày/tên phải lấy từ câu hỏi hiện tại):\n{examples}\n" if examples else ""
    return [{"role": "user", "content": SPEC_PROMPT.format(intents=intents, as_of=as_of, periods=periods_txt or "(không có)", q=q,
                                                           q_norm=q_norm or q, examples=ex)}]


def parse_llm_spec(text, biz, as_of):
    """JSON của LLM -> đặc tả đã kiểm (ngày hợp lệ, intent có trong danh mục, mặt hàng có thật). Sai -> None."""
    import json
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    it = d.get("intent")
    if it not in INTENTS:
        return None
    out = {"intent": it, "_nguon": "llm", "doc_khac": d.get("doc_khac") or []}
    for k in ("tu", "den", "X"):
        if d.get(k):
            try:
                out[k] = pd.Timestamp(str(d[k])[:10])
            except Exception:
                return None
    if d.get("thang"):
        try:
            out["thang"] = [(int(str(x)[:4]), int(str(x)[5:7])) for x in d["thang"]]
        except Exception:
            return None
    if d.get("mat_hang"):
        ma, ten = L.tim_hang(biz.V, str(d["mat_hang"]))
        if ma is None and ten is None:
            return None
        out["mat_hang"] = ten or ma
    for k in ("don_vi", "khach", "dem"):
        if d.get(k):
            out[k] = str(d[k])
    for k in ("n", "N"):
        if d.get(k) is not None:
            try:
                out[k] = int(d[k])
            except Exception:
                return None
    for k in ("ti_trong", "ti_le", "theo_hang"):
        out[k] = bool(d.get(k))
    if isinstance(d.get("loc"), dict):
        out["loc"] = {k: v for k, v in d["loc"].items() if k in ("nhom", "hoat_chat", "ke_don", "size", "ten") and v not in (None, "")}
    return out


def same_answer(x, y):
    """So hai đáp án theo cách chấm §9 (số khớp tuyệt đối, chuỗi bỏ hoa thường/dấu câu, bảng không xét thứ tự dòng)."""
    if not x or not y or x.get("type") != y.get("type"):
        return False
    if x["type"] == "unanswerable":
        return True
    a, b = x.get("answer"), y.get("answer")

    def norm(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            return round(float(v), 6)
        if isinstance(v, str):
            return re.sub(r"[\s\W_]+", "", key(v))
        if isinstance(v, list):
            return [norm(z) for z in v]
        if isinstance(v, dict):
            return sorted(map(str, (norm(r) for r in v.get("rows", []))))
        return v
    return norm(a) == norm(b)
