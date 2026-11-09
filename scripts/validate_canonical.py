"""
Kiểm BỘ BẢNG CHUẨN V + hàm nghiệp vụ (ledger.py) bằng đáp án TRAIN — CHỈ dùng train (luật 6: không dùng public/private).

    python scripts/validate_canonical.py --train /path/train [--cache /tmp/cache] [--show_bad]

Với từng dạng câu dễ nhận ra bằng mẫu chữ, tính thẳng từ V / ledger rồi so với đáp án train:
  doanh thu tháng, tiền nhập tháng / NCC nhập nhiều nhất, công nợ (tổng, số khách, top khách, tuổi nợ, quá N ngày,
  thu nợ trong tháng, nợ thêm / trả nợ của một khách), tồn kho cuối ngày (số / [chẵn, lẻ]), lô và hạn dùng.
Mẫu chữ CHỈ để chọn câu kiểm — pipeline không dùng các mẫu này để trả lời.
Lần chạy 08/11 (tối): xem docs/TOM_TAT_PIPELINE.md.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402

import datamate_pipeline as d  # noqa: E402
import ledger as L  # noqa: E402

BS = {}


def biz(train, bid, cache):
    if bid not in BS:
        BS[bid] = d.Business(os.path.join(train, "data", bid), cache_dir=cache)
    return BS[bid]


def num_eq(a, b, tol=1.01):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(num_eq(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < tol
    return a == b


def month_range(t, asof):
    m = re.search(r"thang (\d{1,2})/(\d{4})", t)
    if m:
        a = pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
        return a, min(a + pd.offsets.MonthEnd(0), asof)
    if "thang nay" in t:
        return asof.replace(day=1), asof
    return None, None


def period_range(t, asof):
    """tháng M/YYYY, 'tháng M năm YYYY', tháng này, tuần trước (thứ Hai–Chủ nhật)."""
    m = re.search(r"thang (\d{1,2})(?:/| nam )(\d{4})", t)
    if m:
        a = pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
        return a, min(a + pd.offsets.MonthEnd(0), asof)
    if "thang nay" in t:
        return asof.replace(day=1), asof
    if re.search(r"tuan (truoc|trc|roi)", t):
        s0 = asof.normalize() - pd.Timedelta(days=asof.weekday() + 7)
        return s0, s0 + pd.Timedelta(days=6)
    return None, None


def run(train, cache, show_bad):
    qs = [json.loads(x) for x in open(os.path.join(train, "questions.jsonl"), encoding="utf-8")]
    an = {json.loads(x)["id"]: json.loads(x).get("answer") for x in open(os.path.join(train, "answers.jsonl"), encoding="utf-8")}
    res = collections.defaultdict(lambda: [0, 0])
    bad = []

    def mark(kind, ok, q, g, v):
        res[kind][0] += bool(ok)
        res[kind][1] += 1
        if not ok:
            bad.append((kind, q["business_id"], q["question"][:70], str(g)[:80], str(v)[:80]))

    for q in qs:
        t, g, asof = d.key(q["question"]), an.get(q["id"]), pd.Timestamp(q["as_of_date"])
        B = None
        a, b = month_range(t, asof)
        a2, b2 = period_range(t, asof)
        mx = re.search(r"het ngay (\d{1,2})/(\d{1,2})/(\d{4})", t)
        X = pd.Timestamp(int(mx.group(3)), int(mx.group(2)), int(mx.group(1))) if mx else None
        try:
            # ---- giá nhập bình quân / doanh thu bình quân đơn / giá trị tồn / hết hàng / lãi gộp, lãi ròng
            if "binh quan" in t and "nhap" in t and a2 is not None:
                B = biz(train, q["business_id"], cache)
                u = L.don_vi_lon_trong_cau(B.V, q["question"], q["question"])
                if u is None:
                    mu = re.search(r"mot (\w+)", t)
                    u = mu.group(1) if mu else None
                v = L.gia_nhap_binh_quan(B.V, q["question"], a2, b2, u)
                v = None if v is None else d.rhu(v, 0)
                mark("gia_nhap_binh_quan", (v is None and g is None) or num_eq(v, g), q, g, v)
                continue
            if re.search(r"binh quan moi (don|bill|order)", t) and a2 is not None:
                B = biz(train, q["business_id"], cache)
                v = L.doanh_thu_binh_quan_don(B.V, a2, b2)
                v = None if v is None else d.rhu(v, 0)
                mark("dt_binh_quan_don", num_eq(v, g), q, g, v)
                continue
            if "gia von" in t and "kho" in t:
                B = biz(train, q["business_id"], cache)
                v = L.gia_tri_ton_kho(B.V, asof, B.data_end)
                mark("gia_tri_ton_kho", (v is None and g is None) or num_eq(v, g), q, g, v)
                continue
            if "het hang" in t:
                B = biz(train, q["business_id"], cache)
                v = L.so_mat_hang_het_hang(B.V, asof, B.data_end)
                mark("so_mat_hang_het_hang", (v is None and g is None) or num_eq(v, g), q, g, v)
                continue
            if re.search(r"lai gop|lai rong", t) and a2 is not None:
                B = biz(train, q["business_id"], cache)
                if re.search(r"hang nao|mat hang nao", t):
                    x = L.lai_gop_ky(B.V, a2, b2, True)
                    v = None if x is None else x.iloc[0]["mat_hang"]
                    mark("hang_lai_gop_cao_nhat", v == g, q, g, v)
                else:
                    v = L.lai_rong_ky(B.V, a2, b2) if "lai rong" in t else L.lai_gop_ky(B.V, a2, b2)
                    mark("lai_rong" if "lai rong" in t else "lai_gop", (v is None and g is None) or num_eq(v, g), q, g, v)
                continue
            # ---- doanh thu tháng (câu đơn giản, không điều kiện phụ)
            if (re.search(r"doanh thu", t) and re.search(r"thang (\d{1,2})/(\d{4})", t) and isinstance(g, (int, float))
                    and not re.search(r"tu \d|hon|kem|xu huong|ti trong|ty trong|binh quan|qua|tra bang|k ke|khong ke|%|so voi|tang|giam|"
                                      r"khach|nhom|mat hang|san pham|ca |kenh|nhan vien|tien mat|chuyen khoan|cod|trung binh|moi|ngay|tuan|"
                                      r"thuoc|size|chenh|tet", t)):
                B = biz(train, q["business_id"], cache)
                mark("doanh_thu_thang", num_eq(L.doanh_thu_ky(B.V, a, b), g, 0.5), q, g, L.doanh_thu_ky(B.V, a, b))
            # ---- nhập hàng
            elif re.search(r"tien nhap|phieu nhap|nhap hang het|bo ra.*nhap", t) and a is not None and not re.search(r"mat hang|binh quan|trung binh", t):
                B = biz(train, q["business_id"], cache)
                if "nha cung cap" in t:
                    pn = B.V.get("phieu_nhap")
                    v = None
                    if pn is not None and "tong_tien" in pn:
                        ng = pd.to_datetime(pn["ngay"], errors="coerce")
                        sel = pn[(ng >= a) & (ng <= L._end(b)) & ~pn["huy"]]
                        col = "ncc_ten" if "ncc_ten" in sel else "ncc"
                        v = sel.groupby(col)["tong_tien"].sum().idxmax() if len(sel) else None
                    mark("nhap_ncc", str(v) == str(g), q, g, v)
                else:
                    v = L.tien_nhap_ky(B.V, a, b)
                    mark("tien_nhap", (v is None and g is None) or (g is not None and num_eq(v, g, 0.5)), q, g, v)
            # ---- công nợ
            elif re.search(r"\bno\b|cong no", t):
                B = biz(train, q["business_id"], cache)
                V = B.V
                if X is not None and re.search(r"tuoi no|0.7", t):
                    mark("no_tuoi", num_eq(L.tuoi_no_nhom(V, X), g), q, g, L.tuoi_no_nhom(V, X))
                elif X is not None and re.search(r"(qua|hon) (\d+) ngay", t):
                    n = int(re.search(r"(qua|hon) (\d+) ngay", t).group(2))
                    mark("no_qua_n", num_eq(L.no_qua_n_ngay(V, X, n), g), q, g, L.no_qua_n_ngay(V, X, n))
                elif X is not None and isinstance(g, dict):
                    n = int(re.search(r"(\d+) khach", t).group(1))
                    tb = L.no_theo_khach(V, X)
                    v = None if tb is None else [[r.khach_hang, r.so_tien_no] for r in tb.head(n).itertuples()]
                    ok = v is not None and len(v) == len(g["rows"]) and all(str(x[0]) == str(y[0]) and abs(x[1] - y[1]) < 1.01 for x, y in zip(v, g["rows"]))
                    mark("no_top_khach", ok, q, g["rows"][:2], (v or [])[:2])
                elif X is not None and re.search(r"bao nhi\w* khach|so khach|bn khach", t):
                    mark("no_so_khach", num_eq(L.so_khach_con_no(V, X), g), q, g, L.so_khach_con_no(V, X))
                elif X is not None:
                    mark("no_tong", (g is None and L.tong_no(V, X) is None) or num_eq(L.tong_no(V, X), g), q, g, L.tong_no(V, X))
                elif a is not None and re.search(r"no them|mua chiu them", t) and isinstance(g, list):
                    names = set()
                    for c in ("khach_ten", "khach_ten_dm"):
                        if c in V["don"]:
                            names |= set(V["don"][c].dropna().astype(str))
                    hit = [n for n in names if len(d.key(n)) > 5 and d.key(n) in t]
                    v = L.no_them_tra_no(V, a, b, max(hit, key=len)) if hit else None
                    mark("no_them_tra_no", num_eq(v, g), q, g, v)
                elif a is not None:
                    v = L.thu_no_ky(V, a, b)
                    mark("thu_no_thang", (g is None and v is None) or num_eq(v, g), q, g, v)
            # ---- lô, hạn dùng
            elif re.search(r"het han", t) or re.search(r"\blo\b.*gan het han", t):
                B = biz(train, q["business_id"], cache)
                V = B.V
                mN = re.search(r"(\d+) ngay toi", t)
                mD = re.search(r"het (?:ngay )?(\d{1,2})/(\d{1,2})/(\d{4})", t)
                N = int(mN.group(1)) if mN else ((pd.Timestamp(int(mD.group(3)), int(mD.group(2)), int(mD.group(1))) - asof).days if mD else None)
                if "gan het han nhat" in t:
                    r = L.lo_gan_het_han_nhat(V, q["question"], asof)
                    v = str(r["han_dung"].date()) if r else None
                    mark("lo_gan_het_han", v == g, q, g, v)
                elif N is not None:
                    x = L.lo_het_han_trong(V, asof, N)
                    if x is None:
                        v = None
                    elif "tien von" in t:
                        v = float(x["tien_von"].sum())
                    elif "may lo" in t:
                        v = len(x)
                    elif "may mat hang" in t:
                        v = x["ma_hang"].nunique()
                    else:
                        v = sorted(x["ten_hang"].dropna().unique().tolist())
                    gg = sorted(g) if isinstance(g, list) else g
                    mark("lo_het_han", num_eq(v, gg), q, g, v)
            # ---- tồn kho cuối ngày
            elif (re.search(r"ton kho|kho con|con ton|ton cuoi|trong kho con|con bao nhieu", t) and re.search(r"cuoi ngay|het hom nay|cuoi bua|hom nay", t)
                  and "het hang" not in t):
                B = biz(train, q["business_id"], cache)
                V = B.V
                if isinstance(g, list) or re.search(r"chan|nguyen", t):
                    u = L.don_vi_lon_trong_cau(V, q["question"], q["question"])
                    v = L.ton_don_vi_lon(V, q["question"], u, asof, B.data_end) if u else None
                else:
                    v = L.ton_tai_ngay(V, q["question"], asof, B.data_end)
                mark("ton_kho", (v is None and g is None) or num_eq(v, g, 0.5), q, g, v)
        except Exception as e:  # lỗi hàm = sai, ghi lại để sửa
            mark("LỖI", False, q, g, f"{type(e).__name__}: {e}")
    for k, (o, n) in sorted(res.items()):
        print(f"{k:<18}: khớp {o}/{n}")
    if show_bad:
        for x in bad:
            print("  ✗", x)
    return res


def run_sales(train, cache, show_bad):
    """Bán hàng: doanh thu theo mặt hàng / nhóm / hoạt chất / kê đơn (cả tỉ trọng), số lượng bán theo đơn vị hỏi,
    bán chạy, khách mua nhiều tiền nhất, ngày bán nhiều tiền nhất, nhập trừ bán."""
    qs = [json.loads(x) for x in open(os.path.join(train, "questions.jsonl"), encoding="utf-8")]
    an = {json.loads(x)["id"]: json.loads(x).get("answer") for x in open(os.path.join(train, "answers.jsonl"), encoding="utf-8")}
    res = collections.defaultdict(lambda: [0, 0])
    bad = []
    for q in qs:
        t=d.key(q['question']); g=an[q['id']]; asof=pd.Timestamp(q['as_of_date']); a,b=period_range(t,asof)
        if a is None: continue
        k=v=None
        try:
            if re.search(r'ngay nao .*(nhieu tien|doanh thu cao)',t):
                B=biz(train,q['business_id'],cache); s=L.doanh_thu_theo_ngay(B.V,a,b); k='ngay_max'; v=str(s.idxmax().date()) if s is not None and len(s) else None
            elif re.search(r'(ban chay|hot|ban (duoc )?nhieu nhat)',t) and re.search(r'mat hang|mon|san pham|hang nao',t) and isinstance(g,str):
                B=biz(train,q['business_id'],cache); x=L.ban_hang_ky(B.V,a,b); k='ban_chay'; v=None if x is None or not len(x) else x.iloc[0]['mat_hang']
            elif re.search(r'thuoc ke don',t) and re.search(r'tien|doanh thu|dthu',t) and isinstance(g,(int,float)) and '%' not in t and 'ti le' not in t:
                B=biz(train,q['business_id'],cache); k='ke_don'; v=L.doanh_thu_loc(B.V,a,b,ke_don=True)
            elif re.search(r'hoat chat (.+?) (ban|thang|trong|duoc)',t) and isinstance(g,(int,float)) and re.search(r'tien|doanh thu',t):
                B=biz(train,q['business_id'],cache); h=B.V.get('hang')
                hcs=sorted(set(h['hoat_chat'].dropna().astype(str)),key=len,reverse=True) if h is not None and 'hoat_chat' in h else []
                hit=[x for x in hcs if d.key(x) in t]; k='hoat_chat'; v=L.doanh_thu_loc(B.V,a,b,hoat_chat=hit[0]) if hit else 'không thấy'
            elif re.search(r'nhom',t) and re.search(r'tien|doanh thu|dthu',t) and isinstance(g,(int,float)) and not re.search(r'%|ti trong|ty trong|khach',t):
                B=biz(train,q['business_id'],cache); h=B.V.get('hang')
                ns=sorted(set(h['nhom'].dropna().astype(str)),key=len,reverse=True) if h is not None and 'nhom' in h else []
                hit=[x for x in ns if d.key(x) in t]; k='nhom'; v=L.doanh_thu_loc(B.V,a,b,nhom=hit[0]) if hit else 'không thấy'
            elif re.search(r'(ai|khach nao) mua .*nhieu tien nhat',t) and isinstance(g,str):
                B=biz(train,q['business_id'],cache); x=L.khach_ky(B.V,a,b); k='khach_top'; v=None if x is None or not len(x) else x.iloc[0]['khach_hang']
            elif re.search(r'nhap tru (so )?ban',t):
                B=biz(train,q['business_id'],cache); k='nhap_tru_ban'; v=L.nhap_tru_ban(B.V,q['question'],a,b); v=None if v is None else d.rhu(v,0)
            elif re.search(r'(ban (ra|duoc|dc|đc)|so) .*(bao nhi|bn|may|nhieu|la)',t) and isinstance(g,(int,float)) and not re.search(r'tien|doanh thu|dthu|don|bill|khach|%',t):
                B=biz(train,q['business_id'],cache); ma,tn=L.tim_hang(B.V,q['question'])
                if tn or ma:
                    u,_f=L.don_vi_trong_cau(B.V,q['question'],q['question'])
                    k='so_luong_ban'; v=L.so_luong_ban(B.V,tn or ma,a,b,u)
                    if re.search(r'size [ml]\b',t): k=None
            elif re.search(r'(mat hang|mon|hang) .*(thu ve|ban duoc|ban dc|tien ban)|tien ban',t) and isinstance(g,(int,float)) and not re.search(r'nhom|hoat chat|ke don|%',t):
                B=biz(train,q['business_id'],cache); ma,tn=L.tim_hang(B.V,q['question'])
                if tn or ma: k='dt_mat_hang'; v=L.doanh_thu_loc(B.V,a,b,ten=tn or ma)
        except Exception as e: v=f'ERR {e}'
        if k in ('ke_don','nhom','hoat_chat') and isinstance(v,(int,float)):
            if re.search(r'phan tram|%|ti trong|ty trong',t):
                tot=L.doanh_thu_ky(B.V,a,b); v=d.rhu(v/tot*100,2) if tot else None
            elif re.search(r'tru thuoc ke don|con lai',t):
                v=L.doanh_thu_loc(B.V,a,b,ke_don=False)
        if k is None: continue
        ok=(v==g) or (isinstance(v,(int,float)) and isinstance(g,(int,float)) and abs(v-g)<1.01) or (v is None and g is None)
        res[k][0]+=ok; res[k][1]+=1
        if not ok: bad.append((k,q['business_id'],q['question'][:70],str(g)[:60],str(v)[:60]))
    for k, (o, n) in sorted(res.items()):
        print(f"{k:<18}: khớp {o}/{n}")
    if show_bad:
        for x in bad:
            print("  ✗", x)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--show_bad", action="store_true")
    a = ap.parse_args()
    run(a.train, a.cache, a.show_bad)
    run_sales(a.train, a.cache, a.show_bad)
