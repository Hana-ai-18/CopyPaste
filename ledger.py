"""Hàm nghiệp vụ dùng chung trên BỘ BẢNG CHUẨN V (công nợ, nhập hàng, doanh thu kỳ).

Mọi hàm chỉ đọc V (đã chuẩn hoá ở datamate_pipeline.build_canonical) và theo rulebook §1, §4, §5.
Không có logic theo mã câu hỏi / mã shop. Hàm trả None khi dữ liệu không đủ (→ câu 'không trả lời được').
"""
import re
import unicodedata

import numpy as np
import pandas as pd


def _k(s):
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(ch for ch in s if unicodedata.category(ch) != "Mn")).strip()


COD_PAT = re.compile(r"\bcod\b|thu ho")
SAN_PAT = re.compile(r"\b(san|shopee|lazada|tiki|tiktok|tmdt|sendo|marketplace)\b")
DEBT_PAY_PAT = re.compile(r"ghi no|cong no|\bdebt\b|tra sau|mua chiu|\bno\b")


def _sid(v):
    if v is None or (isinstance(v, float) and v != v) or v is pd.NA:
        return np.nan
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def _none(v):
    return None if v is None or (isinstance(v, float) and v != v) or v is pd.NA else v


def _end(x):
    """Hết ngày x (23:59:59.999)."""
    return pd.Timestamp(x).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def _cong_no_base(V):
    """Đơn được tính vào công nợ (§4): không huỷ; bỏ COD và đơn bán qua sàn TMĐT; đơn trả hàng VẪN tính."""
    if "don" not in V or "tong_don" not in V["don"].columns:
        return None
    o = V["don"].copy()
    o["ngay"] = pd.to_datetime(o["ngay"], errors="coerce")
    m = o["nhom_tt"].ne("huy")
    if "httt" in o.columns:
        m &= ~o["httt"].astype(str).map(_k).str.contains(COD_PAT)
    if "kenh" in o.columns:
        m &= ~o["kenh"].astype(str).map(_k).str.contains(SAN_PAT)
    return o[m]


def _payments(V):
    if "thu" not in V or not {"ma_don", "so_tien"} <= set(V["thu"].columns):
        return None
    t = V["thu"].copy()
    t["ngay_thu"] = pd.to_datetime(t.get("ngay_thu"), errors="coerce")
    t["so_tien"] = pd.to_numeric(t["so_tien"], errors="coerce").fillna(0)
    return t


def cong_no_don(V, X):
    """Nợ từng đơn tính đến HẾT ngày X. Cột: ma_don, ngay, khach_ma, khach_ten, khach_le, tong_don, da_tra, con_no, tuoi_no.
    Chỉ giữ đơn còn nợ > 0. Tuổi nợ = số ngày từ NGÀY ĐẶT đến X (đặt hôm nay = 0)."""
    o = _cong_no_base(V)
    if o is None:
        return None
    E = _end(X)
    o = o[o["ngay"] <= E]
    t = _payments(V)
    if t is not None:
        paid = t[t["ngay_thu"] <= E].groupby("ma_don")["so_tien"].sum()
        o = o.assign(da_tra=o["ma_don"].map(paid).fillna(0))
    elif "da_tra_file" in o.columns:
        # không có bảng thu: chỉ biết số đã trả tính đến CUỐI dữ liệu (không biết theo ngày)
        o = o.assign(da_tra=pd.to_numeric(o["da_tra_file"], errors="coerce").fillna(0))
    elif "con_no_file" in o.columns:
        o = o.assign(da_tra=o["tong_don"] - pd.to_numeric(o["con_no_file"], errors="coerce").fillna(0))
    else:
        return None
    o = o.assign(con_no=(pd.to_numeric(o["tong_don"], errors="coerce").fillna(0) - o["da_tra"]).round(6))
    o = o[o["con_no"] > 0.5]
    o = o.assign(tuoi_no=(pd.Timestamp(X).normalize() - o["ngay"].dt.normalize()).dt.days)
    keep = [c for c in ("ma_don", "ngay", "khach_ma", "khach_ten", "khach_ten_dm", "khach_le", "tong_don", "da_tra", "con_no", "tuoi_no") if c in o.columns]
    return o[keep].reset_index(drop=True)


def _khach_key(df):
    """§3: có mã khách → theo MÃ; không có mã → theo tên (bỏ dấu/viết hoa: cùng cách viết). Khách lẻ → NaN."""
    if "khach_ma" in df.columns:
        k = df["khach_ma"].map(_sid)
        k = k.where(~k.str.lower().isin(["", "nan", "none", "null"]))
    elif "khach_ten" in df.columns:
        k = df["khach_ten"].astype(str).map(_k)
        k = k.where(~k.isin(["", "nan", "none", "khach le", "khach vang lai", "vang lai", "le"]))
    else:
        return pd.Series(np.nan, index=df.index)
    if "khach_le" in df.columns:
        k = k.where(~df["khach_le"].fillna(False).astype(bool))
    return k


def _khach_name(df):
    for c in ("khach_ten", "khach_ten_dm"):
        if c in df.columns and df[c].notna().any():
            return df[c]
    return df.get("khach_ma", pd.Series(np.nan, index=df.index))


def ten_khach_chuan(V):
    """khach_key -> tên hiển thị. Ưu tiên tên trong DANH MỤC khách (khach_ten_dm); không có thì tên ghi trên đơn
    hay gặp nhất của khách đó (hoà: bản viết hoa đầu từ)."""
    o = V["don"]
    k = _khach_key(o)
    out = {}
    if "khach_ten_dm" in o.columns and o["khach_ten_dm"].notna().any():
        out = o.assign(_k=k).dropna(subset=["_k", "khach_ten_dm"]).groupby("_k")["khach_ten_dm"].first().to_dict()
    if "khach_ten" in o.columns:
        t = o.assign(_k=k, _n=o["khach_ten"].map(_sid)).dropna(subset=["_k"])
        t = t[~t["_n"].str.lower().isin(["", "nan", "none"])]
        for kk, grp in t.groupby("_k")["_n"]:
            if kk in out:
                continue
            vc = grp.value_counts()
            top = vc[vc == vc.max()].index.tolist()
            out[kk] = sorted(top, key=lambda n: (not all(w[:1].isupper() for w in n.split()), n))[0]
    if "khach_ma" in o.columns:
        for kk in k.dropna().unique():
            out.setdefault(kk, kk)
    return out


def tong_no(V, X):
    d = cong_no_don(V, X)
    return None if d is None else float(d["con_no"].sum())


def no_theo_khach(V, X):
    """Bảng nợ theo khách tại hết ngày X (không có dòng khách lẻ), giảm dần. Cột: khach_hang, so_tien_no, khach_key."""
    d = cong_no_don(V, X)
    if d is None:
        return None
    d = d.assign(khach_key=_khach_key(d))
    d = d[d["khach_key"].notna()]
    g = d.groupby("khach_key").agg(so_tien_no=("con_no", "sum")).reset_index()
    g["khach_hang"] = g["khach_key"].map(ten_khach_chuan(V))
    return g.sort_values(["so_tien_no", "khach_hang"], ascending=[False, True]).reset_index(drop=True)


def so_khach_con_no(V, X):
    g = no_theo_khach(V, X)
    return None if g is None else int((g["so_tien_no"] > 0.5).sum())


def tuoi_no_nhom(V, X, moc=(7, 15)):
    """Nợ theo nhóm tuổi [0..moc0], [moc0+1..moc1], > moc1 (mặc định 0–7, 8–15, trên 15 ngày)."""
    d = cong_no_don(V, X)
    if d is None:
        return None
    a, b = moc
    return [float(d.loc[d["tuoi_no"] <= a, "con_no"].sum()), float(d.loc[(d["tuoi_no"] > a) & (d["tuoi_no"] <= b), "con_no"].sum()),
            float(d.loc[d["tuoi_no"] > b, "con_no"].sum())]


def no_qua_n_ngay(V, X, n):
    """Tổng còn nợ của các đơn đặt đã HƠN n ngày (tuổi nợ > n)."""
    d = cong_no_don(V, X)
    return None if d is None else float(d.loc[d["tuoi_no"] > n, "con_no"].sum())


def _ghi_no(V):
    """Đơn ghi nợ (§4 'Nợ thêm, trả nợ'): đơn tính công nợ, có phần tổng đơn CHƯA trả ngay trong ngày mua."""
    o = _cong_no_base(V)
    t = _payments(V)
    if o is None or t is None:
        return None, None
    t = t.merge(o[["ma_don", "ngay"]], on="ma_don", how="inner")
    t["cung_ngay"] = t["ngay_thu"].dt.normalize() <= t["ngay"].dt.normalize()
    tra_ngay = t[t["cung_ngay"]].groupby("ma_don")["so_tien"].sum()
    o = o.assign(tra_luc_mua=o["ma_don"].map(tra_ngay).fillna(0))
    o = o.assign(no_them=(pd.to_numeric(o["tong_don"], errors="coerce").fillna(0) - o["tra_luc_mua"]).clip(lower=0))
    gn = o[o["no_them"] > 0.5]
    return gn, t[t["ma_don"].isin(gn["ma_don"]) & ~t["cung_ngay"]]


def no_them_tra_no(V, tu, den, khach=None):
    """[nợ thêm, trả nợ] trong kỳ [tu, den] (cả hai đầu, theo ngày). khach: mã khách hoặc tên (so bỏ dấu); None = cả shop.
    Nợ thêm = phần chưa trả trong ngày mua của đơn ghi nợ ĐẶT trong kỳ. Trả nợ = khoản trả SAU ngày mua, có ngày trả trong kỳ."""
    gn, tr = _ghi_no(V)
    if gn is None:
        return None
    a, b = pd.Timestamp(tu).normalize(), _end(den)
    if khach is not None:
        kk = _k(khach)
        def pick(df):
            m = pd.Series(False, index=df.index)
            for c in ("khach_ma", "khach_ten", "khach_ten_dm"):
                if c in df.columns:
                    m |= df[c].astype(str).map(_k).eq(kk)
            return df[m]
        gn = pick(gn)
        tr = tr[tr["ma_don"].isin(gn["ma_don"])]
    nt = float(gn.loc[(gn["ngay"] >= a) & (gn["ngay"] <= b), "no_them"].sum())
    tn = float(tr.loc[(tr["ngay_thu"] >= a) & (tr["ngay_thu"] <= b), "so_tien"].sum())
    return [nt, tn]


def thu_no_ky(V, tu, den):
    r = no_them_tra_no(V, tu, den)
    return None if r is None else r[1]


def tien_nhap_ky(V, tu, den):
    """§5: tổng tiền các phiếu nhập có NGÀY PHIẾU trong kỳ, bỏ phiếu huỷ. None nếu file không ghi tiền nhập."""
    if "phieu_nhap" not in V or "tong_tien" not in V["phieu_nhap"].columns:
        return None
    p = V["phieu_nhap"]
    ng = pd.to_datetime(p["ngay"], errors="coerce")
    m = (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & ~p["huy"].astype(bool)
    return float(pd.to_numeric(p.loc[m, "tong_tien"], errors="coerce").sum())


def doanh_thu_ky(V, tu, den):
    """§1: tổng doanh_thu các đơn tính doanh thu (hoàn thành / đang giao; bỏ huỷ, trả hàng) có ngày trong kỳ."""
    if "don" not in V or "doanh_thu" not in V["don"].columns:
        return None
    o = V["don"]
    ng = pd.to_datetime(o["ngay"], errors="coerce")
    m = (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & o["tinh_doanh_thu"]
    return float(o.loc[m, "doanh_thu"].sum())


HELPERS = {f.__name__: f for f in (cong_no_don, tong_no, no_theo_khach, so_khach_con_no, tuoi_no_nhom, no_qua_n_ngay,
                                   no_them_tra_no, thu_no_ky, tien_nhap_ky, doanh_thu_ky)}


# ============================================================================ kho
def tim_hang(V, ten):
    """Tên/mã hàng trong câu hỏi -> (mã hàng, tên hàng). So khớp bỏ dấu: trùng hẳn > tên chứa câu > câu chứa tên (dài nhất)."""
    k = _k(ten)
    rows = []
    for tb in ("hang", "ton", "dong"):
        if tb in V:
            d = V[tb]
            cols = [c for c in ("ma_hang", "ten_hang") if c in d.columns]
            if cols:
                rows.append(d[cols].drop_duplicates())
    if not rows:
        return None, None
    allr = pd.concat(rows, ignore_index=True).drop_duplicates()
    best, score = None, -1
    for r in allr.itertuples(index=False):
        ma = getattr(r, "ma_hang", None)
        tn = getattr(r, "ten_hang", None)
        for v in (tn, ma):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            kv = _k(v)
            sc = 3e6 if kv == k else (2e6 - len(kv) if k in kv else (1e6 + len(kv) if kv and kv in k else -1))
            if sc > score:
                best, score = (ma, tn), sc
    return (_none(best[0]), _none(best[1])) if best else (None, None)


def _match(df, ma, ten):
    ma, ten = _none(ma), _none(ten)
    m = pd.Series(False, index=df.index)
    if ma is not None and "ma_hang" in df.columns:
        m |= df["ma_hang"].map(_sid).eq(str(ma))
    if ten is not None and "ten_hang" in df.columns:
        m |= df["ten_hang"].astype(str).map(_k).eq(_k(ten))
    return m


def bien_dong_kho(V, ma, ten):
    """Các phát sinh (ngày, số lượng cơ bản có dấu, loại) của một mặt hàng: + nhập (bỏ phiếu huỷ), − bán (đơn không huỷ), + khách trả."""
    parts = []
    # nhập: ưu tiên bảng lô (số lượng cơ bản) có mã phiếu -> ngày phiếu
    ln = V.get("lo_nhap")
    if ln is not None and "sl_nhap_co_ban" in ln.columns and _match(ln, ma, ten).any():
        x = ln[_match(ln, ma, ten)].copy()
        if "ma_phieu" in x.columns and "phieu_nhap" in V:
            pn = V["phieu_nhap"].set_index("ma_phieu")
            x["ngay"] = pd.to_datetime(x["ma_phieu"].map(pn["ngay"]), errors="coerce")
            x = x[~x["ma_phieu"].map(pn["huy"]).fillna(False).astype(bool)]
        elif "nhap" in V and "ma_phieu" in x.columns:
            nd = V["nhap"].drop_duplicates("ma_phieu").set_index("ma_phieu")
            x["ngay"] = pd.to_datetime(x["ma_phieu"].map(nd["ngay"]), errors="coerce")
            x = x[~x["ma_phieu"].map(nd["huy"]).fillna(False).astype(bool)]
        if "ngay" in x.columns:
            parts.append(pd.DataFrame({"ngay": x["ngay"], "sl": x["sl_nhap_co_ban"], "loai": "nhap"}))
    elif "nhap" in V:
        nd = V["nhap"]
        x = nd[_match(nd, ma, ten) & ~nd["huy"].astype(bool)]
        q = x["sl_co_ban"] if "sl_co_ban" in x.columns else x.get("so_luong")
        if q is not None and len(x):
            if "sl_co_ban" not in x.columns and "don_vi" in x.columns and "quy_doi" in V:
                qd = V["quy_doi"].set_index(["hang_key", "don_vi_key"])["he_so"].to_dict()
                f = [qd.get((_k(ten or ""), _k(u)), qd.get((_k(ma or ""), _k(u)), 1.0)) for u in x["don_vi"]]
                q = pd.to_numeric(q, errors="coerce") * pd.Series(f, index=x.index)
            parts.append(pd.DataFrame({"ngay": pd.to_datetime(x["ngay"], errors="coerce"), "sl": pd.to_numeric(q, errors="coerce"), "loai": "nhap"}))
    # bán
    if "dong" in V:
        dd = V["dong"]
        x = dd[_match(dd, ma, ten) & dd["nhom_tt"].ne("huy")]
        if "la_hang" in x.columns:
            x = x[x["la_hang"]]
        has_tra = "tra" in V
        if not has_tra and "dong_tra" not in dd.columns:
            # không có số lượng trả từng dòng: đơn trả hàng coi như hàng về kho (không trừ)
            x = x[x["nhom_tt"].ne("tra_hang")]
        # có dòng số lượng âm (hàng trả ghi âm cùng mã đơn): cộng cả dòng bán và dòng trả -> tự bù trừ
        parts.append(pd.DataFrame({"ngay": pd.to_datetime(x["ngay"], errors="coerce"), "sl": -pd.to_numeric(x["sl_co_ban"], errors="coerce"),
                                   "loai": "ban", "ma_don": x["ma_don"]}))
        if has_tra:
            tr = V["tra"]
            if "lo" in tr.columns and ln is not None and "ma_hang" in ln.columns:
                lots = set(ln.loc[_match(ln, ma, ten), "lo"])
                tr = tr[tr["lo"].isin(lots)]
            else:
                tr = tr[tr["ma_don"].isin(x["ma_don"])]
            od = V["don"].set_index("ma_don")["ngay"]
            parts.append(pd.DataFrame({"ngay": pd.to_datetime(tr["ma_don"].map(od), errors="coerce"), "sl": tr["sl_tra_co_ban"], "loai": "tra"}))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).dropna(subset=["ngay", "sl"])


def ton_tai_ngay(V, mat_hang, X, data_end=None):
    """Tồn (đơn vị cơ bản) của một mặt hàng lúc HẾT ngày X.
    - Có thẻ kho: tồn ở dòng cuối của mặt hàng có ngày ≤ X.
    - Có tồn cuối kỳ (tại data_end): tồn_cuối − phát sinh trong (X, data_end].
    - Không có cả hai: cộng mọi phát sinh từ đầu dữ liệu đến hết X."""
    ma, ten = tim_hang(V, mat_hang)
    if ma is None and ten is None:
        return None
    E = _end(X)
    if "the_kho" in V:
        tk = V["the_kho"]
        x = tk[_match(tk, ma, ten) & (tk["ngay"] <= E)].sort_values(["ngay", "_ord"])
        return float(x["ton"].iloc[-1]) if len(x) else 0.0
    mv = bien_dong_kho(V, ma, ten)
    if "ton" in V:
        t = V["ton"]
        r = t[_match(t, ma, ten)]
        if not len(r):
            return None
        ton_end = float(r["ton"].iloc[0])
        if mv is None or data_end is None:
            return ton_end
        later = mv[(mv["ngay"] > E) & (mv["ngay"] <= _end(data_end))]
        return ton_end - float(later["sl"].sum())
    # không có tồn cuối, không có thẻ kho: chỉ tính được khi file ghi NHẬP (tồn = nhập − bán + trả từ đầu dữ liệu)
    if mv is None or not (mv["loai"] == "nhap").any():
        return None
    return float(mv.loc[mv["ngay"] <= E, "sl"].sum())


def he_so(V, mat_hang, don_vi):
    """Số đơn vị cơ bản trong 1 'don_vi' của mặt hàng (đã nhân dồn nhiều cấp). Nguồn: bảng quy đổi / README;
    không có thì SUY từ dòng bán (sl_co_ban / so_luong của chính mặt hàng + đơn vị đó, lấy trung vị). None nếu không biết."""
    ma, ten = tim_hang(V, mat_hang)
    if "quy_doi" in V:
        qd = V["quy_doi"]
        for hk in (_k(ten or ""), _k(ma or ""), _k(mat_hang)):
            r = qd[(qd["hang_key"] == hk) & (qd["don_vi_key"] == _k(don_vi))]
            if len(r):
                return float(r["he_so"].iloc[0])
    if "dong" in V and {"so_luong", "sl_co_ban", "don_vi"} <= set(V["dong"].columns):
        dd = V["dong"]
        x = dd[_match(dd, ma, ten) & (dd["don_vi"].astype(str).map(_k) == _k(don_vi))]
        q = pd.to_numeric(x["so_luong"], errors="coerce")
        r = (pd.to_numeric(x["sl_co_ban"], errors="coerce") / q)[q > 0].dropna()
        if len(r):
            return float(r.median())
    return None


def don_vi_lon_trong_cau(V, mat_hang, cau):
    """Đơn vị lớn (có hệ số) của mặt hàng được nhắc trong câu, ưu tiên cụm dài nhất ('xe 4m3' hơn 'xe')."""
    if "quy_doi" not in V:
        return None
    ma, ten = tim_hang(V, mat_hang)
    kc = _k(cau)
    kt = _k(ten or "")
    kc2 = kc.replace(kt, " ") if kt else kc   # bỏ tên hàng khỏi câu ('hộp 24 gói' trong tên không phải đơn vị hỏi)
    qd = V["quy_doi"]
    us = qd.loc[qd["hang_key"].isin([kt, _k(ma or "")]), "don_vi_key"].tolist()
    hit = [u for u in us if re.search(r"(^|\s)" + re.escape(u) + r"(\s|$)", kc2)]
    return max(hit, key=len) if hit else None


def ton_don_vi_lon(V, mat_hang, don_vi_lon, X, data_end=None):
    """§5: [số đơn vị lớn nguyên, số đơn vị CƠ BẢN lẻ] (hàng nhiều cấp: phần lẻ vẫn tính bằng đơn vị cơ bản)."""
    t = ton_tai_ngay(V, mat_hang, X, data_end)
    f = he_so(V, mat_hang, don_vi_lon)
    if t is None or not f:
        return None
    n = int(t // f)
    return [n, int(round(t - n * f))]


HELPERS.update({f.__name__: f for f in (tim_hang, bien_dong_kho, ton_tai_ngay, he_so, ton_don_vi_lon, don_vi_lon_trong_cau)})


# ============================================================================ lô, hạn dùng
def ton_lo(V, X):
    """Tồn từng lô lúc hết ngày X: nhập (phiếu không huỷ, ngày phiếu ≤ X) − bán từ lô (đơn không huỷ, ngày đơn ≤ X) + khách trả về lô.
    Cột: lo, ma_hang, han_dung, gia_nhap, sl_nhap, sl_ban, sl_tra, con_ton."""
    if "lo_nhap" not in V or "sl_nhap_co_ban" not in V["lo_nhap"].columns:
        return None
    E = _end(X)
    ln = V["lo_nhap"].copy()
    if "ma_phieu" in ln.columns and "phieu_nhap" in V:
        pn = V["phieu_nhap"].set_index("ma_phieu")
        ln["ngay_nhap"] = pd.to_datetime(ln["ma_phieu"].map(pn["ngay"]), errors="coerce")
        ln = ln[~ln["ma_phieu"].map(pn["huy"]).fillna(False).astype(bool)]
        ln = ln[ln["ngay_nhap"].isna() | (ln["ngay_nhap"] <= E)]
    out = ln.groupby("lo").agg(**{c: (c, "first") for c in ("ma_hang", "ten_hang", "han_dung", "gia_nhap") if c in ln.columns},
                               sl_nhap=("sl_nhap_co_ban", "sum"))
    out["sl_ban"] = 0.0
    out["sl_tra"] = 0.0
    if "ban_lo" in V and "don" in V:
        bl = V["ban_lo"]
        od = V["don"].set_index("ma_don")
        ng = pd.to_datetime(bl["ma_don"].map(od["ngay"]), errors="coerce")
        # đơn có trong bảng lô nhưng thiếu ở bảng đơn: không biết ngày → chỉ tính khi X đã qua ngày cuối của dữ liệu đơn
        last = pd.to_datetime(V["don"]["ngay"], errors="coerce").max()
        ok = bl["ma_don"].map(od["nhom_tt"]).ne("huy") & ((ng <= E) | (ng.isna() & (E >= last)))
        out["sl_ban"] = bl[ok].groupby("lo")["sl_ban_co_ban"].sum().reindex(out.index).fillna(0)
        out["sl_tra"] = bl[ok].groupby("lo")["sl_tra_co_ban"].sum().reindex(out.index).fillna(0)
    out["con_ton"] = out["sl_nhap"] - out["sl_ban"] + out["sl_tra"]
    return out.reset_index()


def lo_het_han_trong(V, D, N):
    """Lô CÒN TỒN tại hôm nay D có hạn dùng từ D đến hết D+N. Thêm cột tien_von = còn tồn × giá nhập của lô quy về đơn vị cơ bản
    (giá nhập theo đơn vị nhập / hệ số; nếu không biết hệ số thì để NaN)."""
    t = ton_lo(V, D)
    if t is None:
        return None
    a, b = pd.Timestamp(D).normalize(), _end(pd.Timestamp(D) + pd.Timedelta(days=int(N)))
    x = t[(t["con_ton"] > 0) & (t["han_dung"] >= a) & (t["han_dung"] <= b)].copy()
    if "gia_nhap" in x.columns:
        x["gia_nhap_co_ban"] = x.apply(lambda r: _gia_co_ban(V, r), axis=1)
        x["tien_von"] = x["con_ton"] * x["gia_nhap_co_ban"]
    if "ten_hang" not in x.columns or x["ten_hang"].isna().all():
        x["ten_hang"] = x["ma_hang"].map(_ten_theo_ma(V)) if "ma_hang" in x.columns else np.nan
    return x.reset_index(drop=True)


def _ten_theo_ma(V):
    for tb in ("hang", "ton", "dong"):
        if tb in V and {"ma_hang", "ten_hang"} <= set(V[tb].columns):
            return V[tb].dropna(subset=["ma_hang"]).drop_duplicates("ma_hang").set_index("ma_hang")["ten_hang"].to_dict()
    return {}


def _gia_co_ban(V, r):
    """Giá nhập một đơn vị cơ bản của lô: giá nhập (theo đơn vị nhập của dòng nhập) / hệ số của đơn vị nhập."""
    g = r.get("gia_nhap")
    if g is None or pd.isna(g):
        return np.nan
    if "nhap" in V and "ma_phieu" in V.get("lo_nhap", pd.DataFrame()).columns:
        nd = V["nhap"]
        cand = nd[(nd.get("ma_hang", pd.Series(dtype=str)).astype(str) == str(r.get("ma_hang"))) if "ma_hang" in nd.columns else slice(None)]
        if "don_vi" in cand.columns and len(cand):
            u = cand["don_vi"].dropna().astype(str)
            if len(u):
                f = he_so(V, str(r.get("ten_hang") or r.get("ma_hang")), u.iloc[0])
                if f:
                    return g / f
    return g


def lo_gan_het_han_nhat(V, mat_hang, D):
    """Lô còn tồn, CHƯA quá hạn tại hôm nay D, có hạn sớm nhất của một mặt hàng."""
    t = ton_lo(V, D)
    if t is None:
        return None
    ma, ten = tim_hang(V, mat_hang)
    x = t[(t["con_ton"] > 0) & (t["han_dung"] >= pd.Timestamp(D).normalize())]
    if ma is not None and "ma_hang" in x.columns:
        x = x[x["ma_hang"].astype(str) == str(ma)]
    elif ten is not None and "ten_hang" in x.columns:
        x = x[x["ten_hang"].astype(str).map(_k) == _k(ten)]
    return None if not len(x) else x.sort_values(["han_dung", "lo"]).iloc[0].to_dict()


HELPERS.update({f.__name__: f for f in (ton_lo, lo_het_han_trong, lo_gan_het_han_nhat)})


# ============================================================================ giá, lãi, kho tổng hợp
def danh_sach_hang(V):
    """Mọi mặt hàng biết được: (ma_hang, ten_hang) từ danh mục / tồn / dòng bán, bỏ trùng."""
    rows = []
    for tb in ("hang", "ton", "dong"):
        if tb in V:
            d = V[tb]
            if tb == "dong" and "la_hang" in d.columns:
                d = d[d["la_hang"]]
            cols = [c for c in ("ma_hang", "ten_hang") if c in d.columns]
            if cols:
                rows.append(d[cols].drop_duplicates())
        if rows and tb in ("hang", "ton"):
            break                  # có danh mục / bảng tồn thì lấy danh sách hàng từ đó (đầy đủ, không lẫn dòng lạ)
    if not rows:
        return pd.DataFrame(columns=["ma_hang", "ten_hang"])
    out = pd.concat(rows, ignore_index=True)
    key_col = "ma_hang" if "ma_hang" in out.columns else "ten_hang"
    return out.dropna(subset=[key_col]).drop_duplicates(key_col).reset_index(drop=True)


PHA_CHE = re.compile(r"\b(pha che|che bien|mon|do uong pha|tai cho|nau|phuc vu)\b")


def _hang_mua_ve(V):
    """Mặt hàng MUA VỀ ĐỂ BÁN: có trong nhập / lô / tồn; loại món pha chế, chế biến tại chỗ (theo nhóm/loại trong danh mục)."""
    ds = danh_sach_hang(V)
    if "hang" in V:
        h = V["hang"]
        for c in [c for c in h.columns if c.startswith("nhom") or c.startswith("loai")]:
            bad = h[h[c].astype(str).map(_k).str.contains(PHA_CHE)]
            if len(bad) and "ma_hang" in ds.columns and "ma_hang" in bad.columns:
                ds = ds[~ds["ma_hang"].isin(bad["ma_hang"])]
    return ds


def ton_tat_ca(V, X, data_end=None):
    """Tồn hết ngày X của MỌI mặt hàng: DataFrame[ma_hang, ten_hang, ton] (None nếu shop không có thông tin tồn)."""
    ds = danh_sach_hang(V)
    out = []
    for r in ds.itertuples(index=False):
        ma = getattr(r, "ma_hang", None)
        ten = getattr(r, "ten_hang", None)
        t = ton_tai_ngay(V, str(ten if isinstance(ten, str) else ma), X, data_end)
        out.append((ma, ten, t))
    df = pd.DataFrame(out, columns=["ma_hang", "ten_hang", "ton"])
    return None if df["ton"].isna().all() else df


def so_mat_hang_het_hang(V, X, data_end=None, tra_ve="so"):
    """§5: mặt hàng mua về để bán có tồn LÚC HẾT ngày X bằng 0 (≤ 0). tra_ve='so' -> số mặt hàng; 'ds' -> danh sách tên."""
    t = ton_tat_ca(V, X, data_end)
    if t is None:
        return None
    keep = _hang_mua_ve(V)
    if "ma_hang" in keep.columns and keep["ma_hang"].notna().any():
        t = t[t["ma_hang"].isin(keep["ma_hang"])]
    het = t[t["ton"].notna() & (t["ton"] <= 0)]
    return int(len(het)) if tra_ve == "so" else sorted(het["ten_hang"].fillna(het["ma_hang"]).astype(str).tolist())


def gia_tri_ton_kho(V, X, data_end=None):
    """§5: Σ tồn hết ngày X (đv cơ bản) × GIÁ VỐN một đv cơ bản GHI TRONG DANH MỤC; chỉ hàng còn tồn (> 0)."""
    if "hang" not in V or "gia_von" not in V["hang"].columns:
        return None
    t = ton_tat_ca(V, X, data_end)
    if t is None:
        return None
    h = V["hang"]
    gv = pd.to_numeric(h["gia_von"], errors="coerce")
    if "ma_hang" in h.columns and t["ma_hang"].notna().any():
        mp = dict(zip(h["ma_hang"].astype(str), gv))
        g = t["ma_hang"].astype(str).map(mp)
    else:
        mp = dict(zip(h["ten_hang"].astype(str).map(_k), gv))
        g = t["ten_hang"].astype(str).map(_k).map(mp)
    m = t["ton"] > 0
    return float((t.loc[m, "ton"] * g[m]).sum())


def gia_nhap_binh_quan(V, mat_hang, tu, den, don_vi=None):
    """§5: giá nhập bình quân THEO SỐ LƯỢNG (Σ tiền nhập / Σ số lượng), các phiếu có ngày phiếu trong kỳ, bỏ phiếu huỷ.
    don_vi: đơn vị hỏi (vd 'thùng'); khác đơn vị nhập thì quy qua đơn vị cơ bản. None nếu file không ghi giá/tiền nhập."""
    if "nhap" not in V:
        return None
    nd = V["nhap"]
    ma, ten = tim_hang(V, mat_hang)
    ng = pd.to_datetime(nd["ngay"], errors="coerce") if "ngay" in nd.columns else None
    if ng is None:
        return None
    x = nd[_match(nd, ma, ten) & (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & ~nd["huy"].astype(bool)]
    if not len(x) or not ({"thanh_tien"} <= set(x.columns) or {"don_gia", "so_luong"} <= set(x.columns)):
        return None
    tien = pd.to_numeric(x["thanh_tien"], errors="coerce") if "thanh_tien" in x.columns else \
        pd.to_numeric(x["so_luong"], errors="coerce") * pd.to_numeric(x["don_gia"], errors="coerce")
    if tien.isna().all():
        return None
    q = pd.to_numeric(x["so_luong"], errors="coerce")
    if don_vi is not None and "don_vi" in x.columns:
        fu = he_so(V, mat_hang, don_vi) or 1.0
        fx = pd.Series([he_so(V, mat_hang, u) or 1.0 for u in x["don_vi"].astype(str)], index=x.index)
        if not all(_k(u) == _k(don_vi) for u in x["don_vi"].astype(str)):
            q = q * fx / fu          # số lượng theo đơn vị hỏi
    tq = float(q.sum())
    return None if tq <= 0 else float(tien.sum()) / tq


def lai_gop_ky(V, tu, den, theo_hang=False):
    """§5: doanh thu (§1) − Σ số lượng × giá vốn GHI TRÊN DÒNG BÁN (một đơn vị bán). theo_hang=True -> DataFrame theo mặt hàng."""
    if "dong" not in V or "gia_von_1dv" not in V["dong"].columns:
        return None
    dd = V["dong"]
    ok_don = set(V["don"].loc[V["don"]["tinh_doanh_thu"], "ma_don"]) if "don" in V else None
    ng = pd.to_datetime(dd["ngay"], errors="coerce")
    m = (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den))
    if ok_don is not None:
        m &= dd["ma_don"].isin(ok_don)
    if "la_hang" in dd.columns:
        m &= dd["la_hang"]
    x = dd[m]
    lai = pd.to_numeric(x["thanh_tien"], errors="coerce").fillna(0) - pd.to_numeric(x["so_luong"], errors="coerce").fillna(0) * pd.to_numeric(x["gia_von_1dv"], errors="coerce").fillna(0)
    if not theo_hang:
        return float(lai.sum())
    k = "ten_hang" if "ten_hang" in x.columns else "ma_hang"
    g = lai.groupby(x[k]).sum().rename("lai_gop").reset_index().rename(columns={k: "mat_hang"})
    return g.sort_values(["lai_gop", "mat_hang"], ascending=[False, True]).reset_index(drop=True)


def chi_phi_ky(V, tu, den):
    if "chi_phi" not in V:
        return None
    c = V["chi_phi"]
    ng = pd.to_datetime(c["ngay"], errors="coerce")
    return float(pd.to_numeric(c.loc[(ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)), "so_tien"], errors="coerce").sum())


def lai_rong_ky(V, tu, den):
    """§5: lãi gộp − MỌI khoản trong sổ chi phí cùng kỳ. None nếu thiếu giá vốn hoặc thiếu sổ chi phí."""
    lg, cp = lai_gop_ky(V, tu, den), chi_phi_ky(V, tu, den)
    return None if lg is None or cp is None else lg - cp


def so_don_ky(V, tu, den, chi_tinh_doanh_thu=True):
    if "don" not in V:
        return None
    o = V["don"]
    ng = pd.to_datetime(o["ngay"], errors="coerce")
    m = (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den))
    if chi_tinh_doanh_thu:
        m &= o["tinh_doanh_thu"]
    return int(m.sum())


def doanh_thu_binh_quan_don(V, tu, den):
    n = so_don_ky(V, tu, den)
    dt = doanh_thu_ky(V, tu, den)
    return None if not n or dt is None else dt / n


HELPERS.update({f.__name__: f for f in (danh_sach_hang, ton_tat_ca, so_mat_hang_het_hang, gia_tri_ton_kho, gia_nhap_binh_quan,
                                        lai_gop_ky, chi_phi_ky, lai_rong_ky, so_don_ky, doanh_thu_binh_quan_don)})


# ============================================================================ nhân viên / dịch vụ
def doanh_thu_theo_nhan_vien(V, tu, den):
    """Doanh thu (đơn tính doanh thu) theo nhân viên / thợ trong kỳ: DataFrame[nhan_vien, doanh_thu, so_don] giảm dần."""
    if "don" not in V or "nhan_vien" not in V["don"].columns:
        return None
    o = V["don"]
    ng = pd.to_datetime(o["ngay"], errors="coerce")
    x = o[(ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & o["tinh_doanh_thu"] & o["nhan_vien"].notna()]
    g = x.groupby("nhan_vien").agg(doanh_thu=("doanh_thu", "sum"), so_don=("ma_don", "count")).reset_index()
    return g.sort_values(["doanh_thu", "nhan_vien"], ascending=[False, True]).reset_index(drop=True)


def hoa_hong_ky(V, tu, den):
    """Hoa hồng theo thợ = % hoa hồng (danh sách thợ) × tổng tiền các phiếu thợ làm trong kỳ (README thường ghi 'trong tháng')."""
    if "nhan_vien" not in V or "hoa_hong_pct" not in V["nhan_vien"].columns:
        return None
    g = doanh_thu_theo_nhan_vien(V, tu, den)
    if g is None:
        return None
    pct = V["nhan_vien"].set_index("ho_ten")["hoa_hong_pct"]
    g["hoa_hong_pct"] = g["nhan_vien"].map(pct)
    g["hoa_hong"] = g["doanh_thu"] * g["hoa_hong_pct"] / 100.0
    return g


HELPERS.update({f.__name__: f for f in (doanh_thu_theo_nhan_vien, hoa_hong_ky)})


# ============================================================================ bán hàng theo mặt hàng / nhóm / ngày / khách
def _dong_tinh_dt(V, tu, den, mat_hang_only=True):
    """Dòng hàng của các đơn TÍNH DOANH THU có ngày trong kỳ (bỏ huỷ, trả). mat_hang_only: bỏ phí phục vụ / voucher / ship."""
    if "dong" not in V:
        return None
    dd = V["dong"]
    ng = pd.to_datetime(dd["ngay"], errors="coerce")
    m = (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den))
    if "don" in V:
        m &= dd["ma_don"].isin(set(V["don"].loc[V["don"]["tinh_doanh_thu"], "ma_don"]))
    col = "la_mat_hang" if mat_hang_only and "la_mat_hang" in dd.columns else ("la_hang" if "la_hang" in dd.columns else None)
    if col:
        m &= dd[col].astype(bool)
    return dd[m]


def _gan_thuoc_tinh(V, x):
    """Gắn nhom / hoat_chat / ke_don từ danh mục vào dòng bán (theo mã, không có mã thì theo tên)."""
    if "hang" not in V:
        return x
    h = V["hang"]
    x = x.copy()
    for c in ("nhom", "hoat_chat", "ke_don"):
        if c not in h.columns or c in x.columns:
            continue
        if "ma_hang" in h.columns and "ma_hang" in x.columns:
            x[c] = x["ma_hang"].map(_sid).map(dict(zip(h["ma_hang"].map(_sid), h[c])))
        elif "ten_hang" in h.columns and "ten_hang" in x.columns:
            x[c] = x["ten_hang"].map(_k).map(dict(zip(h["ten_hang"].map(_k), h[c])))
    return x


def ban_hang_ky(V, tu, den, nhom=None, hoat_chat=None, ke_don=None, size=None, ten=None):
    """Bán theo MẶT HÀNG trong kỳ: DataFrame[mat_hang, ma_hang, so_luong_co_ban, doanh_thu, so_don].
    Lọc: nhom (đúng tên nhóm, so bỏ dấu), hoat_chat (ĐÚNG chuỗi hoạt chất, phối hợp là loại riêng), ke_don (True/False),
    size ('M'/'L': tên kết thúc '(M)'/'(L)'), ten (tên/mã một mặt hàng). Xếp 'bán chạy' theo so_luong_co_ban (rulebook)."""
    x = _dong_tinh_dt(V, tu, den)
    if x is None:
        return None
    x = _gan_thuoc_tinh(V, x)
    if nhom is not None:
        if "nhom" not in x.columns:
            return None
        x = x[x["nhom"].astype(str).map(_k) == _k(nhom)]
    if hoat_chat is not None:
        if "hoat_chat" not in x.columns:
            return None
        x = x[x["hoat_chat"].astype(str).map(_k) == _k(hoat_chat)]
    if ke_don is not None:
        if "ke_don" not in x.columns:
            return None
        x = x[x["ke_don"].fillna(False).astype(bool) == bool(ke_don)]
    if size is not None:
        x = x[x["ten_hang"].astype(str).str.strip().str.lower().str.endswith(f"({str(size).lower()})")]
    if ten is not None:
        ma, tn = tim_hang(V, ten)
        x = x[_match(x, ma, tn)]
    k = "ten_hang" if "ten_hang" in x.columns and x["ten_hang"].notna().any() else "ma_hang"
    if k not in x.columns or not {"sl_co_ban", "thanh_tien", "ma_don"} <= set(x.columns):
        return None              # shop không ghi mặt hàng / số lượng trên dòng bán
    g = x.groupby(x[k].astype(str)).agg(so_luong_co_ban=("sl_co_ban", "sum"), doanh_thu=("thanh_tien", "sum"), so_don=("ma_don", "nunique"))
    g = g.reset_index().rename(columns={k: "mat_hang"})
    if "ma_hang" in x.columns and k != "ma_hang":
        g["ma_hang"] = g["mat_hang"].map(x.drop_duplicates(k).set_index(x.drop_duplicates(k)[k].astype(str))["ma_hang"])
    return g.sort_values(["so_luong_co_ban", "mat_hang"], ascending=[False, True]).reset_index(drop=True)


def doanh_thu_loc(V, tu, den, **loc):
    """Doanh thu (§1) của các mặt hàng thoả bộ lọc (nhom / hoat_chat / ke_don / size / ten). None nếu shop không có cột lọc."""
    g = ban_hang_ky(V, tu, den, **loc)
    return None if g is None else float(g["doanh_thu"].sum())


def so_luong_ban(V, ten_hang, tu, den, don_vi=None):
    """Số lượng bán (đơn tính doanh thu) của một mặt hàng; don_vi khác đv cơ bản thì chia hệ số."""
    g = ban_hang_ky(V, tu, den, ten=ten_hang)
    if g is None:
        return None
    q = float(g["so_luong_co_ban"].sum())
    if don_vi is not None:
        f = he_so(V, ten_hang, don_vi)
        if f:
            q = q / f
    return q


def doanh_thu_theo_ngay(V, tu, den):
    """Series doanh thu theo NGÀY (đơn tính doanh thu); ngày nhiều tiền nhất = .idxmax()."""
    if "don" not in V:
        return None
    o = V["don"]
    ng = pd.to_datetime(o["ngay"], errors="coerce")
    x = o[(ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & o["tinh_doanh_thu"]]
    return x.groupby(pd.to_datetime(x["ngay"]).dt.normalize())["doanh_thu"].sum().sort_index()


def doanh_thu_theo(V, cot, tu, den):
    """Doanh thu theo một cột của V['don'] (kenh, httt, nhan_vien, ...): DataFrame[cot, doanh_thu, so_don] giảm dần."""
    if "don" not in V or cot not in V["don"].columns:
        return None
    o = V["don"]
    ng = pd.to_datetime(o["ngay"], errors="coerce")
    x = o[(ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & o["tinh_doanh_thu"]]
    g = x.groupby(x[cot].astype(str)).agg(doanh_thu=("doanh_thu", "sum"), so_don=("ma_don", "count")).reset_index()
    return g.sort_values(["doanh_thu", cot], ascending=[False, True]).reset_index(drop=True)


def khach_ky(V, tu, den):
    """§3: theo khách (bỏ khách lẻ) trong kỳ: DataFrame[khach_hang, khach_key, doanh_thu, so_don] giảm theo doanh thu, hoà theo mã/tên."""
    if "don" not in V:
        return None
    o = V["don"]
    ng = pd.to_datetime(o["ngay"], errors="coerce")
    x = o[(ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & o["tinh_doanh_thu"]]
    x = x.assign(khach_key=_khach_key(x))
    x = x[x["khach_key"].notna()]
    g = x.groupby("khach_key").agg(doanh_thu=("doanh_thu", "sum"), so_don=("ma_don", "count")).reset_index()
    g["khach_hang"] = g["khach_key"].map(ten_khach_chuan(V))
    return g.sort_values(["doanh_thu", "khach_key"], ascending=[False, True]).reset_index(drop=True)


HELPERS.update({f.__name__: f for f in (ban_hang_ky, doanh_thu_loc, so_luong_ban, doanh_thu_theo_ngay, doanh_thu_theo, khach_ky)})


def don_vi_trong_cau(V, mat_hang, cau):
    """Đơn vị câu hỏi nhắc (lớn hoặc cơ bản), bỏ phần tên hàng ra trước. Trả (don_vi_key, he_so) — he_so 1 nếu là đv cơ bản."""
    ma, ten = tim_hang(V, mat_hang)
    kc = _k(cau)
    kt = _k(ten or "")
    kc2 = kc.replace(kt, " ") if kt else kc
    u = don_vi_lon_trong_cau(V, mat_hang, cau)
    if u:
        return u, he_so(V, mat_hang, u)
    base = None
    if "quy_doi" in V:
        qd = V["quy_doi"]
        r = qd[qd["hang_key"].isin([kt, _k(ma or "")])]
        if len(r) and r["dv_co_ban"].notna().any():
            base = _k(r["dv_co_ban"].dropna().iloc[0])
    if base is None and "hang" in V and "dv_co_ban" in V["hang"].columns:
        h = V["hang"]
        r = h[_match(h, ma, ten)]
        if len(r) and r["dv_co_ban"].notna().any():
            base = _k(r["dv_co_ban"].dropna().iloc[0])
    if base and re.search(r"(^|\s)" + re.escape(base) + r"(\s|$)", kc2):
        return base, 1.0
    return None, None


def nhap_tru_ban(V, mat_hang, tu, den):
    """§5 'nhập trừ bán', 'chênh nhập và bán': số lượng NHẬP (phiếu không huỷ, ngày phiếu trong kỳ) − số lượng BÁN
    (đơn tính doanh thu), CẢ HAI quy về ĐƠN VỊ CƠ BẢN (đối chiếu đáp án train: đúng 4/4 theo cách này)."""
    if "nhap" not in V:
        return None
    nd = V["nhap"]
    ma, ten = tim_hang(V, mat_hang)
    ng = pd.to_datetime(nd["ngay"], errors="coerce")
    x = nd[_match(nd, ma, ten) & (ng >= pd.Timestamp(tu).normalize()) & (ng <= _end(den)) & ~nd["huy"].astype(bool)]
    ban = so_luong_ban(V, ten or ma, tu, den)
    if ban is None or "so_luong" not in x.columns:
        return None
    if "sl_co_ban" in x.columns and x["sl_co_ban"].notna().all():
        nhap = float(pd.to_numeric(x["sl_co_ban"], errors="coerce").sum())
    else:
        q = pd.to_numeric(x["so_luong"], errors="coerce").fillna(0)
        f = [he_so(V, ten or ma, u) or 1.0 for u in x["don_vi"].astype(str)] if "don_vi" in x.columns else [1.0] * len(x)
        nhap = float((q * pd.Series(f, index=x.index)).sum())
    return nhap - ban


HELPERS.update({f.__name__: f for f in (don_vi_trong_cau, nhap_tru_ban)})
