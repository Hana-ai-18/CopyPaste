#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataMate pipeline v0 — bản chạy đầu-cuối (end-to-end) theo DataMate_Pipeline_v2.md

Luồng:
  rulebook ─┐
  data/<biz> ─▶ đọc + nhận header + bỏ dòng rác + định kiểu cột (tiền/ngày/số lượng) ─▶ "thẻ dữ liệu"
  câu hỏi ─▶ chuẩn hoá (vn_normalize) ─▶ kỳ thời gian tính sẵn (P) ─▶ cảnh báo bài học (16 cách hiểu nhầm)
          ─▶ LLM viết hàm solve() ─▶ sandbox (AST guard + timeout) ─▶ kiểm format ─▶
          ─▶ (tuỳ chọn) đường B độc lập ─▶ đồng thuận / abstain ─▶ submission.jsonl + log từng câu
  score ─▶ chấm theo rulebook §9 (CHỈ dùng với answers.jsonl của TRAIN)

Lệnh:
  python datamate_pipeline.py selftest
  python datamate_pipeline.py run   --split_dir /path/train --out out_train --llm vllm --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ [--biz biz_101,biz_102] [--limit 50]
  python datamate_pipeline.py score --split_dir /path/train --pred out_train/submission.jsonl
  python datamate_pipeline.py card  --split_dir /path/train --biz biz_101      # xem thẻ dữ liệu của 1 shop

Tuân thủ luật: không có logic theo id câu hỏi / business; không đọc đáp án test; chạy offline được (model đã tải sẵn).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import functools
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import ledger
except Exception:  # ledger.py nằm cạnh pipeline; thiếu thì model tự tính từ V/T
    ledger = None
try:
    import compiler as CMP           # bước 3 (đặc tả) + bước 4 đường 1 (máy tính theo công thức)
except Exception:
    CMP = None
try:
    import rule_delta as RDL         # so rulebook đang chạy với vân tay rulebook train (luật đổi/bỏ/mới)
except Exception:
    RDL = None
PROC_T0 = __import__("time").time()  # mốc thời gian TỪ KHI tiến trình bắt đầu (gồm cả nạp model) — ngân sách 6 giờ tính từ đây
import sqlite3
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

try:
    from vn_normalize import Normalizer  # module chuẩn hoá đi kèm
except Exception:  # pragma: no cover
    Normalizer = None
try:
    from vn_normalize import ALIAS_GROUPS
except Exception:  # pragma: no cover
    ALIAS_GROUPS = []
try:
    from vn_context import ContextNormalizer, resolve_unsure  # sửa chữ theo ngữ cảnh cả câu (+ LLM chọn chỗ chưa chắc)
except Exception:  # pragma: no cover
    ContextNormalizer = resolve_unsure = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

# =============================================================================
# 0. Tiện ích chung
# =============================================================================


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def strip_accents(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def sid(v):
    """Mã dạng chữ, giữ ô trống là NaN (pandas 2 biến NaN thành 'nan', pandas 3 giữ NaN — chuẩn hoá để kết quả như nhau)."""
    if v is None or (isinstance(v, float) and v != v) or v is pd.NA or v is pd.NaT:
        return np.nan
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def key(s):
    """Khoá so tên: thường, bỏ dấu, gom khoảng trắng."""
    return re.sub(r"\s+", " ", strip_accents(nfc(s)).lower()).strip()


def rhu(x, n=0):
    """Làm tròn HALF_UP (không dùng round() của Python — round() là banker's rounding)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return x
    q = Decimal(1).scaleb(-n)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return int(v) if n == 0 else float(v)


def pct(a, b):
    """% thay đổi của a so với kỳ gốc b, làm tròn 2 chữ số. b = 0 -> None."""
    if b in (0, None) or (isinstance(b, float) and math.isnan(b)):
        return None
    return rhu((a - b) / b * 100, 2)


def now():
    return time.time()


def jdump(o):
    return json.dumps(o, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================
# 1. Parser tiền / ngày / số lượng (quyết định theo CẢ CỘT)
# =============================================================================

_MONEY_SUFFIX = re.compile(r"^-?\(?[\d.,]+(tr\d*|k|đ|d|vnd|vnđ|đồng|dong|triệu|trieu|nghìn|ngàn)?\)?$", re.I)


def parse_money_one(s, thousands=None):
    """Parse 1 giá trị tiền. thousands ∈ {'.', ',', None} do quyết định theo cột."""
    if s is None:
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    t = nfc(s).strip().lower().replace("\xa0", "").replace(" ", "")
    if t in ("", "nan", "none", "-"):
        return np.nan
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1]
    if t.startswith("-"):
        neg, t = True, t[1:]
    t = re.sub(r"(vnđ|vnd|đồng|dong)$", "", t)
    try:
        m = re.fullmatch(r"(\d+)tr(\d+)", t)  # 1tr380 = 1.380.000 ; 1tr5 = 1.500.000
        if m:
            v = int(m[1]) * 1_000_000 + int(m[2].ljust(3, "0")[:3]) * 1000
        elif re.fullmatch(r"[\d.,]+(tr|triệu|trieu|củ)", t):
            v = float(re.sub(r"(tr|triệu|trieu|củ)$", "", t).replace(",", ".")) * 1e6
        elif re.fullmatch(r"[\d.,]+(k|nghìn|ngàn|nghin|ngan)", t):
            v = float(re.sub(r"(k|nghìn|ngàn|nghin|ngan)$", "", t).replace(",", ".")) * 1e3
        elif re.fullmatch(r"[\d.,]+(đ|d)", t):
            v = float(t[:-1].replace(".", "").replace(",", ""))
        else:
            if thousands == ".":
                v = float(t.replace(".", "").replace(",", "."))
            elif thousands == ",":
                v = float(t.replace(",", ""))
            else:
                v = float(t.replace(",", "")) if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t) else float(t.replace(",", "."))
    except ValueError:
        return np.nan
    v = round(v, 6)
    return -v if neg else v


def decide_thousands(values):
    """Quyết định dấu phân cách nghìn theo cả cột."""
    dot_k = comma_k = dot_dec = comma_dec = 0
    for t in values:
        t = str(t).strip().replace(" ", "")
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", t):
            dot_k += 1
        elif re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
            comma_k += 1
        elif re.fullmatch(r"-?\d+\.\d{1,2}|-?\d+\.\d{4,}", t):
            dot_dec += 1
        elif re.fullmatch(r"-?\d+,\d{1,2}", t):
            comma_dec += 1
    if dot_k > comma_k and dot_k >= dot_dec:
        return "."
    if comma_k >= dot_k and (comma_k > 0 or dot_dec > 0):
        return ","
    if comma_dec > dot_dec:
        return "."
    return None


_DATE_RES = [
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?"), "ymd"),
    (re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?"), "dmy"),
    (re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$"), "dmy2"),
    (re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(\d{4})", re.I), "thg"),
    (re.compile(r"^(\d{1,2})/(\d{1,2})$"), "dm"),
]


def parse_date_one(s, year_hint=None, dayfirst=True, allow_serial=False):
    if s is None:
        return pd.NaT
    if isinstance(s, (pd.Timestamp, dt.datetime)):
        return pd.Timestamp(s)
    if isinstance(s, (int, float, np.integer, np.floating)) and not (isinstance(s, float) and math.isnan(s)):
        if allow_serial and 20000 < float(s) < 80000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(s))
        return pd.NaT
    t = nfc(s).strip()
    if not t or t.lower() in ("nan", "none", "nat"):
        return pd.NaT
    if allow_serial and re.fullmatch(r"\d{5}(\.\d+)?", t) and 20000 < float(t) < 80000:
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(t))
    t = re.sub(r"^(t[2-7]|cn|thứ\s*\w+)[,\s]+", "", t, flags=re.I)
    for rx, kind in _DATE_RES:
        m = rx.match(t)
        if not m:
            continue
        try:
            g = m.groups()
            hh = mm_ = ss = 0
            if kind == "ymd":
                y, mo, d = int(g[0]), int(g[1]), int(g[2])
                hh, mm_, ss = int(g[3] or 0), int(g[4] or 0), int(g[5] or 0)
            elif kind in ("dmy", "dmy2"):
                a, b, y = int(g[0]), int(g[1]), int(g[2])
                if kind == "dmy2":
                    y += 2000
                d, mo = (a, b) if dayfirst else (b, a)
                hh, mm_, ss = int(g[3] or 0), int(g[4] or 0), int(g[5] or 0)
                if re.search(r"\b(ch|pm)\b", t, re.I) and hh < 12:
                    hh += 12
            elif kind == "thg":
                d, mo, y = int(g[0]), int(g[1]), int(g[2])
            else:  # dm
                if not year_hint:
                    return pd.NaT
                d, mo, y = int(g[0]), int(g[1]), int(year_hint)
            return pd.Timestamp(y, mo, d, hh, mm_, ss)
        except (ValueError, TypeError):
            return pd.NaT
    try:
        return pd.Timestamp(t)
    except Exception:
        return pd.NaT


def decide_dayfirst(values):
    first_gt12 = second_gt12 = 0
    for t in values:
        m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}", str(t).strip())
        if m:
            if int(m[1]) > 12:
                first_gt12 += 1
            if int(m[2]) > 12:
                second_gt12 += 1
    if second_gt12 > first_gt12:
        return False
    return True  # mặc định VN: ngày/tháng


# =============================================================================
# 2. Đọc dữ liệu shop: header, dòng rác, định kiểu cột, thẻ dữ liệu
# =============================================================================

ID_HINT_RAW = re.compile(r"(^|\b)(mã|ma|code|id|sđt|sdt|phone|điện thoại|dien thoai|imei|barcode|mã vạch|số ct|so ct|bill|sku|biển số|bien so|số hđ|so hd)\b", re.I)
MONEY_NAME = re.compile(r"(tổng|tong|tiền|tien|giá|gia\b|trị giá|tri gia|thành tiền|thanh tien|amount|total|phải thu|phai thu|nợ|\bno\b)", re.I)


class _IdHint:
    """'Tổng bill', 'Tiền bill' là cột TIỀN, không phải mã: tên có chữ tiền/tổng thì không coi là mã."""
    def search(self, c):
        c = str(c)
        if MONEY_NAME.search(c) and not re.search(r"(^|\b)(mã|ma|id|code)\b", c, re.I):
            return None
        return ID_HINT_RAW.search(c)


ID_HINT = _IdHint()
DATE_HINT = re.compile(r"(ngày|ngay|date|thời gian|thoi gian|time|giờ|gio|hạn|han|lúc|luc)", re.I)
MONEY_HINT = re.compile(r"(tiền|tien|giá|gia|tổng|tong|thành|thanh|nợ|no\b|trả|tra\b|thu\b|chi\b|phí|phi\b|giảm|giam|vốn|von|amount|price|total|cost|cộng|cong)", re.I)
# dòng tổng: ô chỉ ghi 'Tổng', 'Tổng cộng', 'Cộng:', 'Total'… — KHÔNG phải tên người họ Tống viết không dấu ('Tong Huong Chau')
TOTAL_ROW = re.compile(r"^\s*(tổng|tong|cộng|cong|total|sum|grand total)(\s*(cộng|cong|tiền|tien|số|so|kết|ket|chung|tháng|thang|ngày|ngay|năm|nam|quý|quy)?)?\s*[:：.\-]?\s*(\S{0,12})?\s*$", re.I)


def _dedupe_cols(cols):
    out, seen = [], Counter()
    for i, c in enumerate(cols):
        c = nfc(c).strip() if c is not None and str(c) != "nan" else f"col_{i}"
        c = re.sub(r"\s+", " ", c)
        if seen[c]:
            out.append(f"{c}_{seen[c]}")
        else:
            out.append(c)
        seen[c] += 1
    return out


def detect_header(raw: pd.DataFrame, max_scan=25):
    """Chọn dòng header: nhiều ô chữ (không phải số/ngày), các ô khác nhau, sau đó là dữ liệu."""
    best, best_i = -1, 0
    ncol = raw.shape[1]
    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i]
        vals = [str(v).strip() for v in row if v is not None and str(v).strip() not in ("", "nan", "None")]
        if len(vals) < max(2, 0.5 * ncol):
            continue
        texty = sum(1 for v in vals if not re.fullmatch(r"[-\d.,/: %đ]+", v))
        score = texty + 0.5 * len(set(vals)) - (len(vals) - len(set(vals)))
        if score > best:
            best, best_i = score, i
    return best_i


def read_tables(biz_dir):
    """Trả dict tên_bảng -> DataFrame (cột dtype object, đã strip), kèm ghi chú."""
    tables, notes = {}, defaultdict(list)
    files = sorted(glob.glob(os.path.join(biz_dir, "*")))
    for f in files:
        base = os.path.basename(f)
        ext = base.lower().rsplit(".", 1)[-1]
        year_hint = None
        m = re.search(r"(20\d{2})", base)
        if m:
            year_hint = int(m[1])
        try:
            if ext == "csv":
                df = None
                for enc in ("utf-8-sig", "utf-8", "cp1258", "utf-16", "latin-1"):
                    try:
                        df = pd.read_csv(f, header=None, dtype=object, encoding=enc, sep=None, engine="python")
                        break
                    except Exception:
                        continue
                if df is None:
                    notes[base].append("không đọc được")
                    continue
                tables[base] = (df, year_hint)
            elif ext in ("xlsx", "xlsm", "xls"):
                engine = None
                try:
                    import python_calamine  # noqa: F401
                    engine = "calamine"
                except Exception:
                    engine = None
                sheets = pd.read_excel(f, sheet_name=None, header=None, dtype=object, engine=engine)
                for sh, df in sheets.items():
                    yh = year_hint
                    m2 = re.search(r"(20\d{2})", str(sh))
                    if m2:
                        yh = int(m2[1])
                    tables[f"{base}::{sh}"] = (df, yh)
            elif ext in ("sqlite", "db", "sqlite3"):
                con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                names = [r[0] for r in con.execute("select name from sqlite_master where type in ('table','view')")]
                for n in names:
                    df = pd.read_sql(f'select * from "{n}"', con)
                    df = df.astype(object)
                    tables[f"{base}::{n}"] = ("SQL", df, year_hint)
                con.close()
        except Exception as e:  # pragma: no cover
            notes[base].append(f"lỗi đọc: {e}")
    out = {}
    for name, val in tables.items():
        if isinstance(val[0], str) and val[0] == "SQL":
            df, yh = val[1], val[2]
            df.columns = _dedupe_cols(df.columns)
        else:
            raw, yh = val
            raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            if raw.empty:
                continue
            h = detect_header(raw)
            if h > 0:
                notes[name].append(f"bỏ {h} dòng tiêu đề phía trên header")
            cols = _dedupe_cols(list(raw.iloc[h]))
            df = raw.iloc[h + 1:].copy()
            df.columns = cols
            df = df.reset_index(drop=True)
        df = clean_table(df, name, notes)
        out[name] = (df, yh)
    return out, notes


def clean_table(df, name, notes):
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):   # pandas 2 (object) và pandas 3 (str)
            df[c] = df[c].map(lambda v: nfc(v).strip() if isinstance(v, str) else v)
            df[c] = df[c].replace({"": None, "nan": None})
    df = df.dropna(how="all")
    cols = [str(c) for c in df.columns]
    # header lặp giữa file
    hdr_mask = df.apply(lambda r: sum(str(v) == str(c) for v, c in zip(r.values, cols)) >= max(2, len(cols) // 2), axis=1)
    if hdr_mask.any():
        notes[name].append(f"bỏ {int(hdr_mask.sum())} dòng header lặp")
        df = df[~hdr_mask]
    # dòng tổng: ô đầu tiên có chữ bắt đầu bằng Tổng/Cộng/Total
    def is_total(r):
        hit = any(isinstance(v, str) and TOTAL_ROW.match(v) for v in r.values)   # ô 'Tổng' có thể ở cột bất kỳ
        if not hit:
            return False
        # dòng tổng thật: ngoài ô 'Tổng' thì gần như chỉ có SỐ (không có tên, mã, ngày dạng chữ khác)
        txt = [v for v in r.values if isinstance(v, str) and v.strip() and not re.fullmatch(r"[\d\s.,%đkKtrđ\-+()]+", v.strip())]
        return len(txt) <= 1
    tot = df.apply(is_total, axis=1) if len(df) else pd.Series(dtype=bool)
    if len(df) and tot.any():
        notes[name].append(f"bỏ {int(tot.sum())} dòng tổng/cộng")
        df = df[~tot]
    df = df.reset_index(drop=True)
    return df


def type_columns(df, year_hint, name, notes):
    """Định kiểu cột theo cả cột. Trả df mới + mô tả kiểu từng cột."""
    types = {}
    for c in list(df.columns):
        s = df[c]
        nn = s.dropna()
        if len(nn) == 0:
            types[c] = "trống"
            continue
        head = nn.head(200)
        if sum(isinstance(v, (pd.Timestamp, dt.datetime, dt.date, np.datetime64)) for v in head) > 0.9 * len(head):
            df[c] = pd.to_datetime(s, errors="coerce")
            types[c] = "ngày"
            continue
        if not all(isinstance(v, str) for v in nn.head(50)):
            # sqlite/excel số thực sự
            if pd.api.types.is_numeric_dtype(pd.to_numeric(nn, errors="coerce")) and pd.to_numeric(nn, errors="coerce").notna().mean() > 0.98 and not ID_HINT.search(c):
                if DATE_HINT.search(c) and pd.to_numeric(nn, errors="coerce").between(20000, 80000).mean() > 0.9:
                    df[c] = s.map(lambda v: parse_date_one(v, year_hint, allow_serial=True))
                    types[c] = "ngày"
                else:
                    df[c] = pd.to_numeric(s, errors="coerce")
                    types[c] = "số"
                continue
        sv = nn.astype(str)
        sample = sv.sample(min(len(sv), 3000), random_state=SEED) if len(sv) > 3000 else sv
        if ID_HINT.search(str(c)):
            df[c] = s.map(lambda v: re.sub(r"\s+", "", v) if isinstance(v, str) and re.fullmatch(r"\s*[A-Za-z]{0,6}\d+\s*", v) else v)
            types[c] = "mã/chữ"
            continue
        # ngày
        is_dateish = sample.str.match(r"^\s*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/\-.]\d{1,2}([/\-.]\d{2,4})?|\d{1,2}\s*thg|\d{5}(\.\d+)?$)").mean()
        if (DATE_HINT.search(str(c)) and is_dateish > 0.8) or is_dateish > 0.97:
            dayfirst = decide_dayfirst(sample)
            allow_serial = bool(DATE_HINT.search(str(c)))
            parsed = s.map(lambda v: parse_date_one(v, year_hint, dayfirst, allow_serial))
            ok = parsed.notna().sum() / max(1, s.notna().sum())
            if ok > 0.9:
                df[c] = pd.to_datetime(parsed)
                types[c] = "ngày" + ("" if dayfirst else " (tháng/ngày)")
                if ok < 1:
                    notes[name].append(f"cột '{c}': {int(s.notna().sum() - parsed.notna().sum())} ô ngày không đọc được")
                continue
        # số lượng kèm đơn vị: "1 ly", "2 hộp"
        with_unit = sample.str.match(r"^\s*-?\d+(\.\d+)?\s+[^\d\s].{0,15}$")
        plain_num = sample.str.match(r"^\s*-?\d+(\.\d+)?\s*$")
        units_seen = sample[with_unit].map(lambda v: re.sub(r"^\s*-?\d+(?:\.\d+)?\s*", "", v).strip().lower())
        money_units = units_seen.isin(["k", "tr", "đ", "d", "vnd", "vnđ", "triệu", "trieu", "nghìn", "ngàn", "đồng"]).mean() if len(units_seen) else 0
        qty_unit = with_unit.mean()
        # cột lẫn "3 chai" và "3" (đơn vị chỉ ghi ở một số dòng) cũng là số lượng kèm đơn vị
        mixed_qty = (with_unit | plain_num).mean() > 0.97 and qty_unit >= 0.05 and money_units < 0.2 and not MONEY_HINT.search(str(c))
        if (qty_unit > 0.9 and money_units < 0.2) or mixed_qty:
            def _qnum(v):
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                    return float(v)          # ô Excel là số thật (không kèm đơn vị)
                if isinstance(v, str) and re.match(r"\s*-?\d", v):
                    return float(re.match(r"\s*(-?\d+(?:\.\d+)?)", v)[1])
                return np.nan
            num = s.map(_qnum)
            unit = s.map(lambda v: re.sub(r"^\s*-?\d+(?:\.\d+)?\s*", "", v).strip() if isinstance(v, str) else None)
            if num.notna().sum() < 0.97 * s.notna().sum():
                types[c] = "chữ"
                notes[name].append(f"cột '{c}': trông như số lượng kèm đơn vị nhưng {int(s.notna().sum() - num.notna().sum())} ô không đọc được số → giữ dạng chữ")
                continue
            df[c] = num
            df[f"{c}__dv"] = unit.replace("", None)
            types[c] = "số (đã tách đơn vị sang cột __dv)"
            types[f"{c}__dv"] = "đơn vị" + ("" if qty_unit > 0.9 else " (chỉ ghi ở một số dòng; trống = đơn vị ở cột đơn vị tính/mặc định)")
            if qty_unit <= 0.9:
                notes[name].append(f"cột '{c}': {qty_unit:.0%} ô ghi kèm đơn vị ('3 chai'), còn lại chỉ có số → đã tách số và đơn vị")
            continue
        # tiền / số
        moneyish = sample.map(lambda v: bool(_MONEY_SUFFIX.match(v.replace(" ", "").replace("\xa0", "").lower()))).mean()
        if moneyish > 0.95:
            th = decide_thousands(sample)
            parsed = s.map(lambda v: parse_money_one(v, th))
            ok = parsed.notna().sum() / max(1, s.notna().sum())
            if ok > 0.97:
                df[c] = parsed.astype(float)
                types[c] = "tiền/số" if MONEY_HINT.search(str(c)) else "số"
                if ok < 1:
                    notes[name].append(f"cột '{c}': {int(s.notna().sum() - parsed.notna().sum())} ô số không đọc được")
                continue
        types[c] = "chữ"
    return df, types


ORDER_ID_HINT = re.compile(r"(đơn|don\b|hđ|\bhd\b|hoá đơn|hóa đơn|hoa don|bill|order|invoice|phiếu|phieu|chứng từ|chung tu|\bct\b)", re.I)


def _colnorm(c):
    """'MaHoaDon' / 'ma_hoa_don' / 'Mã hoá đơn' -> 'ma hoa don' (để nhận tên cột kiểu sqlite/camelCase)."""
    c = strip_accents(str(c)).replace("đ", "d").replace("Đ", "D")
    c = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", c)
    c = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", c)     # 'SLNhapQuyDoi' -> 'SL Nhap Quy Doi', 'IDCuaHang' -> 'ID Cua Hang'
    c = re.sub(r"[_\-.]+", " ", c)
    return re.sub(r"\s+", " ", c.lower()).strip()


ORDER_ID_NORM = re.compile(r"\b(ma|so|id)\s*(don|don hang|hoa don|hd|bill|order|phieu|chung tu|ct|giao dich|gd|invoice)\b|\border\s*id\b|\binvoice\b")


ORDER_ID_REV = re.compile(r"^(don|hoa don|hd|phieu|bill|chung tu)\s*(so|ma|#)$")
ORDER_ID_DESC = re.compile(r"^(ma don|ma hoa don|so hoa don|so don|ma phieu|so phieu|ma chung tu)\b")


def order_id_col(df, types):
    """Cột mã đơn: 'Mã đơn hàng', 'Mã hóa đơn', 'MaHoaDon', 'Mã phiếu', 'Mã giao dịch', 'order_id', 'Đơn số'…;
    không nhận ra theo tên thì theo MÔ TẢ README ('HĐ': số hoá đơn). Cuối cùng: cột __ma_don dựng từ STT đơn."""
    for c in df.columns:
        if str(c).startswith("__"):
            continue
        cn = _colnorm(c)
        if ORDER_ID_NORM.search(cn) or ORDER_ID_REV.match(cn):
            return c
    for c in df.columns:
        d_ = col_desc(df, c) if not str(c).startswith("__") else None
        if d_ and ORDER_ID_DESC.match(d_) and not re.search(r"thu tu dong|phieu thu|phieu nhap|phieu tra", d_):
            return c
    return "__ma_don" if "__ma_don" in df.columns else None


def fill_row_groups(df, types, name, notes):
    """Rulebook §8: ô trống ở các dòng sau của cùng một nhóm dòng = giống dòng đầu nhóm.
    Nhận diện: cột mã đơn có ô trống (không phải dòng đầu), các cột cấp-đơn trống ĐÚNG ở những dòng đó.
    Chỉ điền các ô trống thuộc dòng tiếp nối; ô trống ở dòng đầu nhóm (vd khách lẻ không có mã) giữ nguyên."""
    oid = order_id_col(df, types)
    if oid is None or not len(df):
        return df
    na = df[oid].isna()
    if not (0.02 < na.mean() < 0.97) or na.iloc[0]:
        return df
    grp = (~na).cumsum()
    cols = []
    for c in df.columns:
        if c == oid or str(c).startswith("__"):
            continue
        cn = df[c].isna()
        if cn[na].mean() >= 0.95 and cn[~na].mean() < 0.9:
            cols.append(c)
    if not cols:
        return df
    df = df.copy()
    for c in [oid] + cols:
        head = df.loc[~na, c]
        head.index = grp[~na].values
        df.loc[na, c] = grp[na].map(head).values
    notes[name].append(f"nhóm dòng: {int(na.sum())} dòng tiếp nối được điền theo dòng đầu nhóm cho cột {[oid] + cols}")
    return df


def drop_same_order_duplicates(df, types, name, notes):
    """Rulebook §8: hai dòng của CÙNG một đơn giống nhau mọi ô = lỗi xuất, tính một lần.
    Bảng không có mã đơn: giữ mọi dòng (hai dòng giống nhau vẫn tính cả hai)."""
    oid = order_id_col(df, types)
    if oid is None or not len(df) or df[oid].isna().mean() > 0.5:
        return df
    cols = [c for c in df.columns if c != "__nguon"]
    dup = df.duplicated(subset=cols, keep="first") & df[oid].notna()
    if dup.any():
        notes[name].append(f"bỏ {int(dup.sum())} dòng trùng hoàn toàn trong cùng một đơn (cột mã đơn '{oid}')")
        df = df[~dup].reset_index(drop=True)
    return df


# =============================================================================
# 2b. LỚP NGHĨA: thêm cột chuẩn để model tra cứu dễ, ít suy luận lại mỗi câu
#     __nhom_trang_thai  hoan_thanh | huy | tra_hang | dang_giao | chua_ro   (rulebook §1, §8)
#     __sl_co_ban / __dv_co_ban  số lượng quy về đơn vị cơ bản theo bảng quy đổi của shop (rulebook §5, §10)
#     __khach_le         True nếu khách lẻ / vãng lai / trống ở dòng đầu đơn (rulebook §3)
# Model vẫn phải đối chiếu README: ghi chú làm sạch nói rõ cách ánh xạ.
# =============================================================================

STATUS_COL = re.compile(r"(trang thai|tinh trang|status|tinhtrang|trangthai|ket qua|\bxu ly\b)(?!.*\b(kh|khach|nv|dong|hddt)\b)")
NOTE_COL = re.compile(r"(ghi chu|note|ghichu)")
STATUS_RULES = [  # thứ tự quan trọng: 'hoàn thành' xét trước 'hoàn' (trả); so theo TỪ (\b) để 'chuyển' không khớp 'huy'
    ("hoan_thanh", r"\bhoan thanh\b|\bthanh cong\b|\bda giao\b|\bda lay\b|\bda nhan\b|\bxong\b|\bdone\b|\bcomplete|\bsuccess|"
                   r"\bda thanh toan\b|\bok\b|\bda nhap\b|\breceived\b|\bnhap kho\b"),
    ("huy", r"\bhuy\b|\bcancel|\bvoid\b|\bbom\b|\bkhong nhan\b|\btu choi\b|\bthat bai\b|\bfail|\bkhong giao duoc\b"),
    ("tra_hang", r"\btra hang\b|\btra ve\b|\bhoan tra\b|\bhoan hang\b|\bhoan tien\b|\breturn|\brefund|\bdoi tra\b|\btra lai\b|"
                 r"^tra$|\bhoan\b|^khach tra$|^khach tra lai\b|\bkhach tra lai\b|\bkhach doi\b"),
    ("dang_giao", r"\bdang giao\b|\bdang ship|\bvan chuyen\b|\bcho giao\b|\bchua giao\b|\bdang xu ly\b|\bcho xu ly\b|\bpending\b|"
                  r"\bdelivering\b|\bshipping\b|\bdang giao dich\b|\bcho lay\b|\bcho xac nhan\b|\bdang gui\b"),
]
WALKIN = re.compile(r"^(khach le|khach vang lai|vang lai|khach ghe|walk ?in|khach|kl|le)$")


def status_group(v):
    k = key(v)
    if not k:
        return None
    for g, pat in STATUS_RULES:
        if re.search(pat, k):
            return g
    return "chua_ro"


COL_DESC = {}          # tên cột -> mô tả README (toàn shop, mô tả đầu tiên thắng) — đặt trong Business.__init__
COL_DESC_FILE = {}     # tên file (theo README '## Cột trong `file`') -> {tên cột: mô tả}


def parse_readme_cols(readme):
    """README: '- **Cột**: mô tả' (theo từng mục '## Cột trong `file`') và '`bang.Cot` là mô tả'.
    Trả (toàn_shop, theo_file). Mô tả đã key hoá (bỏ dấu, chữ thường)."""
    glob_, by_file, cur = {}, {}, None
    for line in (readme or "").splitlines():
        h = re.match(r"^\s*#+\s*.*?`([^`]+)`", line)
        if h and re.search(r"(cot|cột)", key(line)):
            cur = h.group(1).strip()
            by_file.setdefault(cur, {})
            continue
        if re.match(r"^\s*#", line):
            cur = None
        m = re.match(r"^\s*[-*]\s*\*\*([^*\n]+?)\*\*\s*[:：]\s*(.+)$", line)
        if m:
            c, d_ = m.group(1).strip(), key(m.group(2))
            glob_.setdefault(c, d_)
            if cur:
                by_file[cur].setdefault(c, d_)
    for m in re.finditer(r"`([\w.]+)\.([\w ]+)`\s+là\s+([^;.\n`]+)", readme or ""):
        t_, c, d_ = m.group(1).strip(), m.group(2).strip(), key(m.group(3))
        glob_.setdefault(c, d_)
        by_file.setdefault(t_, {}).setdefault(c, d_)
    return glob_, by_file


def _tfile(tname):
    """'pos.sqlite::don_hang' -> ['pos.sqlite', 'don_hang']; 'SoBanHang* (gộp 17 file)' -> ['SoBanHang*']."""
    parts = re.sub(r"\s*\(gộp.*?\)\s*$", "", str(tname)).split("::")
    return [p.strip() for p in parts if p.strip()]


def col_desc(df, c):
    """Mô tả README của cột c thuộc bảng df: ưu tiên mục README của ĐÚNG file/bảng đó, rồi mô tả toàn shop."""
    tn = df.attrs.get("tname", "") if hasattr(df, "attrs") else ""
    if tn and COL_DESC_FILE:
        for part in reversed(_tfile(tn)):
            base = part.rstrip("*")
            for f, mp in COL_DESC_FILE.items():
                fb = re.sub(r"\.(csv|xlsx|xls|sqlite|db|json)$", "", f)
                pb = re.sub(r"\.(csv|xlsx|xls|sqlite|db|json)$", "", base)
                if (f == part or fb == pb or (part.endswith("*") and fb.startswith(pb)) or fb.startswith(pb + "_") or pb.startswith(fb)) and str(c) in mp:
                    return mp[str(c)]
    return COL_DESC.get(str(c))


def _desc_match(df, pat, exclude=None):
    """Dự phòng khi tên cột không nói gì: tìm theo MÔ TẢ cột trong README (chỉ dùng cho trạng thái / SL cơ bản)."""
    for c in df.columns:
        d_ = col_desc(df, c)
        if d_ and not str(c).startswith("__") and re.search(pat, d_) and not (exclude and re.search(exclude, d_)):
            return c
    return None


def _find_col(df, pat, types=None, want=None, use_desc=False):
    for c in df.columns:
        if str(c).startswith("__"):
            continue
        if re.search(pat, _colnorm(c)) and (want is None or str((types or {}).get(c, "")).startswith(want)):
            return c
    return _desc_match(df, pat) if use_desc else None


STT_ORDER_README = re.compile(r"(cung|cùng)\s*\**\s*(stt|so thu tu|số thứ tự)\s*\**\s*(la|là)\s*(mot|một)\s*(don|đơn)", re.I)


def stt_order_key(b, name, df):
    """Sổ không có mã đơn nhưng README nói 'các dòng cùng STT là một đơn' (hoặc mô tả cột 'số thứ tự đơn'):
    dựng __ma_don = nguồn(sheet/file) + STT (+ ngày nếu STT đánh lại mỗi ngày). Trả mô tả hoặc None."""
    if order_id_col(df, b.types[name]) is not None:
        return None
    sc = None
    for c in df.columns:
        if str(c).startswith("__"):
            continue
        cn, d_ = _colnorm(c), col_desc(df, c) or ""
        if re.search(r"so thu tu don|thu tu hoa don|so thu tu (hoa don|bill)", d_) or (re.fullmatch(r"stt|so thu tu|so tt", cn) and STT_ORDER_README.search(getattr(b, "readme", "") or "")):
            if re.search(r"thu tu dong", d_):
                continue
            sc = c
            break
    if sc is None or df[sc].isna().mean() > 0.5:
        return None
    parts = [df["__nguon"].astype(str)] if "__nguon" in df.columns else []
    dc = _date_col(df, b.types[name])
    if dc is not None:
        g = df.assign(_k=df[sc].astype(str)).groupby((["__nguon"] if "__nguon" in df.columns else []) + ["_k"])[dc].nunique()
        if (g > 1).mean() > 0.05:      # STT đánh lại theo ngày -> thêm ngày vào khoá
            parts.append(pd.to_datetime(df[dc], errors="coerce").dt.strftime("%Y%m%d"))
    k = df[sc].map(lambda v: str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v) and float(v).is_integer() else str(v).strip())
    for p_ in reversed(parts):
        k = p_ + "|" + k
    df["__ma_don"] = k.where(df[sc].notna())
    b.types[name]["__ma_don"] = "mã đơn (tự dựng)"
    return f"`{name}`: không có mã đơn → __ma_don = " + (" + ".join(["nguồn"] * ("__nguon" in df.columns) + ([ "ngày"] if len(parts) > ("__nguon" in df.columns) else []) + [f"'{sc}'"])) + " (README: các dòng cùng STT là một đơn)"


def enrich_semantics(b):
    """Thêm cột nghĩa vào các bảng của shop b (sửa tại chỗ) + ghi chú để model biết."""
    sem = []
    for name, df in b.T.items():
        try:
            m_ = stt_order_key(b, name, df)
        except Exception:
            m_ = None
        if m_:
            sem.append(m_)
    # --- 1. nhóm trạng thái
    for name, df in b.T.items():
        sc = _find_col(df, STATUS_COL.pattern, use_desc=True)
        if order_id_col(df, b.types[name]) is None:
            # chỉ bảng đơn / phiếu (bảng khách, danh mục có 'trạng thái' khác nghĩa) — NGOẠI LỆ: sổ bán không ghi mã đơn
            # nhưng có cột 'Trạng thái đơn' / là bảng dòng bán (có hàng + số lượng + tiền)
            try:
                kd = _table_kind(name, df, b.types[name])
            except Exception:
                kd = "khac"
            if not (sc is not None and (re.search(r"\b(don|hoa don|bill|order|phieu)\b", _colnorm(sc)) or kd in ("dong_ban", "don_ban", "nhap"))
                    and kd not in ("khach", "danh_muc", "ton", "chi_phi", "han_no", "quy_doi")):
                continue
        if sc is not None and NOTE_COL.search(_colnorm(sc)) and not re.match(r"^(trang thai|tinh trang)", col_desc(df, sc) or ""):
            sc = None          # cột GHI CHÚ (dù mô tả có nhắc trạng thái) là ghi chú tự do: không nhắc gì = hoàn thành
        src = sc
        if sc is None and order_id_col(df, b.types[name]) is not None:
            nc = _find_col(df, NOTE_COL.pattern)
            if nc is not None and "ghi chu" in key(getattr(b, "readme", "")) + " " + key(" ".join(map(str, df.columns))):
                src = nc
        if src is None:
            continue
        vals = df[src].dropna().astype(str).unique()
        if len(vals) > 60 and src == sc:
            continue
        mp = {v: status_group(v) for v in vals}
        if src != sc:   # suy từ Ghi chú: không nhắc trạng thái -> hoàn thành (luật 'Sổ không có cột trạng thái')
            mp = {v: (g if g != "chua_ro" else "hoan_thanh") for v, g in mp.items()}
        df["__nhom_trang_thai"] = df[src].astype(str).map(mp).where(df[src].notna(), "hoan_thanh" if src != sc else None)
        b.status_src = dict(getattr(b, "status_src", {}) or {}, **{name: src})      # cho bài kiểm từ luật (chèn đơn huỷ giả)
        b.types[name]["__nhom_trang_thai"] = "nhóm trạng thái (tự động)"
        by = defaultdict(list)
        for v, g in mp.items():
            by[g].append(v)
        desc = "; ".join(f"{g}: {', '.join(sorted(vs)[:8])}" for g, vs in sorted(by.items()))
        sem.append(f"`{name}`: cột __nhom_trang_thai lấy từ '{src}'" + (" (suy từ Ghi chú, không nhắc trạng thái = hoàn thành)" if src != sc else "") + f" → {desc}")
        if by.get("chua_ro"):
            sem.append(f"  ⚠ giá trị trạng thái CHƯA RÕ nhóm (đọc README để xếp): {by['chua_ro'][:10]}")
    # --- 2. bảng quy đổi đơn vị
    conv = None
    for name, df in b.T.items():
        fc = _find_col(df, r"(ti le|he so|quy doi|factor|so luong quy doi)", b.types[name], "số")
        bc = _find_col(df, r"(don vi co ban|dv co ban|base)")
        uc = _find_col(df, r"(don vi quy doi|don vi lon|dv lon|don vi)(?! co ban)")
        pc = _find_col(df, r"(ten san pham|ten hang|san pham|mat hang|ma hang|ma sp|ten sp)")
        if fc and bc and uc and pc and uc != bc and len(df) < 20000:
            conv = (name, pc, uc, fc, bc)
            break
    if conv:
        name, pc, uc, fc, bc = conv
        cdf = b.T[name]
        table = {}
        base_of = {}
        for _, r in cdf.iterrows():
            if pd.isna(r[fc]):
                continue
            table[(key(r[pc]), key(r[uc]))] = float(r[fc])
            base_of[key(r[pc])] = str(r[bc])
        n_tab = 0
        for tname, df in b.T.items():
            if tname == name:
                continue
            p2 = _find_col(df, r"(ten san pham|ten hang|san pham|mat hang|ma hang|ma sp|ten sp)")
            q2 = _find_col(df, r"(so luong|sl\b|qty)", b.types[tname], "số")
            u2 = next((c for c in df.columns if str(c).endswith("__dv")), None) if q2 and f"{q2}__dv" in df.columns else None
            u_main = _find_col(df, r"(don vi tinh|dvt|don vi)(?! co ban)")
            if not (p2 and q2 and (u2 or u_main)):
                continue
            pk = df[p2].map(key)
            if u2 and u_main:
                unit = df[u2].where(df[u2].notna(), df[u_main])
            else:
                unit = df[u2 or u_main]
            uk = unit.map(key)
            fac = [table.get((a, u), 1.0) for a, u in zip(pk, uk)]
            df["__sl_co_ban"] = df[q2] * pd.Series(fac, index=df.index)
            df["__dv_co_ban"] = [base_of.get(a, u) for a, u in zip(pk, unit.astype(str))]
            b.types[tname]["__sl_co_ban"] = "số (quy về đơn vị cơ bản, tự động)"
            b.types[tname]["__dv_co_ban"] = "đơn vị cơ bản (tự động)"
            n_conv = int((pd.Series(fac) != 1.0).sum())
            sem.append(f"`{tname}`: __sl_co_ban = '{q2}' × hệ số theo `{name}` (khớp tên hàng + đơn vị '{(u2 or u_main)}'); {n_conv} dòng có đơn vị lớn được quy đổi")
            n_tab += 1
        if not n_tab:
            sem.append(f"có bảng quy đổi `{name}` nhưng chưa ghép tự động được vào bảng bán/nhập: tự quy đổi trong code")
    # --- 3. khách lẻ
    for name, df in b.T.items():
        cc = _find_col(df, r"(ten khach|khach hang|ten kh|ma kh|ma khach|customer)")
        if cc is None or order_id_col(df, b.types[name]) is None:
            continue
        kv = df[cc].map(lambda v: bool(WALKIN.match(key(v))) if pd.notna(v) and str(v).strip() else True)
        if kv.any() and kv.mean() < 0.95:
            df["__khach_le"] = kv
            b.types[name]["__khach_le"] = "khách lẻ (tự động: tên kiểu 'khách lẻ/vãng lai' hoặc trống)"
            sem.append(f"`{name}`: __khach_le từ '{cc}' ({int(kv.sum())} dòng); khách lẻ không là khách khi đếm/xếp hạng nhưng vẫn tính doanh thu, tổng nợ")
    b.semantics = sem
    return sem


# =============================================================================
# 2c. BỘ BẢNG CHUẨN dùng chung mọi shop: V['don'] (mỗi đơn 1 dòng), V['dong'] (dòng hàng bán),
#     V['nhap'] (dòng nhập hàng), V['thu'] (khoản thu / trả nợ). Model dùng V thay vì tự ghép bảng gốc.
#     Mỗi cột chuẩn ghi rõ lấy từ cột gốc nào (b.V_src) để model / người kiểm đối chiếu.
# =============================================================================

PROD_PAT = r"(ten hang|ten san pham|ten sp|san pham|mat hang|hang ban|hang hoa|chi tiet hang|ma hang|ma sp|ma sku|ma mat hang|ma hang ban|id sp|ma hh|ten mon|\bmon\b|product|\bhang\b|noi dung)"
QTY_PAT = r"(so luong|\bsl\b|s l|sluong|slg|soluong|qty|sl ban|sl nhap)"
BASEQTY_PAT = r"(so luong quy doi|quy ra dv le|sl quy doi|so luong le|sl le|sl co ban|slquydoi|sl quy ra le|so luong \(dvt co ban\)|so luong dvt co ban|sl nhap quy doi)"
AMT_PAT = r"(thanh tien|t tien|tien ban|tri gia|tong tien|thanh_tien|amount|\btien\b|\btong\b|tritia|trigia)"
PRICE_PAT = r"(don gia|gia ban|gia le|gia don vi|gia 1 dv|gia san pham|gia niem yet|\bgia\b|price|giadonvi|gia nhap|dongianhap)"
DISC_PAT = r"(giam gia|chiet khau|khuyen mai|\bck\b|chietkhau)"
DATE_PREF = r"(ngay dat|ngay ban|ngay lap|ngaylap|ngay giao dich|ngay mua|ngay thang|ngay chung tu|ngayban|^thoi gian$|^ngay$|ngay nhap|ngaynhapkho|thoi gian)"
CUST_CODE = r"(ma kh|ma khach|customer id|ma khach mua|so kh|makhachhang)"
CUST_NAME = r"(ten khach|khach hang|ten kh|\bkhach\b|ben mua|doi tac|tenkh)"
HEAD_TOTAL = r"(tong sau giam|tong thanh toan|tongthanhtoan|phai thu|tong don|tong bill|tong tien hang|tien hang|tongtien|tong tien)"
PAID_PAT = r"(khach da tra|khach tra|da tra|da thu|khachdatra|thanh toan$)"
DEBT_PAT = r"(con no|du no|chua tra|con lai|conphaithu|con phai thu)"
CHANNEL_PAT = r"(kenh|ban qua|nguon don|loai ban|kenhbanhang)"
PAYTYPE_PAT = r"(phuong thuc|hinh thuc thanh toan|tra bang|pttt|httt|phuongthuctt)"
STAFF_PAT = r"(nguoi ban|nhan vien|ma nv|manhanvien)"


def _col(df, pat, types=None, want=None, exclude=None):
    for c in df.columns:
        cn = _colnorm(c)
        if str(c).startswith("__") or (exclude and re.search(exclude, cn)):
            continue
        if re.search(pat, cn) and (want is None or str((types or {}).get(c, "")).startswith(want)):
            return c
    return None


PROD_EX = r"cua hang|chi nhanh|khach hang|don hang|ma don|nhom|loai|kho hang|giao hang|tra hang|tien hang|^id$|nha cung cap"
LINE_EX = r"|ngay|thoi gian|dat hang|so luong|\bgia\b|tien|trang thai"
CUST_NAME_EX = r"nhom|loai|^ma |\btra\b|tien|thanh toan|\bno\b|sdt|dien thoai|dia chi|so kh|\bid\b|ma kh"
# trường chuẩn của bảng ĐƠN: (mẫu theo MÔ TẢ README — ưu tiên, mẫu theo TÊN cột, loại trừ theo tên, kiểu)
HEAD_FIELDS = {
    "ngay":          (r"^ngay ban|^ngay lap|^ngay dat", None, None, "ngày"),
    "khach_ma":      (r"^ma khach", CUST_CODE, None, None),
    "khach_ten":     (r"^ten khach", CUST_NAME, CUST_NAME_EX, None),
    "kenh":          (r"^kenh ban", CHANNEL_PAT, None, None),
    "httt":          (r"hinh thuc thanh toan", PAYTYPE_PAT, None, None),
    "nhan_vien":     (r"nhan vien", STAFF_PAT, None, None),
    "tien_hang":     (r"tong tien hang|tien hang cua don", r"(tong tien hang|tien hang)", r"sau", None),
    "giam_gia_don":  (r"^giam gia|^chiet khau", DISC_PAT, r"ma |phan tram|%", None),
    "phi_giao":      (r"phi giao|phi van chuyen|phi ship", r"(phi van chuyen|phi giao|phi ship|tien ship|ship)", r"ma |hinh thuc|don vi|ngay", None),
    "tong_don":      (r"^tong tien don|phai tra cho ca don", r"(tong sau giam|tong thanh toan|tongthanhtoan|phai thu|phai tra|tong don|tong bill|^tong tien$|^tongtien$)", r"tien hang|truoc giam|da tra|con", None),
    "da_tra_file": (r"^da thanh toan|^da tra|^khach da tra", r"(khach da tra|^da tra$|^da thu$|khachdatra|^khach tra$|^thanh toan$)", r"tong|phai|hinh thuc|phuong thuc|ngay|cach", None),
    "con_no_file":   (r"con no|du no", DEBT_PAT, r"han|ngay|tuoi", None),
}


def _pick(df, ty, field, spec=None):
    """Chọn cột cho một trường chuẩn: MÔ TẢ README trước (đáng tin nhất), rồi TÊN cột; kiểm kiểu nếu cần."""
    dpat, npat, ex, want = spec or HEAD_FIELDS[field]
    ok = lambda c: want is None or str(ty.get(c, "")).startswith(want) or (want == "số" and str(ty.get(c, "")).startswith("tiền"))
    if dpat:
        for c in df.columns:
            d_ = col_desc(df, c)
            if d_ and not str(c).startswith("__") and re.search(dpat, d_) and ok(c):
                return c
    if npat:
        for c in df.columns:
            cn = _colnorm(c)
            if str(c).startswith("__") or (ex and re.search(ex, cn)):
                continue
            # cột đã có mô tả README nói nghĩa KHÁC thì không lấy theo tên (vd 'Khách trả' = đã thanh toán, không phải tên khách)
            d_ = col_desc(df, c)
            if d_ and dpat and any(re.search(o[0], d_) for f2, o in HEAD_FIELDS.items() if f2 != field and o[0]):
                continue
            if re.search(npat, cn) and ok(c):
                return c
    return None


def _date_col(df, types):
    dcols = [c for c in df.columns if str(types.get(c, "")).startswith("ngày")
             and not re.search(r"\b(sinh|han|hsd|ap dung|tao)\b|giao hang|\bthu$|ngay thu", _colnorm(c))]   # \b: 'hàng' không khớp 'hạn' 
    for c in dcols:
        if re.search(DATE_PREF, _colnorm(c)):
            return c
    return dcols[0] if dcols else None


def _table_kind(name, df, types):
    tn = _colnorm(name)            # tách camelCase / gạch dưới: 'DanhSachQuyDoi' -> 'danh sach quy doi'
    n = tn + " " + " ".join(_colnorm(c) for c in df.columns)
    if re.search(r"(tra hang|phieu tra|hang tra|tra lai|khach tra hang)", tn) and not re.search(r"(nhap|ncc)", tn):
        return "tra"
    if re.search(r"\blo\b", tn) and not re.search(r"(hoa don|don hang|ban hang)", tn.replace("don hang lo", "")):
        return "lo"
    if re.search(r"(nha cung cap|\bncc\b|supplier)", tn) and not _col(df, QTY_PAT, types, "số"):
        return "danh_muc"
    if (re.search(r"(quy doi|don vi tinh|don vi sp|donviquydoi|don vi quy doi)", tn) or _col(df, r"(he so quy doi|hesoquydoi|ti le quy doi)")) \
            and len(df) < 5000 and not _col(df, QTY_PAT, types, "số"):
        return "quy_doi"
    if re.search(r"(nhap hang|nhap kho|phieu nhap|don nhap|chi tiet nhap|chitietnhap)", tn) or re.search(r"\b(nha cung cap|ncc|sl nhap|gia nhap|so luong nhap)\b", n) and not re.search(r"hoa don|don hang|ban hang", tn):
        return "nhap"
    if re.search(r"(phieu thu|so quy|thanh toan|thu no|soquy|thanh_toan)", tn) or (re.search(r"(phieu thu|ma phieu thu|thoi gian thu|ngay thu)", n) and not _col(df, PROD_PAT)):
        return "thu"
    if re.search(r"(giao hang|lan giao|luot giao|so giao|ngay cho|dot giao)", n) and re.search(r"(giao|cho)", tn):
        return "giao"
    if re.search(r"(\blo\b|ban theo lo)", tn) or (_col(df, r"(so luong tra|sl tra|sltraquydoi)") and not _col(df, AMT_PAT, types)):
        return "lo"
    if re.search(r"(cong no theo han|han no|han tra)", n) and len(df.columns) <= 5:
        return "han_no"
    if re.search(r"(khach hang|danh sach khach|khach_hang)", tn) and not _col(df, QTY_PAT):
        return "khach"
    if re.search(r"(quy doi|don vi tinh|don_vi_sp|donviquydoi|danh muc)", tn) and len(df) < 5000 and not order_id_col(df, types):
        return "quy_doi"
    if re.search(r"(ton kho|bao cao ton|ton_kho|the kho|the_kho)", tn):
        return "ton"
    if re.search(r"(chi phi|so tay chi)", tn):
        return "chi_phi"
    if re.search(r"(san pham|hang hoa|danh sach thuoc|thuoc|bang gia|lich su gia|chi nhanh|nhan vien|nha cung cap|phong ban)", tn) and not order_id_col(df, types):
        return "danh_muc"
    has_prod = _col(df, PROD_PAT) or _desc_match(df, r"^(ten hang|ma hang|ten san pham|hang ban|ten mon|mat hang)")
    has_qty = _col(df, QTY_PAT, types, "số") or _desc_match(df, r"^so luong")
    if has_prod and has_qty:
        return "dong_ban"
    if order_id_col(df, types) is not None and _col(df, HEAD_TOTAL + "|" + AMT_PAT, types):
        return "don_ban"
    # sổ DỊCH VỤ (sửa xe, spa, giặt là…): mỗi dòng một phiếu/lượt, có ngày + tiền cả phiếu + nội dung việc, không có số lượng hàng
    has_date = any(str(types.get(c, "")).startswith("ngày") for c in df.columns)
    money = _desc_match(df, r"tien ca phieu|tong tien|so tien|thanh tien") or _col(df, r"(^tien$|tong tien|thanh tien|so tien|tien cong|thu)", types, "tiền") \
        or _col(df, r"(^tien$|tong tien|thanh tien|so tien)", types, "số")
    if has_date and money and not has_qty and len(df) >= 30 and \
            re.search(r"(sua|dich vu|phieu|luot|lam gi|cong viec|\bviec\b|bao duong|ky thuat|tho\b)", n + " " + " ".join(col_desc(df, c) or "" for c in df.columns)):
        return "dich_vu"
    return "khac"


CONV_LINE = re.compile(r"^\s*[-*]\s*(.+?)\s*:\s*1\s+(.+?)\s*=\s*([\d][\d.,]*)\s+(.+?)\s*\.?\s*$")


def readme_conversions(readme):
    """README '- Bia Tiger lon: 1 thùng = 24 lon' -> [(khoá hàng, khoá đơn vị lớn, hệ số, đơn vị nhỏ)]."""
    out = []
    for line in (readme or "").splitlines():
        m = CONV_LINE.match(line)
        if not m:
            continue
        f = m.group(3).replace(",", ".")
        if f.count(".") > 1 or (f.count(".") == 1 and len(f.split(".")[1]) == 3):
            f = f.replace(".", "")
        try:
            out.append((key(m.group(1)), key(m.group(2)), float(f), m.group(4).strip()))
        except ValueError:
            pass
    return out


def _chain(tab, base_u):
    """Nhân dồn nhiều cấp: thùng = 4 lốc, lốc = 6 lon -> thùng = 24 lon (tối đa 5 cấp). Trả (bảng, đơn vị cơ bản, số mục nhiều cấp)."""
    multi, final, base = 0, {}, {}
    for (pk, uk), f in tab.items():
        bu, seen = base_u.get((pk, uk)), {uk}
        hop = False
        while bu and (pk, key(bu)) in tab and key(bu) not in seen and len(seen) < 6:
            seen.add(key(bu))
            f *= tab[(pk, key(bu))]
            bu = base_u.get((pk, key(bu)))
            hop = True
        multi += hop
        final[(pk, uk)] = f
        if bu and (pk not in base or hop):
            base[pk] = bu
    return final, base, multi


def _conversion(b):
    """Quy đổi -> {"tab": (khoá hàng, khoá đơn vị) -> hệ số về đơn vị CƠ BẢN, "base": khoá hàng -> đơn vị cơ bản, ...}.
    Nguồn: bảng quy đổi của shop (nếu có) + dòng quy đổi trong README. Xử lý NHIỀU CẤP (thùng → lốc → lon)."""
    tab, base_u, srcs = {}, {}, []
    for name, df in b.T.items():
        ty = b.types[name]
        if _table_kind(name, df, ty) != "quy_doi":
            continue
        fc = _pick(df, ty, "he_so", (r"he so|so don vi co ban trong|ti le", r"(ti le|he so|quy doi|factor|\bbang\b|hesoquydoi)", r"^ma |gia", "số"))
        pcs = [c for c in (_col(df, r"(ten san pham|ten hang|ten mon|ten thuoc)"), _col(df, r"(ma hang|ma sp|mahanghoa|ma sku|ma hh|\bhang\b)")) if c]
        uc = _pick(df, ty, "dv_lon", (r"^don vi lon|^don vi quy doi", r"(don vi quy doi|don vi lon|dv lon|dv nhap|don vi tinh|dvt|donvitinh|^don vi$)", r"co ban|goc|nho|gia", None))
        bc = _pick(df, ty, "dv_cb", (r"^don vi co ban", r"(don vi co ban|dv co ban|don vi goc|dv goc|dv nho|base)", None, None))
        if not (fc and pcs and uc):
            continue
        srcs.append(name)
        for _, r in df.iterrows():
            if pd.isna(r[fc]) or pd.isna(r[uc]):
                continue
            for pc in pcs:
                if pd.isna(r[pc]):
                    continue
                k_ = (key(r[pc]), key(r[uc]))
                tab[k_] = float(r[fc])
                if bc and pd.notna(r[bc]):
                    base_u[k_] = str(r[bc])
    rc = readme_conversions(getattr(b, "readme", ""))
    if rc:
        srcs.append("README")
        for pk, uk, f, small in rc:
            tab.setdefault((pk, uk), f)
            base_u.setdefault((pk, uk), small)
    if not tab:
        return None
    final, base, multi = _chain(tab, base_u)
    return {"bang": " + ".join(srcs), "tab": final, "base": base, "nhieu_cap": multi}


PHONE_TAIL = re.compile(r"\s*[-–(,]?\s*\+?[\d][\d .]{7,}\)?\s*$")


def clean_person_name(v):
    """'Dương Hùng Trung - 0328 548 455' -> 'Dương Hùng Trung' (README: cột tên có thể dính số điện thoại)."""
    if not isinstance(v, str):
        return v
    return PHONE_TAIL.sub("", v).strip()


def _name_lookup(b, code_col, skip_table, kind_hint=None, codes=None):
    """Mã -> tên từ bảng danh mục. Tìm cột mã theo (1) cùng tên cột, (2) mô tả README 'mã ...', (3) trùng giá trị với mã cần tra.
    Cột tên theo mô tả README 'tên' trước, rồi theo tên cột. None nếu không chắc."""
    cn = _colnorm(code_col)
    codes = set(map(str, codes)) if codes is not None else None
    for n, t in b.T.items():
        if n == skip_table or len(t) > 50000:
            continue
        if kind_hint and not re.search(kind_hint, _colnorm(n)):
            continue
        cands = [c for c in t.columns if _colnorm(c) == cn]
        cands += [c for c in t.columns if c not in cands and re.search(r"^ma ", col_desc(t, c) or "")]
        if codes:
            cands += [c for c in t.columns if c not in cands and not str(c).startswith("__")
                      and t[c].map(sid).isin(codes).mean() > 0.5]
        for cc in cands:
            vals = t[cc].map(sid)
            if not vals.is_unique or (codes and len(codes & set(vals)) < 0.5 * min(len(codes), len(vals))):
                continue
            nm = next((c for c in t.columns if c != cc and re.fullmatch(r"(ten|ho ten|ten khach|ten ncc|ten nha cung cap)( .*)?", col_desc(t, c) or "")), None) \
                or next((c for c in t.columns if c != cc and re.search(r"^(ten|ho ten|ten goi|tengoi|name|ten ncc|ten nha cung cap|ten khach|khach hang \(ten\))", _colnorm(c))), None)
            if nm is not None:
                return dict(zip(vals, t[nm].map(clean_person_name))), f"{n}::{nm}"
    return None, None


PLATE = re.compile(r"\b(\d{2}\s?[A-Z]{1,2}\d?\s?-?\s?\d{3}\.?\d{2})\b", re.I)
HONORIFIC = re.compile(r"^(anh|chi|chị|em|co|cô|chu|chú|bac|bác|ong|ông|ba|bà|a|c|e)\s*[.\s]\s*", re.I)


def _goi_key(v):
    """'a Đức', 'A.Đức', 'anh Đức', 'Đức' -> 'duc' (tên gọi, bỏ danh xưng)."""
    if not isinstance(v, str):
        return None
    x = HONORIFIC.sub("", v.strip())
    return key(x) or None


def build_service(b, kinds, S, V, src, notes):
    """Shop DỊCH VỤ không có bảng bán hàng: mỗi dòng sổ là MỘT PHIẾU (một đơn). V['don'] + V['viec'] + V['nhan_vien']."""
    df, ty = b.T[S], b.types[S]
    dc = _date_col(df, ty) or next((c for c in df.columns if str(ty.get(c, "")).startswith("ngày")), None)
    ac = _desc_match(df, r"tien ca phieu|tong tien") or _col(df, r"(^tien$|tong tien|thanh tien|so tien)", ty, "tiền") or _col(df, r"(^tien$|tong tien|thanh tien|so tien)", ty, "số")
    wc = _desc_match(df, r"viec|dich vu|noi dung") or _col(df, r"(lam gi|cong viec|noi dung|dich vu|hang muc|\bviec\b)")
    sc = _desc_match(df, r"^tho|nhan vien|ky thuat") or _col(df, r"(\btho\b|nhan vien|ky thuat|nguoi lam)")
    xc = _desc_match(df, r"dong xe|bien so|khach|thiet bi|may") or _col(df, r"(^xe$|bien so|khach|may|thiet bi)")
    nc = _col(df, r"(ghi chu|note)")
    od = pd.DataFrame({"ma_don": (df["__nguon"].astype(str) + "|" if "__nguon" in df.columns else "") + "P" + df.index.astype(str)})
    od["ngay"] = pd.to_datetime(df[dc], errors="coerce") if dc else pd.NaT
    od["tong_don"] = df[ac].map(lambda v: parse_money_one(v) if isinstance(v, str) else v).astype(float) if ac else np.nan
    od["doanh_thu"] = od["tong_don"]
    if wc:
        od["noi_dung"] = df[wc]
        viec = df[wc].map(lambda v: "" if pd.isna(v) else str(v)).map(lambda v: [w.strip() for w in re.split(r"\s*(?:\+|,|;|/| và )\s*", v) if w.strip() and w.strip().lower() != "nan"])
        od["so_viec"] = viec.map(len)
        vv = pd.DataFrame({"ma_don": od["ma_don"].values, "viec": viec.values}).explode("viec").dropna()
        vv["viec_key"] = vv["viec"].map(key)
        V["viec"] = vv.reset_index(drop=True)
        src["viec"] = f"tách {S}::{wc} theo '+', ',', ';' (một phiếu nhiều việc; tiền chỉ có ở cả phiếu)"
    if xc:
        od["doi_tuong"] = df[xc]
        pl = df[xc].astype(str).str.extract(PLATE)[0]
        if pl.notna().mean() > 0.5:
            od["bien_so"] = pl.str.upper().str.replace(r"\s", "", regex=True)
            od["dong_xe"] = [str(a).replace(str(p), "").strip() if isinstance(p, str) else a for a, p in zip(df[xc], pl)]
    if nc:
        od["ghi_chu"] = df[nc]
    od["bao_hanh"] = (od.get("ghi_chu", pd.Series("", index=od.index)).astype(str).map(key).str.contains(r"\bbao hanh\b")) & od["tong_don"].fillna(0).eq(0)
    od["nhom_tt"] = "hoan_thanh"
    od["tinh_doanh_thu"] = True
    # thợ: tên gọi trong sổ -> họ tên trong danh sách thợ (khớp tên gọi = chữ cuối họ tên); trùng tên gọi thì để trống
    staff = None
    for n_, t_ in b.T.items():
        if n_ == S:
            continue
        hc = _desc_match(t_, r"ho ten|ten tho|ten nhan vien") or _col(t_, r"(ho ten|ten tho|ten nhan vien|^ten$)")
        pc = _desc_match(t_, r"hoa hong|phan tram") or _col(t_, r"(hoa hong|phan tram|% )")
        if hc and len(t_) < 500:
            staff = (n_, hc, pc)
            break
    if sc:
        od["nhan_vien_goi"] = df[sc]
        gk = df[sc].map(_goi_key)
        if staff:
            n_, hc, pc = staff
            st = b.T[n_]
            last = st[hc].map(lambda v: key(v).split()[-1] if isinstance(v, str) and key(v) else None)
            cnt = last.value_counts()
            mp = {k_: v_ for k_, v_ in zip(last, st[hc]) if cnt.get(k_, 0) == 1}
            full = {key(v_): v_ for v_ in st[hc] if isinstance(v_, str)}
            od["nhan_vien"] = [(full.get(g) or mp.get(g) or mp.get(g.split()[-1])) if isinstance(g, str) and g else None for g in gk]
            miss = od["nhan_vien"].isna() & gk.notna()
            if miss.any():
                notes.append(f"{int(miss.sum())} phiếu có tên thợ không khớp danh sách thợ (hoặc trùng tên gọi): {sorted(set(df.loc[miss, sc].astype(str)))[:6]}")
            nv = pd.DataFrame({"ho_ten": st[hc]})
            if pc:
                nv["hoa_hong_pct"] = pd.to_numeric(st[pc], errors="coerce")
            for c in st.columns:
                if str(b.types[n_].get(c, "")).startswith("ngày"):
                    nv["ngay_vao_lam"] = st[c]
                    break
            V["nhan_vien"] = nv
            src["nhan_vien"] = f"{n_}::{hc}" + (f", {pc}" if pc else "")
        else:
            od["nhan_vien"] = gk
    V["don"] = od.reset_index(drop=True)
    src["don"] = f"{S}: MỖI DÒNG MỘT PHIẾU (shop dịch vụ, không có bảng bán hàng); tong_don = {ac}"
    src["don.doanh_thu"] = f"{S}::{ac} (tiền cả phiếu; phiếu bảo hành tiền 0 vẫn là một lượt)"
    notes.append("shop DỊCH VỤ: V['don'] mỗi dòng một phiếu; không có dòng hàng / tồn kho; tiền từng việc KHÔNG có (chỉ tiền cả phiếu) → câu hỏi tiền theo từng việc là không trả lời được trừ khi rulebook chỉ cách")
    m_ = re.search(r"có đến hết\s*(?:ngày\s*)?(\d{1,2})/(\d{1,2})/(\d{4})", getattr(b, "readme", "") or "")
    b.data_end = pd.Timestamp(int(m_.group(3)), int(m_.group(2)), int(m_.group(1))) if m_ else None
    b.V, b.V_src, b.V_notes = V, src, notes


def self_check_V(b, V, notes):
    """Tự đối soát bộ bảng chuẩn (không cần đáp án): lệch thì ghi CẢNH BÁO để model / người xem biết chỗ đáng ngờ."""
    de = getattr(b, "data_end", None)
    o = V.get("don")
    if o is not None and len(o):
        if o["ma_don"].duplicated().any():
            notes.append(f"KIỂM: {int(o['ma_don'].duplicated().sum())} mã đơn lặp trong V['don']")
        ng = pd.to_datetime(o.get("ngay"), errors="coerce") if "ngay" in o.columns else None
        if ng is not None:
            if ng.isna().mean() > 0.005:
                notes.append(f"KIỂM: {ng.isna().mean():.1%} đơn không có ngày (không vào được kỳ nào)")
            if de is not None and (ng > de + pd.Timedelta(days=1)).any():
                notes.append(f"KIỂM: {int((ng > de + pd.Timedelta(days=1)).sum())} đơn có ngày SAU ngày cuối dữ liệu {de.date()} (ngày đọc sai?)")
            if (ng < pd.Timestamp("2015-01-01")).any():
                notes.append(f"KIỂM: {int((ng < pd.Timestamp('2015-01-01')).sum())} đơn có ngày trước 2015 (ngày đọc sai?)")
        if "doanh_thu" in o.columns:
            bad = o["tinh_doanh_thu"] & (pd.to_numeric(o["doanh_thu"], errors="coerce").fillna(0) <= 0)
            if bad.mean() > 0.01:
                notes.append(f"KIỂM: {int(bad.sum())} đơn tính doanh thu nhưng doanh_thu ≤ 0 (thiếu dòng hàng / sai cột tiền?)")
        if {"tien_hang", "doanh_thu"} <= set(o.columns) and "dong" in V:
            th = pd.to_numeric(o["tien_hang"], errors="coerce")
            lech = (th - o["doanh_thu"]).abs() > 1
            if th.notna().any() and lech[th.notna()].mean() > 0.02:
                notes.append(f"KIỂM: {lech[th.notna()].mean():.1%} đơn có tổng thành tiền dòng ≠ tiền hàng ghi ở đầu đơn (đọc README: dòng có trừ giảm giá? có dòng lạ?)")
        if "thu" in V and "tong_don" in o.columns and {"ma_don", "so_tien"} <= set(V["thu"].columns):
            paid = pd.to_numeric(V["thu"]["so_tien"], errors="coerce").groupby(V["thu"]["ma_don"]).sum()
            over = o["ma_don"].map(paid) - pd.to_numeric(o["tong_don"], errors="coerce") > 1
            if over.mean() > 0.01:
                notes.append(f"KIỂM: {int(over.sum())} đơn có tổng tiền thu > tổng đơn (bảng thu có gồm tiền hoàn / phí khác?)")
            nomatch = ~V["thu"]["ma_don"].isin(set(o["ma_don"]))
            if nomatch.mean() > 0.05:
                notes.append(f"KIỂM: {nomatch.mean():.0%} khoản thu không khớp mã đơn nào (cột mã đơn của bảng thu chọn sai?)")
    d_ = V.get("dong")
    if d_ is not None and len(d_):
        if "ten_hang" in d_.columns:
            tn = d_["ten_hang"]
            if pd.api.types.is_datetime64_any_dtype(tn) or pd.api.types.is_numeric_dtype(tn):
                notes.append("KIỂM: cột ten_hang của V['dong'] là ngày/số — chọn sai cột tên hàng")
        if "sl_co_ban" in d_.columns and "so_luong" in d_.columns:
            r = pd.to_numeric(d_["sl_co_ban"], errors="coerce") / pd.to_numeric(d_["so_luong"], errors="coerce")
            if (r < 1 - 1e-9).mean() > 0.01:
                notes.append("KIỂM: có dòng sl_co_ban < so_luong (hệ số quy đổi < 1?) — kiểm cột số lượng quy đổi")
    if "ton" in V and "lo_nhap" in V and "ban_lo" in V and de is not None:
        try:
            import ledger as _L
            t = _L.ton_lo(V, de)
            if t is not None and "ma_hang" in t.columns and "ma_hang" in V["ton"].columns:
                g = t.groupby("ma_hang")["con_ton"].sum()
                tv = V["ton"].set_index("ma_hang")["ton"]
                ok = (g.reindex(tv.index) - tv).abs() < 0.5
                if ok.mean() < 0.9:
                    notes.append(f"KIỂM: tồn cộng từ lô khớp bảng tồn cuối chỉ {ok.mean():.0%} mặt hàng (lô / bán theo lô đọc sai?)")
        except Exception:
            pass


def data_family(biz_dir):
    """'Họ' định dạng dữ liệu của shop = tập tên file đã bỏ số thứ tự / năm / tháng (vd DonHang_ChiTiet_1, _2026 -> donhang_chitiet).
    Dùng để kiểm chéo / hiệu chỉnh THEO HỌ (giữ lại cả một họ khi kiểm) — mô phỏng shop có định dạng lạ của private."""
    stems = set()
    for f in sorted(os.listdir(biz_dir)):
        if f.lower().endswith(".md"):
            continue
        st = os.path.splitext(f)[0]
        st = re.sub(r"(?i)[_\-\s]*(thang|t)?\d{1,4}(?=[_\-\s]|$)", "", st)
        stems.add(key(st).replace(" ", ""))
    return "|".join(sorted(stems))


def shop_quality(b):
    """ĐỐI SOÁT CHÉO không cần đáp án (dữ liệu do bộ mô phỏng sinh nên tự nhất quán): mỗi đẳng thức shop có dữ liệu
    -> tỉ lệ khớp. Dùng để phát hiện cách đọc/làm sạch sai TRƯỚC khi trả lời, và làm đặc trưng bỏ trống theo shop.
      ton_khop : Σ nhập (đv cơ bản, bỏ phiếu huỷ) − Σ bán (đv cơ bản, mọi đơn KHÔNG huỷ, dòng trả âm cộng lại kho) = tồn cuối
      tien_dong: tiền hàng đầu đơn = Σ thành tiền dòng hàng
    Trả dict {tên: tỉ lệ khớp (0..1) hoặc None nếu shop không có dữ liệu cho đẳng thức đó}."""
    q = getattr(b, "_quality", None)
    if q is not None:
        return q
    V = getattr(b, "V", None) or {}
    q = {"ton_khop": None, "tien_dong": None}
    try:
        if {"nhap", "dong", "ton"} <= set(V) and "sl_co_ban" in V["dong"].columns:
            nd, dd, tv = V["nhap"], V["dong"], V["ton"]
            kcol = "ma_hang" if all("ma_hang" in x.columns and x["ma_hang"].notna().mean() > 0.95 for x in (nd, dd, tv)) else "ten_hang"
            if all(kcol in x.columns for x in (nd, dd, tv)):
                kf = (lambda x: x.map(sid)) if kcol == "ma_hang" else (lambda x: x.astype(str).map(key))
                nq = pd.to_numeric(nd["sl_co_ban"], errors="coerce") if "sl_co_ban" in nd.columns and nd["sl_co_ban"].notna().all() else None
                if nq is None and "so_luong" in nd.columns:
                    qd = V.get("quy_doi")
                    fac = 1.0
                    if qd is not None and "don_vi" in nd.columns and "ten_hang" in nd.columns:
                        tab = {(h, u): f for h, u, f in zip(qd["hang_key"], qd["don_vi_key"], qd["he_so"])}
                        fac = pd.Series([tab.get((key(h), key(u)), 1.0) for h, u in zip(nd["ten_hang"].astype(str), nd["don_vi"].astype(str))], index=nd.index)
                    nq = pd.to_numeric(nd["so_luong"], errors="coerce") * fac
                if nq is not None:
                    nh = nq[~nd["huy"].astype(bool)].groupby(kf(nd[kcol])[~nd["huy"].astype(bool)]).sum()
                    m = dd["nhom_tt"].ne("huy") if "nhom_tt" in dd.columns else pd.Series(True, index=dd.index)
                    if "la_mat_hang" in dd.columns:
                        m &= dd["la_mat_hang"].astype(bool)
                    bn = pd.to_numeric(dd["sl_co_ban"], errors="coerce")[m].groupby(kf(dd[kcol])[m]).sum()
                    tt = pd.to_numeric(tv["ton"], errors="coerce").groupby(kf(tv[kcol])).sum()
                    calc = nh.reindex(tt.index).fillna(0) - bn.reindex(tt.index).fillna(0)
                    if len(tt) and nh.reindex(tt.index).notna().mean() > 0.5:
                        q["ton_khop"] = round(float(((calc - tt).abs() < 0.5).mean()), 3)
        o = V.get("don")
        if o is not None and {"tien_hang", "doanh_thu"} <= set(o.columns) and "dong" in V:
            th = pd.to_numeric(o["tien_hang"], errors="coerce")
            ok = th.notna()
            if ok.any():
                q["tien_dong"] = round(float(((th - o["doanh_thu"]).abs() <= 1)[ok].mean()), 3)
    except Exception as e:  # đối soát lỗi không được làm hỏng việc trả lời
        q["loi"] = f"{type(e).__name__}: {e}"[:200]
    b._quality = q
    return q


def build_canonical(b):
    V, src, notes = {}, {}, []
    kinds = {n: _table_kind(n, df, b.types[n]) for n, df in b.T.items()}
    b.table_kinds = kinds
    lines = [n for n, k in kinds.items() if k == "dong_ban"]
    heads = [n for n, k in kinds.items() if k == "don_ban"]
    if not lines and not heads:
        svc = [n for n, k in kinds.items() if k == "dich_vu"]
        if svc:
            build_service(b, kinds, max(svc, key=lambda n: len(b.T[n])), V, src, notes)
            return
        b.V, b.V_src, b.V_notes = V, src, ["không nhận ra bảng bán hàng: dùng T[...] gốc"]
        return
    L = max(lines, key=lambda n: len(b.T[n])) if lines else None
    H = max(heads, key=lambda n: len(b.T[n])) if heads else None
    conv = _conversion(b)
    # ---- dòng hàng
    if L:
        df, ty = b.T[L], b.types[L]
        oid = order_id_col(df, ty)
        # README trước (mô tả cột), tên cột sau; kiểu cột phải hợp (tên hàng không thể là cột ngày / số)
        notdate = lambda c: c is not None and not str(ty.get(c, "")).startswith(("ngày", "số", "tiền"))
        pc = _pick(df, ty, "ten_hang", (r"^ten hang|^ten san pham|^ten mon|^ten thuoc|^mat hang|^ten$", None, None, None))
        if not notdate(pc):
            pc = next((c for c in (_col(df, r"(ten hang|ten san pham|ten sp|san pham|mat hang|hang ban|hang hoa|chi tiet hang|ten mon|\bmon\b|product|\bhang\b|noi dung)", exclude=r"^ma |" + PROD_EX + LINE_EX),
                                   _col(df, PROD_PAT, exclude=r"^ma |^id " + PROD_EX + LINE_EX)) if notdate(c)), None)
        mc = _pick(df, ty, "ma_hang", (r"^ma hang|^ma san pham|^ma sp", r"(ma hang|ma sp|ma sku|ma mat hang|ma hang ban|id sp|ma hh|mahanghoa|ma san pham|ma mon)", r"lien quan|nhom|vach", None))
        if pc is not None and pc == mc:
            pc = None          # cột mã không phải cột tên; tên sẽ tra từ danh mục
        qc = _pick(df, ty, "so_luong", (r"^so luong$|^so luong ban|^so luong \(theo don vi", QTY_PAT, r"quy doi|quy ra|le\b|co ban|tra", "số"))
        bq = _col(df, BASEQTY_PAT, ty, "số")
        if bq is None:
            bq = _desc_match(df, r"(doi ra|quy ra|theo) don vi co ban")
            if bq is not None and not str(ty.get(bq, "")).startswith("số"):
                bq = None
        ac = _pick(df, ty, "thanh_tien", (r"^thanh tien|^tri gia", None, None, None)) or \
            _col(df, r"(thanh tien|t tien|tien ban|tri gia|trigia|amount|thanh_tien)", ty, "tiền") or _col(df, AMT_PAT, ty, "tiền") or _col(df, AMT_PAT, ty, "số")
        prc = _pick(df, ty, "don_gia", (r"^don gia|^gia ban", PRICE_PAT, r"von|nhap", None))
        dc = _pick(df, ty, "giam_gia_dong", (r"^giam gia|^chiet khau", None, None, None)) or _col(df, DISC_PAT, ty, "tiền") or _col(df, DISC_PAT, ty, "số")
        uc = next((c for c in df.columns if str(c).endswith("__dv")), None)
        um = _pick(df, ty, "don_vi", (r"^don vi tinh|^don vi$|^dvt|^don vi ban", r"(don vi tinh|dvt|don vi|\bunit\b|donvi|d vi|dv tinh)", r"co ban|__dv", None))
        gv = _pick(df, ty, "gia_von", (r"gia von", r"(gia von|giavon)", None, None))
        dd = pd.DataFrame(index=df.index)
        dd["ma_don"] = df[oid].map(sid) if oid else ("ROW" + df.index.astype(str))
        if pc:
            dd["ten_hang"] = df[pc]
        if mc:
            dd["ma_hang"] = df[mc].map(sid)
        dd["so_luong"] = df[qc] if qc else np.nan
        unit = None
        if uc and um:
            unit = df[uc].where(df[uc].notna(), df[um])
        elif uc or um:
            unit = df[uc or um]
        if unit is not None:
            dd["don_vi"] = unit
        # số lượng đơn vị cơ bản: ưu tiên cột của chính shop, rồi bảng quy đổi, rồi cột __sl_co_ban đã tính
        if bq:
            dd["sl_co_ban"] = df[bq]
            src["dong.sl_co_ban"] = f"{L}::{bq} (cột quy đổi của shop)"
        elif conv and qc and unit is not None:
            k1 = df[pc].map(key) if pc else pd.Series("", index=df.index)
            k2 = df[mc].map(key) if mc else pd.Series("", index=df.index)
            fac = [conv["tab"].get((h1, key(u)), conv["tab"].get((h2, key(u)), 1.0)) for h1, h2, u in zip(k1, k2, unit)]
            dd["sl_co_ban"] = df[qc] * pd.Series(fac, index=df.index)
            src["dong.sl_co_ban"] = f"{L}::{qc} × hệ số `{conv['bang']}`" + (f" (nhiều cấp: {conv['nhieu_cap']} mục)" if conv["nhieu_cap"] else "")
        elif "__sl_co_ban" in df.columns:
            dd["sl_co_ban"] = df["__sl_co_ban"]
        elif qc:
            dd["sl_co_ban"] = df[qc]
            src["dong.sl_co_ban"] = f"{L}::{qc} (không thấy bảng quy đổi: coi là đơn vị cơ bản)"
        if prc:
            dd["don_gia"] = df[prc]
        if dc:
            dd["giam_gia_dong"] = df[dc]
        if ac:
            dd["thanh_tien"] = df[ac]
            src["dong.thanh_tien"] = f"{L}::{ac}"
        elif prc and qc:
            dd["thanh_tien"] = df[qc] * df[prc] - (df[dc].fillna(0) if dc else 0)
            src["dong.thanh_tien"] = f"{L}: {qc} × {prc}" + (f" − {dc}" if dc else "")
        if gv:
            dd["gia_von_1dv"] = df[gv]
        for k_, c_ in (("ngay", _pick(df, ty, "ngay") or _date_col(df, ty)), ("khach_ma", _pick(df, ty, "khach_ma")), ("khach_ten", _pick(df, ty, "khach_ten")),
                       ("kenh", _pick(df, ty, "kenh")), ("httt", _pick(df, ty, "httt")), ("nhan_vien", _pick(df, ty, "nhan_vien"))):
            if c_:
                dd[k_] = df[c_]
                src[f"dong.{k_}"] = f"{L}::{c_}"
        if "__nhom_trang_thai" in df.columns:
            dd["nhom_tt"] = df["__nhom_trang_thai"]
        if "__khach_le" in df.columns:
            dd["khach_le"] = df["__khach_le"]
        V["dong"] = dd
        src["dong"] = L
        # dòng KHÔNG phải hàng (README: 'Voucher', 'Dùng n điểm', 'Phí ship'… ghi chung sổ bán) -> la_hang = False, loai_dong
        if "ten_hang" in dd.columns:
            kn = dd["ten_hang"].astype(str).map(key)
            loai = pd.Series("hang", index=dd.index, dtype=object)
            loai[kn.str.match(r"^(voucher|ma giam|giam gia (hoa )?don|chiet khau (hoa )?don|khuyen mai)\b")] = "voucher"
            loai[kn.str.match(r"^(dung|tru|doi) \d+ diem|^dung diem|^tru diem")] = "diem"
            loai[kn.str.match(r"^(phi ship|phi giao( hang)?|phi van chuyen|tien ship|ship)$")] = "phi_ship"
            loai[kn.str.match(r"^(phi phuc vu|phu thu|phi dich vu|service charge)\b")] = "phi_phuc_vu"
            dd["loai_dong"] = loai
            dd["la_hang"] = loai.isin(["hang", "phi_phuc_vu"])      # phí phục vụ VẪN tính doanh thu (rulebook) ...
            dd["la_mat_hang"] = loai.eq("hang")                     # ... nhưng không phải mặt hàng: không xếp hạng, không đếm
            if (loai != "hang").any():
                notes.append("V['dong']: " + ", ".join(f"{int(v)} dòng {k}" for k, v in loai[loai != "hang"].value_counts().items())
                             + " → voucher/diem/phi_ship: la_hang = False, KHÔNG vào doanh_thu (V['don'] có cột voucher/diem/phi_ship, số dương); "
                             + "phi_phuc_vu: vào doanh_thu nhưng la_mat_hang = False (không xếp hạng / đếm mặt hàng)")
        if "thanh_tien" in dd.columns:
            neg = (pd.to_numeric(dd["thanh_tien"], errors="coerce") < 0) & dd.get("la_hang", pd.Series(True, index=dd.index))
            if neg.any():
                notes.append(f"V['dong'] có {int(neg.sum())} dòng HÀNG thành tiền ÂM (hàng trả?) — đọc README")
        if "so_luong" in dd.columns:
            negq = pd.to_numeric(dd["so_luong"], errors="coerce") < 0
            if negq.any():
                dd["dong_tra"] = negq
                notes.append(f"V['dong'] có {int(negq.sum())} dòng SỐ LƯỢNG ÂM = hàng khách trả (cùng mã đơn) → các đơn này nhom_tt = tra_hang")
    # ---- đơn
    if H:
        hf, hty = b.T[H], b.types[H]
        hid = order_id_col(hf, hty)
        od = pd.DataFrame({"ma_don": hf[hid].map(sid)})
        used = set()
        for k_ in HEAD_FIELDS:
            c_ = _pick(hf, hty, k_) if k_ != "ngay" else (_pick(hf, hty, k_) or _date_col(hf, hty))
            if c_ and c_ not in used:
                used.add(c_)
                od[k_] = hf[c_].values
                src[f"don.{k_}"] = f"{H}::{c_}"
        if "__nhom_trang_thai" in hf.columns:
            od["nhom_tt"] = hf["__nhom_trang_thai"].values
        if "__khach_le" in hf.columns:
            od["khach_le"] = hf["__khach_le"].values
        for c in ("tien_hang", "giam_gia_don", "phi_giao", "tong_don", "da_tra_file", "con_no_file"):
            if c in od.columns and not pd.api.types.is_numeric_dtype(od[c]):
                od[c] = od[c].map(lambda v: parse_money_one(v) if isinstance(v, str) else v).astype(float)
        if "tong_don" not in od.columns and "tien_hang" in od.columns:
            od["tong_don"] = od["tien_hang"].fillna(0) - (od["giam_gia_don"].fillna(0) if "giam_gia_don" in od.columns else 0) \
                + (od["phi_giao"].fillna(0) if "phi_giao" in od.columns else 0)
            src["don.tong_don"] = "tính = tien_hang − giam_gia_don + phi_giao (file không có cột tổng phải trả)"
        od = od.drop_duplicates("ma_don")
    elif "dong" in V:
        dd = V["dong"]
        agg = {c: "first" for c in ("ngay", "khach_ma", "khach_ten", "kenh", "httt", "nhan_vien", "khach_le") if c in dd.columns}
        od = dd.groupby("ma_don", sort=False).agg(agg).reset_index() if agg else dd[["ma_don"]].drop_duplicates()
        if "nhom_tt" in dd.columns:
            # trạng thái là của ĐƠN: dòng nào có trả hàng -> cả đơn trả hàng; có huỷ -> huỷ (ghi chú thường chỉ ở dòng đầu đơn)
            pri = {"tra_hang": 5, "huy": 4, "chua_ro": 3, "dang_giao": 2, "hoan_thanh": 1}
            st = dd.assign(_p=dd["nhom_tt"].map(pri).fillna(0)).sort_values("_p", ascending=False).drop_duplicates("ma_don").set_index("ma_don")["nhom_tt"]
            od["nhom_tt"] = od["ma_don"].map(st)
            nmix = int((dd.groupby("ma_don")["nhom_tt"].nunique() > 1).sum())
            if nmix:
                notes.append(f"{nmix} đơn có trạng thái khác nhau giữa các dòng → lấy theo ưu tiên trả hàng > huỷ > chưa rõ > đang giao > hoàn thành")
        src["don"] = f"gom từ {L} theo ma_don"
    else:
        od = None
    if od is not None:
        if "dong" in V and "thanh_tien" in V["dong"].columns:
            dl = V["dong"]
            hang = dl["la_hang"] if "la_hang" in dl.columns else pd.Series(True, index=dl.index)
            rev = pd.to_numeric(dl["thanh_tien"], errors="coerce")[hang].groupby(dl.loc[hang, "ma_don"]).sum()
            od["doanh_thu"] = od["ma_don"].map(rev).fillna(0.0)
            if "loai_dong" in dl.columns:
                for lk in ("voucher", "diem", "phi_ship"):
                    m_ = dl["loai_dong"].eq(lk)
                    if m_.any():
                        od[lk] = od["ma_don"].map(pd.to_numeric(dl.loc[m_, "thanh_tien"], errors="coerce").abs().groupby(dl.loc[m_, "ma_don"]).sum()).fillna(0.0)
            if "dong_tra" in dl.columns:
                rset = set(dl.loc[dl["dong_tra"], "ma_don"])
                if "nhom_tt" not in od.columns:
                    od["nhom_tt"] = "hoan_thanh"
                od.loc[od["ma_don"].isin(rset) & od["nhom_tt"].ne("huy"), "nhom_tt"] = "tra_hang"
            src["don.doanh_thu"] = "tổng dong.thanh_tien các DÒNG HÀNG theo ma_don (rulebook: không trừ voucher/giảm giá cả đơn, không gồm phí giao)"
            # đơn có ngày / trạng thái ở bảng đầu đơn: chép sang dòng
            dd = V["dong"]
            for c in ("ngay", "nhom_tt", "khach_ma", "khach_ten", "khach_le", "kenh", "httt", "nhan_vien"):
                if c in od.columns and c not in dd.columns:
                    dd[c] = dd["ma_don"].map(od.set_index("ma_don")[c])
        elif "tien_hang" in od.columns:
            od["doanh_thu"] = od["tien_hang"]
            src["don.doanh_thu"] = "don.tien_hang (không có dòng hàng; rulebook: doanh thu = tổng thành tiền các dòng, không trừ giảm giá cả đơn, không gồm phí giao)"
        elif "tong_don" in od.columns:
            od["doanh_thu"] = od["tong_don"]
            src["don.doanh_thu"] = "lấy don.tong_don (không có dòng hàng, không có cột tiền hàng) — CẢNH BÁO: có thể đã trừ giảm giá cả đơn / gồm phí giao"
            notes.append("doanh_thu lấy từ tổng đơn: kiểm README xem tổng đơn có gồm phí giao / giảm giá cả đơn không")
        # đơn có trả hàng (kể cả một phần) theo bảng lô / bảng trả hàng
        ret = set()
        for n_, k_ in kinds.items():
            t_ = b.T[n_]
            rc = _col(t_, r"(so luong tra|sl tra|sltraquydoi|tien hoan tra)", b.types[n_])
            # bảng trả / bảng lô: cột MÃ ĐƠN BÁN (không lấy 'Mã phiếu trả')
            oc = _col(t_, r"(ma hoa don|mahoadon|ma don hang|ma don\b|order id|ma bill|so ct|ma giao dich|so hd)") or order_id_col(t_, b.types[n_])
            if k_ in ("lo", "tra", "khac") and oc and rc:
                ret |= set(t_.loc[pd.to_numeric(t_[rc], errors="coerce").fillna(0) > 0, oc].map(sid))
            elif k_ == "tra" and oc:
                ret |= set(t_[oc].map(sid))
        if ret:
            if "nhom_tt" not in od.columns:
                od["nhom_tt"] = "hoan_thanh"
            m = od["ma_don"].isin(ret) & (od["nhom_tt"] != "huy")
            od.loc[m, "nhom_tt"] = "tra_hang"
            notes.append(f"{int(m.sum())} đơn có trả hàng theo bảng lô/trả hàng → nhom_tt = tra_hang (rulebook: trả một phần cũng bỏ cả đơn)")
            if "dong" in V:
                V["dong"]["nhom_tt"] = V["dong"]["ma_don"].map(od.set_index("ma_don")["nhom_tt"])
        if "khach_ten" in od.columns:
            od["khach_ten"] = od["khach_ten"].map(clean_person_name)
        km = src.get("don.khach_ma") or src.get("dong.khach_ma")
        if km and "khach_ma" in od.columns:
            mp, ms = _name_lookup(b, km.split("::")[-1], "::".join(km.split("::")[:-1]), r"khach", codes=od["khach_ma"].dropna().map(sid).unique()[:5000])
            if mp:
                od["khach_ten_dm"] = od["khach_ma"].map(sid).map(mp)
                src["don.khach_ten_dm"] = f"{ms} (tên theo danh mục khách, nối bằng mã khách)"
        if "nhom_tt" in od.columns and od["nhom_tt"].isna().any():
            n_na = int(od["nhom_tt"].isna().sum())
            od["nhom_tt"] = od["nhom_tt"].fillna("chua_ro")
            notes.append(f"{n_na} đơn trạng thái TRỐNG → nhom_tt = chua_ro (tinh_doanh_thu = False): đọc README xem ô trống nghĩa là gì")
        if "nhom_tt" not in od.columns:
            od["nhom_tt"] = "hoan_thanh"
            notes.append("không có cột trạng thái: coi mọi đơn là hoàn thành (kiểm README)")
        od["tinh_doanh_thu"] = od["nhom_tt"].isin(["hoan_thanh", "dang_giao"])
        if "dong" in V:      # trạng thái / ngày của ĐƠN áp cho mọi dòng (dòng voucher, dòng tiếp nối có thể trống)
            m_od = od.drop_duplicates("ma_don").set_index("ma_don")
            V["dong"]["nhom_tt"] = V["dong"]["ma_don"].map(m_od["nhom_tt"])
            if "ngay" in m_od.columns:
                V["dong"]["ngay"] = V["dong"]["ngay"].where(V["dong"]["ngay"].notna(), V["dong"]["ma_don"].map(m_od["ngay"])) if "ngay" in V["dong"].columns \
                    else V["dong"]["ma_don"].map(m_od["ngay"])
        V["don"] = od.reset_index(drop=True)
        src.setdefault("don", H or L)
    # ---- nhập hàng: V['nhap'] (dòng) + V['phieu_nhap'] (mỗi phiếu một dòng). README trước, tên cột sau.
    PID_PAT = r"(so phieu|ma phieu|ma nhap|ma don nhap|maphieunhap|ma phieu nhap)"
    NHAP_LINE = {
        "ten_hang":  (r"^ten hang", r"(ten hang|ten san pham|ten sp|san pham|mat hang|hang hoa|product|ten thuoc)", PROD_EX + r"|^ma |nhap hang|ma nhap|ma phieu|ma lo", None),
        "ma_hang":   (r"^ma hang", r"(ma hang|ma sp|ma sku|mahanghoa|ma hh|ma thuoc)", r"nhap hang|ma nhap|ma phieu", None),
        "so_luong":  (r"^so luong nhap|^sl nhap|^so luong", QTY_PAT, r"quy doi|co ban", "số"),
        "sl_co_ban": (r"so luong .*don vi co ban|quy doi", BASEQTY_PAT, None, "số"),
        "don_vi":    (r"^don vi nhap|^don vi tinh|^dvt", r"(dvt|don vi)", r"co ban|gia|__dv", None),
        "don_gia":   (r"^gia nhap", r"(gia nhap|don gia|dongianhap)", r"ban|von", None),
        "thanh_tien": (r"tien nhap cua ca dong|thanh tien", r"(thanh tien|tri gia)", r"tong|can tra", None),
        "ncc":       (r"nha cung cap", r"(nha cung cap|\bncc\b|manhacungcap)", None, None),
    }
    NHAP_HEAD = {
        "ngay":     (r"^ngay nhap|^ngay phieu", None, None, "ngày"),
        "ncc":      (r"nha cung cap", r"(nha cung cap|\bncc\b|manhacungcap)", None, None),
        "tong_tien": (r"tong tien cua ca phieu|tong tien phieu", r"(tong gia tri nhap|tonggiatrinhap|can tra ncc|tong tien|tong gia tri)", None, None),
    }
    imp = [n for n, k in kinds.items() if k == "nhap"]
    if imp:
        det = [n for n in imp if _col(b.T[n], QTY_PAT, b.types[n], "số")]
        hed = [n for n in imp if n not in det]
        if det:
            D_ = max(det, key=lambda n: len(b.T[n]))
            df, ty = b.T[D_], b.types[D_]
            pid = _col(df, PID_PAT) or order_id_col(df, ty)
            nd = pd.DataFrame(index=df.index)
            nd["ma_phieu"] = df[pid].map(sid) if pid else ("ROW" + df.index.astype(str))
            for k_, spec in NHAP_LINE.items():
                c_ = _pick(df, ty, k_, spec)
                if c_ and c_ not in nd.attrs.setdefault("used", set()):
                    nd.attrs["used"].add(c_)
                    nd[k_] = df[c_]
                    src[f"nhap.{k_}"] = f"{D_}::{c_}"
            dcol = _pick(df, ty, "ngay", NHAP_HEAD["ngay"]) or _date_col(df, ty)
            if dcol and not re.search(r"\b(han|hsd)\b", _colnorm(dcol)):
                nd["ngay"] = df[dcol]
                src["nhap.ngay"] = f"{D_}::{dcol}"
            if "thanh_tien" not in nd.columns and {"so_luong", "don_gia"} <= set(nd.columns):
                nd["thanh_tien"] = pd.to_numeric(nd["so_luong"], errors="coerce") * pd.to_numeric(nd["don_gia"], errors="coerce")
                src["nhap.thanh_tien"] = f"{D_}: số lượng × giá nhập (theo đơn vị nhập)"
            head_tot = None
            if hed and pid:
                Hn = max(hed, key=lambda n: len(b.T[n]))
                hf, hty = b.T[Hn], b.types[Hn]
                hp = _col(hf, PID_PAT) or order_id_col(hf, hty)
                if hp:
                    hk = hf.drop_duplicates(hp).set_index(hf.drop_duplicates(hp)[hp].map(sid))
                    for k_ in ("ngay", "ncc"):
                        c_ = _pick(hf, hty, k_, NHAP_HEAD[k_]) or (_date_col(hf, hty) if k_ == "ngay" else None)
                        if c_ and k_ not in nd.columns:
                            nd[k_] = nd["ma_phieu"].map(hk[c_].to_dict())
                            src[f"nhap.{k_}"] = f"{Hn}::{c_}"
                    if "__nhom_trang_thai" in hf.columns:
                        nd["huy"] = nd["ma_phieu"].map(hk["__nhom_trang_thai"].to_dict()).eq("huy")
                        src["nhap.huy"] = f"{Hn}::trạng thái phiếu"
                    tc = _pick(hf, hty, "tong_tien", NHAP_HEAD["tong_tien"])
                    if tc:
                        head_tot = hk[tc]
                        src["phieu_nhap.tong_tien"] = f"{Hn}::{tc} (tổng cả phiếu ghi ở đầu phiếu)"
            if "huy" not in nd.columns and "__nhom_trang_thai" in df.columns:
                nd["huy"] = df["__nhom_trang_thai"].eq("huy")
                src["nhap.huy"] = f"{D_}::trạng thái/ghi chú"
            if "huy" not in nd.columns:
                gc = _col(df, r"(ghi chu|noi dung|mo ta|dien giai)")
                nd["huy"] = df[gc].astype(str).map(lambda v: status_group(v) == "huy") if gc else False
                src["nhap.huy"] = f"{D_}::{gc} (chữ 'huỷ')" if gc else "không có thông tin huỷ phiếu: coi như không huỷ"
            nd["huy"] = nd["huy"].fillna(False).astype(bool)
            nd.attrs.pop("used", None)
            V["nhap"] = nd
            src["nhap"] = D_
            # mỗi phiếu một dòng: tiền phiếu = tổng đầu phiếu nếu có, không thì cộng dòng
            agg = {"ngay": "first", "huy": "max"}
            if "ncc" in nd.columns:
                agg["ncc"] = "first"
            pn = nd.groupby("ma_phieu", sort=False).agg(agg)
            if "thanh_tien" in nd.columns:
                pn["tien_dong"] = pd.to_numeric(nd["thanh_tien"], errors="coerce").groupby(nd["ma_phieu"]).sum(min_count=1)
            if head_tot is not None:
                pn["tong_tien"] = pn.index.map(pd.to_numeric(head_tot, errors="coerce").to_dict())
                if "tien_dong" in pn.columns:
                    lech = (pn["tong_tien"] - pn["tien_dong"]).abs() > 1
                    if lech.any():
                        notes.append(f"phieu_nhap: {int(lech.sum())} phiếu có tổng đầu phiếu ≠ cộng dòng (dùng tổng đầu phiếu; cộng dòng ở cột tien_dong)")
            elif "tien_dong" in pn.columns:
                pn["tong_tien"] = pn["tien_dong"]
                src["phieu_nhap.tong_tien"] = "cộng nhap.thanh_tien theo phiếu"
            else:
                notes.append("nhập hàng: file không ghi giá/tiền nhập → câu hỏi tiền nhập là KHÔNG trả lời được (luật: không suy từ cột khác)")
            # mã NCC -> tên NCC (câu hỏi 'nhà cung cấp nào' cần TÊN)
            ncc_c = src.get("nhap.ncc", "").split("::")[-1]
            if "ncc" in pn.columns and ncc_c:
                mp, ms = _name_lookup(b, ncc_c, D_, r"nha cung cap|ncc|supplier", codes=pn["ncc"].dropna().map(sid).unique())
                if mp:
                    pn["ncc_ten"] = pn["ncc"].map(sid).map(mp)
                    nd["ncc_ten"] = nd["ncc"].map(sid).map(mp)
                    src["phieu_nhap.ncc_ten"] = ms
            V["phieu_nhap"] = pn.reset_index()
            src["phieu_nhap"] = f"gom từ nhap theo ma_phieu"
    # ---- khoản thu (trả tiền / trả nợ)
    pays = [n for n, k in kinds.items() if k == "thu"]
    if pays:
        P_ = max(pays, key=lambda n: len(b.T[n]))
        df, ty = b.T[P_], b.types[P_]
        # README trước: 'Mã GD' có thể là mã PHIẾU THU, còn 'Mã giao dịch' mới là MÃ ĐƠN
        oc = _pick(df, ty, "ma_don", (r"^ma don|^ma hoa don|^so hoa don", r"(ma hoa don|ma don|so hd|so ct|ma bill|mahoadon|so hoa don|ma chung tu|ma giao dich)",
                                      r"phieu thu|ma pt\b", None))
        if oc is not None and col_desc(df, oc) and re.search(r"phieu thu", col_desc(df, oc)):
            oc = None
        dts = [c for c in df.columns if str(ty.get(c, "")).startswith("ngày") and not str(c).startswith("__")]
        dt = _pick(df, ty, "ngay_thu", (r"thoi diem thu|ngay thu|thoi gian thu|ngay tra", None, None, "ngày")) or \
            next((c for c in dts if re.search(r"thu|thoi gian|thoi diem|ngay", _colnorm(c))), dts[0] if dts else None)
        stc = _pick(df, ty, "so_tien", (r"^so tien", r"(so tien|tien|gia tri|sotienthu)", r"ma |hinh thuc|phuong thuc", None))
        htc = _pick(df, ty, "httt", (r"hinh thuc thanh toan|phuong thuc", PAYTYPE_PAT, None, None))
        td = pd.DataFrame(index=df.index)
        for k_, c_ in (("ma_don", oc), ("ngay_thu", dt), ("so_tien", stc), ("httt", htc)):
            if c_:
                td[k_] = df[c_].map(sid) if k_ == "ma_don" else df[c_]
                src[f"thu.{k_}"] = f"{P_}::{c_}"
        V["thu"] = td
        src["thu"] = P_
    # ---- ngày cuối có dữ liệu (README 'Dữ liệu có đến hết dd/mm/yyyy')
    m_ = re.search(r"có đến hết\s*(?:ngày\s*)?(\d{1,2})/(\d{1,2})/(\d{4})", getattr(b, "readme", "") or "")
    b.data_end = pd.Timestamp(int(m_.group(3)), int(m_.group(2)), int(m_.group(1))) if m_ else None
    if b.data_end is not None:
        src["data_end"] = "README: dữ liệu có đến hết " + str(b.data_end.date())
    # ---- danh mục hàng V['hang']: mã, tên, đơn vị cơ bản, giá vốn
    cats = [n for n, k in kinds.items() if k in ("danh_muc", "ton")]
    best = None
    for n_ in cats:
        t_, ty_ = b.T[n_], b.types[n_]
        mc_ = _pick(t_, ty_, "ma_hang", (r"^ma hang|^ma san pham|^ma sp", r"(ma hang|ma sp|ma sku|mahanghoa|ma hh|ma san pham|id sp|ma mon)", r"lien quan|nhom|barcode|vach", None))
        tc_ = _pick(t_, ty_, "ten_hang", (r"^ten hang|^ten san pham|^ten$", r"(ten hang|ten san pham|ten sp|hang hoa|san pham|ten thuoc|^mon$|product|^ten goi$|^ten$)", r"^ma |nhom|loai|lien quan", None))
        if (mc_ or tc_) and len(t_) < 20000 and not re.search(r"(lich su gia|thuoc|bang gia)", _colnorm(n_)):
            sc_ = (mc_ is not None) + (tc_ is not None) + (kinds[n_] == "danh_muc")
            if best is None or sc_ > best[0]:
                best = (sc_, n_, mc_, tc_)
    if best:
        _, n_, mc_, tc_ = best
        t_, ty_ = b.T[n_], b.types[n_]
        hg = pd.DataFrame(index=t_.index)
        if mc_:
            hg["ma_hang"] = t_[mc_].map(sid)
        if tc_:
            hg["ten_hang"] = t_[tc_]
        for k_, spec in (("dv_co_ban", (r"^don vi co ban", r"(dvt|don vi tinh|don vi co ban|^don vi$)", r"lon|quy doi|nhap", None)),
                         ("gia_von", (r"gia von", r"(gia von|giavon)", r"lan cuoi", None))):
            c_ = _pick(t_, ty_, k_, spec)
            if c_:
                hg[k_] = t_[c_]
                src[f"hang.{k_}"] = f"{n_}::{c_}"
        # nhóm hàng: mô tả README trước; theo tên thì chọn cột có NHIỀU nhóm nhất (bỏ cột hằng 'Hàng hóa')
        c_ = _desc_match(t_, r"^nhom hang|^nhom$|^phan loai")
        if c_ is None:
            cands = [c for c in t_.columns if not str(c).startswith("__") and re.search(r"(nhom hang|^nhom$|danh muc|loai hang|nhom sp|category|phan loai|^nhom )", _colnorm(c))
                     and not re.search(r"khach|ma |lien quan", _colnorm(c)) and 1 < t_[c].nunique() <= max(2, len(t_) // 2)]
            c_ = max(cands, key=lambda c: t_[c].nunique()) if cands else None
        if c_:
            hg["nhom"] = t_[c_]
            src["hang.nhom"] = f"{n_}::{c_}"
        # danh mục thuốc (hoạt chất, kê đơn) nối theo mã/tên — chọn bảng có NHIỀU thông tin nhất
        best_t = None
        for n2, t2 in b.T.items():
            if n2 == n_ or len(t2) > 20000:
                continue
            ty2 = b.types[n2]
            hc2 = _pick(t2, ty2, "hoat_chat", (r"hoat chat|thanh phan", r"(hoat chat|hoatchat|thanh phan|active)", r"gia|ham luong", None))
            kc2 = _pick(t2, ty2, "ke_don", (r"ke don|thuoc ke don", r"(ke don|kedon|ban theo don|bantheodon|\brx\b)", r"gia|don vi|quy doi", None))
            if hc2 or kc2:
                sc2 = (hc2 is not None) + (kc2 is not None)
                if best_t is None or sc2 > best_t[0]:
                    best_t = (sc2, n2, hc2, kc2)
        if best_t:
            _, n2, hc2, kc2 = best_t
            t2, ty2 = b.T[n2], b.types[n2]
            k2 = _pick(t2, ty2, "ma_hang", (r"^ma hang|^ma san pham", r"(ma hang|ma sp|ma sku|mahanghoa|ma hh|ma san pham|ma thuoc)", None, None))
            n2c = _pick(t2, ty2, "ten_hang", (r"^ten", r"(ten hang|ten san pham|ten thuoc|^ten$)", r"^ma ", None))
            kk = kh = None
            if k2 and "ma_hang" in hg.columns:
                kk, kh = t2[k2].map(sid), hg["ma_hang"]
            elif n2c and "ten_hang" in hg.columns:
                kk, kh = t2[n2c].map(key), hg["ten_hang"].map(key)
            if kk is not None:
                if hc2:
                    hg["hoat_chat"] = kh.map(dict(zip(kk, t2[hc2])))
                if kc2:
                    yn = t2[kc2].map(lambda v: (str(v).strip().lower() in ("1", "true", "x", "co", "có", "yes", "y", "1.0", "kê đơn", "ke don", "rx"))
                                     if pd.notna(v) else False)
                    hg["ke_don"] = kh.map(dict(zip(kk, yn))).fillna(False).astype(bool)
                src["hang.thuoc"] = f"{n2}: " + ", ".join(x for x in (hc2, kc2) if x) + " (nối theo mã/tên hàng)"
        V["hang"] = hg.reset_index(drop=True)
        src["hang"] = n_
    # ---- tồn kho V['ton']: (a) tồn cuối kỳ theo mặt hàng (tại data_end) hoặc (b) thẻ kho (tồn sau mỗi phát sinh)
    for n_ in [n for n, k in kinds.items() if k == "ton"]:
        t_, ty_ = b.T[n_], b.types[n_]
        tc = _pick(t_, ty_, "ton", (r"ton kho|ton cuoi|ton sau", r"(ton cuoi|ton kho|ton thuc|toncuoi|^ton$)", r"toi thieu|toi da|nho nhat|lon nhat|dinh muc", None))
        mc_ = _pick(t_, ty_, "ma_hang", (r"^ma hang|^ma san pham", r"(ma hang|ma sp|ma sku|mahanghoa|ma hh|id sp|ma san pham|ma mon)", r"chung tu|giao dich|chi nhanh", None))
        nc_ = _pick(t_, ty_, "ten_hang", (r"^ten hang|^ten", r"(ten hang|ten san pham|hang hoa|san pham|product|mat hang)", r"^ma |id", None))
        if not tc or not (mc_ or nc_):
            continue
        tv = pd.DataFrame({"ton": pd.to_numeric(t_[tc], errors="coerce")})
        if mc_:
            tv["ma_hang"] = t_[mc_].map(sid)
        if nc_:
            tv["ten_hang"] = t_[nc_]
        dc_ = _date_col(t_, ty_) or next((c for c in t_.columns if str(ty_.get(c, "")).startswith("ngày")), None)
        if dc_ is not None and len(t_) > 3 * max(1, (tv.get("ma_hang", tv.get("ten_hang"))).nunique()):
            tv["ngay"] = pd.to_datetime(t_[dc_], errors="coerce")
            tv["_ord"] = range(len(tv))
            V["the_kho"] = tv
            src["the_kho"] = f"{n_}: {tc} là tồn SAU mỗi phát sinh; tồn tại X = dòng cuối của mặt hàng có ngày ≤ X"
        else:
            V["ton"] = tv
            src["ton"] = f"{n_}::{tc} = tồn CUỐI ngày cuối có dữ liệu ({b.data_end.date() if b.data_end is not None else '?'}), đơn vị cơ bản"
        break
    # ---- dòng trả hàng V['tra']: số lượng trả (đơn vị cơ bản) theo đơn / lô (bảng lô có cột 'SL trả')
    for n_, k_ in kinds.items():
        if k_ not in ("lo", "tra"):
            continue
        t_, ty_ = b.T[n_], b.types[n_]
        rq = _pick(t_, ty_, "sl_tra", (r"so luong khach tra|so luong tra", r"(so luong tra|sl tra|sltraquydoi)", None, "số"))
        oc = _pick(t_, ty_, "ma_don", (r"^ma don|^ma hoa don", r"(ma hoa don|mahoadon|ma don hang|ma don\b)", r"phieu tra|nhap", None))
        lc = _pick(t_, ty_, "lo", (r"^so lo|^lo\b|^ma lo", r"(^lo$|ma lo|so lo|malo)", None, None))
        if rq and oc:
            tr_ = pd.DataFrame({"ma_don": t_[oc].map(sid), "sl_tra_co_ban": pd.to_numeric(t_[rq], errors="coerce").fillna(0)})
            if lc:
                tr_["lo"] = t_[lc].map(sid)
            V["tra"] = tr_[tr_["sl_tra_co_ban"] > 0].reset_index(drop=True)
            src["tra"] = f"{n_}::{rq} (số lượng khách trả, đơn vị cơ bản; ngày trả: không ghi → theo ngày đơn)"
            break
    # ---- lô nhập V['lo_nhap']: lô, mã hàng, hạn dùng, số lượng nhập (cơ bản), giá nhập, mã phiếu.
    # Có thể có nhiều bảng cùng ghi lô (bảng lô của phần mềm + sổ tay nhập): chọn bảng có mã hàng + số lượng theo ĐƠN VỊ CƠ BẢN.
    cand = []
    for n_, k_ in kinds.items():
        t_, ty_ = b.T[n_], b.types[n_]
        if k_ not in ("lo", "nhap"):
            continue
        hc = _pick(t_, ty_, "hsd", (r"han su dung|han dung", r"(han su dung|\bhsd\b|han dung|hansudung)", None, "ngày"))
        lc = _pick(t_, ty_, "lo", (r"^so lo|^lo\b|^ma lo", r"(^lo$|ma lo|so lo|malo)", None, None))
        qc_ = _pick(t_, ty_, "sl_nhap", (r"so luong nhap", r"(so luong|sl nhap|slnhapquydoi|\bsl\b)", r"\btra\b|sl ban|so luong ban|da ban", "số"))
        if not (hc and lc and qc_):
            continue
        cols = {"lo": lc, "han_dung": hc, "sl_nhap": qc_}
        for k2, spec in (("ma_hang", (r"^ma hang", r"(ma hang|ma sp|ma sku|mahanghoa|ma hh|ma mon|ma san pham)", None, None)),
                         ("ten_hang", (r"^ten hang", r"(ten hang|ten san pham|mat hang|product|hang hoa)", r"^ma ", None)),
                         ("ma_phieu", (r"ma phieu nhap", PID_PAT, None, None)), ("gia_nhap", (r"^gia nhap", r"(gia nhap|don gia|dongianhap)", None, None)),
                         ("don_vi", (r"^don vi nhap|^dvt", r"(dvt|don vi)", r"co ban|__dv", None))):
            c_ = _pick(t_, ty_, k2, spec)
            if c_:
                cols[k2] = c_
        base_q = bool(re.search(r"co ban|quy doi", _colnorm(qc_) + " " + (col_desc(t_, qc_) or "")))
        cand.append(((("ma_hang" in cols) * 2 + base_q * 2 + (k_ == "lo") + ("gia_nhap" in cols)), n_, cols, base_q))
    if cand:
        _, n_, cols, base_q = max(cand, key=lambda x: x[0])
        t_ = b.T[n_]
        ln = pd.DataFrame({"lo": t_[cols["lo"]].map(sid), "han_dung": pd.to_datetime(t_[cols["han_dung"]], errors="coerce")})
        q = pd.to_numeric(t_[cols["sl_nhap"]], errors="coerce")
        for k2 in ("ma_hang", "ten_hang", "ma_phieu", "don_vi"):
            if k2 in cols:
                ln[k2] = t_[cols[k2]].map(sid) if k2 != "ten_hang" else t_[cols[k2]]
        if "gia_nhap" in cols:
            ln["gia_nhap"] = pd.to_numeric(t_[cols["gia_nhap"]], errors="coerce")
        if base_q:
            ln["sl_nhap_co_ban"] = q
        else:
            ln["sl_nhap"] = q
            if conv and "don_vi" in ln.columns:
                k1 = ln["ten_hang"].map(key) if "ten_hang" in ln.columns else pd.Series("", index=ln.index)
                k2_ = ln["ma_hang"].map(key) if "ma_hang" in ln.columns else pd.Series("", index=ln.index)
                ln["sl_nhap_co_ban"] = q * pd.Series([conv["tab"].get((a, key(u)), conv["tab"].get((c, key(u)), 1.0))
                                                      for a, c, u in zip(k1, k2_, ln["don_vi"])], index=ln.index)
        V["lo_nhap"] = ln.reset_index(drop=True)
        src["lo_nhap"] = f"{n_}::{cols['sl_nhap']}" + (" (đơn vị cơ bản)" if base_q else " × hệ số quy đổi theo đơn vị nhập")
    # ---- bán theo lô V['ban_lo']: mỗi dòng = một phần của dòng bán lấy từ một lô (+ số lượng khách trả về lô đó)
    for n_, k_ in kinds.items():
        if k_ != "lo":
            continue
        t_, ty_ = b.T[n_], b.types[n_]
        lc = _pick(t_, ty_, "lo", (r"^so lo|^lo\b|^ma lo", r"(^lo$|ma lo|so lo|malo)", None, None))
        oc = _pick(t_, ty_, "ma_don", (r"^ma don|^ma hoa don", r"(ma hoa don|mahoadon|ma don hang|ma don\b)", r"nhap|phieu tra", None))
        qb = _pick(t_, ty_, "sl_ban", (r"so luong ban lay tu lo|so luong ban", r"(so luong|sl quy ra le|slquydoi|sl quy doi|\bsl\b)", r"tra|nhap", "số"))
        if not (lc and oc and qb):
            continue
        bl = pd.DataFrame({"ma_don": t_[oc].map(sid), "lo": t_[lc].map(sid),
                           "sl_ban_co_ban": pd.to_numeric(t_[qb], errors="coerce").fillna(0)})
        rq = _pick(t_, ty_, "sl_tra", (r"so luong khach tra|so luong tra", r"(so luong tra|sl tra|sltraquydoi)", None, "số"))
        bl["sl_tra_co_ban"] = pd.to_numeric(t_[rq], errors="coerce").fillna(0) if rq else 0.0
        V["ban_lo"] = bl
        src["ban_lo"] = f"{n_}: {qb} (bán lấy từ lô)" + (f", {rq} (khách trả về lô)" if rq else "")
        break
    # tên hàng cho các bảng chỉ có mã (dòng bán, nhập, tồn, lô, thẻ kho) — tra từ danh mục; câu hỏi / đáp án dùng TÊN
    if "hang" in V and {"ma_hang", "ten_hang"} <= set(V["hang"].columns):
        mp_ = V["hang"].dropna(subset=["ma_hang"]).drop_duplicates("ma_hang").set_index("ma_hang")["ten_hang"].to_dict()
        for tb in ("dong", "nhap", "ton", "lo_nhap", "the_kho"):
            if tb in V and "ma_hang" in V[tb].columns and "ten_hang" not in V[tb].columns:
                V[tb]["ten_hang"] = V[tb]["ma_hang"].map(sid).map(mp_)
                src[f"{tb}.ten_hang"] = f"tra từ danh mục {src.get('hang', '')} theo ma_hang"
    # sổ chi phí V['chi_phi']: ngày, số tiền, nhóm/mục (lãi ròng = lãi gộp − mọi khoản chi)
    for n_, k_ in kinds.items():
        if k_ != "chi_phi":
            continue
        t_, ty_ = b.T[n_], b.types[n_]
        dc_ = _date_col(t_, ty_) or next((c for c in t_.columns if str(ty_.get(c, "")).startswith("ngày")), None)
        ac_ = _pick(t_, ty_, "so_tien", (r"so tien|tien chi", r"(so tien|tien chi|tien|chi phi|gia tri)", r"ma |ngay", None))
        if dc_ and ac_:
            cp = pd.DataFrame({"ngay": pd.to_datetime(t_[dc_], errors="coerce"),
                               "so_tien": t_[ac_].map(lambda v: parse_money_one(v) if isinstance(v, str) else v).astype(float)})
            nc_ = _pick(t_, ty_, "nhom", (r"^nhom|^muc|^loai", r"(nhom|muc|loai|khoan|noi dung|dien giai)", None, None))
            if nc_:
                cp["nhom"] = t_[nc_]
            V["chi_phi"] = cp
            src["chi_phi"] = f"{n_}: {dc_}, {ac_}"
            break
    if conv:
        V["quy_doi"] = pd.DataFrame([{"hang_key": pk, "don_vi_key": uk, "he_so": f, "dv_co_ban": conv["base"].get(pk)}
                                     for (pk, uk), f in conv["tab"].items()])
        src["quy_doi"] = f"{conv['bang']} (hệ số về ĐƠN VỊ CƠ BẢN, đã nhân dồn nhiều cấp: {conv['nhieu_cap']} mục)"
    try:
        self_check_V(b, V, notes)
    except Exception as e:
        notes.append(f"(tự kiểm lỗi: {type(e).__name__}: {e})")
    b.V, b.V_src, b.V_notes = V, src, notes


BIZ_CACHE_VER = "v44"


class Business:
    """Dữ liệu 1 shop đã đọc + thẻ mô tả cho LLM."""

    def activate(self):
        """Đặt mô tả cột README của shop này làm ngữ cảnh toàn cục (col_desc / _pick dùng). Gọi khi chuyển shop."""
        global COL_DESC, COL_DESC_FILE
        COL_DESC, COL_DESC_FILE = getattr(self, "col_desc", {}) or {}, getattr(self, "col_desc_file", {}) or {}

    def __init__(self, biz_dir, cache_dir=None):
        self.dir = biz_dir
        self.id = os.path.basename(biz_dir.rstrip("/"))
        # đổi BIZ_CACHE_VER mỗi khi đổi logic đọc/làm sạch để không dùng nhầm cache cũ
        # khoá cache gồm CHỮ KÝ NỘI DUNG thư mục (tên + cỡ file): hai split có thể trùng mã shop (vd dev_001) — không được nạp nhầm
        sig = hashlib.sha1("|".join(f"{os.path.relpath(f, biz_dir)}:{os.path.getsize(f)}" for f in sorted(glob.glob(os.path.join(biz_dir, "**", "*"), recursive=True))
                                     if os.path.isfile(f)).encode("utf-8")).hexdigest()[:12]
        # + phiên bản pandas: pickle DataFrame của pandas 3 không nạp đúng ở pandas 2 (và ngược lại)
        pv = ".".join(pd.__version__.split(".")[:2])
        cp = os.path.join(cache_dir, f"{self.id}.{BIZ_CACHE_VER}.pd{pv}.{sig}.pkl") if cache_dir else None
        if cp and os.path.exists(cp):
            try:
                with open(cp, "rb") as fh:
                    d = pickle.load(fh)
                self.__dict__.update(d)
                self.dir = biz_dir                 # thư mục có thể đã đổi chỗ (cache chép từ máy khác)
                self.activate()
                return
            except Exception as e:             # cache hỏng / ghi dở -> đọc lại từ dữ liệu gốc
                print(f"  [cảnh báo] cache {os.path.basename(cp)} hỏng ({type(e).__name__}) -> đọc lại", flush=True)
        t0 = now()
        # README TRƯỚC: mô tả cột dùng ngay từ bước làm sạch (nhận mã đơn theo mô tả...) — không để sót ngữ cảnh shop trước
        self.readme = ""
        for rf in sorted(glob.glob(os.path.join(biz_dir, "*.md"))):
            with open(rf, encoding="utf-8", errors="ignore") as fh:
                self.readme += fh.read()
        self.col_desc, self.col_desc_file = parse_readme_cols(self.readme)
        self.activate()
        raw, notes = read_tables(biz_dir)
        self.T, self.types, self.notes = {}, {}, notes
        for name, (df, yh) in raw.items():
            df, ty = type_columns(df, yh, name, notes)
            df.attrs["tname"] = name
            df = fill_row_groups(df, ty, name, notes)
            df.attrs["tname"] = name
            self.T[name] = df
            self.types[name] = ty
        self.T, self.types = merge_same_structure(self.T, self.types, self.notes)
        for name in list(self.T):
            self.T[name].attrs["tname"] = name
            self.T[name] = drop_same_order_duplicates(self.T[name], self.types[name], name, self.notes)
        for n_, t_ in self.T.items():
            t_.attrs["tname"] = n_
        try:
            enrich_semantics(self)
        except Exception as e:  # lớp nghĩa lỗi không được làm hỏng việc đọc shop
            self.semantics = [f"(lớp nghĩa lỗi: {e})"]
        try:
            build_canonical(self)
        except Exception as e:  # bộ bảng chuẩn lỗi: model dùng T gốc
            self.V, self.V_src, self.V_notes = {}, {}, [f"(bộ bảng chuẩn lỗi: {type(e).__name__}: {e})"]
        self.load_sec = now() - t0
        self.card = build_card(self)
        self.terms = collect_terms(self)
        if cp:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{cp}.{os.getpid()}.tmp"        # ghi nguyên tử: bị ngắt giữa chừng không để lại pickle dở
            with open(tmp, "wb") as fh:
                pickle.dump({k: v for k, v in self.__dict__.items()}, fh)
            os.replace(tmp, cp)


def merge_same_structure(T, types, notes):
    """Gộp các bảng cùng bộ cột (file/sheet theo tháng, _1/_2) thành 1 bảng, thêm cột '__nguon'."""
    # khoá nhóm = TẬP tên cột đã chuẩn hoá (không phụ thuộc thứ tự cột, hoa/thường, dấu): file _2 đảo cột vẫn gộp đúng
    groups = defaultdict(list)
    for name, df in T.items():
        norm = [_colnorm(c).replace(" ", "") for c in df.columns]      # 'DonVi' ~ 'DONVI' ~ 'don_vi
        k = tuple(sorted(norm)) if len(set(norm)) == len(norm) else tuple(map(str, df.columns))
        groups[k].append(name)
    T2, ty2 = {}, {}
    for _, names in groups.items():
        if len(names) == 1:
            T2[names[0]], ty2[names[0]] = T[names[0]], types[names[0]]
            continue
        # đưa mọi bảng về tên + thứ tự cột của bảng đầu
        ref = list(T[names[0]].columns)
        nk = lambda c: _colnorm(c).replace(" ", "")
        rmap = {nk(c): c for c in ref}
        for n in names[1:]:
            ren = {c: rmap.get(nk(c), c) for c in T[n].columns}
            T[n] = T[n].rename(columns=ren)[ref]
            types[n] = {ren.get(c, c): v for c, v in types[n].items()}
        cols = ref
        bases = sorted({n.split("::")[0] for n in names})
        if len(bases) == 1:
            new = f"{bases[0]} (gộp {len(names)} sheet)"
        else:
            pref = os.path.commonprefix(bases).rstrip("_-. 0123456789") or bases[0]
            new = f"{pref}* (gộp {len(names)} file)"
        parts = []
        for n in names:
            d = T[n].copy()
            d["__nguon"] = n
            parts.append(d)
        merged = pd.concat(parts, ignore_index=True)
        ty = dict(types[names[0]])
        for c in cols:
            kinds = {types[n].get(c) for n in names}
            if len(kinds) > 1:
                ty[c] = "hỗn hợp: " + ", ".join(sorted(k for k in kinds if k))
        ty["__nguon"] = "chữ (tên file/sheet gốc)"
        T2[new], ty2[new] = merged, ty
        notes[new].append(f"gộp {len(names)} bảng cùng cấu trúc: {', '.join(names[:4])}{' …' if len(names) > 4 else ''}")
    return T2, ty2


NAME_COL_HINT = re.compile(r"(tên|ten|hàng|hang|khách|khach|sản phẩm|san pham|món|mon|nhóm|nhom|ncc|nhà cung cấp|nhân viên|item|product|customer|hoạt chất|thuốc|dịch vụ|loại|kênh|danh mục)", re.I)


def collect_terms(b):
    """Tên hàng/khách/nhóm... (cột chữ, nhiều giá trị lặp) để bảo vệ khi chuẩn hoá câu hỏi."""
    cnt = Counter()
    for name, df in b.T.items():
        for c, ty in b.types[name].items():
            if ty == "chữ" and c in df.columns:
                vc = df[c].dropna().astype(str)
                if not len(vc) or (vc.str.len().median() < 3 and not NAME_COL_HINT.search(str(c))):
                    continue
                if 2 <= vc.nunique() <= 5000:
                    for v, n in vc.value_counts().items():
                        if 3 <= len(v) <= 80 and not re.fullmatch(r"[\d\s.,/:-]+", v):
                            cnt[v] += n
    return [t for t, _ in cnt.most_common(6000)]


V_DESC = {
    "don": "mỗi ĐƠN một dòng", "dong": "mỗi DÒNG HÀNG một dòng", "nhap": "dòng phiếu nhập", "phieu_nhap": "mỗi PHIẾU NHẬP một dòng",
    "thu": "mỗi LẦN KHÁCH TRẢ TIỀN một dòng (gồm cả trả lúc mua)", "hang": "danh mục hàng", "ton": "tồn CUỐI ngày cuối có dữ liệu (đv cơ bản)",
    "the_kho": "thẻ kho: tồn SAU mỗi phát sinh", "tra": "số lượng khách trả (đv cơ bản)", "lo_nhap": "mỗi LÔ nhập một dòng",
    "ban_lo": "phần dòng bán lấy từ từng lô", "chi_phi": "mỗi khoản CHI một dòng",
    "viec": "mỗi VIỆC của một phiếu dịch vụ một dòng", "nhan_vien": "danh sách thợ / nhân viên", "quy_doi": "hệ số về đv CƠ BẢN (đã nhân dồn nhiều cấp)",
}


def canonical_card(b, max_chars=3500):
    """Mục 'BỘ BẢNG CHUẨN V' của thẻ dữ liệu: bảng chuẩn dùng chung mọi shop + nguồn từng cột (để model kiểm lại)."""
    V = getattr(b, "V", None) or {}
    if not V:
        return []
    src = getattr(b, "V_src", {}) or {}
    out = ["### BỘ BẢNG CHUẨN V (tự động, cùng tên cột cho MỌI shop — ưu tiên dùng; nguồn ghi trong ngoặc, README nói khác thì theo README)"]
    if getattr(b, "data_end", None) is not None:
        out.append(f"  DỮ LIỆU ĐẾN HẾT: {b.data_end.date()} (DATA_END)")
    for k, df in V.items():
        cols = []
        for c in df.columns:
            if str(c).startswith("_"):
                continue
            sc = src.get(f"{k}.{c}")
            cols.append(f"{c}" + (f" ({sc.split('::')[-1][:40]})" if sc else ""))
        out.append(f"  - V[{k!r}] {len(df)} dòng — {V_DESC.get(k, '')}; nguồn: {str(src.get(k, '?'))[:70]}")
        out.append(f"      cột: {', '.join(cols)}")
    for n_ in getattr(b, "V_notes", []) or []:
        out.append(f"  ⚠ {n_}")
    txt = "\n".join(out)
    return (txt[:max_chars] + "\n  …").split("\n") if len(txt) > max_chars else out


def build_card(b, max_chars=16500, max_cols_detail=30):
    """Thẻ dữ liệu ngắn gọn cho LLM: bảng, cột, kiểu, ví dụ, các giá trị (nếu ít), ghi chú làm sạch."""
    lines = []
    for name, df in b.T.items():
        n = len(df)
        dup = int(df.duplicated().sum()) if n else 0
        lines.append(f"### Bảng T[{name!r}] — {n} dòng, {df.shape[1]} cột" + (f"; {dup} dòng trùng hoàn toàn" if dup else ""))
        for nt in b.notes.get(name, []):
            lines.append(f"  (làm sạch: {nt})")
        for c in list(df.columns)[:max_cols_detail]:
            s = df[c]
            ty = b.types[name].get(c, "?")
            nn = s.dropna()
            null = 1 - len(nn) / max(1, n)
            nun = nn.astype(str).nunique() if len(nn) else 0
            if ty.startswith("ngày") and len(nn):
                ex = f"từ {nn.min()} đến {nn.max()}"
            elif ty.startswith(("tiền", "số")) and len(nn):
                ex = f"min {nn.min():g}, max {nn.max():g}, ví dụ {', '.join(f'{v:g}' for v in nn.head(3))}"
            else:
                vals = nn.astype(str)
                if 0 < nun <= 25:
                    vc = vals.value_counts()
                    ex = "CÁC GIÁ TRỊ: " + "; ".join(f"{k} ({v})" for k, v in vc.items())
                else:
                    ex = "ví dụ: " + " | ".join(vals.drop_duplicates().head(4).tolist())
            extra = ""
            if 0.05 < null < 0.97 and ty in ("chữ", "mã/chữ"):
                extra = f"; trống {null:.0%} (nếu trống theo nhóm dòng thì là ô gộp → ffill trong phạm vi bảng)"
            elif null >= 0.97:
                extra = "; gần như trống"
            lines.append(f"  - {c!r} [{ty}] {ex[:300]}{extra}")
        if df.shape[1] > max_cols_detail:
            lines.append(f"  … và {df.shape[1] - max_cols_detail} cột khác: {list(df.columns)[max_cols_detail:]}")
    sem = getattr(b, "semantics", [])
    if sem:
        lines = ["### LỚP NGHĨA (tự động — đối chiếu README; README nói khác thì theo README)"] + [f"  - {x}" for x in sem] + lines
    vcard = canonical_card(b)
    if vcard:
        lines = vcard + lines
    card = "\n".join(lines)
    if len(card) > max_chars:
        card = card[:max_chars] + "\n…(thẻ bị cắt bớt; dùng T[...].columns / .head() trong code nếu cần)"
    return card


# =============================================================================
# 1b. ĐỌC LUẬT: tách rulebook thành luật nguyên tử, chọn luật liên quan cho từng câu, rút tham số
#     (rulebook mỗi bộ test KHÁC NHAU — bộ public thêm ngành Mẹ và bé, Sửa xe, Điện thoại, thêm định nghĩa
#      'khách sộp', 'giờ cao điểm'… → mọi thứ phải đọc từ rulebook được phát, không viết cứng)
# =============================================================================

_RB_STOP = set("la cua va cho cac mot nhung trong theo khi thi duoc khong co de tu den ra vao nay do cung neu ma voi "
               "hay hoac tinh ngay ky so bao nhieu nao gi dang da se luat mac dinh".split())


def parse_rulebook(text):
    """-> list dict(id, muc, tieu_de, text). Mỗi dòng bảng '| luật | mặc định |' hoặc gạch đầu dòng là một luật."""
    rules, sec, sub, n = [], "", "", 0
    for line in text.splitlines():
        h = re.match(r"^(#{2,4})\s*(.+)", line)
        if h:
            if len(h.group(1)) == 2:
                sec, sub, n = h.group(2).strip(), "", 0
            else:
                sub = h.group(2).strip()
            continue
        t = line.strip()
        if t.startswith("|"):
            cells = [c.strip() for c in t.strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= set("-: ") or key(cells[0]) in ("luat", "truong hop", "muc"):
                continue
            body = f"{cells[0]}: {' | '.join(cells[1:])}"
        elif re.match(r"^[-*]\s+", t) or re.match(r"^\d+\.\s+", t):
            body = re.sub(r"^([-*]|\d+\.)\s+", "", t)
        else:
            continue
        n += 1
        m = re.match(r"(\d+)", sec)
        rid = f"§{m.group(1) if m else sec[:10]}.{n}"
        cell = body.split(":", 1)[0] if t.startswith("|") else body[:60]
        rkey = ((RDL.rule_key(sub) + " :: ") if sub else "") + RDL.rule_key(cell) if RDL is not None else ""
        rules.append({"id": rid, "muc": sec, "tieu_de": sub, "text": re.sub(r"\*\*", "", body), "rkey": rkey})
    return rules


class RuleIndex:
    """Chọn luật liên quan nhất cho một câu hỏi (chồng lấp từ có trọng số idf + cụm 2 từ + ngành của shop)."""

    def __init__(self, text):
        self.rules = parse_rulebook(text)
        self.toks = [self._tok(r["muc"] + " " + r["tieu_de"] + " " + r["text"]) for r in self.rules]
        df = Counter(w for t in self.toks for w in set(t))
        N = max(1, len(self.rules))
        self.idf = {w: math.log((N + 1) / (c + 0.5)) for w, c in df.items()}

    @staticmethod
    def _tok(s):
        ws = [w for w in key(s).split() if w not in _RB_STOP and len(w) > 1]
        return ws + [a + "_" + b for a, b in zip(ws, ws[1:])]

    EXPAND = [(r"\b(thung|hop|loc|ket|bao|vi|chai|lon|goi)\b", "don vi lon don vi co ban quy doi"),
              (r"\bno\b|\bnợ\b", "cong no no"), (r"lau nhat|qua han", "tuoi no"), (r"\bhuy\b", "don huy"),
              (r"\btra hang\b|\btra lai\b", "don tra hang"), (r"\btet\b", "tet dip tet"), (r"\bca (sang|chieu)\b", "ca sang ca chieu"),
              (r"ban chay|nhieu nhat|it nhat|top", "xep hang ban chay"), (r"\bkhach\b", "xac dinh khach khach le")]

    def select(self, question, readme="", k=6):
        qk = key(question)
        extra = " ".join(add for pat, add in self.EXPAND if re.search(pat, qk))
        q = set(self._tok(question + " " + extra))
        rk = key(readme)[:3000]
        forced = set(RULE_DELTA.new_terms_in(qk)) if RULE_DELTA is not None else set()
        scored = []
        for r, t in zip(self.rules, self.toks):
            sc = sum(self.idf.get(w, 0) for w in set(t) & q)
            if r["tieu_de"] and key(r["tieu_de"]) and key(r["tieu_de"]) in rk:
                sc += 2.0                      # luật ngành của đúng ngành shop
            if r.get("rkey") in forced:
                sc += 100.0                    # câu dùng đúng thuật ngữ của một luật MỚI: luật đó phải đứng đầu
            elif RULE_DELTA is not None and sc > 0 and RULE_DELTA.status.get(r.get("rkey")) in ("new", "changed"):
                sc *= 1.5                      # luật mới / đổi so với train: ưu tiên nhắc lại
            if sc > 0:
                scored.append((sc, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:k]]

    def look_alike(self, r, chosen_ids, thr=0.34):
        """Luật GẦN GIỐNG nhất với r nhưng khác nghĩa (RuleArena: LLM hay nhầm luật na ná) — None nếu không có."""
        i = self.rules.index(r)
        a = {w for w in self.toks[i] if "_" not in w}
        best = None
        for j, (r2, t2) in enumerate(zip(self.rules, self.toks)):
            if j == i or r2["id"] in chosen_ids:
                continue
            b = {w for w in t2 if "_" not in w}
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= thr and (best is None or jac > best[0]):
                best = (jac, r2)
        return best[1] if best else None

    def block(self, question, readme="", k=6, width=400, contrast=True):
        """Khối 'LUẬT LIÊN QUAN NHẤT' cho prompt: đánh dấu luật MỚI/ĐỔI so với train + luật dễ nhầm đi kèm."""
        rel = self.select(question, readme, k=k)
        if not rel:
            return ""
        ids = {r["id"] for r in rel}
        out = []
        for r in rel:
            mk = RULE_DELTA.mark(r.get("rkey")) if RULE_DELTA is not None else ""
            out.append(f"- [{r['id']}{' ' + r['tieu_de'] if r['tieu_de'] else ''}{' — ' + mk if mk else ''}] {r['text'][:width]}")
        if contrast:
            pairs = []
            for r in rel[:3]:
                r2 = self.look_alike(r, ids)
                if r2 is not None:
                    ids.add(r2["id"])
                    pairs.append(f"- [{r['id']}] «{r['text'].split(':', 1)[0]}» KHÁC [{r2['id']}] «{r2['text'][:220]}» — đọc kỹ, đừng lẫn hai luật")
            if pairs:
                out.append("LUẬT DỄ NHẦM (gần chữ nhưng khác nghĩa):")
                out += pairs
        return "\n".join(out)


RULE_PARAMS = {"tet_start": (12, 23), "tet_end": (1, 10), "week_start": 0}
RULE_DELTA = None   # rule_delta.RuleDelta của rulebook đang chạy (đặt trong run())
_WEEKDAYS = {"hai": 0, "ba": 1, "tu": 2, "nam": 3, "sau": 4, "bay": 5, "chu nhat": 6}


def extract_rule_params(text):
    """Rút tham số số học từ rulebook được phát; thiếu thì giữ mặc định và báo. Trả (params, cảnh_báo)."""
    p, warn = dict(RULE_PARAMS), []
    k = "\n".join(re.sub(r"[*_`]+", "", key(line)) for line in text.splitlines())   # giữ xuống dòng; bỏ ** in đậm
    m = re.search(r"tet[^\n]{0,160}?(\d{1,2}) thang chap[^\n]{0,160}?mung (\d{1,2}) thang gieng", k)
    if m:
        p["tet_start"], p["tet_end"] = (12, int(m.group(1))), (1, int(m.group(2)))
    else:
        warn.append("không tìm thấy khoảng Tết (ngày Chạp → mùng Giêng) trong rulebook: dùng mặc định 23 Chạp → mùng 10")
    m = re.search(r"tuan[^\n]{0,40}?bat dau (?:tu )?thu (hai|ba|tu|nam|sau|bay)|tuan[^\n]{0,40}?bat dau (?:tu )?(chu nhat)", k)
    if m:
        p["week_start"] = _WEEKDAYS[m.group(1) or m.group(2)]
    else:
        warn.append("không tìm thấy ngày bắt đầu tuần trong rulebook: dùng mặc định thứ Hai")
    return p, warn


# =============================================================================
# 3. Bộ giải thời gian: kỳ tính sẵn P (code thuần, có test)
# =============================================================================

VN_MONTH_WORDS = {"mot": 1, "gieng": 1, "hai": 2, "ba": 3, "tu": 4, "bon": 4, "sau": 6, "bay": 7, "tam": 8,
                  "chin": 9, "muoi mot": 11, "muoi hai": 12, "chap": 12, "muoi": 10}


def month_range(y, m):
    a = pd.Timestamp(y, m, 1)
    return a, a + pd.offsets.MonthEnd(0)


def _resolve_year(y, start_month, as_of):
    if y is None:
        y = as_of.year
        if pd.Timestamp(y, start_month, 1) > as_of:
            y -= 1
    elif y < 100:
        y += 2000
    return y


def tet_window(year, cal):
    """Tết năm Y: 23 tháng Chạp (năm âm Y-1) → mùng 10 tháng Giêng (năm âm Y)."""
    if cal is None:
        return None
    (ms, ds), (me, de) = RULE_PARAMS["tet_start"], RULE_PARAMS["tet_end"]     # đọc từ rulebook (extract_rule_params)
    a = cal[(cal.lunar_year == year - 1) & (cal.lunar_month == ms) & (cal.lunar_day == ds) & (~cal.lunar_leap)]
    b = cal[(cal.lunar_year == year) & (cal.lunar_month == me) & (cal.lunar_day == de) & (~cal.lunar_leap)]
    if a.empty or b.empty:
        return None
    return a.date.iloc[0], b.date.iloc[0]


def resolve_periods(question, as_of, cal=None):
    """Trả dict tên -> (start, end) (pd.Timestamp, gồm cả 2 đầu). Đầu cuối không vượt hôm nay."""
    D = pd.Timestamp(as_of).normalize()
    P = {}
    P["hom_nay"] = (D, D)
    P["hom_qua"] = (D - pd.Timedelta(days=1),) * 2
    mon = D - pd.Timedelta(days=(D.weekday() - RULE_PARAMS["week_start"]) % 7)   # ngày đầu tuần theo rulebook
    P["tuan_nay"] = (mon, D)
    P["tuan_truoc"] = (mon - pd.Timedelta(days=7), mon - pd.Timedelta(days=1))
    P["thang_nay"] = (D.replace(day=1), D)
    pm_start = (D.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
    pm_end_full = D.replace(day=1) - pd.Timedelta(days=1)
    P["thang_truoc"] = (pm_start, pm_end_full)
    P["thang_truoc_cung_ky"] = (pm_start, min(pm_start + pd.Timedelta(days=D.day - 1), pm_end_full))
    P["tu_dau_nam"] = (pd.Timestamp(D.year, 1, 1), D)
    P["nam_truoc"] = (pd.Timestamp(D.year - 1, 1, 1), pd.Timestamp(D.year - 1, 12, 31))
    qs = (D.month - 1) // 3 * 3 + 1
    P["quy_nay"] = (pd.Timestamp(D.year, qs, 1), D)

    q = key(question)
    if CMP is not None and hasattr(CMP, "canon_period_text"):
        try:   # cách ghi tắt ('5-2026', '01/2026', '2/26', 'th5', 'Q2/25') -> thêm dạng chuẩn để các mẫu dưới bắt được
            q = q + " | " + CMP.canon_period_text(q, D)
        except Exception:
            pass
    # N ngày gần nhất
    for m in re.finditer(r"(\d{1,3})\s*ngay\s*(gan nhat|qua|vua qua|tro lai|gan day|vua roi|truoc)?", q):
        n = int(m[1])
        if 1 <= n <= 400 and m[2]:
            P[f"{n}_ngay_gan_nhat"] = (D - pd.Timedelta(days=n - 1), D)
    # quý: Q1, quý 2, quy 3/2025, Q4-25
    for m in re.finditer(r"(?<![a-z0-9])(?:quy|q)\s*([1-4])(?:\s*(?:/|-|nam)\s*(\d{4}|\d{2}))?(?![0-9])", q):
        qq = int(m[1])
        y = _resolve_year(int(m[2]) if m[2] else None, (qq - 1) * 3 + 1, D)
        a = pd.Timestamp(y, (qq - 1) * 3 + 1, 1)
        b = a + pd.offsets.QuarterEnd(0)
        P[f"quy_{qq}_{y}"] = (a, min(b, D))
    # tháng: tháng 3/2026, th3/26, T3, thang 3 nam 2025, tháng ba
    for m in re.finditer(r"(?<![a-z0-9])(?:thang|th|t)\s*\.?\s*(\d{1,2})(?:\s*(?:/|-|nam)\s*(\d{4}|\d{2}))?(?![0-9/])", q):
        mo = int(m[1])
        if not 1 <= mo <= 12:
            continue
        y = _resolve_year(int(m[2]) if m[2] else None, mo, D)
        a, b = month_range(y, mo)
        P[f"thang_{mo}_{y}"] = (a, min(b, D))
    # danh sách tháng: 'các tháng 11/2025, 12/2025 và 1/2026' — tháng thứ 2, 3… không có chữ 'tháng' đứng trước
    for m in re.finditer(r"(?:thang|th)\s*\d{1,2}\s*/\s*\d{4}((?:\s*(?:,|va|và|;|&|-|->|→)\s*\d{1,2}\s*/\s*\d{4})+)", q):
        for mo_s, y_s in re.findall(r"(\d{1,2})\s*/\s*(\d{4})", m[1]):
            mo, y = int(mo_s), int(y_s)
            if 1 <= mo <= 12:
                a, b = month_range(y, mo)
                P.setdefault(f"thang_{mo}_{y}", (a, min(b, D)))
    # khoảng tháng: 'từ tháng 8/2025 tới/đến/qua tháng 10/2025' -> mọi tháng ở giữa + P['khoang_thang']
    for m in re.finditer(r"(?:tu|từ)\s*(?:thang|th)\s*(\d{1,2})\s*/\s*(\d{4})\s*(?:den|toi|qua|->|→|-)\s*(?:het\s*)?(?:thang|th)?\s*(\d{1,2})\s*/\s*(\d{4})", q):
        a0 = pd.Timestamp(int(m[2]), int(m[1]), 1)
        b0 = pd.Timestamp(int(m[4]), int(m[3]), 1)
        if a0 <= b0 and (b0.year - a0.year) * 12 + b0.month - a0.month <= 24:
            cur = a0
            while cur <= b0:
                a, b = month_range(cur.year, cur.month)
                P.setdefault(f"thang_{cur.month}_{cur.year}", (a, min(b, D)))
                cur = cur + pd.offsets.MonthBegin(1)
            P["khoang_thang"] = (a0, min(b0 + pd.offsets.MonthEnd(0), D))
    ql_acc = nfc(question).lower()
    for w, mo in sorted(VN_MONTH_WORDS.items(), key=lambda x: -len(x[0])):
        # 'tháng sau' (tháng kế tiếp) và 'tháng sáu' (tháng 6) trùng nhau khi bỏ dấu: chỉ nhận 'sau' khi câu có dấu 'sáu'
        if w == "sau" and not re.search(r"tháng\s+sáu", ql_acc):
            continue
        for m in re.finditer(rf"thang {w}(?![a-z])(?:\s*(?:/|-|nam)\s*(\d{{4}}|\d{{2}}))?", q):
            y = _resolve_year(int(m[1]) if m[1] else None, mo, D)
            a, b = month_range(y, mo)
            P.setdefault(f"thang_{mo}_{y}", (a, min(b, D)))
    # năm tường minh
    for m in re.finditer(r"nam\s*(20\d{2})", q):
        y = int(m[1])
        P[f"nam_{y}"] = (pd.Timestamp(y, 1, 1), min(pd.Timestamp(y, 12, 31), D))
    # ngày tường minh dd/mm/yyyy
    for m in re.finditer(r"(?<![0-9/])(\d{1,2})/(\d{1,2})/(\d{4})(?![0-9])", q):
        try:
            d0 = pd.Timestamp(int(m[3]), int(m[2]), int(m[1]))
            P[f"ngay_{d0.date()}"] = (d0, d0)
        except ValueError:
            pass
    # Tết
    if re.search(r"(?<![a-z])tet(?![a-z])", q) and cal is not None:
        years = {int(y) for y in re.findall(r"tet\s*(?:nam\s*)?(20\d{2})", q)} or {D.year, D.year - 1}
        for y in sorted(years):
            w = tet_window(y, cal)
            if w and w[0] <= D:
                a, b = w
                L = (b - a).days + 1
                P[f"tet_{y}"] = (a, min(b, D))
                P[f"truoc_tet_{y}"] = (a - pd.Timedelta(days=L), a - pd.Timedelta(days=1))
    return P


def periods_text(P):
    return "\n".join(f"  P[{k!r}] = ({a.date()} → {b.date()})" for k, (a, b) in P.items())


# =============================================================================
# 4. Cảnh báo "bài học" từ 16 cách hiểu nhầm (mục 6.2 của thiết kế)
# =============================================================================


def lesson_warnings(question):
    q = key(question)
    W = []
    if re.search(r"(?<![a-z])so\s+(voi|vs|sanh|cung|thang|ky|nam|tuan|hom|quy|ca|t\d|th\d|q\d)(?![a-z])", q) or "so với" in nfc(question).lower():
        W.append("SO SÁNH: vế đứng sau 'so với/so' là KỲ GỐC. % thay đổi = (kỳ được hỏi − kỳ gốc) / kỳ gốc × 100. "
                 "Câu 'So với X, Y …' thì X là gốc, Y là kỳ được hỏi. Kiểm lại dấu: kỳ hỏi nhỏ hơn gốc thì % âm.")
    if re.search(r"(?<![a-z0-9])(quy|q)\s*[1-4]", q):
        W.append("QUÝ: 'Q1/quý 1' là CẢ QUÝ (3 tháng), KHÔNG phải tháng 1. Dùng P['quy_..'].")
    if re.search(r"(phai khong|dung khong|phai ko|dung ko|co phai|hay khong|\bko\b\s*\??$|khong\s*\??$|chu\s*\??$|ha\s*\??$)", q) and \
            re.search(r"(?<![a-z])(va|roi|lai|lien tuc|luon|ca hai|deu)(?![a-z])", q):
        W.append("ĐÚNG/SAI NHIỀU VẾ: phải kiểm TẤT CẢ các vế; chỉ trả true khi mọi vế đúng.")
    if re.search(r"nhap", q) and re.search(r"(tru|ban)", q):
        W.append("NHẬP − BÁN: quy CẢ HAI vế về CÙNG đơn vị (đơn vị câu hỏi nêu, ví dụ 'cái'); dùng hệ số quy đổi (hộp, thùng, bó…).")
    ql = nfc(question).lower()
    neg_word = re.search(r"(?<!\w)(ngoài|trừ|bỏ ra|bỏ qua|không tính|ko tính|không kể|chẳng|ngoai|tru|khong tinh|khong ke|ko ke)(?!\w)", ql)
    neg_mid = re.search(r"(?<!\w)(không|ko|hông|khong|hong|hổng)\s+(?!(ạ|vậy|vay|nhỉ|nhi|ha|hả|nhé|nhe|z|zậy|ta|nữa)(\W|$))\w", ql)
    if neg_word or neg_mid:
        W.append("PHỦ ĐỊNH: câu có từ phủ định (không/ngoài/trừ/bỏ) — áp đúng vào điều kiện mà nó bổ nghĩa.")
    if re.search(r"(tu \d+.{0,20}tro len|it nhat|toi thieu)", q):
        W.append("NGƯỠNG: 'từ N trở lên/ít nhất N' là >= N.")
    if re.search(r"(?<![a-z])(hon|tren|qua|vuot)\s+\d", q):
        W.append("NGƯỠNG: 'hơn/trên/quá N' là > N (không gồm N).")
    if re.search(r"(trung binh|binh quan)", q):
        W.append("TRUNG BÌNH: mẫu số là thứ đứng sau 'mỗi' (mỗi đơn / mỗi ngày / mỗi khách); không nói thì theo rulebook. Tiền trung bình làm tròn về đồng.")
    if re.search(r"(thu nhi|hang nhi|dung nhi|(xep|dung|hang|chay|nhat|nhieu|cao|thap|lon)\s+thu\s+(hai|ba|tu|nam|\d))", q):
        W.append("THỨ HẠNG: 'thứ hai/nhì' là MỘT phần tử ở hạng đó, không phải top-k.")
    if re.search(r"(chiem|ti trong|ty trong)", q):
        W.append("TỈ TRỌNG: mẫu số theo rulebook (thường là doanh thu cả shop, gồm cả khách lẻ và phí phục vụ) trừ khi câu nói 'trong nhóm…'.")
    if re.search(r"(?<![a-z])ai(?![a-z])", q):
        W.append("'AI': nếu ngữ cảnh là nhân viên (NV, bán giỏi, chăm chỉ, đứng ca) thì là nhân viên; còn lại là khách. Khách lẻ không tính là khách.")
    if re.search(r"(con no|con ton|den het ngay|den cuoi ngay|con lai)", q):
        W.append("SỐ CÒN LẠI: tính trạng thái tích luỹ đến hết ngày được hỏi (không chỉ phát sinh trong kỳ).")
    if re.search(r"(thang truoc|thang roi)", q) and re.search(r"(thang|th|t)\s*\d", q):
        W.append("'THÁNG TRƯỚC' trong câu đã có tháng cụ thể = tháng liền trước THÁNG ĐƯỢC HỎI, không phải tháng trước của hôm nay.")
    return W


# =============================================================================
# 5. LLM: backend vLLM / HF / dummy, có cache
# =============================================================================


class LLM:
    def __init__(self, backend, model, cache_dir, max_model_len=20000, tp=1, gpu_mem=0.90, max_new_tokens=1400):
        self.backend, self.model_name = backend, model
        self.max_new = max_new_tokens
        self.max_model_len = max_model_len
        self.cache_path = os.path.join(cache_dir, "llm_cache.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["v"]
                    except Exception:
                        pass
        self.tok = None
        if backend == "vllm":
            # GPU cũ (T4 = sm75): FlashInfer phải BIÊN DỊCH kernel lúc chạy và hỏng (đã gặp trên Kaggle: ninja/ld lỗi ở
            # batch_prefill head_dim 256) -> tắt bộ lấy mẫu FlashInfer, ép backend attention Triton (chạy được trên mọi GPU)
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            from vllm import LLM as VLLM, SamplingParams
            import torch
            cap = torch.cuda.get_device_capability()[0]
            dtype = "float16" if cap < 8 else "bfloat16"
            base = dict(model=model, dtype=dtype, max_model_len=max_model_len, tensor_parallel_size=tp,
                        gpu_memory_utilization=gpu_mem, seed=SEED, trust_remote_code=True)
            # model đa phương thức (vd Qwen3.5) chỉ dùng chữ: không dành bộ nhớ cho bộ mã hoá ảnh.
            # Thử từ cấu hình đầy đủ -> bớt dần tham số mà bản vLLM đang cài không hiểu / không hỗ trợ cho kiến trúc này.
            mm = any(k in model.lower() for k in ("qwen3.5", "qwen3_5", "qwen3.6", "-vl"))   # model ảnh+chữ
            no_img = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if mm else {}
            # Thử từ cấu hình đầy đủ -> bớt dần. Engine V1 chạy lõi trong TIẾN TRÌNH CON: mọi lỗi khởi tạo (vd model lai
            # Gated DeltaNet + prefix caching -> "Mamba cache align mode" assert, CUDA graph không chụp được trên T4) chỉ
            # hiện ra ở đây là RuntimeError "Engine core initialization failed" -> cũng phải thử cấu hình sau.
            tries = [dict(enable_prefix_caching=True, **no_img),
                     dict(enable_prefix_caching=False, **no_img),
                     dict(enable_prefix_caching=False, enforce_eager=True, **no_img)]
            if cap < 8:
                # attention_config: vLLM ≥ 0.12 (V1). Bản cũ không hiểu -> TypeError -> các cấu hình KHÔNG ép backend ở cuối
                att = [{"attention_config": {"backend": "TRITON_ATTN"}}] * 3 + [{"attention_config": {"backend": "FLEX_ATTENTION"}}]
                tries = [dict(t, **a) for t, a in zip(tries + [dict(enable_prefix_caching=False, enforce_eager=True, **no_img)], att)] + tries
            if mm:
                tries.append(dict(enable_prefix_caching=False, enforce_eager=True))      # bản vLLM không hiểu limit_mm_per_prompt
            err = None
            for extra in tries:
                try:
                    self.engine = VLLM(**base, **extra)
                    self.vllm_extra = extra
                    break
                except (TypeError, ValueError, NotImplementedError, RuntimeError, AssertionError) as e:
                    err = e
                    print(f"  [vLLM] cấu hình {extra} lỗi ({type(e).__name__}: {str(e)[:200]}) -> thử cấu hình sau", flush=True)
                    try:                                # dọn bộ nhớ GPU của lần thử hỏng trước khi thử lần sau
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            else:
                raise err
            self.sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, seed=SEED)
            self.tok = self.engine.get_tokenizer()
        elif backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            # sdpa: attention không dựng cả ma trận seq×seq (Kaggle: 'eager' đòi 11–35 GiB cho prompt ~10k token -> OOM mọi câu)
            kw = dict(device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="sdpa")
            if "awq" not in model.lower() and "gptq" not in model.lower():
                try:
                    from transformers import BitsAndBytesConfig
                    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                except Exception:
                    pass
            try:
                self.hf = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
            except (ValueError, KeyError):             # model đa phương thức (Qwen3.5…): lớp ảnh-chữ, vẫn chỉ đưa chữ vào
                from transformers import AutoModelForImageTextToText
                self.hf = AutoModelForImageTextToText.from_pretrained(model, **kw).eval()
        elif backend == "dummy":
            pass
        else:
            raise ValueError(backend)

    def _fmt(self, messages, think=False):
        if self.tok is not None:
            # enable_thinking=False (mặc định): Qwen3/3.5 tự "nghĩ" trước khi trả lời -> tiêu hết max_tokens vào phần nghĩ.
            # think=True chỉ dùng ở vòng XEM LẠI (max_tokens lớn). Template không có biến này (Qwen2.5…) thì Jinja bỏ qua.
            try:
                return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(think))
            except TypeError:
                return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)

    def _k(self, prompt):
        return hashlib.sha256((self.model_name + "\x00" + prompt).encode("utf-8")).hexdigest()

    def n_tokens(self, prompt):
        if self.tok is None:
            return len(prompt) // 3
        return len(self.tok(prompt, add_special_tokens=False)["input_ids"])

    def generate(self, list_of_messages, think=False, max_new=None):
        """think=True: cho model (Qwen3/3.5) SUY NGHĨ trước khi trả lời (dùng ở vòng xem lại câu bỏ trống);
        max_new: số token sinh tối đa cho lượt này (phần nghĩ dài hơn code nhiều)."""
        self._cur_max = int(max_new or self.max_new)
        prompts = [self._fmt(m, think=think) for m in list_of_messages]
        keys = [self._k(p) + (f"#max{self._cur_max}" if self._cur_max != self.max_new else "") for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        st = self.__dict__.setdefault("stats", Counter())
        st["goi_dai"] += len(prompts)
        st["goi_dai_that"] += len(todo)
        st["ky_tu_prompt"] += sum(len(prompts[i]) for i in todo)
        # prompt quá dài sẽ làm vLLM báo lỗi CẢ LÔ -> loại riêng từng prompt quá dài (trả rỗng = không có code = abstain)
        limit = self.max_model_len - self._cur_max - 32
        too_long = [i for i in todo if self.n_tokens(prompts[i]) > limit]
        for i in too_long:
            print(f"  [cảnh báo] prompt {self.n_tokens(prompts[i])} token > giới hạn {limit} -> bỏ qua câu này", flush=True)
            self.cache[keys[i]] = ""
        todo = [i for i in todo if i not in set(too_long)]
        if todo:
            outs = self._gen_safe([prompts[i] for i in todo])
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                for i, o in zip(todo, outs):
                    if o is None:                       # lỗi sinh (OOM…): không ghi cache, câu này coi như không có code
                        continue
                    self.cache[keys[i]] = o
                    fh.write(jdump({"k": keys[i], "v": o}) + "\n")
            for i, o in zip(todo, outs):
                if o is None:
                    self.cache.setdefault(keys[i] + "#loi", "")
        return [self.cache.get(k, "") for k in keys]

    def _gen_safe(self, prompts):
        """Sinh cả lô; lô lỗi -> sinh LẺ từng prompt; prompt lỗi -> None. Một prompt hỏng không được làm hỏng cả lô câu hỏi
        (Kaggle backend hf: OOM ở 1 prompt dài làm 'fatal' toàn bộ 1000 câu, mất hết đường code)."""
        st = self.__dict__.setdefault("stats", Counter())
        try:
            return self._gen(prompts)
        except Exception as e:
            st["loi_lo"] += 1
            print(f"  [LLM] lô {len(prompts)} prompt lỗi ({type(e).__name__}: {str(e)[:160]}) -> sinh lẻ từng prompt", flush=True)
            self._free_gpu()
        outs = []
        for p in prompts:
            try:
                outs.append(self._gen([p])[0])
            except Exception as e:
                st["loi_prompt"] += 1
                print(f"  [LLM] prompt lỗi ({type(e).__name__}: {str(e)[:120]}) -> bỏ", flush=True)
                self._free_gpu()
                outs.append(None)
        return outs

    @staticmethod
    def _free_gpu():
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate_short(self, list_of_messages, max_tokens=4):
        """Câu trả lời rất ngắn (trắc nghiệm 1 chữ cái), greedy, có cache riêng."""
        prompts = [self._fmt(m) for m in list_of_messages]
        keys = [self._k("short\x00" + p) for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        st = self.__dict__.setdefault("stats", Counter())
        st["goi_ngan"] += len(prompts)
        st["goi_ngan_that"] += len(todo)
        if todo:
            if self.backend == "vllm":
                from vllm import SamplingParams
                sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=SEED)
                res = self.engine.generate([prompts[i] for i in todo], sp, use_tqdm=False)
                outs = [self.strip_think(r.outputs[0].text) for r in res]
            elif self.backend == "hf":
                import torch
                outs = []
                for i in todo:
                    try:
                        ids = self.tok(prompts[i], return_tensors="pt").to(self.hf.device)
                        with torch.no_grad():
                            g = self.hf.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
                        outs.append(self.strip_think(self.tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))
                    except Exception as e:
                        print(f"  [LLM] câu hỏi ngắn lỗi ({type(e).__name__}) -> bỏ", flush=True)
                        self._free_gpu()
                        outs.append("")
            else:
                outs = ["A"] * len(todo)        # dummy: luôn chọn phương án đầu (= lựa chọn của bộ sửa chữ)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                for i, o in zip(todo, outs):
                    self.cache[keys[i]] = o
                    fh.write(jdump({"k": keys[i], "v": o}) + "\n")
        return [self.cache[k] for k in keys]

    @staticmethod
    def strip_think(t):
        """Bỏ phần <think>…</think> (model có chế độ nghĩ). Thẻ mở mà không đóng (nghĩ đến hết max_tokens) -> rỗng."""
        if "<think>" not in t and "</think>" not in t:
            return t
        if "</think>" in t:
            return t.split("</think>")[-1].lstrip()
        return ""

    def _gen(self, prompts):
        return [self.strip_think(o) for o in self._gen_raw(prompts)]

    def _gen_raw(self, prompts):
        mx = getattr(self, "_cur_max", None) or self.max_new
        if self.backend == "vllm":
            sp = self.sp
            if mx != self.max_new:
                from vllm import SamplingParams
                # lượt dài (xem lại, có suy nghĩ): giải mã tham lam dễ lặp vòng trong phần nghĩ -> phạt lặp nhẹ (vẫn tất định)
                sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=mx, seed=SEED, repetition_penalty=1.05)
            res = self.engine.generate(prompts, sp, use_tqdm=False)
            return [r.outputs[0].text for r in res]
        if self.backend == "hf":
            import torch
            outs = []
            for p in prompts:
                ids = self.tok(p, return_tensors="pt").to(self.hf.device)
                with torch.no_grad():
                    g = self.hf.generate(**ids, max_new_tokens=mx, do_sample=False,
                                         **({"repetition_penalty": 1.05} if mx != self.max_new else {}))
                outs.append(self.tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
            return outs
        # dummy: luôn abstain (dùng để test luồng chạy)
        return ["```python\ndef solve():\n    return {'type': 'abstain'}\n```"] * len(prompts)


# =============================================================================
# 6. Prompt
# =============================================================================

SYSTEM_TMPL = """Bạn là chuyên viên phân tích dữ liệu bán hàng Việt Nam, cực kỳ cẩn thận. Nhiệm vụ: viết MỘT hàm Python `solve()` trả lời đúng câu hỏi trên dữ liệu shop, áp dụng ĐÚNG rulebook.

QUY TẮC TUYỆT ĐỐI:
1. Chỉ dùng các biến có sẵn: T (dict tên_bảng -> pandas.DataFrame, cột ĐÃ chuẩn kiểu: ngày là datetime64, tiền/số là float), P (dict kỳ thời gian đã tính sẵn: tên -> (start, end) gồm 2 đầu), AS_OF (pd.Timestamp hôm nay), CAL (lịch âm dương), pd, np, re, math, key(s) (chuẩn hoá tên: thường, bỏ dấu), rhu(x, n) (làm tròn half-up), pct(a, b) (% của a so với gốc b, 2 chữ số), parse_money(series), parse_date(series).
2. KHÔNG import os/sys/subprocess, KHÔNG đọc/ghi file, KHÔNG in ra. Được import pandas/numpy/re/math/datetime/collections/itertools/unicodedata.
3. Thẻ dữ liệu có mục LỚP NGHĨA: cột tự động __nhom_trang_thai (hoan_thanh/huy/tra_hang/dang_giao/chua_ro, đã ánh xạ từ cột trạng thái của shop), __sl_co_ban (số lượng quy về đơn vị cơ bản theo bảng quy đổi của shop), __khach_le (khách lẻ/vãng lai). DÙNG các cột này thay vì tự ánh xạ lại, trừ khi README nói khác; giá trị chua_ro thì tự xếp theo README/rulebook. Đọc kỹ thẻ dữ liệu và GHI CHÚ LÀM SẠCH: nhóm dòng (ô trống ở dòng tiếp nối) và dòng trùng trong cùng đơn ĐÃ được xử lý sẵn nếu ghi chú nói vậy — KHÔNG tự bỏ trùng thêm; bảng không có mã đơn thì hai dòng giống nhau vẫn tính cả hai; nếu còn ô trống theo nhóm dòng chưa xử lý thì lấy theo dòng đầu nhóm; ánh xạ MỌI giá trị trạng thái theo rulebook (hoàn thành / huỷ / trả hàng / đang giao); so tên hàng/khách bằng key(); KHÔNG dùng str.contains với chuỗi ngắn; khách lẻ không phải khách; mã có khoảng trắng thì strip.
3b. BỘ BẢNG CHUẨN V (xem thẻ dữ liệu) và DATA_END: V['don'] (ma_don, ngay, khach_ma, khach_ten, kenh, httt, tien_hang, giam_gia_don, phi_giao, tong_don = số khách phải trả cả đơn, da_tra_file, nhom_tt, tinh_doanh_thu, doanh_thu = tổng thành tiền dòng), V['dong'] (dòng hàng: ten_hang, ma_hang, so_luong, don_vi, sl_co_ban, don_gia, thanh_tien, ngay, nhom_tt…), V['phieu_nhap'] (ma_phieu, ngay, ncc, ncc_ten, tong_tien, huy), V['nhap'], V['thu'] (ma_don, ngay_thu, so_tien), V['ton'], V['the_kho'], V['lo_nhap'], V['ban_lo'], V['quy_doi'], V['hang']. Hàm có sẵn (đã gắn V, đã ĐỐI CHIẾU đúng rulebook trên dữ liệu mẫu): doanh_thu_ky(tu, den); tien_nhap_ky(tu, den); tong_no(X); so_khach_con_no(X); no_theo_khach(X) -> DataFrame[khach_hang, so_tien_no]; tuoi_no_nhom(X, (7, 15)); no_qua_n_ngay(X, n); no_them_tra_no(tu, den, khach=None) -> [nợ thêm, trả nợ]; thu_no_ky(tu, den); cong_no_don(X); tim_hang(ten) -> (ma, ten); ton_tai_ngay(ten_hang, X); ton_don_vi_lon(ten_hang, don_vi_lon, X) -> [chẵn, lẻ cơ bản]; don_vi_lon_trong_cau(ten_hang, cau); he_so(ten_hang, don_vi); ton_lo(X); lo_het_han_trong(D, N); lo_gan_het_han_nhat(ten_hang, D); so_don_ky(tu, den); doanh_thu_binh_quan_don(tu, den); gia_nhap_binh_quan(ten_hang, tu, den, don_vi); ton_tat_ca(X); so_mat_hang_het_hang(X); gia_tri_ton_kho(X); lai_gop_ky(tu, den, theo_hang=False); chi_phi_ky(tu, den); lai_rong_ky(tu, den); doanh_thu_theo_nhan_vien(tu, den); hoa_hong_ky(tu, den). Shop dịch vụ (sửa xe…): V['don'] mỗi dòng một phiếu, V['viec'] (việc đã tách, ghi thô), V['nhan_vien']. Hàm trả None = file không có thông tin → unanswerable. Ưu tiên V và các hàm này; câu hỏi có điều kiện khác (kênh, nhóm khách, mặt hàng…) thì lọc V rồi tự tính theo rulebook.
4. Doanh thu, số đơn… theo rulebook (loại đơn huỷ và đơn có trả hàng, giữ đơn đang giao). Tỉ lệ huỷ: mẫu số là mọi đơn.
5. Dùng P[...] cho kỳ thời gian; nếu cần kỳ khác thì tự tính theo rulebook. Bản ghi sau AS_OF coi như chưa xảy ra.
6. solve() TRẢ VỀ dict đúng format: {{"type": "number"|"percent"|"date"|"string"|"list"|"table"|"boolean", "answer": ...}}
   - number: số (tiền là số nguyên đồng, làm tròn rhu); percent: số % 2 chữ số (12.4 chứ không 0.124); date: "YYYY-MM-DD", hỏi tháng thì "YYYY-MM";
   - string: tên đúng như trong dữ liệu; list: mảng có thứ tự (1 phần tử vẫn là mảng); table: {{"columns": [...], "rows": [[...], ...]}}; boolean: True/False.
   - Nếu dữ liệu KHÔNG có thông tin câu hỏi cần (giới tính, địa chỉ, lý do, đánh giá chủ quan, cột không tồn tại…) hoặc kỳ hỏi nằm ngoài dữ liệu/ở tương lai: {{"type": "unanswerable", "reason": "..."}}.
   - Nếu không chắc chắn: {{"type": "abstain"}}.
7. Bằng nhau thì xếp theo MÃ tăng dần (không có mã thì theo tên).
8. Trả lời bằng MỘT khối ```python ...``` chứa hàm solve() và các hàm phụ nếu cần. Trước khối code, viết 2–4 dòng ghi chú ngắn: tính gì, kỳ nào, bảng/cột nào, luật nào.

===== RULEBOOK (bắt buộc tuân theo) =====
{rulebook}
"""

USER_TMPL = """===== SHOP: {biz} =====
README:
{readme}

THẺ DỮ LIỆU (T[tên_bảng]):
{card}

===== CÂU HỎI =====
Hôm nay (as_of): {as_of}
Câu hỏi gốc: {q_raw}
{q_norm_line}
Kỳ thời gian đã tính sẵn (dùng P[...]):
{periods}
{warnings}
Viết hàm solve()."""

USER_TMPL_B = """===== SHOP: {biz} =====
THẺ DỮ LIỆU (T[tên_bảng]):
{card}

README:
{readme}

===== NHIỆM VỤ =====
Ngày hôm nay là {as_of}. Chủ shop hỏi nguyên văn: «{q_raw}»
Trước hết hãy xác định rõ: (1) hỏi đại lượng gì, (2) của đối tượng nào, (3) kỳ nào (và kỳ gốc nếu so sánh), (4) điều kiện/phủ định, (5) kiểu đáp án. Sau đó viết solve().
Các kỳ có sẵn trong P:
{periods}
{warnings}"""


RULE_INDEX = None   # RuleIndex của rulebook đang chạy (đặt trong run())


RETHINK_TMPL = """
=== XEM LẠI KỸ: câu này các lần giải trước CHƯA đủ chắc để trả lời ===
{prev}
Lần này đừng làm nhanh. Trước khi viết code, suy nghĩ chậm từng bước và GHI phần phân tích thành chú thích # ở đầu solve():
1. Câu hỏi hỏi CHÍNH XÁC đại lượng gì, của ai / mặt hàng nào / nhân viên nào / kênh nào, kỳ nào (dùng đúng KỲ ĐÃ TÍNH SẴN),
   đáp án thuộc loại nào (số, %, ngày, tên, danh sách, bảng, đúng/sai, không trả lời được).
2. Luật nào trong rulebook áp dụng (đơn huỷ, trả hàng, khách lẻ, quy đổi đơn vị, làm tròn, hoà, luật riêng của ngành) — ghi tên luật.
3. Dữ liệu ở bảng/cột nào. Tên mặt hàng / khách / nhân viên trong câu có thể viết KHÁC dữ liệu (viết tắt, từ địa phương:
   phụ = hũ, rán = chiên…): in thử các giá trị gần giống rồi chọn đúng, KHÔNG được bỏ qua bộ lọc mà tính cả shop.
4. Tự kiểm kết quả: độ lớn, dấu, đơn vị, kỳ có dữ liệu không. Chỉ trả unanswerable khi rulebook nói vậy hoặc dữ liệu thật sự không có.
{judge}"""

RETHINK_JUDGE = ("Các kết quả trên LỆCH nhau hoặc chưa được xác nhận. Đừng tin kết quả nào chỉ vì nó có sẵn: tự tính lại từ đầu "
                 "theo rulebook; nếu cách nào ở trên hiểu sai câu hỏi hay luật thì chỉ rõ trong chú thích.")


def build_messages(rulebook, biz, q, q_norm, as_of, P, warnings, variant="A", readme_chars=3500, prev=None):
    sys_msg = SYSTEM_TMPL.format(rulebook=rulebook)
    w = ("LƯU Ý (bài học hay sai):\n" + "\n".join(f"- {x}" for x in warnings)) if warnings else ""
    if RULE_INDEX is not None:
        blk = RULE_INDEX.block(q + " " + (q_norm or ""), biz.readme or "", k=6)
        if blk:
            w = ("LUẬT LIÊN QUAN NHẤT (trích rulebook; vẫn phải theo TOÀN BỘ rulebook):\n" + blk + ("\n" + w if w else ""))
    common = dict(biz=biz.id, readme=(biz.readme or "(không có)")[:readme_chars], card=biz.card, as_of=str(pd.Timestamp(as_of).date()),
                  q_raw=q, periods=periods_text(P), warnings=w)
    if variant in ("A", "D1", "D2"):
        q_norm_line = f"Câu hỏi đã chuẩn hoá chữ: {q_norm}" if q_norm and q_norm.strip().lower() != q.strip().lower() else ""
        user = USER_TMPL.format(q_norm_line=q_norm_line, **common)
    else:
        user = USER_TMPL_B.format(**common)
    if variant == "D1":        # xem lại CÓ đối chiếu các kết quả trước
        user += RETHINK_TMPL.format(prev=prev or "(các lần trước không ra kết quả)", judge=RETHINK_JUDGE)
    elif variant == "D2":      # xem lại ĐỘC LẬP (không cho thấy kết quả trước -> phiếu bầu không bị dắt)
        user += RETHINK_TMPL.format(prev="", judge="")
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


# =============================================================================
# 7. Trích code, kiểm AST, chạy sandbox có timeout
# =============================================================================

BANNED_MODULES = {"os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pathlib", "pickle", "importlib", "ctypes", "multiprocessing", "threading", "builtins"}
BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "exit", "quit", "globals", "setattr", "delattr", "breakpoint"}
ALLOWED_MODULES = {"pandas", "numpy", "re", "math", "datetime", "collections", "itertools", "unicodedata", "decimal", "functools",
                   "statistics", "calendar", "string", "operator", "heapq", "bisect", "json", "typing", "dateutil"}


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        for b in blocks[::-1]:
            if "def solve" in b:
                return b
        return blocks[-1]
    if "def solve" in text:
        return text[text.index("def solve"):]
    return None


def ast_guard(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    has_solve = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            has_solve = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            for md in mods:
                if md in BANNED_MODULES or md not in ALLOWED_MODULES:
                    return f"import bị cấm: {md}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            return f"hàm bị cấm: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr in ("to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "system", "popen"):
            return f"thao tác bị cấm: {node.attr}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__name__",):
            return f"thuộc tính bị cấm: {node.attr}"
    return None if has_solve else "không có hàm solve()"


def _shuffled_ns(ns, code, seed=12345):
    """Bản sao namespace với các bảng MÀ CODE NHẮC TÊN (T['...'], V['...']) bị đảo thứ tự dòng (giữ chỉ số) — kiểm metamorphic:
    câu trả lời đúng theo rulebook không phụ thuộc thứ tự dòng trong file (hoà đã có luật xếp theo mã).
    Hàm ledger giữ nguyên bảng gốc (đã kiểm riêng bằng train); chỉ đảo bảng code tự đọc -> nhanh."""
    g = dict(ns)
    used = set(re.findall(r"""\[\s*['"]([^'"]+)['"]\s*\]""", code))
    shuf = lambda d: {k: (v.sample(frac=1.0, random_state=seed) if k in used and isinstance(v, pd.DataFrame) and len(v) > 1 else v) for k, v in d.items()}
    g["T"] = shuf(ns.get("T", {}))
    if "V" in ns:
        g["V"] = shuf(ns["V"])
    return g


RULE_TESTS = {
    "tuong_lai": "§ thời gian: bản ghi có ngày SAU hôm nay coi như chưa xảy ra",
    "huy": "§1: đơn HUỶ không tính doanh thu / số đơn / số lượng bán / công nợ / tồn",
}


def _inject_order(ns, kind, seed_date, code=""):
    """Bài kiểm từ LUẬT (metamorphic): chèn một đơn GIẢ chép từ một đơn thật vào mọi bảng gốc có cột mã đơn + bộ bảng chuẩn V,
    - 'tuong_lai': ngày = hôm nay + 45 ngày  -> câu hỏi đến hôm nay KHÔNG được đổi đáp án;
    - 'huy'      : ngày trong kỳ hỏi, trạng thái HUỶ -> câu hỏi không về huỷ KHÔNG được đổi đáp án.
    Trả namespace mới hoặc None nếu không chèn được (shop không có mã đơn / không có trạng thái huỷ)."""
    info = ns.get("__inject_info__") or {}
    V = ns.get("V") or {}
    o = V.get("don")
    if o is None or not len(o) or not info.get("oid_cols"):
        return None
    pool = o[o["tinh_doanh_thu"]]["ma_don"].astype(str)
    pool = sorted(set(pool) & set(info.get("raw_ids", ())))
    if not pool:
        return None
    sid0 = pool[len(pool) // 2]
    new_id = "ZZTEST" + kind.upper()
    nd = (ns["AS_OF"].normalize() + pd.Timedelta(days=45)) if kind == "tuong_lai" else pd.Timestamp(seed_date)
    g = dict(ns)
    T2 = dict(ns["T"])
    done = False
    used = set(re.findall(r"""\[\s*['"]([^'"]+)['"]\s*\]""", code)) if code else None
    for tn, oc in info["oid_cols"].items():
        if used is not None and tn not in used:
            continue                      # chỉ chép vào bảng gốc mà code thật sự đọc (bảng lớn chép lại rất chậm)
        t = T2.get(tn)
        if t is None or oc not in t.columns:
            continue
        rows = t[t[oc].astype(str).str.strip() == sid0]
        if not len(rows):
            continue
        rows = rows.copy()
        rows[oc] = new_id
        for dc in info["date_cols"].get(tn, ()):
            rows[dc] = nd if pd.api.types.is_datetime64_any_dtype(t[dc]) else nd.strftime("%Y-%m-%d")
        if kind == "huy":
            sc = info["status_cols"].get(tn)
            if sc is None or info["huy_raw"].get(tn) is None:
                if "__nhom_trang_thai" in t.columns:
                    return None          # bảng có trạng thái mà không biết chữ 'huỷ' gốc -> không chèn được đúng
            else:
                rows[sc] = info["huy_raw"][tn]
                rows["__nhom_trang_thai"] = "huy"
        T2[tn] = pd.concat([t, rows], ignore_index=True)
        done = True
    uses_v = (not code) or ("V[" in code) or (ledger is not None and any(re.search(r"\b" + n + r"\s*\(", code) for n in ledger.HELPERS))
    if not done and not uses_v:
        return None
    g["T"] = T2
    if not uses_v:
        return g
    V2 = dict(V)
    od = o[o["ma_don"].astype(str) == sid0].copy()
    od["ma_don"], od["ngay"] = new_id, nd
    if kind == "huy":
        od["nhom_tt"], od["tinh_doanh_thu"] = "huy", False
    V2["don"] = pd.concat([o, od], ignore_index=True)
    if "dong" in V:
        dl = V["dong"][V["dong"]["ma_don"].astype(str) == sid0].copy()
        dl["ma_don"] = new_id
        if "ngay" in dl.columns:
            dl["ngay"] = nd
        if kind == "huy" and "nhom_tt" in dl.columns:
            dl["nhom_tt"] = "huy"
        V2["dong"] = pd.concat([V["dong"], dl], ignore_index=True)
    g["V"] = V2
    if ledger is not None:
        de = ns.get("DATA_END")
        for name, f in ledger.HELPERS.items():
            if name in g:
                g[name] = functools.partial(f, V2, data_end=de) if name in ("ton_tai_ngay", "ton_don_vi_lon", "ton_tat_ca", "so_mat_hang_het_hang",
                                                                             "gia_tri_ton_kho") else functools.partial(f, V2)
    return g


def inject_info(biz):
    """Chuẩn bị (một lần mỗi shop) thông tin để chèn đơn giả: bảng nào có cột mã đơn, cột ngày, cột trạng thái + chữ 'huỷ' gốc."""
    if getattr(biz, "_inject_info", None) is not None:
        return biz._inject_info
    info = {"oid_cols": {}, "date_cols": {}, "status_cols": {}, "huy_raw": {}, "raw_ids": set()}
    for tn, t in biz.T.items():
        oc = order_id_col(t, biz.types[tn])
        if oc is None or oc == "__ma_don" or len(t) > 400000:
            continue
        info["oid_cols"][tn] = oc
        info["raw_ids"] |= set(t[oc].astype(str).str.strip())
        info["date_cols"][tn] = [c for c in t.columns if str(biz.types[tn].get(c, "")).startswith("ngày")]
        sc = (getattr(biz, "status_src", {}) or {}).get(tn)
        if sc is not None and "__nhom_trang_thai" in t.columns:
            info["status_cols"][tn] = sc
            hv = t.loc[t["__nhom_trang_thai"] == "huy", sc].dropna()
            info["huy_raw"][tn] = hv.iloc[0] if len(hv) else None
    biz._inject_info = info
    return info


def _child(conn, code, ns):
    try:
        import warnings as _w
        _w.filterwarnings("ignore")
        g = dict(ns)
        exec(compile(code, "<solve>", "exec"), g)
        res = g["solve"]()
        out = json.loads(jdump(res))
        for kind in ns.get("__rule_tests__") or ():
            try:
                g3 = _inject_order(ns, kind, ns.get("__seed_date__"), code)
                if g3 is None:
                    continue
                exec(compile(code, "<solve>", "exec"), g3)
                out3 = json.loads(jdump(g3["solve"]()))
                if isinstance(out, dict) and isinstance(out3, dict) and canon_raw(out) != canon_raw(out3):
                    out = dict(out, __rule_diff__=f"{kind}|{str(out3.get('answer'))[:150]}")
                    break
            except Exception:
                pass
        if ns.get("__metamorphic__"):
            try:
                g2 = _shuffled_ns(ns, code)
                exec(compile(code, "<solve>", "exec"), g2)
                out2 = json.loads(jdump(g2["solve"]()))
                if isinstance(out, dict) and isinstance(out2, dict) and canon_raw(out) != canon_raw(out2):
                    out = dict(out, __meta_diff__=str(out2.get("answer"))[:200])
            except Exception:
                pass
        conn.send(("ok", out))
    except Exception:
        conn.send(("err", traceback.format_exc(limit=3)[-1500:]))
    finally:
        conn.close()


_CTX = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()


def canon_raw(r):
    """So hai kết quả solve() bỏ qua khác biệt trình bày (thứ tự dòng bảng, số float rất gần)."""
    a = r.get("answer")
    if isinstance(a, dict) and "rows" in a:
        a = sorted(map(str, a["rows"]))
    elif isinstance(a, float):
        a = round(a, 6)
    return (r.get("type"), json.dumps(a, ensure_ascii=False, sort_keys=True, default=str))


def run_sandbox(code, ns, timeout=20):
    err = ast_guard(code)
    if err:
        return "err", err
    parent, child = _CTX.Pipe(duplex=False)
    p = _CTX.Process(target=_child, args=(child, code, ns))
    p.start()
    child.close()
    if parent.poll(timeout):
        try:
            status, val = parent.recv()
        except EOFError:
            status, val = "err", "tiến trình con kết thúc không trả kết quả"
    else:
        status, val = "err", f"quá thời gian {timeout}s"
    p.join(1)
    if p.is_alive():
        p.kill()
        p.join()
    return status, val


def make_namespace(biz, P, as_of, cal):
    def parse_money(series):
        th = decide_thousands(series.dropna().astype(str).head(3000))
        return series.map(lambda v: parse_money_one(v, th))

    def parse_date(series, year_hint=None):
        df_ = decide_dayfirst(series.dropna().astype(str).head(3000))
        return pd.to_datetime(series.map(lambda v: parse_date_one(v, year_hint, df_, True)))

    biz.activate() if hasattr(biz, "activate") else None
    # không copy: code chạy trong tiến trình con (fork) nên sửa V/T trong solve() không ảnh hưởng tiến trình cha
    V = dict(getattr(biz, "V", None) or {})
    de = getattr(biz, "data_end", None)
    ns = {"T": {k: v for k, v in biz.T.items()}, "V": V, "DATA_END": de, "P": P, "AS_OF": pd.Timestamp(as_of), "CAL": cal, "pd": pd, "np": np,
          "re": re, "math": math, "key": key, "rhu": rhu, "pct": pct, "parse_money": parse_money, "parse_date": parse_date,
          "__builtins__": __builtins__}
    if ledger is not None and V:
        for name, f in ledger.HELPERS.items():
            if name in ("ton_tai_ngay", "ton_don_vi_lon", "ton_tat_ca", "so_mat_hang_het_hang", "gia_tri_ton_kho"):
                ns[name] = functools.partial(f, V, data_end=de)
            else:
                ns[name] = functools.partial(f, V)
    return ns


# =============================================================================
# 8. Kiểm + chuẩn hoá đáp án theo rulebook §9
# =============================================================================

VALID_TYPES = {"number", "percent", "date", "string", "list", "table", "boolean", "unanswerable", "abstain"}


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(round(f)) if abs(f - round(f)) < 1e-9 else rhu(f, 2)
    return None


def format_answer(res):
    """Trả (line_dict, lỗi). line_dict theo đúng format nộp."""
    if not isinstance(res, dict) or res.get("type") not in VALID_TYPES:
        return None, f"đầu ra không hợp lệ: {str(res)[:200]}"
    t = res["type"]
    if t == "abstain":
        return {"type": "abstain"}, None
    if t == "unanswerable":
        return {"type": "unanswerable", "reason": str(res.get("reason", ""))[:300]}, None
    a = res.get("answer")
    if t == "number":
        v = _num(a)
        return ({"type": t, "answer": v}, None) if v is not None else (None, f"number sai: {a!r}")
    if t == "percent":
        if isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, float, np.integer, np.floating)) or not math.isfinite(float(a)):
            return None, f"percent sai: {a!r}"
        return {"type": t, "answer": rhu(float(a), 2)}, None
    if t == "date":
        a = str(a)[:10] if a is not None else ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a) or re.fullmatch(r"\d{4}-\d{2}", str(res.get("answer"))[:7]) and len(str(res.get("answer")).strip()) == 7:
            return {"type": t, "answer": a if len(str(res.get('answer')).strip()) != 7 else str(res.get('answer')).strip()}, None
        return None, f"date sai: {a!r}"
    if t == "boolean":
        return ({"type": t, "answer": bool(a)}, None) if isinstance(a, (bool, np.bool_)) else (None, f"boolean sai: {a!r}")
    if t == "string":
        if isinstance(a, list) and len(a) == 1:
            a = a[0]
        return ({"type": t, "answer": nfc(a).strip()}, None) if isinstance(a, str) and a.strip() else (None, f"string sai: {a!r}")
    if t == "list":
        if not isinstance(a, list):
            return None, f"list sai: {a!r}"
        out = []
        for x in a:
            if isinstance(x, str):
                out.append(nfc(x).strip())
            elif _num(x) is not None:
                out.append(_num(x))
            else:
                return None, f"phần tử list sai: {x!r}"
        return {"type": t, "answer": out}, None
    if t == "table":
        if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
            return None, "table sai cấu trúc"
        rows = []
        for r in a["rows"]:
            if not isinstance(r, list) or len(r) != len(a["columns"]):
                return None, "dòng table sai độ dài"
            rows.append([nfc(x).strip() if isinstance(x, str) else _num(x) for x in r])
        return {"type": t, "answer": {"columns": [str(c) for c in a["columns"]], "rows": rows}}, None
    return None, "type lạ"


def clean_final(line):
    """Đáp án ghi vào submission: chỉ giữ type / answer / reason, qua lại format_answer (bỏ khoá nội bộ như _tie)."""
    if not isinstance(line, dict) or line.get("type") not in VALID_TYPES:
        return {"type": "abstain"}
    if line["type"] == "abstain":
        return {"type": "abstain"}
    out, err = format_answer({k: v for k, v in line.items() if k in ("type", "answer", "reason")})
    return out if out is not None and not err else {"type": "abstain"}


def _vocab(biz):
    """Từ vựng tên của shop (mọi ô chữ trong T + V): chuẩn hoá BTC -> có; bỏ dấu -> các cách viết gốc. Dựng một lần mỗi shop."""
    vb = getattr(biz, "_vocab", None)
    if vb is not None:
        return vb
    exact, by_key = set(), defaultdict(set)
    frames = list(getattr(biz, "T", {}).values()) + list((getattr(biz, "V", None) or {}).values())
    for df in frames:
        if not isinstance(df, pd.DataFrame):
            continue
        for c in df.columns:
            col = df[c]
            if not (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)):   # pandas 3: chuỗi có dtype 'str'
                continue
            vals = col.dropna().astype(str).unique()
            if len(vals) > 200000:
                continue
            for v in vals:
                v = nfc(v).strip()
                if not v or len(v) > 120:
                    continue
                n = _snorm(v)
                exact.add(n)
                by_key[key(v)].add(v)
    biz._vocab = (exact, by_key)
    return biz._vocab


def snap_name(v, biz):
    """Tên trong đáp án phải TRÙNG một cách viết có trong file (BTC nhận mọi cách viết trong file + mã; ngoài ra phải đúng dấu).
    Đã trùng -> giữ. Không trùng mà bỏ dấu khớp DUY NHẤT một cách viết -> thay bằng cách viết trong file. Còn lại giữ nguyên."""
    if not isinstance(v, str) or biz is None:
        return v, None
    exact, by_key = _vocab(biz)
    if _snorm(v) in exact:
        return v, None
    cands = by_key.get(key(v)) or set()
    forms = {_snorm(c): c for c in sorted(cands)}
    if len(forms) == 1:
        nv = next(iter(forms.values()))
        return nv, f"{v} -> {nv}"
    return v, ("không có trong dữ liệu" if not cands else f"{len(forms)} cách viết khác dấu")


def snap_final(line, biz):
    """Áp snap_name cho string / list tên / cột đầu của table. Trả (line, ghi_chú)."""
    if not isinstance(line, dict) or biz is None:
        return line, None
    t, a, notes = line.get("type"), line.get("answer"), []
    try:
        if t == "string":
            a2, n = snap_name(a, biz)
            notes.append(n)
            line = dict(line, answer=a2)
        elif t == "list" and isinstance(a, list):
            out = []
            for x in a:
                x2, n = snap_name(x, biz) if isinstance(x, str) else (x, None)
                out.append(x2)
                notes.append(n)
            line = dict(line, answer=out)
        elif t == "table" and isinstance(a, dict) and isinstance(a.get("rows"), list):
            rows = []
            for r in a["rows"]:
                if isinstance(r, list) and r and isinstance(r[0], str):
                    x2, n = snap_name(r[0], biz)
                    notes.append(n)
                    r = [x2] + r[1:]
                rows.append(r)
            line = dict(line, answer=dict(a, rows=rows))
    except Exception as e:  # snap lỗi không được làm hỏng đáp án
        notes.append(f"lỗi snap: {e}")
    notes = [n for n in notes if n]
    return line, ("; ".join(notes)[:300] if notes else None)


def canon(line):
    """Dạng chuẩn để so 2 đáp án (đồng thuận A/B)."""
    if line is None:
        return None
    t = line["type"]
    a = line.get("answer")
    if t in ("unanswerable", "abstain"):
        return (t,)
    if t == "string":
        return (t, key(a))
    if t == "list":
        return (t, tuple(key(x) if isinstance(x, str) else x for x in a))
    if t == "table":
        return (t, tuple(sorted(tuple(key(x) if isinstance(x, str) else x for x in r) for r in a["rows"])))
    return (t, a)


# =============================================================================
# 9. Runner
# =============================================================================


def load_calendar(split_dir):
    p = os.path.join(split_dir, "calendar.csv")
    if not os.path.exists(p):
        return None
    c = pd.read_csv(p)
    c["date"] = pd.to_datetime(c["date"])
    c["lunar_leap"] = c["lunar_leap"].astype(str).str.lower().isin(["true", "1"])
    return c


def alias_hints(question, biz):
    """Từ đồng nghĩa vùng miền (lexicon/gioitre_vungmien_lamtay.tsv, nhóm dong_nghia_*): câu nói 'lạc' mà dữ liệu shop
    chỉ có 'đậu phộng' -> nhắc model. Không sửa câu, chỉ thêm gợi ý. So khớp GIỮ DẤU (bỏ dấu thì 'lợn' trùng 'lon')."""
    if not ALIAS_GROUPS:
        return []

    def low(t):
        return " " + re.sub(r"[^\w]+", " ", nfc(str(t)).lower()).strip() + " "
    blob = getattr(biz, "_terms_blob", None)
    if blob is None:
        blob = " | ".join(low(t) for t in biz.terms)
        biz._terms_blob = blob
    qk = low(question)
    out = []
    for grp in ALIAS_GROUPS:
        in_data = [w for w in grp if low(w) in blob]
        for w in grp:
            if low(w) in qk and low(w) not in blob and in_data:
                h = f"Từ '{w}' trong câu có thể là '{in_data[0]}' trong dữ liệu shop (cách gọi theo vùng miền)."
                if h not in out:
                    out.append(h)
    return out[:3]


def _prep(qrow, biz, cal, normalizer, nr=None, nr_log=None):
    q, as_of = qrow["question"], qrow["as_of_date"]
    log = {"id": qrow["id"], "business_id": biz.id, "as_of": as_of}
    q_norm = q
    if normalizer is not None:
        try:
            nr = nr if nr is not None else normalizer.normalize(q)
            q_norm = nr["text"]
            log["normalize_changes"] = nr["changes"]
            if nr.get("unsure"):
                log["normalize_unsure"] = nr["unsure"]
            if nr_log:
                log["normalize_llm"] = nr_log
        except Exception as e:  # lỗi chuẩn hoá không được làm hỏng câu
            log["normalize_error"] = str(e)
    P = resolve_periods(q_norm + " " + q, as_of, cal)
    W = lesson_warnings(q_norm + " " + q) + alias_hints(q_norm + " " + q, biz)
    log.update(question=q, normalized=q_norm, periods={k: [str(x.date()), str(y.date())] for k, (x, y) in P.items()}, warnings=W)
    return {"q": q, "q_norm": q_norm, "as_of": as_of, "P": P, "W": W, "log": log, "t0": now()}


def _run_variant_batch(llm, rulebook, biz, items, cal, variant, args):
    """Chạy 1 đường (A hoặc B) cho cả lô câu: sinh code theo lô, chạy sandbox, sửa lỗi theo lô."""
    msgs = [build_messages(rulebook, biz, it["q"], it["q_norm"], it["as_of"], it["P"], it["W"], variant=variant,
                           prev=it.get("_prev_text")) for it in items]
    res = [{"line": None, "error": None, "code": None} for _ in items]
    pending = list(range(len(items)))
    deep = variant.startswith("D")
    for attempt in range(args.max_repair + 1 + getattr(args, "sanity_rounds", 1)):
        if not pending:
            break
        if deep:   # vòng xem lại: cho model nghĩ (Qwen3/3.5), nhiều token hơn
            outs = llm.generate([msgs[i] for i in pending], think=getattr(args, "rethink_think", "on") == "on",
                                max_new=getattr(args, "rethink_tokens", 6144))
        else:
            outs = llm.generate([msgs[i] for i in pending])
        nxt = []
        for i, out in zip(pending, outs):
            code = extract_code(out)
            line, err = None, None
            if not code:
                err = "không tìm thấy code"
            else:
                ns = make_namespace(biz, items[i]["P"], items[i]["as_of"], cal)
                ns["__metamorphic__"] = getattr(args, "metamorphic", "on") == "on"
                rts = rule_tests_for(items[i], args)
                if rts:
                    ns["__rule_tests__"], ns["__inject_info__"], ns["__seed_date__"] = rts, inject_info(biz), items[i].get("_seed_date")
                st, val = run_sandbox(code, ns, timeout=args.exec_timeout * (1 + ns["__metamorphic__"] + len(rts)))
                meta_diff = None
                if st == "ok":
                    if isinstance(val, dict) and "__meta_diff__" in val:
                        meta_diff = val.pop("__meta_diff__")
                    if isinstance(val, dict) and "__rule_diff__" in val:
                        items[i].setdefault("_rule", {})[variant] = val.pop("__rule_diff__")
                    line, err = format_answer(val)
                else:
                    err = val
                if line is not None and meta_diff is not None:
                    items[i].setdefault("_meta", {})[variant] = meta_diff
            if line is not None and attempt < args.max_repair + getattr(args, "sanity_rounds", 1):
                why = sanity_issue(line, items[i], biz)
                if why and not items[i].get("_sanity_done_" + variant):
                    items[i]["_sanity_done_" + variant] = True
                    res[i] = {"line": line, "error": None, "code": code, "sanity": why}
                    msgs[i] = msgs[i] + [{"role": "assistant", "content": out},
                                         {"role": "user", "content": f"Kết quả chạy: {json.dumps(line, ensure_ascii=False)[:300]}\nKIỂM TRA THẤY ĐÁNG NGỜ: {why}\n"
                                          "Hãy đọc lại câu hỏi + rulebook, sửa nếu sai (nếu chắc kết quả đúng thì trả lại đúng code cũ). Trả lại toàn bộ khối ```python```."}]
                    nxt.append(i)
                    continue
            res[i] = {"line": line, "error": err, "code": code, "attempts": attempt + 1,
                      **({"sanity_prev": res[i].get("sanity")} if res[i].get("sanity") else {})}
            if line is None:
                hint = repair_hint(err or "", code or "", biz, items[i])
                msgs[i] = msgs[i] + [{"role": "assistant", "content": out},
                                     {"role": "user", "content": f"Code lỗi:\n{err}\n" + (f"CHẨN ĐOÁN: {hint}\n" if hint else "") +
                                      "Hãy sửa và trả lại toàn bộ khối ```python``` với solve() đúng format."}]
                res[i]["repair_hint"] = hint
                nxt.append(i)
        pending = nxt
    return res


FORMAT_EXAMPLE = {
    "number": '{"type": "number", "answer": 123456}', "percent": '{"type": "percent", "answer": 12.35}',
    "date": '{"type": "date", "answer": "2025-03-14"}  (tháng: "2025-03")', "boolean": '{"type": "boolean", "answer": true}',
    "string": '{"type": "string", "answer": "Tên"}', "list": '{"type": "list", "answer": ["A", "B"]}',
    "table": '{"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 1000]]}}',
}


def repair_hint(err, code, biz, item):
    """Chẩn đoán lỗi để vòng sửa có thêm thông tin (prompt gốc + code lỗi + thông báo lỗi + CHẨN ĐOÁN):
    cột/bảng không có -> liệt kê cột thật và tên gần nhất; lỗi ngày/kiểu; chia 0; quá giờ; sai format."""
    import difflib
    hints = []
    tables = {**{f"T['{k}']": v for k, v in biz.T.items()}, **{f"V['{k}']": v for k, v in (getattr(biz, "V", None) or {}).items()}}
    m = re.search(r"KeyError: ['\"]([^'\"]+)['\"]", err)
    if m:
        miss = m.group(1)
        used = [k for k in tables if k.split("'")[1] in code]
        if miss in biz.T or miss in (getattr(biz, "V", None) or {}):
            pass
        elif not used:
            near = difflib.get_close_matches(miss, list(biz.T) + list((getattr(biz, "V", None) or {})), n=3, cutoff=0.3)
            hints.append(f"không có bảng/cột '{miss}'. Bảng gần nhất: {near}")
        for k in used[:3]:
            cols = [str(c) for c in tables[k].columns if not str(c).startswith("__") or str(c) in ("__nhom_trang_thai", "__sl_co_ban", "__khach_le")]
            near = difflib.get_close_matches(miss, cols, n=3, cutoff=0.3) or difflib.get_close_matches(key(miss), [key(c) for c in cols], n=3, cutoff=0.3)
            hints.append(f"{k} không có cột '{miss}'. Cột gần nhất: {near}. Các cột: {cols[:40]}")
    if re.search(r"Can only use \.dt accessor|'str' object has no attribute|Invalid comparison|Cannot compare|not supported between instances of 'str' and 'Timestamp'|dtype=datetime", err):
        hints.append("so sánh / .dt trên cột chưa phải kiểu ngày: dùng pd.to_datetime(cột, errors='coerce') hoặc parse_date(cột); "
                     "V[...]['ngay'] đã là kiểu ngày. Hết ngày b = b + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)")
    if "ZeroDivisionError" in err or "division by zero" in err:
        hints.append("mẫu số = 0 (kỳ không có dữ liệu): theo §7 trả {'type': 'unanswerable', 'reason': '...'} thay vì chia")
    if "quá thời gian" in err:
        hints.append("chạy quá lâu: bỏ vòng for/iterrows/apply theo dòng, dùng groupby / lọc vector; lọc kỳ TRƯỚC rồi mới gộp")
    if re.search(r"đầu ra không hợp lệ|sai:|type lạ|table sai|dòng table", err):
        exp = item.get("expect_type")
        hints.append("solve() phải trả dict đúng format, ví dụ: " + (FORMAT_EXAMPLE.get(exp) or "; ".join(FORMAT_EXAMPLE.values())))
    if re.search(r"NameError: name '(\w+)'", err):
        nm = re.search(r"NameError: name '(\w+)'", err).group(1)
        hints.append(f"'{nm}' chưa định nghĩa. Có sẵn: T, V, P, AS_OF, DATA_END, CAL, pd, np, re, math, key, rhu, pct, parse_money, parse_date"
                     + (", các hàm ledger (doanh_thu_ky, tong_no, ton_tai_ngay, …)" if ledger is not None else ""))
    if "không tìm thấy code" in err:
        hints.append("trả lời bằng MỘT khối ```python``` chứa def solve(): ... return {...}")
    return " | ".join(hints)[:1500]


def answer_features(log):
    """Đặc trưng của một câu ĐÃ quyết định (đọc từ log) -> dict số, dùng cho mô hình xác suất đúng (bỏ trống theo PA)."""
    dec = str(log.get("decision", ""))
    paths = log.get("paths") or {}
    lines = [v.get("line") for v in paths.values() if isinstance(v, dict)]
    ok_lines = [l for l in lines if l]
    sp = log.get("spec") or {}
    fin = log.get("final") or {}
    f = {
        "bias": 1.0,
        "dec_compiler": float(dec.startswith("compiler")),
        "dec_compiler_agree": float("agree" in dec and dec.startswith("compiler")),
        "dec_agree": float(dec == "agree"),
        "dec_single": float(dec in ("single", "one_path")),
        "dec_choice": float(dec.startswith("choice")),
        "spec_luat": float(sp.get("_nguon") == "luat"),
        "spec_llm": float(sp.get("_nguon") == "llm"),
        "spec_tin_thap": float(sp.get("tin_cay") == "thap"),
        "n_paths_ok": float(len(ok_lines)),
        "paths_same": float(len(ok_lines) >= 2 and all(canon(ok_lines[0]) == canon(l) for l in ok_lines[1:])),
        "repaired": float(any(isinstance(v, dict) and v.get("attempts", 0) > 1 for v in paths.values())),
        "sanity_flag": float(any(isinstance(v, dict) and (v.get("sanity") or v.get("sanity_prev")) for v in paths.values())),
        "audit_misread": float(bool((log.get("audit") or {}).get("duong_code_hieu_theo"))),
        "t_number": float(fin.get("type") in ("number", "percent")),
        "t_list_table": float(fin.get("type") in ("list", "table")),
        "t_unans": float(fin.get("type") == "unanswerable"),
        "norm_changes": float(min(5, len(log.get("normalize_changes") or []))),
        # mới: tín hiệu từ Rule-Delta, oracle hoà, snap tên, đối soát chéo của shop
        "rule_delta": float(bool(log.get("rule_delta"))),
        "tie": float(bool((log.get("audit") or {}).get("hoa_o_bien"))),
        "snap_miss": float("không có trong dữ liệu" in str((log.get("audit") or {}).get("snap", ""))),
        "shop_ton_lech": float(((log.get("shop_quality") or {}).get("ton_khop") or 1.0) < 0.8),
    }
    return f


def prob_correct(model, log):
    """Xác suất đáp án đúng theo mô hình logistic đã fit trên log TRAIN (scripts/fit_abstain.py)."""
    import math as _m
    f = answer_features(log)
    z = sum(model["w"].get(k, 0.0) * v for k, v in f.items())
    return 1.0 / (1.0 + _m.exp(-z))


def rule_tests_for(item, args):
    """Chọn bài kiểm từ luật áp được cho câu: 'tuong_lai' cho mọi câu hỏi đến hôm nay; 'huy' cho câu không hỏi về huỷ / trạng thái."""
    if getattr(args, "rule_tests", "on") != "on":
        return []
    q = key(item["q_norm"] + " " + item["q"])
    out = []
    if not re.search(r"sap toi|tuong lai|du kien|ke hoach|thang sau|tuan sau|ngay mai|het han", q):
        out.append("tuong_lai")
    if not re.search(r"\bhuy\b|cancel|trang thai|tat ca (cac )?don|don dat|moi don|ti le|ty le|bom hang|that bai", q):
        # ngày của đơn huỷ giả phải NẰM TRONG kỳ câu hỏi: lấy từ đặc tả nếu có, không thì kỳ do câu nêu rõ (không lấy các kỳ
        # tương đối mặc định như 'tuần này' mà câu không nhắc)
        sd = None
        sp = item.get("spec") or {}
        if sp.get("tu") is not None and sp.get("den") is not None and sp["tu"] <= sp["den"]:
            sd = sp["tu"] + (sp["den"] - sp["tu"]) / 2
        elif sp.get("X") is not None:
            sd = sp["X"] - pd.Timedelta(days=1)
        if sd is None:
            rel = {"hom_nay": "hom nay", "hom_qua": "hom qua", "tuan_nay": "tuan nay", "tuan_truoc": "tuan (truoc|roi|trc)",
                   "thang_nay": "thang nay", "thang_truoc": "thang (truoc|roi|trc)", "thang_truoc_cung_ky": "cung ky",
                   "tu_dau_nam": "dau nam", "nam_truoc": "nam (truoc|ngoai)", "quy_nay": "quy nay"}
            for k, (a, b) in (item.get("P") or {}).items():
                if a is None or b is None or a > b:
                    continue
                if k in rel and not re.search(rel[k], q):
                    continue
                sd = a + (b - a) / 2
                break
        if sd is None:
            sd = pd.Timestamp(item["as_of"]).normalize()
        item["_seed_date"] = pd.Timestamp(sd).normalize()
        out.append("huy")
    return out


def sanity_issue(line, item, biz):
    """Agent tự soát KẾT QUẢ (không chỉ lỗi chạy): sai kiểu so với đặc tả, số đếm âm/lẻ, % vô lý, ngày ngoài dữ liệu, rỗng.
    Trả lý do (chuỗi) hoặc None. Chỉ cho model MỘT vòng xem lại — không ép theo đường máy tính."""
    t, a = line.get("type"), line.get("answer")
    if item.get("_rule"):
        kind, other = list(item["_rule"].values())[0].split("|", 1)
        return (f"VI PHẠM LUẬT — {RULE_TESTS.get(kind, kind)}. Kiểm bằng cách chèn một đơn {'có ngày SAU hôm nay' if kind == 'tuong_lai' else 'đã HUỶ trong kỳ hỏi'}"
                f" (mã ZZTEST…): đáp án đổi thành {other[:80]}. Code chưa lọc đúng; hãy lọc theo luật (ngày ≤ hôm nay / bỏ đơn huỷ, dùng cột nhom_tt / tinh_doanh_thu).")
    if item.get("_meta"):
        return ("kết quả ĐỔI khi đảo thứ tự dòng các bảng (lần đảo ra: " + str(list(item["_meta"].values())[0])[:80] +
                ") — code đang dựa vào thứ tự dòng trong file (iloc[0], head, first, drop_duplicates không sắp xếp…); "
                "hãy sắp xếp rõ ràng theo luật (hoà xếp theo mã) hoặc tính theo nhóm")
    exp = item.get("expect_type")
    if exp and t not in (exp, "unanswerable") and not (exp == "string" and t == "list") and not (exp == "number" and t == "percent"):
        return f"kiểu đáp án '{t}' khác kiểu câu hỏi cần ('{exp}')"
    q = key(item["q_norm"] + " " + item["q"])
    if t == "number" and isinstance(a, (int, float)):
        if a < 0 and not re.search(r"chenh|lech|thay doi|tang|giam|lai|lo\b|tru|con lai|so voi|hon|kem", q):
            return "số âm cho một đại lượng không thể âm (đếm / tiền bán / tồn)"
        if re.search(r"(bao nhieu|may|bn|so) (don|khach|lan|ngay|mat hang|lo)\b", q) and abs(a - round(a)) > 1e-9:
            return "số đếm (đơn, khách, ngày…) phải là số nguyên"
    if t == "percent" and isinstance(a, (int, float)) and abs(a) > 1000:
        return "phần trăm vượt ±1000% — kiểm lại mẫu số / đã nhân 100 hai lần?"
    if t == "percent" and re.search(r"ti le (don )?huy|chiem|ti trong", q) and isinstance(a, (int, float)) and not 0 <= a <= 100:
        return "tỉ lệ / tỉ trọng phải nằm trong 0–100"
    if t in ("list", "string") and (a in ([], "") or a is None):
        return "kết quả rỗng"
    if t == "date" and isinstance(a, str) and CMP is not None:
        try:
            s0, e0 = CMP._data_range(biz)
            d0 = pd.Timestamp(a + ("-01" if len(a) == 7 else ""))
            if (s0 is not None and d0 < s0 - pd.Timedelta(days=31)) or (e0 is not None and d0 > e0 + pd.Timedelta(days=31) and not re.search(r"han", q)):
                return "ngày kết quả nằm ngoài khoảng có dữ liệu"
        except Exception:
            pass
    return None


def ask_choice(llm, item, rulebook_rules, opt_a, opt_b):
    """Hỏi model ĐÚNG MỘT câu trắc nghiệm ở chỗ phân vân: cách tính nào đúng theo rulebook. Trả 'A', 'B' hoặc None."""
    msg = [{"role": "user", "content":
            "Hai cách tính cho cùng câu hỏi ra hai kết quả khác nhau. Theo rulebook, cách nào ĐÚNG?\n"
            f"Câu hỏi: {item['q']}\nHôm nay: {item['as_of']}\nLuật liên quan:\n{rulebook_rules}\n\n"
            f"A) {opt_a}\nB) {opt_b}\nC) Không chắc\n\nChỉ trả một chữ cái A, B hoặc C."}]
    try:
        out = llm.generate_short([msg], max_tokens=3)[0]
    except Exception:
        return None
    m = re.search(r"\b([ABC])\b", (out or "").strip().upper())
    return m.group(1) if m and m.group(1) in "AB" else None


def spec_text(spec):
    """Mô tả đặc tả bằng lời (để hỏi model chọn cách hiểu)."""
    if not spec or CMP is None:
        return ""
    parts = [CMP.INTENTS.get(spec.get("intent"), spec.get("intent", ""))]
    for k in ("tu", "den", "X"):
        if spec.get(k) is not None:
            parts.append(f"{k}={pd.Timestamp(spec[k]).date()}")
    for k in ("mat_hang", "don_vi", "n", "N", "khach", "loc", "thang", "ti_trong", "ti_le", "dem"):
        if spec.get(k) not in (None, False, {}, []):
            parts.append(f"{k}={spec[k]}")
    return "; ".join(map(str, parts))


def decide(results, n_variants, args):
    """Quyết định cuối từ kết quả các đường."""
    decision, final = "abstain", {"type": "abstain"}
    if n_variants == 1:
        if results["A"]["line"] is not None:
            final, decision = results["A"]["line"], "single"
    else:
        a, b = results["A"]["line"], results["B"]["line"]
        if a is not None and b is not None:
            if canon(a) == canon(b):
                final, decision = a, "agree"
            else:
                decision = "disagree"
                if args.policy == "lenient":
                    final = a
        elif a is not None or b is not None:
            decision = "one_path"
            if args.on_single == "answer":
                final = a or b
    # unanswerable chỉ nhận khi đủ chắc (2 đường cùng nói không trả lời được)
    if final.get("type") == "unanswerable" and n_variants == 2 and decision != "agree":
        final, decision = {"type": "abstain"}, decision + "+unans_not_confirmed"
    return decision, final


PHO_PRE = {}   # id câu -> (kết quả sửa chữ, log) do PhoGPT chấm ở pha trước khi bật vLLM


def normalize_batch(llm, normalizer, qrows, args):
    """Sửa chữ cả lô; chỗ chưa chắc: Qwen trắc nghiệm ('main'), PhoGPT chấm câu ('pho'), hoặc chỉ đổi khi CẢ HAI cùng chọn
    ('both'). Model chỉ CHỌN trong phương án do luật sinh — không viết lại câu."""
    mode = getattr(args, "norm_llm", "main")
    if mode in ("pho", "both") and all(q["id"] in PHO_PRE for q in qrows):
        if mode == "pho" or resolve_unsure is None or args.llm == "dummy":
            return [PHO_PRE[q["id"]][0] for q in qrows], [PHO_PRE[q["id"]][1] for q in qrows]
        try:
            _, qlog = resolve_unsure(normalizer, [q["question"] for q in qrows], lambda msgs: llm.generate_short(msgs))
        except Exception as e:
            print(f"  [cảnh báo] Qwen trắc nghiệm lỗi: {e} -> dùng PhoGPT", flush=True)
            return [PHO_PRE[q["id"]][0] for q in qrows], [PHO_PRE[q["id"]][1] for q in qrows]
        import vn_lm_scorer as VS
        plog = []
        for i, q in enumerate(qrows):
            plog += [dict(x, q=i) for x in PHO_PRE[q["id"]][1]]
        forced = VS.combine(plog, qlog)
        res = [normalizer.normalize(q["question"], forced=forced[i]) if i in forced else normalizer.normalize(q["question"]) for i, q in enumerate(qrows)]
        per_q = [[] for _ in qrows]
        for x in plog + [dict(x, nguon="qwen") for x in qlog]:
            per_q[x["q"]].append({k: v for k, v in x.items() if k != "q"})
        return res, per_q
    if mode in ("pho", "both"):
        mode = "main"          # thiếu kết quả PhoGPT (không nạp được model) -> quay về Qwen trắc nghiệm
    if normalizer is None or not hasattr(normalizer, "normalize") or resolve_unsure is None or mode == "off" \
            or not isinstance(normalizer, ContextNormalizer):
        return [None] * len(qrows), [None] * len(qrows)
    try:
        res, log = resolve_unsure(normalizer, [q["question"] for q in qrows], lambda msgs: llm.generate_short(msgs))
    except Exception as e:  # LLM lỗi -> dùng kết quả sửa chữ không có LLM
        print(f"  [cảnh báo] sửa chữ bằng LLM lỗi: {e}", flush=True)
        return [None] * len(qrows), [None] * len(qrows)
    per_q = [[] for _ in qrows]
    for x in log:
        per_q[x["q"]].append({k: v for k, v in x.items() if k != "q"})
    return res, per_q


COMPILER_VOCAB = None          # từ vựng câu hỏi train (lexicon/compiler_vocab.json) cho chặn chữ lạ


def load_compiler_vocab(path=None):
    global COMPILER_VOCAB
    path = path or os.path.join(HERE, "lexicon", "compiler_vocab.json")
    try:
        with open(path, encoding="utf-8") as fh:
            COMPILER_VOCAB = set(json.load(fh)["tokens"])
    except Exception:
        COMPILER_VOCAB = None
    return COMPILER_VOCAB


def shop_profile_text(biz):
    """README + tên hàng của shop (để nhận NGÀNH: cà phê, điện thoại, sửa xe, mẹ và bé…)."""
    if getattr(biz, "_profile_txt", None) is None:
        names = []
        for tn in ("hang", "dong"):
            df = biz.V.get(tn) if isinstance(getattr(biz, "V", None), dict) else None
            if df is not None and "ten_hang" in df.columns:
                names += [str(x) for x in df["ten_hang"].dropna().unique()[:3000]]
        biz._profile_txt = key((getattr(biz, "readme", "") or "") + " " + " ".join(names))
    return biz._profile_txt


def _demote(it, why, field):
    sp = it.get("spec")
    if sp and sp.get("tin_cay", "cao") == "cao":
        it["spec"] = dict(sp, tin_cay="thap", **{field: why})
        it["log"][field] = why


def _rule_delta_guard(it, biz=None):
    """Chặn máy tính tự quyết khi nó có thể đang tính theo HIỂU BIẾT CŨ:
    1) Rule-Delta: luật nền của intent đã đổi/bỏ, hoặc câu dùng thuật ngữ của luật MỚI -> bỏ hẳn kết quả máy tính.
    2) Ngành mới: shop thuộc ngành mà rulebook đang chạy có luật mới/đổi (vd cửa hàng điện thoại: thu máy cũ, trả góp)
       -> hạ tin cậy (chỉ dùng khi đường code đồng ý).
    3) Chữ lạ: câu còn chữ ngoài từ vựng câu hỏi train và ngoài tên thực thể đã gắn (tên món/nhân viên/nhóm không tìm
       thấy…) -> có thể bộ lọc bị bỏ sót -> hạ tin cậy.
    4) 'Không trả lời được' do máy tính KHÔNG THẤY dữ liệu (không phải câu hỏi thông tin shop không ghi): định dạng shop
       mới có thể chứa dữ liệu ở chỗ máy tính không biết -> hạ tin cậy."""
    if not it.get("spec"):
        return
    sp = it["spec"]
    if RULE_DELTA is not None:
        why = RULE_DELTA.block_reason(sp, key(it["q_norm"] + " " + it["q"]))
        if why:
            it["log"]["rule_delta"] = why
            it["log"]["compiler_blocked"] = it.get("C")
            it["C"] = None
            it["spec"] = dict(sp, tin_cay="thap", _blocked=why)
            return
        biz = biz or it.get("_biz")
        if biz is not None:
            ind = RULE_DELTA.industry_hits(shop_profile_text(biz))
            if ind:
                _demote(it, "ngành có luật mới/đổi: " + ", ".join(ind), "_nganh_moi")
    biz = biz or it.get("_biz")
    if COMPILER_VOCAB and sp.get("intent") != "unanswerable" and biz is not None:
        rest = CMP.residual_tokens(sp, key(it["q_norm"] or it["q"]), COMPILER_VOCAB, CMP.shop_entity_tokens(biz))
        if rest:
            _demote(it, " ".join(rest[:6]), "_chu_la")
    C = it.get("C")
    if C is not None and C.get("type") == "unanswerable" and sp.get("intent") != "unanswerable" \
            and "ngoai khoang" not in key(C.get("reason", "")):        # kỳ ngoài dữ liệu (§7) vẫn chắc chắn
        _demote(it, C.get("reason", ""), "_khong_thay_du_lieu")


def _compiler_paths(llm, biz, items, args):
    """Bước 3 + đường 1: đặc tả (luật chữ; không nhận ra thì LLM viết đặc tả JSON) -> máy tính theo công thức."""
    if CMP is None or getattr(args, "compiler", "on") == "off":
        return
    need_llm = []
    for i, it in enumerate(items):
        it["spec"], it["C"] = None, None
        try:
            sp = CMP.spec_rules(it["q"], it["as_of"], biz, it["q_norm"])
            if sp:
                sp["_nguon"] = "luat"
                it["spec"] = sp
                it["C"] = CMP.compile_spec(sp, biz, it["as_of"])
        except Exception as e:
            it["log"]["compiler_error"] = f"{type(e).__name__}: {e}"
        _rule_delta_guard(it)
        if it["spec"] is None:
            need_llm.append(i)
        if it["spec"] and it["spec"].get("intent") in CMP.ANSWER_TYPE:
            it["expect_type"] = CMP.ANSWER_TYPE[it["spec"]["intent"]]
    if need_llm and getattr(args, "llm_spec", "on") == "on" and args.llm != "dummy":
        ret = None
        try:
            import retriever as RT
            ret = RT.bank_retriever(getattr(args, "spec_bank", "") or os.path.join(HERE, "lexicon", "spec_bank.jsonl"), getattr(args, "embed_model", None))
        except Exception as e:
            print(f"  [cảnh báo] không dựng được bộ truy hồi ví dụ: {e}", flush=True)
        msgs = []
        for i in need_llm:
            ex = RT.few_shot_text(ret, items[i]["q_norm"] or items[i]["q"], k=4, exclude_biz=biz.id) if ret is not None else ""
            items[i]["log"]["spec_examples"] = len(ex.split("\n- Câu")) if ex else 0
            msgs.append(CMP.spec_messages(items[i]["q"], items[i]["q_norm"], str(pd.Timestamp(items[i]["as_of"]).date()), periods_text(items[i]["P"]), ex))
        try:
            outs = llm.generate(msgs)
        except Exception as e:
            outs = [None] * len(need_llm)
            print(f"  [cảnh báo] LLM viết đặc tả lỗi: {e}", flush=True)
        for i, out in zip(need_llm, outs):
            it = items[i]
            try:
                sp = CMP.parse_llm_spec(out, biz, it["as_of"])
                if sp:
                    it["spec"] = sp
                    it["C"] = CMP.compile_spec(sp, biz, it["as_of"])
                    it["expect_type"] = CMP.ANSWER_TYPE.get(sp["intent"])
                    _rule_delta_guard(it)
                it["log"]["llm_spec_raw"] = (out or "")[:600]
            except Exception as e:
                it["log"]["compiler_error"] = f"{type(e).__name__}: {e}"


def decide_all(llm, it, results, n_variants, args):
    """Quyết định cuối = máy tính theo công thức (đường 1) + model tự viết code (đường A, B) + kiểm toán.
    - Đặc tả bằng LUẬT, độ tin cao, máy tính ra kết quả: lấy máy tính (đã đối chiếu train). Đường A lệch -> ghi kiểm toán.
    - Đặc tả độ tin thấp / do LLM viết: chỉ lấy khi khớp đường A hoặc B; lệch -> hỏi model chọn A/B đúng một câu; không chắc -> bỏ trống.
    - Không có máy tính: quyết định cũ theo A/B."""
    C, sp = it.get("C"), it.get("spec")
    base_dec, base_final = decide(results, n_variants, args)
    audit = {}
    if CMP is None or C is None or sp is None:
        return base_dec, base_final, audit
    if C.get("_tie"):
        # BTC đã loại mọi câu xếp hạng có HOÀ ở biên k / hạng 1 -> máy tính ra hoà nghĩa là cách đọc/hiểu đang lệch đáp án
        audit["hoa_o_bien"] = True
        return "compiler_tie_abstain", {"type": "abstain"}, audit
    lines = [results[v]["line"] for v in results if results[v]["line"] is not None]
    agree = [l for l in lines if CMP.same_answer(C, l)]
    if lines and not agree:
        # kiểm toán: con số của đường A có trùng một CÁCH HIỂU SAI hay gặp không?
        for name, alt in CMP.alternative_readings(sp):
            try:
                ca = CMP.compile_spec(alt, it["_biz"], it["as_of"])
            except Exception:
                ca = None
            if ca is not None and any(CMP.same_answer(ca, l) for l in lines):
                audit["duong_code_hieu_theo"] = name
                break
    high = sp.get("_nguon") == "luat" and sp.get("tin_cay", "cao") == "cao"
    if C.get("type") == "unanswerable":
        if high or (lines and all(l.get("type") == "unanswerable" for l in lines)):
            return ("compiler_unans" + ("+agree" if agree else "")), C, audit
        return base_dec, base_final, audit
    if high:
        return ("compiler+agree" if agree else ("compiler_wins" if lines else "compiler_only")), C, audit
    if agree:
        return "compiler_lowconf+agree", C, audit
    a_line = results.get("A", {}).get("line")
    if a_line is not None and getattr(args, "ask_choice", "on") == "on" and args.llm != "dummy":
        rules = ""
        if RULE_INDEX is not None:
            rules = RULE_INDEX.block(it["q"] + " " + it["q_norm"], it["_biz"].readme or "", k=4, width=300)
        ch = ask_choice(llm, it, rules, f"{spec_text(sp)} → {json.dumps(C.get('answer'), ensure_ascii=False)[:200]}",
                        f"code model tự viết → {json.dumps(a_line.get('answer'), ensure_ascii=False)[:200]}")
        audit["hoi_chon"] = ch
        if ch == "A":
            return "choice_compiler", C, audit
        if ch == "B" and base_dec in ("agree", "single", "one_path"):
            return "choice_code", a_line, audit
        return "abstain_choice_unsure", {"type": "abstain"}, audit
    return base_dec, base_final, audit


def provisional_answer(qrow, biz, cal, normalizer):
    """Đáp án TẠM không cần model: đặc tả bằng LUẬT, tin cậy cao, máy tính ra kết quả, không bị Rule-Delta chặn, không hoà
    ở biên. Đây đúng là nhánh decide_all lấy máy tính (compiler_*) nên đáp án tạm = đáp án cuối của các câu đó."""
    if CMP is None:
        return None
    it = _prep(qrow, biz, cal, normalizer)
    sp = CMP.spec_rules(it["q"], it["as_of"], biz, it["q_norm"])
    if not sp or sp.get("tin_cay", "cao") != "cao":
        return None
    sp["_nguon"] = "luat"
    it["spec"], it["C"] = sp, CMP.compile_spec(sp, biz, it["as_of"])
    _rule_delta_guard(it, biz)
    C = it.get("C")
    if C is None or C.get("_tie") or (it.get("spec") or {}).get("tin_cay") == "thap":
        return None
    out = clean_final(C)
    return None if out.get("type") == "abstain" else out


def solve_batch(llm, rulebook, biz, qrows, cal, normalizer, args):
    nrs, nlogs = normalize_batch(llm, normalizer, qrows, args)
    items = [_prep(q, biz, cal, normalizer, nr, nl) for q, nr, nl in zip(qrows, nrs, nlogs)]
    sq = shop_quality(biz)
    for it in items:
        it["_biz"] = biz
        it["log"]["shop_quality"] = sq
    _compiler_paths(llm, biz, items, args)
    variants = ["A", "B"] if args.n_paths >= 2 else ["A"]
    per_var = {}
    for v in variants:
        if v == "B" and getattr(args, "b_when_sure", "off") == "off":
            # đường B chỉ cần khi chưa có kết quả chắc: máy tính theo luật (tin cậy cao) đã ra đáp án thì bỏ B (tiết kiệm ~1 lượt gọi/câu)
            need = [i for i, it in enumerate(items) if not (it.get("C") is not None and (it.get("spec") or {}).get("_nguon") == "luat"
                                                            and (it.get("spec") or {}).get("tin_cay", "cao") == "cao")]
            sub = _run_variant_batch(llm, rulebook, biz, [items[i] for i in need], cal, v, args) if need else []
            full = [{"line": None, "error": "bỏ qua: máy tính theo luật đã chắc", "code": None, "skipped": True} for _ in items]
            for i, r in zip(need, sub):
                full[i] = r
            per_var[v] = full
        else:
            per_var[v] = _run_variant_batch(llm, rulebook, biz, items, cal, v, args)
    first = []
    for i, it in enumerate(items):
        results = {v: per_var[v][i] for v in variants}
        first.append(list(decide_all(llm, it, results, len(variants), args)) + [results])
    rethink_batch(llm, rulebook, biz, items, first, cal, args)
    out = []
    for i, it in enumerate(items):
        decision, final, audit, results = first[i]
        sp = it.get("spec")
        if final.get("type") == "abstain":
            # CỨU câu bỏ trống: luôn ghi ứng viên tốt nhất (để đo trên train); chỉ TRẢ LỜI bằng nó khi loại ứng viên này
            # đã đo trên TRAIN đúng > 50% chắc chắn (cận dưới Wilson > 0,5) -> mỗi câu cứu được kỳ vọng +điểm (PA = đúng − sai)
            cand, lab = rescue_candidate(it, results, decision)
            if cand is not None:
                audit = dict(audit or {}, ung_vien=cand, ung_vien_nhan=lab)
                tab = getattr(args, "_calibration", None) or {}
                kk = f"cuu:{lab}|{(sp or {}).get('intent', '-')}"
                row = tab.get(kk) if (tab.get(kk) or {}).get("n", 0) >= getattr(args, "_cal_min_n", 15) else tab.get(f"cuu:{lab}")
                if row and row.get("tra_loi"):
                    audit["cuu"] = f"{kk if row is tab.get(kk) else 'cuu:' + lab}: p={row.get('p')} (n={row.get('n')}) trên train"
                    decision, final = f"{decision}->cuu:{lab}", cand
        pmodel = getattr(args, "_abstain_model", None)
        if pmodel and final.get("type") != "abstain":
            # bỏ trống theo XÁC SUẤT ĐÚNG: PA = đúng − sai, nên chỉ trả lời khi p > ngưỡng (mặc định 0,5)
            tmp = dict(it["log"], paths=results, decision=decision, final=final, audit=audit,
                       spec={k: v for k, v in (sp or {}).items() if k in ("_nguon", "tin_cay", "intent")})
            pc = prob_correct(pmodel, tmp)
            audit = dict(audit or {}, p_dung=round(pc, 3))
            if pc <= pmodel.get("nguong", 0.5):
                decision, final = decision + "->p_abstain", {"type": "abstain"}
        cal_tab = getattr(args, "_calibration", None)
        if cal_tab and final.get("type") != "abstain" and not pmodel:
            # bỏ trống theo PA: nhãn quyết định (hoặc nhãn|intent) đo trên TRAIN có tỉ lệ đúng ≤ 0,5 -> trả lời là lỗ điểm
            kk = f"{decision}|{(sp or {}).get('intent', '-')}"
            row = cal_tab.get(kk) or cal_tab.get(decision)
            if row and row.get("bo_trong"):
                audit = dict(audit or {}, hieu_chinh=f"{kk if kk in cal_tab else decision}: p={row['p']} trên train -> bỏ trống")
                decision, final = decision + "->calib_abstain", {"type": "abstain"}
        if final.get("type") in ("string", "list", "table"):
            final, snap_note = snap_final(final, biz)
            if snap_note:
                audit = dict(audit or {}, snap=snap_note)
        final = clean_final(final)
        it["log"].update(paths=results, decision=decision, final=final, elapsed_s=round(now() - it["t0"], 2),
                         spec={k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in (sp or {}).items() if k != "q"} if sp else None,
                         compiler=it.get("C"), audit=audit or None)
        out.append((final, it["log"]))
    return out


def rethink_allowed(args):
    """Còn thời gian cho vòng xem lại? Ước tổng thời gian nếu giữ tốc độ hiện tại; quá 90% ngân sách -> thôi xem lại
    (ưu tiên mọi câu đều được giải lượt đầu)."""
    if getattr(args, "rethink", "on") != "on" or args.llm in ("none", "dummy") and not getattr(args, "_rethink_test", False):
        return False
    tb = getattr(args, "time_budget", 0)
    if not tb:
        return True
    done, total, t0 = getattr(args, "_progress", (0, 0, PROC_T0))
    el = now() - PROC_T0
    if el > tb * getattr(args, "rethink_until", 0.85):
        return False
    if done and total:
        projected = (now() - t0) / (done / total) + (t0 - PROC_T0)
        return projected <= 0.9 * tb
    return True


def _vote(cands):
    """cands: [(tên, line)] -> [(số phiếu, [tên], line)] nhóm theo đáp án chuẩn hoá, nhiều phiếu trước."""
    groups = {}
    for name, line in cands:
        if isinstance(line, dict) and line.get("type") not in (None, "abstain"):
            k = canon(line)
            g = groups.setdefault(k, [0, [], line])
            g[0] += 1
            g[1].append(name)
    return sorted(groups.values(), key=lambda g: (-g[0], -sum(n.startswith("D") for n in g[1])))


def rethink_batch(llm, rulebook, biz, items, first, cal, args):
    """VÒNG XEM LẠI cho câu đang BỎ TRỐNG (không lướt qua một lần là thôi):
    - D1: giải lại có cho xem các kết quả trước (máy tính / A / B) và yêu cầu phân tích từng bước, tự phán xử cách nào đúng;
    - D2: giải lại ĐỘC LẬP (không thấy kết quả trước) cũng với phân tích từng bước;
      cả hai cho model SUY NGHĨ (Qwen3/3.5 thinking) với nhiều token hơn, vẫn qua sandbox + kiểm luật + vòng sửa lỗi như A/B.
    - Bỏ phiếu mọi kết quả (máy tính, A, B, D1, D2): trả lời khi một đáp án có ≥ 2 phiếu trong đó có ≥ 1 phiếu xem lại
      và không có đáp án khác ngang phiếu; 'không trả lời được' phải được CẢ D1 và D2 nói.
    Không xem lại câu hoà ở biên (BTC đã loại câu hoà)."""
    idx = [i for i, (dec, fin, au, res) in enumerate(first) if fin.get("type") == "abstain" and "tie" not in dec]
    if not idx or not rethink_allowed(args):
        if idx:
            for i in idx:
                items[i]["log"]["nghi_lai"] = "bỏ qua (hết thời gian dự kiến)"
        return
    sub = [items[i] for i in idx]
    for i, it in zip(idx, sub):
        res = first[i][3]
        prev = []
        C = it.get("C")
        if isinstance(C, dict) and C.get("type") not in (None, "abstain"):
            prev.append(f"- Máy tính theo công thức{' (tin cậy thấp)' if (it.get('spec') or {}).get('tin_cay') == 'thap' else ''}: "
                        f"{jdump(clean_final(C))[:300]}")
        for v in ("A", "B"):
            ln = (res.get(v) or {}).get("line")
            err = (res.get(v) or {}).get("error")
            if ln is not None:
                prev.append(f"- Cách {v}: {jdump(ln)[:300]}")
            elif err:
                prev.append(f"- Cách {v}: lỗi ({str(err)[:120]})")
        it["_prev_text"] = "Các lần trước:\n" + "\n".join(prev) if prev else ""
    d1 = _run_variant_batch(llm, rulebook, biz, sub, cal, "D1", args)
    d2 = _run_variant_batch(llm, rulebook, biz, sub, cal, "D2", args)
    for j, i in enumerate(idx):
        dec, fin, au, res = first[i]
        res = dict(res, D1=d1[j], D2=d2[j])
        it = items[i]
        C = it.get("C")
        cands = ([("C", clean_final(C))] if isinstance(C, dict) and not C.get("_tie") else []) + \
            [(v, (res.get(v) or {}).get("line")) for v in ("A", "B", "D1", "D2")]
        groups = _vote(cands)
        info = {"phieu": [(g[0], g[1]) for g in groups][:4], "quyet_dinh_truoc": dec}
        ok = False
        if groups:
            n, names, line = groups[0]
            tie = len(groups) > 1 and groups[1][0] == n
            deep = any(x.startswith("D") for x in names)
            ok = n >= 2 and deep and not tie
            if ok and line.get("type") == "unanswerable":
                ok = "D1" in names and "D2" in names
        it["log"]["nghi_lai"] = info
        if ok:
            lab = f"nghi_lai_{n}phieu" + ("_dd" if "D1" in names and "D2" in names else "")
            first[i] = [lab, line, dict(au or {}, nghi_lai=info), res]
        else:
            first[i] = [dec + "+nghi_lai_khong_dong_thuan", fin, dict(au or {}, nghi_lai=info), res]


def rescue_candidate(it, results, decision):
    """Ứng viên tốt nhất cho một câu đang bỏ trống + nhãn loại (để hiệu chỉnh trên train):
    máy tính (tin cậy thấp / bị hạ) > A khác B (lấy A) > chỉ A > chỉ B. Không cứu câu hoà ở biên (BTC đã loại hoà)."""
    if "tie" in decision:
        return None, None
    C, sp = it.get("C"), it.get("spec") or {}

    def ok(x):
        return isinstance(x, dict) and x.get("type") not in (None, "abstain")
    if ok(C) and not C.get("_tie"):
        lab = "may_tinh_" + ("khong_tl" if C.get("type") == "unanswerable" else "thap" if sp.get("tin_cay") == "thap" else "cao")
        return clean_final(C), lab
    a, b = (results.get("A") or {}).get("line"), (results.get("B") or {}).get("line")
    suf = lambda x: "_khong_tl" if x.get("type") == "unanswerable" else ""   # noqa: E731
    d1 = (results.get("D1") or {}).get("line")
    if ok(d1):                                   # lần xem lại có phán xử (đã nghĩ kỹ) đứng trước các lượt nhanh
        return clean_final(d1), "xem_lai_d1" + suf(d1)
    if ok(a) and ok(b):
        return clean_final(a), "a_khac_b" + suf(a)
    if ok(a):
        return clean_final(a), "chi_a" + suf(a)
    if ok(b):
        return clean_final(b), "chi_b" + suf(b)
    return None, None


def solve_question(llm, rulebook, biz, qrow, cal, normalizer, args, log):
    """Giữ lại cho tương thích: giải 1 câu (dùng solve_batch)."""
    final, lg = solve_batch(llm, rulebook, biz, [qrow], cal, normalizer, args)[0]
    log.update(lg)
    return final


SHED = [("sanity_rounds", 0, "tắt vòng tự soát kết quả"), ("rule_tests", "off", "tắt bài kiểm từ luật"), ("metamorphic", "off", "tắt kiểm đảo dòng"),
        ("ask_choice", "off", "tắt hỏi chọn A/B"), ("n_paths", 1, "chỉ chạy đường A (bỏ đường B)"), ("max_repair", 0, "không sửa lỗi code nữa")]


def shed_steps(args, elapsed, done, total):
    """Ngân sách thời gian: dự báo tổng thời gian theo tốc độ hiện tại; vượt 90% ngân sách thì TẮT DẦN bước phụ
    (theo thứ tự SHED) thay vì để hết giờ rồi bỏ trống cả loạt câu cuối."""
    if getattr(args, "shed", "off") != "on" or not args.time_budget or done < 20 or done >= total:
        return            # mặc định TẮT: quyết định theo tốc độ máy làm kết quả khác nhau giữa máy nhóm và máy BTC (|ΔPA| ≤ 0,03)
    proj = elapsed / done * total
    if not hasattr(args, "_shed"):
        args._shed = 0
    while proj > 0.9 * args.time_budget and args._shed < len(SHED):
        k, v, why = SHED[args._shed]
        args._shed += 1
        if getattr(args, k, None) != v:
            setattr(args, k, v)
            print(f"[thời gian] dự báo {proj / 3600:.2f}h > 90% ngân sách {args.time_budget / 3600:.2f}h -> {why}", flush=True)
            proj *= 0.8          # ước lượng: mỗi bước tắt tiết kiệm ~20%; đo lại ở lô sau
            break


def run(args):
    split = args.split_dir.rstrip("/")
    out_dir = args.out
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    cache_dir = args.cache or os.path.join(out_dir, "cache")
    with open(os.path.join(split, "rulebook.md"), encoding="utf-8") as fh:
        rulebook = fh.read()
    global RULE_INDEX, RULE_PARAMS, RULE_DELTA
    if RDL is not None:
        try:
            RULE_DELTA = RDL.RuleDelta(rulebook)
            print("Rule-Delta (so với rulebook train):", jdump(RULE_DELTA.summary())[:1500], flush=True)
        except Exception as e:
            RULE_DELTA = None
            print(f"  [cảnh báo] Rule-Delta lỗi: {e}", flush=True)
    RULE_INDEX = RuleIndex(rulebook)
    RULE_PARAMS, rp_warn = extract_rule_params(rulebook)
    if CMP is not None:
        CMP.WEEK_START = RULE_PARAMS.get("week_start", 0)
    print(f"Rulebook: {len(RULE_INDEX.rules)} luật nguyên tử; tham số {RULE_PARAMS}", flush=True)
    if getattr(args, "vocab_guard", "on") == "on":
        v = load_compiler_vocab()
        print(f"Chặn chữ lạ cho máy tính: {'%d chữ' % len(v) if v else 'TẮT (thiếu lexicon/compiler_vocab.json)'}", flush=True)
    if RULE_DELTA is not None:
        print("Ngành có luật mới/đổi:", RULE_DELTA.flagged_sections() or "không", flush=True)
    for w_ in rp_warn:
        print("  [cảnh báo rulebook]", w_, flush=True)
    with open(os.path.join(out_dir, "rulebook_rules.json"), "w", encoding="utf-8") as fh:
        json.dump({"params": RULE_PARAMS, "warnings": rp_warn, "rules": RULE_INDEX.rules,
                   "rule_delta": RULE_DELTA.summary() if RULE_DELTA is not None else None}, fh, ensure_ascii=False, indent=1)
    cal = load_calendar(split)
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8")]
    if args.biz:
        keep = set(args.biz.split(","))
        qs = [q for q in qs if q["business_id"] in keep]
    qs.sort(key=lambda q: (q["business_id"], q["id"]))
    if args.limit:
        per = defaultdict(int)
        qs2 = []
        for q in qs:
            if per[q["business_id"]] < args.limit:
                qs2.append(q)
                per[q["business_id"]] += 1
        qs = qs2
    sub_path = os.path.join(out_dir, "submission.jsonl")
    fin_path = os.path.join(out_dir, "answers_final.jsonl")    # nhật ký đáp án CUỐI (append + fsync) — nguồn để chạy tiếp
    if args.fresh:
        for f_ in (sub_path, fin_path):
            if os.path.exists(f_):
                os.remove(f_)
    finals = {}
    src_resume = fin_path if os.path.exists(fin_path) else (sub_path if os.path.exists(sub_path) else None)
    if src_resume:
        for l in open(src_resume, encoding="utf-8"):
            try:
                r = json.loads(l)
                if src_resume == fin_path or r.get("type") != "abstain":
                    finals[r["id"]] = {k: v for k, v in r.items() if k != "id"}
            except Exception:
                pass                       # dòng cuối ghi dở khi bị ngắt: bỏ
    open(fin_path, "a", encoding="utf-8").close()   # có file này thì lần sau KHÔNG đọc submission (có đáp án tạm) làm đáp án cuối
    done = set(finals)
    all_ids = [q["id"] for q in qs]
    provisional = {}

    def write_submission():
        """Ghi LẠI TOÀN BỘ submission (tmp + fsync + os.replace): lúc nào bị ngắt, file cũng đủ mọi câu, không trùng id —
        câu đã xong lấy đáp án cuối, câu chưa xong lấy đáp án tạm (máy tính chắc chắn) hoặc abstain."""
        tmp = sub_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for i_ in all_ids:
                fh.write(jdump({"id": i_, **clean_final(finals.get(i_) or provisional.get(i_) or {"type": "abstain"})}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, sub_path)
    lm_extra = []
    if args.lm_texts and os.path.exists(args.lm_texts):   # ví dụ: train/questions.jsonl (CHỈ train)
        for line in open(args.lm_texts, encoding="utf-8"):
            line = line.strip()
            if line:
                lm_extra.append(json.loads(line).get("question", "") if line.startswith("{") else line)
        print(f"Mô hình ngôn ngữ sửa chữ: thêm {len(lm_extra)} dòng từ {args.lm_texts}", flush=True)

    def make_normalizer(biz):
        nz = Normalizer(biz.terms, ["bill", "voucher", "top", "order", "ship", "cod", "size", "zalo", "ok", "sale"], [rulebook, biz.readme]) if Normalizer else None
        if nz is not None and ContextNormalizer is not None and not args.no_ctx_norm:
            # sửa chữ theo ngữ cảnh cả câu: mô hình 3-gram học từ rulebook + README + tên hàng (+ câu hỏi TRAIN nếu có)
            nz = ContextNormalizer(nz, [rulebook, biz.readme] + lm_extra)
        return nz

    # pha sửa chữ bằng PhoGPT (chạy TRƯỚC vLLM rồi giải phóng GPU)
    if getattr(args, "norm_llm", "main") in ("pho", "both") and getattr(args, "pho_model", None) and ContextNormalizer is not None:
        t_p = now()
        try:
            import vn_lm_scorer as VS
            scorer = VS.LMScorer(args.pho_model)
            todo = defaultdict(list)
            for q in qs:
                if q["id"] not in done:
                    todo[q["business_id"]].append(q)
            for bid, items in todo.items():
                biz = Business(os.path.join(split, "data", bid), cache_dir=os.path.join(cache_dir, "biz"))
                nz = make_normalizer(biz)
                res, log = VS.resolve_unsure_lm(nz, [q["question"] for q in items], scorer, margin=args.pho_margin)
                per_q = defaultdict(list)
                for x in log:
                    per_q[x["q"]].append({k: v for k, v in x.items() if k != "q"})
                for i, q in enumerate(items):
                    PHO_PRE[q["id"]] = (res[i], per_q[i])
            scorer.close()
            del scorer
            print(f"Sửa chữ bằng PhoGPT: {len(PHO_PRE)} câu, {now() - t_p:.0f}s", flush=True)
        except Exception as e:
            print(f"  [cảnh báo] PhoGPT không chạy được ({type(e).__name__}: {e}) -> sửa chữ bằng Qwen trắc nghiệm", flush=True)
            PHO_PRE.clear()
    # pha TẠM: đường máy tính theo luật (không cần GPU) cho mọi câu chưa xong -> ghi submission NGAY, trước khi nạp model.
    # Nạp model lỗi / hết giờ / bị ngắt giữa chừng thì vẫn còn các đáp án chắc chắn này.
    if getattr(args, "provisional", "on") == "on" and CMP is not None:
        t_pv = now()
        todo_b = defaultdict(list)
        for q in qs:
            if q["id"] not in done:
                todo_b[q["business_id"]].append(q)
        pv_cap = getattr(args, "provisional_budget", 2400)
        for bid, items in todo_b.items():
            # trần thời gian cho pha tạm (dữ liệu shop đọc ở đây được cache -> pha chính đọc lại gần như miễn phí)
            if pv_cap and now() - t_pv > pv_cap:
                print(f"  Pha tạm vượt {pv_cap}s -> dừng pha tạm, các shop còn lại vào thẳng pha chính", flush=True)
                break
            try:
                biz = Business(os.path.join(split, "data", bid), cache_dir=os.path.join(cache_dir, "biz"))
                nz = make_normalizer(biz)
                for q in items:
                    pv = provisional_answer(q, biz, cal, nz)
                    if pv is not None:
                        provisional[q["id"]] = pv
            except Exception as e:
                print(f"  [cảnh báo] pha tạm lỗi ở {bid}: {type(e).__name__}: {e}", flush=True)
            write_submission()                 # bị ngắt GIỮA pha tạm vẫn có file đủ id
        print(f"Pha tạm (máy tính theo luật, không model): {len(provisional)}/{sum(len(v) for v in todo_b.values())} câu có đáp án chắc, {now() - t_pv:.0f}s", flush=True)
    write_submission()
    args._abstain_model = None
    if getattr(args, "abstain_model", None) and os.path.exists(args.abstain_model):
        args._abstain_model = json.load(open(args.abstain_model, encoding="utf-8"))
        print(f"Bỏ trống theo xác suất đúng: ngưỡng {args._abstain_model.get('nguong')} (mô hình fit trên train, {args._abstain_model.get('n')} câu)", flush=True)
    args._calibration = None
    if getattr(args, "calibration", None) and os.path.exists(args.calibration):
        _cj = json.load(open(args.calibration, encoding="utf-8"))
        args._calibration, args._cal_min_n = _cj.get("bang"), _cj.get("min_n", 15)
        print(f"Hiệu chỉnh: {sum(1 for v in args._calibration.values() if v.get('bo_trong'))} nhãn đổi sang bỏ trống, "
              f"{sum(1 for k, v in args._calibration.items() if k.startswith('cuu:') and v.get('tra_loi'))} loại ứng viên được CỨU "
              f"(đo trên train)", flush=True)
    if args.llm == "none":
        # không có LLM chạy được (vd máy không GPU): chỉ nộp đáp án CHẮC CHẮN của đường máy tính, còn lại abstain
        write_submission()
        with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
            fh.write(jdump({"llm": "none", "tam": len(provisional), "n": len(all_ids),
                            "tong_tu_dau_tien_trinh_s": round(now() - PROC_T0, 1)}))
        print(f"--llm none: chỉ đáp án chắc chắn của đường máy tính ({len(provisional)}/{len(all_ids)} câu)", flush=True)
        return
    try:
        llm = LLM(args.llm, args.model, cache_dir, max_model_len=args.max_model_len, tp=args.tp, gpu_mem=args.gpu_mem)
    except Exception:
        # không nạp được model (sai phiên bản vLLM / CUDA, thiếu VRAM…): GIỮ submission tạm (đáp án máy tính) thay vì không có gì
        print("LỖI NẠP MODEL — giữ submission tạm (chỉ đường máy tính):\n" + traceback.format_exc(limit=6), flush=True)
        write_submission()
        with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
            fh.write(jdump({"loi_nap_model": traceback.format_exc(limit=6), "tam": len(provisional), "n": len(all_ids)}))
        return
    t_start = now()
    stats = Counter()
    by_biz = defaultdict(list)
    for q in qs:
        if q["id"] not in done:
            by_biz[q["business_id"]].append(q)
    n_total_run = sum(len(v) for v in by_biz.values())
    n_done_run = 0
    with open(fin_path, "a", encoding="utf-8") as fout:
        out_of_time = False
        for bid, items in by_biz.items():
            if out_of_time:
                break
            tb = now()
            biz = Business(os.path.join(split, "data", bid), cache_dir=os.path.join(cache_dir, "biz"))
            normalizer = make_normalizer(biz)
            print(f"[{bid}] đọc dữ liệu {now() - tb:.1f}s, {len(biz.T)} bảng, thẻ {len(biz.card)} ký tự, {len(items)} câu", flush=True)
            for k in range(0, len(items), args.batch):
                # ngân sách tính TỪ LÚC TIẾN TRÌNH BẮT ĐẦU (gồm nạp model, pha tạm) — BTC tính 6 giờ gồm cả nạp model
                if args.time_budget and now() - PROC_T0 > args.time_budget:
                    print("Hết ngân sách thời gian → dừng (câu còn lại = đáp án tạm của máy tính hoặc abstain)", flush=True)
                    out_of_time = True
                    break
                chunk = items[k:k + args.batch]
                args._progress = (n_done_run, n_total_run, t_start)      # cho vòng xem lại ước thời gian còn đủ không
                try:
                    results = solve_batch(llm, rulebook, biz, chunk, cal, normalizer, args)
                except Exception:
                    tb_ = traceback.format_exc(limit=5)
                    results = [({"type": "abstain"}, {"id": q["id"], "business_id": bid, "fatal": tb_}) for q in chunk]
                n_done_run += len(chunk)
                shed_steps(args, now() - t_start, n_done_run, n_total_run)
                for q, (final, log) in zip(chunk, results):
                    final = clean_final(final)
                    if final.get("type") == "abstain" and q["id"] in provisional and "fatal" in log:
                        final = provisional[q["id"]]          # lô lỗi bất ngờ: giữ đáp án tạm (máy tính chắc chắn)
                        log["decision"] = "fatal->provisional"
                    stats[log.get("decision", "fatal")] += 1
                    finals[q["id"]] = final
                    fout.write(jdump({"id": q["id"], **final}) + "\n")
                    log.setdefault("final_written", final)
                    log["elapsed_total_s"] = round(now() - PROC_T0, 1)
                    with open(os.path.join(out_dir, "logs", f"{q['id']}.json"), "w", encoding="utf-8") as lf:
                        lf.write(jdump(log))
                fout.flush()
                os.fsync(fout.fileno())
                write_submission()
                print(f"  [{bid}] {min(k + args.batch, len(items))}/{len(items)} câu, {now() - t_start:.0f}s", flush=True)
            print(f"[{bid}] xong {now() - tb:.1f}s | thống kê: {dict(stats)}", flush=True)
    write_submission()
    n_q = max(1, sum(stats.values()))
    ls = dict(getattr(llm, "stats", {}) or {})
    summary = {"total_s": round(now() - t_start, 1), "decisions": dict(stats), "n": sum(stats.values()),
               "llm_goi": ls, "llm_goi_moi_cau": {k: round(v / n_q, 2) for k, v in ls.items() if k.startswith("goi")},
               "giay_moi_cau": round((now() - t_start) / n_q, 2), "da_tat_buoc": [SHED[i][2] for i in range(getattr(args, "_shed", 0))],
               "tong_tu_dau_tien_trinh_s": round(now() - PROC_T0, 1), "dap_an_tam": len(provisional), "het_gio": out_of_time}
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
        fh.write(jdump(summary))
    print("TÓM TẮT:", summary)


# =============================================================================
# 10. Chấm điểm theo rulebook §9 (dùng với answers.jsonl của TRAIN)
# =============================================================================

def _snorm(s):
    """Chuẩn hoá chuỗi ĐÚNG như BTC chấm: NFC, chữ thường, mọi ký tự không phải chữ/số/khoảng trắng -> khoảng trắng,
    gộp khoảng trắng. GIỮ dấu tiếng Việt. (Bản cũ xoá hết khoảng trắng -> dễ dãi hơn BTC, đo trên train bị lạc quan.)"""
    s = nfc(str(s)).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def _num_eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) < 1e-6


def _str_eq(pred, gold, aliases):
    if not isinstance(pred, str):
        return False
    p = _snorm(pred)
    return p in {_snorm(gold)} | {_snorm(x) for x in (aliases or []) if isinstance(x, str)}


def _elem_eq(p, g, al):
    if isinstance(g, str):
        return _str_eq(p, g, al)
    return _num_eq(p, g)


def score_one(pred, gold):
    """+1 đúng, -1 sai, 0 bỏ trống."""
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    gt, pt_ = gold["type"], pred.get("type")
    if gt == "unanswerable" or gold.get("answer") is None and gt == "unanswerable":
        return 1 if pt_ == "unanswerable" else -1
    if pt_ != gt:
        return -1
    g, p, al = gold.get("answer"), pred.get("answer"), gold.get("alias") or []
    if gt in ("number", "percent"):
        return 1 if _num_eq(p, g) else -1
    if gt == "date":
        return 1 if isinstance(p, str) and p.strip() == g else -1
    if gt == "boolean":
        return 1 if isinstance(p, bool) and p == g else -1
    if gt == "string":
        flat = [x for x in al if isinstance(x, str)] + [y for x in al if isinstance(x, list) for y in x]
        return 1 if _str_eq(p, g, flat) else -1
    if gt == "list":
        if not isinstance(p, list) or len(p) != len(g):
            return -1
        for i, (pi, gi) in enumerate(zip(p, g)):
            ali = al[i] if i < len(al) and isinstance(al[i], list) else []
            if not _elem_eq(pi, gi, ali):
                return -1
        return 1
    if gt == "table":
        if not isinstance(p, dict) or not isinstance(p.get("rows"), list):
            return -1
        gcols, pcols = g["columns"], p.get("columns", [])
        if len(pcols) != len(gcols) or len(p["rows"]) != len(g["rows"]):
            return -1
        if set(map(_snorm, gcols)) == set(map(_snorm, pcols)):
            idx = [list(map(_snorm, pcols)).index(_snorm(c)) for c in gcols]
        else:
            idx = list(range(len(gcols)))
        used = set()
        for ri, grow in enumerate(g["rows"]):
            ali = al[ri] if ri < len(al) and isinstance(al[ri], list) else []
            found = False
            for pj, prow in enumerate(p["rows"]):
                if pj in used or len(prow) != len(gcols):
                    continue
                prow2 = [prow[k] for k in idx]
                if _elem_eq(prow2[0], grow[0], ali) and all(_elem_eq(a, b, []) for a, b in zip(prow2[1:], grow[1:])):
                    used.add(pj)
                    found = True
                    break
            if not found:
                return -1
        return 1
    return -1


def score(args):
    split = args.split_dir.rstrip("/")
    gold_path = args.gold or os.path.join(split, "answers.jsonl")
    if "public" in os.path.abspath(gold_path).lower() or "private" in os.path.abspath(gold_path).lower():
        print("CẢNH BÁO: bộ chấm này dành cho đáp án TRAIN. Dùng đáp án test để chỉnh hệ thống là vi phạm luật 6 của cuộc thi.")
    gold = {}
    for l in open(gold_path, encoding="utf-8"):
        r = json.loads(l)
        gold[r["id"]] = r
    qs = {}
    qp = os.path.join(split, "questions.jsonl")
    if os.path.exists(qp):
        for l in open(qp, encoding="utf-8"):
            r = json.loads(l)
            qs[r["id"]] = r
    pred = {}
    for l in open(args.pred, encoding="utf-8"):
        r = json.loads(l)
        pred[r["id"]] = r
    ids = [i for i in gold if (not args.only_pred or i in pred)]
    tot = Counter()
    by_biz = defaultdict(Counter)
    by_type = defaultdict(Counter)
    errs = []
    for i in ids:
        s = score_one(pred.get(i), gold[i])
        lab = {1: "đúng", -1: "sai", 0: "bỏ trống"}[s]
        tot[lab] += 1
        b = qs.get(i, {}).get("business_id", "?")
        by_biz[b][lab] += 1
        by_type[gold[i]["type"]][lab] += 1
        if s == -1:
            errs.append({"id": i, "business_id": b, "question": qs.get(i, {}).get("question"), "as_of": qs.get(i, {}).get("as_of_date"),
                         "gold": {k: gold[i].get(k) for k in ("type", "answer")}, "pred": pred.get(i)})
    N = len(ids)

    def pa(c):
        n = sum(c.values())
        return (c["đúng"] - c["sai"]) / n if n else 0

    print(f"Số câu: {N} | đúng {tot['đúng']} | sai {tot['sai']} | bỏ trống {tot['bỏ trống']}")
    print(f"PA = {pa(tot):.4f} | precision khi trả lời = {tot['đúng'] / max(1, tot['đúng'] + tot['sai']):.4f}")
    print("\nTheo shop:")
    for b, c in sorted(by_biz.items()):
        print(f"  {b}: PA {pa(c):.3f}  ({dict(c)})")
    print("\nTheo loại đáp án:")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: PA {pa(c):.3f}  ({dict(c)})")
    out = args.errors or os.path.join(os.path.dirname(os.path.abspath(args.pred)), "errors.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for e in errs:
            fh.write(jdump(e) + "\n")
    print(f"\nĐã ghi {len(errs)} câu sai vào {out}")
    return pa(tot)


# =============================================================================
# 11. Self-test (parser, thời gian, chấm điểm, sandbox)
# =============================================================================

def selftest(args):
    ok = True

    def check(name, got, exp):
        nonlocal ok
        if got != exp:
            ok = False
            print(f"  FAIL {name}: got {got!r} expected {exp!r}")

    print("1) parser tiền")
    for s, e in [("29k", 29000.0), ("43.000đ", 43000.0), ("1tr380", 1380000.0), ("1tr5", 1500000.0), ("2,2605tr", 2260500.0),
                 ("-4tr300", -4300000.0), ("(50.000đ)", -50000.0), ("129 000", 129000.0), ("1,05tr", 1050000.0), ("3,24tr", 3240000.0)]:
        check(s, parse_money_one(s), e)
    check("thousands '.'", decide_thousands(["1.500", "12.000", "300"]), ".")
    check("thousands ','", decide_thousands(["1,500", "12,000.5"]), ",")
    print("2) parser ngày")
    for s, e in [("01/03/2025", pd.Timestamp(2025, 3, 1)), ("2025-03-01 08:15:00", pd.Timestamp(2025, 3, 1, 8, 15)),
                 ("1 thg 3 2025", pd.Timestamp(2025, 3, 1)), ("3/26", pd.NaT)]:
        got = parse_date_one(s, None)
        if not (pd.isna(got) and pd.isna(e)):
            check(s, got, e)
    check("serial", parse_date_one("46023", allow_serial=True), pd.Timestamp(2026, 1, 1))
    check("d/m year_hint", parse_date_one("5/2", 2025), pd.Timestamp(2025, 2, 5))
    print("3) rhu / pct")
    check("rhu 2.675", rhu(2.675, 2), 2.68)
    check("rhu 0.5", rhu(0.5), 1)
    check("pct", pct(90, 100), -10.0)
    print("4) bộ giải thời gian")
    P = resolve_periods("tuần trước bán được bao nhiêu", "2025-06-23")  # thứ Hai
    check("tuần trước (hôm nay thứ Hai)", (str(P["tuan_truoc"][0].date()), str(P["tuan_truoc"][1].date())), ("2025-06-16", "2025-06-22"))
    P = resolve_periods("Q1/2026 có bao nhiêu đơn", "2026-07-31")
    check("Q1/2026", (str(P["quy_1_2026"][0].date()), str(P["quy_1_2026"][1].date())), ("2026-01-01", "2026-03-31"))
    P = resolve_periods("so voi th2/2025, doanh thu th2/2026", "2026-09-19")
    check("th2/2025 & th2/2026", ("thang_2_2025" in P, "thang_2_2026" in P), (True, True))
    P = resolve_periods("tháng 11 doanh thu", "2026-03-10")
    check("tháng 11 không ghi năm → năm trước", "thang_11_2025" in P, True)
    P = resolve_periods("tháng này so với tháng trước", "2025-03-31")
    check("cùng kỳ tháng trước bị chặn", str(P["thang_truoc_cung_ky"][1].date()), "2025-02-28")
    P = resolve_periods("7 ngày gần nhất", "2025-03-10")
    check("7 ngày", str(P["7_ngay_gan_nhat"][0].date()), "2025-03-04")
    P = resolve_periods("tháng ba năm 2026", "2026-08-01")
    check("tháng ba", "thang_3_2026" in P, True)
    print("5) cảnh báo bài học")
    check("so sánh", any("SO SÁNH" in w for w in lesson_warnings("So với T4/2025 thì T4/2026 tăng bao nhiêu %")), True)
    check("quý", any("QUÝ" in w for w in lesson_warnings("Q1/2026 có bao nhiêu đơn")), True)
    check("'so' = 'số' không dấu → không cảnh báo so sánh", any("SO SÁNH" in w for w in lesson_warnings("e so don hang quy 4/2025 la bao nhiu")), False)
    check("'nhỉ' không phải thứ hạng", any("THỨ HẠNG" in w for w in lesson_warnings("Giá trị trung bình một đơn tháng 10/2025 là bao nhiêu nhỉ?")), False)
    check("thứ hạng nhì", any("THỨ HẠNG" in w for w in lesson_warnings("món bán chạy thứ hai tháng 3 là gì")), True)
    check("'không?' cuối câu không phải phủ định", any("PHỦ ĐỊNH" in w for w in lesson_warnings("doanh thu có tăng liên tục không?")), False)
    check("'bơ' không phải 'bỏ'", any("PHỦ ĐỊNH" in w for w in lesson_warnings("Sinh tố bơ (M) giá bao nhiêu")), False)
    check("'hông kê đơn' là phủ định", any("PHỦ ĐỊNH" in w for w in lesson_warnings("dthu hàng hông kê đơn là bao nhiêu")), True)
    check("so voi không dấu", any("SO SÁNH" in w for w in lesson_warnings("so voi th2/2025, doanh thu th2/2026 tang may %")), True)
    check("nhiều vế", any("NHIỀU VẾ" in w for w in lesson_warnings("T3 cao hơn T2 và T4 cũng cao hơn T3, phải ko?")), True)
    check("nhập − bán", any("NHẬP" in w for w in lesson_warnings("nhập trừ bán còn bao nhiêu cái")), True)
    print("6) chấm điểm")
    g = {"type": "string", "answer": "Sting dâu 330ml", "alias": ["SP0001"]}
    check("string alias", score_one({"type": "string", "answer": "SP0001"}, g), 1)
    check("string hoa thường", score_one({"type": "string", "answer": "sting DÂU 330ml"}, g), 1)
    check("string sai dấu", score_one({"type": "string", "answer": "Sting dầu 330ml"}, g), -1)
    check("number str", score_one({"type": "number", "answer": "4820000"}, {"type": "number", "answer": 4820000}), -1)
    check("percent", score_one({"type": "percent", "answer": -12.34}, {"type": "percent", "answer": -12.34}), 1)
    check("abstain", score_one({"type": "abstain"}, {"type": "number", "answer": 1}), 0)
    check("unanswerable", score_one({"type": "unanswerable"}, {"type": "unanswerable", "answer": None}), 1)
    gt = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 10], ["B", 5]]}, "alias": [["KH1"], ["KH2"]]}
    check("table đảo dòng + alias", score_one({"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["KH2", 5], ["A", 10]]}}, gt), 1)
    check("list thứ tự", score_one({"type": "list", "answer": ["b", "a"]}, {"type": "list", "answer": ["a", "b"], "alias": []}), -1)
    print("7) sandbox")
    st, v = run_sandbox("def solve():\n    return {'type':'number','answer': int(T['x'].a.sum())}", {"T": {"x": pd.DataFrame({"a": [1, 2]})}, "pd": pd, "__builtins__": __builtins__})
    check("sandbox chạy", (st, v), ("ok", {"type": "number", "answer": 3}))
    st, v = run_sandbox("import os\ndef solve():\n    return 1", {})
    check("sandbox chặn import os", st, "err")
    st, v = run_sandbox("def solve():\n    while True: pass", {"__builtins__": __builtins__}, timeout=2)
    check("sandbox timeout", (st, "quá thời gian" in str(v)), ("err", True))
    check("format percent", format_answer({"type": "percent", "answer": 12.345})[0], {"type": "percent", "answer": 12.35})
    check("format list 1 phần tử", format_answer({"type": "list", "answer": ["x"]})[0], {"type": "list", "answer": ["x"]})
    check("format date tháng", format_answer({"type": "date", "answer": "2025-09"})[0], {"type": "date", "answer": "2025-09"})
    if args.split_dir:
        print("8) chấm đáp án train với chính nó (phải = 1.0)")
        split = args.split_dir.rstrip("/")
        gold = [json.loads(l) for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8")]
        s = [score_one({k: v for k, v in r.items() if k != "alias"}, r) for r in gold]
        check("gold vs gold", (sum(s), len(s)), (len(s), len(s)))
    print("SELFTEST:", "OK" if ok else "CÓ LỖI")
    return ok


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="DataMate pipeline v0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split_dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cache", default=None)
    r.add_argument("--llm", default="vllm", choices=["vllm", "hf", "dummy", "none"], help="none = không LLM: chỉ đáp án chắc chắn của đường máy tính")
    r.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    r.add_argument("--max_model_len", type=int, default=28000)
    r.add_argument("--tp", type=int, default=1)
    r.add_argument("--gpu_mem", type=float, default=0.90)
    r.add_argument("--n_paths", type=int, default=2, help="1 = chỉ đường A; 2 = A + B và yêu cầu đồng thuận")
    r.add_argument("--policy", default="strict", choices=["strict", "lenient"], help="A/B lệch: strict = abstain, lenient = lấy A")
    r.add_argument("--on_single", default="answer", choices=["answer", "abstain"], help="chỉ 1 đường ra kết quả thì làm gì")
    r.add_argument("--max_repair", type=int, default=1)
    r.add_argument("--abstain_model", default=None, help="lexicon/abstain_model.json (scripts/fit_abstain.py trên log train): bỏ trống khi p(đúng) ≤ ngưỡng")
    r.add_argument("--calibration", default=None, help="lexicon/decision_calibration.json (scripts/calibrate_decisions.py trên train)")
    r.add_argument("--b_when_sure", default="off", choices=["on", "off"], help="chạy đường B cả khi máy tính theo luật đã chắc (tốn thêm ~1 lượt gọi/câu)")
    r.add_argument("--rule_tests", default="on", choices=["on", "off"], help="bài kiểm từ luật: chèn đơn tương lai / đơn huỷ giả, đáp án không được đổi")
    r.add_argument("--metamorphic", default="on", choices=["on", "off"], help="chạy lại code trên bảng đảo thứ tự dòng; khác kết quả -> cho model xem lại")
    r.add_argument("--sanity_rounds", type=int, default=1, help="số vòng agent tự soát KẾT QUẢ (kiểu, số âm, % vô lý…) — 0 = tắt")
    r.add_argument("--compiler", default="on", choices=["on", "off"], help="đường 1: đặc tả + máy tính theo công thức (ledger)")
    r.add_argument("--llm_spec", default="on", choices=["on", "off"], help="câu luật chữ không nhận ra: LLM viết đặc tả JSON")
    r.add_argument("--spec_bank", default=None, help="kho ví dụ đặc tả từ train (mặc định lexicon/spec_bank.jsonl)")
    r.add_argument("--embed_model", default=None, help="thư mục bge-m3 offline (không có -> TF-IDF ký tự)")
    r.add_argument("--ask_choice", default="on", choices=["on", "off"], help="lệch nhau: hỏi model chọn cách đúng (1 chữ cái)")
    r.add_argument("--batch", type=int, default=48, help="số câu gửi model cùng lúc (vLLM xử lý song song)")
    r.add_argument("--exec_timeout", type=int, default=40, help="giây cho một lần chạy solve() (nhân thêm theo số bài kiểm); rộng để máy chậm hơn không đổi kết quả")
    r.add_argument("--biz", default=None, help="danh sách business_id, phân tách bằng dấu phẩy")
    r.add_argument("--limit", type=int, default=0, help="tối đa N câu mỗi shop (để đo nhanh)")
    r.add_argument("--time_budget", type=int, default=0, help="giây; hết thì dừng")
    r.add_argument("--fresh", action="store_true", help="ghi đè submission cũ thay vì chạy tiếp")
    r.add_argument("--provisional", default="on", choices=["on", "off"], help="pha tạm: đáp án máy tính cho mọi câu TRƯỚC khi nạp model")
    r.add_argument("--rethink", default="on", choices=["on", "off"], help="câu bỏ trống: giải lại kỹ 2 lần (có/không xem kết quả trước) rồi bỏ phiếu")
    r.add_argument("--rethink_tokens", type=int, default=6144, help="token sinh tối đa ở vòng xem lại (gồm phần suy nghĩ)")
    r.add_argument("--rethink_think", default="on", choices=["on", "off"], help="bật chế độ suy nghĩ của model (Qwen3/3.5) ở vòng xem lại")
    r.add_argument("--rethink_until", type=float, default=0.85, help="ngừng xem lại khi đã dùng quá tỉ lệ này của ngân sách thời gian")
    r.add_argument("--vocab_guard", default="on", choices=["on", "off"], help="máy tính không tự quyết câu còn chữ ngoài từ vựng train")
    r.add_argument("--provisional_budget", type=int, default=2400, help="giây tối đa cho pha tạm (0 = không giới hạn)")
    r.add_argument("--shed", default="off", choices=["on", "off"],
                   help="tắt dần bước phụ khi dự báo vượt ngân sách (theo TỐC ĐỘ MÁY -> có thể làm kết quả khác giữa hai máy; mặc định tắt)")
    r.add_argument("--lm_texts", default=None, help="file văn bản/jsonl để học mô hình sửa chữ theo ngữ cảnh (chỉ dùng câu hỏi TRAIN)")
    r.add_argument("--no_ctx_norm", action="store_true", help="tắt sửa chữ theo ngữ cảnh, chỉ sửa từng chữ")
    r.add_argument("--norm_llm", choices=["main", "pho", "both", "off"], default="main",
                   help="chỗ sửa chữ chưa chắc: Qwen trắc nghiệm (main) | PhoGPT chấm câu (pho) | chỉ đổi khi cả hai cùng chọn (both) | tắt (off)")
    r.add_argument("--pho_model", default=None, help="thư mục PhoGPT-4B(-Chat) offline (cho --norm_llm pho/both)")
    r.add_argument("--pho_margin", type=float, default=1.0, help="PhoGPT phải hơn lựa chọn hiện tại ít nhất bấy nhiêu nat mới đổi")
    s = sub.add_parser("score")
    s.add_argument("--split_dir", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--gold", default=None, help="mặc định <split>/answers.jsonl (CHỈ train)")
    s.add_argument("--errors", default=None)
    s.add_argument("--only_pred", action="store_true", help="chỉ chấm các câu có trong file dự đoán (khi chạy một phần)")
    t = sub.add_parser("selftest")
    t.add_argument("--split_dir", default=None)
    c = sub.add_parser("card")
    c.add_argument("--split_dir", required=True)
    c.add_argument("--biz", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "score":
        score(args)
    elif args.cmd == "selftest":
        sys.exit(0 if selftest(args) else 1)
    elif args.cmd == "card":
        b = Business(os.path.join(args.split_dir, "data", args.biz))
        print(b.card)
        print(f"\n(đọc {b.load_sec:.1f}s; {len(b.terms)} tên được bảo vệ khi chuẩn hoá)")


if __name__ == "__main__":
    main()
