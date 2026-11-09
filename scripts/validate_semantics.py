"""
Kiểm LỚP NGHĨA (cột __nhom_trang_thai, __sl_co_ban, __khach_le) bằng đáp án TRAIN — chỉ dùng train.

    python scripts/validate_semantics.py --train /path/train

Với từng dạng câu đơn giản (số đơn huỷ / tỉ lệ huỷ theo tháng; số lượng bán một mặt hàng theo tháng; số khách mua
≥ N đơn), tính thẳng từ cột nghĩa rồi so với đáp án train. Khớp gần hết = lớp nghĩa + làm sạch đúng.
Lần chạy 08/11: huỷ 164/164, số lượng quy đổi 10/10, khách ≥ N đơn 3/3.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import datamate_pipeline as d  # noqa: E402

TR = CACHE = None


def check_status(qs, an):
    tot=ok=0; bad=[]
    for b in sorted(glob.glob(os.path.join(TR, 'data', '*'))):
        B=d.Business(b,cache_dir=CACHE)
        tabs=[(n,df) for n,df in B.T.items() if '__nhom_trang_thai' in df.columns and d.order_id_col(df,B.types[n])]
        if not tabs: continue
        n,df=max(tabs,key=lambda x: len(x[1]))
        oid=d.order_id_col(df,B.types[n])
        dcols=[c for c in df.columns if str(B.types[n].get(c,'')).startswith('ngày')]
        dc=next((c for c in dcols if re.search(r'(dat|tao|ban|hoa don|ngay$|^ngay)', d._colnorm(c))), dcols[0] if dcols else None)
        if dc is None: continue
        o=df.groupby(oid).agg(ng=(dc,'first'),g=('__nhom_trang_thai','first'))
        for i,q in qs.items():
            if q['business_id']!=B.id: continue
            t=d.key(q['question']); m=re.search(r'thang (\d{1,2})/(\d{4})',t)
            if not m or 'huy' not in t.split() or 'vi' in t.split(): continue
            mm,yy=int(m.group(1)),int(m.group(2)); g=an[i].get('answer')
            if g is None or isinstance(g,(list,dict,str)): continue
            sel=o[(o.ng.dt.month==mm)&(o.ng.dt.year==yy)]
            val = round(100*(sel.g=='huy').sum()/len(sel),2) if ('ti le' in t or '%' in q['question'] or 'phan tram' in t) else int((sel.g=='huy').sum())
            tot+=1; ok+= abs(float(val)-float(g))<1e-6
            if abs(float(val)-float(g))>=1e-6: bad.append((B.id,q['question'][:60],g,val,dc))
    return ok, tot, bad


def check_unit(qs, an):
    tot=ok=0; bad=[]
    for b in sorted(glob.glob(os.path.join(TR, 'data', '*'))):
        B=d.Business(b,cache_dir=CACHE)
        for n,df in B.T.items():
            if '__sl_co_ban' not in df.columns or '__nhom_trang_thai' not in df.columns: continue
            pc=d._find_col(df,r"(ten san pham|ten hang|san pham|mat hang)")
            dcols=[c for c in df.columns if str(B.types[n].get(c,'')).startswith('ngày')]
            dc=dcols[0]
            names={d.key(x):x for x in df[pc].dropna().unique()}
            for i,q in qs.items():
                if q['business_id']!=B.id: continue
                t=d.key(q['question']); m=re.search(r'thang (\d{1,2})/(\d{4})',t)
                g=an[i].get('answer')
                if not m or not isinstance(g,(int,float)) or 'bao nhieu' not in t and 'may' not in t.split(): continue
                hit=[k for k in names if len(k)>6 and k in t]
                if len(hit)!=1 or not re.search(r'\b(ban|ban duoc|ban ra)\b',t) or re.search(r'tien|doanh|%|don\b',t): continue
                mm,yy=int(m.group(1)),int(m.group(2))
                sel=df[(df[dc].dt.month==mm)&(df[dc].dt.year==yy)&(df[pc].map(d.key)==hit[0])&(df['__nhom_trang_thai'].isin(['hoan_thanh','dang_giao']))]
                val=sel['__sl_co_ban'].sum(); raw=sel[d._find_col(df,r"(so luong)",B.types[n],'số')].sum()
                tot+=1; ok+=abs(val-g)<1e-6
                if abs(val-g)>=1e-6: bad.append((B.id,q['question'][:80],g,val,raw))
    return ok, tot, bad


def check_khach(qs, an):
    tot=ok=0; bad=[]
    for b in sorted(glob.glob(os.path.join(TR, 'data', '*'))):
        B=d.Business(b,cache_dir=CACHE)
        tabs=[(n,df) for n,df in B.T.items() if '__khach_le' in df.columns and '__nhom_trang_thai' in df.columns]
        if not tabs: continue
        n,df=max(tabs,key=lambda x: len(x[1]))
        oid=d.order_id_col(df,B.types[n])
        cc=d._find_col(df,r"(ma kh|ma khach|customer id|ma khach mua)") or d._find_col(df,r"(ten khach|khach hang|ten kh|customer)")
        dcols=[c for c in df.columns if str(B.types[n].get(c,'')).startswith('ngày')]
        dc=next((c for c in dcols if re.search(r'(dat|tao|ban|hoa don|ngay$|^ngay)', d._colnorm(c))), dcols[0])
        for i,q in qs.items():
            if q['business_id']!=B.id: continue
            t=d.key(q['question']); m=re.search(r'thang (\d{1,2})/(\d{4})',t); g=an[i].get('answer')
            if not m or not isinstance(g,(int,float)) or not re.search(r'(bao nhieu|may|so) khach',t) or re.search(r'moi|quay lai|lan dau|no|tien|%|ca sang|ca chieu|nhom|tro len|it nhat|hon',t): continue
            mm,yy=int(m.group(1)),int(m.group(2))
            sel=df[(df[dc].dt.month==mm)&(df[dc].dt.year==yy)&(df['__nhom_trang_thai'].isin(['hoan_thanh','dang_giao']))&(~df['__khach_le'])]
            mN=re.search(r'≥\s*(\d+)\s*don', q['question'].lower().replace('đơn','don'))
            if mN:
                per=sel.groupby(sel[cc].map(d.key))[oid].nunique(); val=int((per>=int(mN.group(1))).sum())
            else:
                val=sel[cc].map(d.key).nunique()
            tot+=1; ok+=val==g
            if val!=g: bad.append((B.id,q['question'][:70],g,val,cc))
    return ok, tot, bad


def main():
    global TR, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--cache', default='/tmp/validate_sem_cache')
    a = ap.parse_args()
    TR, CACHE = a.train, a.cache
    qs = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(TR, 'questions.jsonl'), encoding='utf-8')}
    an = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(TR, 'answers.jsonl'), encoding='utf-8')}
    for name, fn in (('đơn huỷ / tỉ lệ huỷ', check_status), ('số lượng quy về đơn vị cơ bản', check_unit), ('số khách', check_khach)):
        ok, tot, bad = fn(qs, an)
        print(f'{name:32s}: khớp {ok}/{tot}')
        for x in bad[:10]:
            print('   lệch:', x)


if __name__ == '__main__':
    main()
