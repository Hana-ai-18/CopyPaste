"""
Soát tuân thủ trước khi commit / nộp.

    python scripts/check_compliance.py [--public /path/public_test]

Kiểm:
  1. Không có ID câu hỏi public/private (q_xxxxxxxxxxxx) nào trong repo (code, lexicon, docs).
  2. Không có logic theo ID: chuỗi 'q_…', 'biz_…', 'dev_…' trong code lõi (.py ngoài tests/ và scripts đánh giá train).
  3. Không có file đáp án / file nộp trong repo (answers*.jsonl, submission*.jsonl).
  4. Nhắc giấy phép: lexicon/gpl/ (GPL v2) nếu có.
Thoát mã 1 nếu có vi phạm (1)–(3).
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {'.py', '.md', '.tsv', '.txt', '.yaml', '.yml', '.ipynb', '.sh', '.json'}
CORE = ['datamate_pipeline.py', 'vn_normalize.py', 'vn_context.py', 'compiler.py', 'ledger.py', 'retriever.py', 'vn_lm_scorer.py']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--public', default=None, help='thư mục public_test (chỉ đọc questions.jsonl để lấy ID cần cấm)')
    a = ap.parse_args()
    bad = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and p.suffix in TEXT_EXT]
    ids = set()
    if a.public:
        for line in open(Path(a.public) / 'questions.jsonl', encoding='utf-8'):
            ids.add(json.loads(line)['id'])
    for p in files:
        txt = p.read_text(encoding='utf-8', errors='ignore')
        hit = [i for i in ids if i in txt] if ids else []
        if hit:
            bad.append(f'[1] {p.relative_to(ROOT)} chứa ID câu public: {hit[:3]}')
        if re.search(r'\bq_[0-9a-f]{12}\b', txt) and p.name != 'check_compliance.py':
            bad.append(f'[1] {p.relative_to(ROOT)} chứa chuỗi dạng ID câu hỏi (q_ + 12 ký tự hex)')
    for name in CORE:
        txt = (ROOT / name).read_text(encoding='utf-8')
        for m in re.finditer(r'["\'](?:biz|dev)_\d{3}["\']', txt):
            bad.append(f'[2] {name}: chuỗi ID shop {m.group(0)} trong code lõi')
    pub_q = set()
    if a.public:
        pub_q = {json.loads(l)['question'].strip() for l in open(Path(a.public) / 'questions.jsonl', encoding='utf-8')}
    for p in ROOT.rglob('*.jsonl'):
        if '.git' in p.parts:
            continue
        if p.relative_to(ROOT).as_posix() == 'lexicon/spec_bank.jsonl':
            # kho ví dụ đặc tả: CHỈ câu train (shop biz_*), không có trường đáp án, không trùng câu public nào
            same = 0
            for i, line in enumerate(open(p, encoding='utf-8')):
                e = json.loads(line)
                if not str(e.get('biz', '')).startswith('biz_') or 'answer' in e:
                    bad.append(f'[3] lexicon/spec_bank.jsonl dòng {i + 1}: không phải câu train (biz_*) hoặc có trường đáp án')
                    break
                same += e.get('question', '').strip() in pub_q
            if same:
                # câu mẫu chung chung ('Doanh thu hôm qua của cửa hàng là bao nhiêu?') có ở CẢ train và public: đây là câu TRAIN,
                # đặc tả dựng từ dữ liệu + đáp án TRAIN — không phải dùng dữ liệu public. Chỉ nhắc, không phải vi phạm.
                print(f'[nhắc] spec_bank: {same} câu train có chữ trùng câu public (câu mẫu chung) — nguồn vẫn là train')
            continue
        bad.append(f'[3] file .jsonl trong repo: {p.relative_to(ROOT)}')
    if (ROOT / 'lexicon' / 'gpl').exists():
        print('[nhắc] lexicon/gpl/ phái sinh từ Viet22K/Viet74K (GPL v2). Xoá thư mục nếu nhóm không muốn ràng buộc GPL.')
    if bad:
        print('VI PHẠM:')
        print('\n'.join(bad))
        sys.exit(1)
    print(f'OK: đã soát {len(files)} file' + (f', {len(ids)} ID câu public' if ids else ''))


if __name__ == '__main__':
    main()
