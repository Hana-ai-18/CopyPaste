"""
Đo trên GPU (Kaggle): LLM có giúp sửa teencode / câu sai chính tả không, và cách dùng nào tốt nhất.
Chỉ dùng câu hỏi TRAIN (làm bẩn có kiểm soát), không dùng public/private.

    python scripts/eval_norm_llm.py --train /path/train --llm vllm --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --n 300
    python scripts/eval_norm_llm.py --train /path/train --llm hf --model vinai/PhoGPT-4B-Chat --n 300   # thử model tiếng Việt

So 3 cách (tỉ lệ câu sửa đúng hoàn toàn, theo từng kiểu bẩn, gồm cả câu SẠCH — không được sửa nhầm):
  ngu_canh      : chỉ luật + mô hình ngôn ngữ 3-gram (không LLM)
  llm_chon      : như trên + LLM chọn trắc nghiệm ở chỗ chưa chắc (cách pipeline đang dùng, --norm_llm main)
  llm_viet_lai  : LLM tự viết lại cả câu (để so: thường đổi bừa tên hàng / con số)
  pho_cham      : PhoGPT CHẤM các câu ứng viên ở chỗ chưa chắc (--norm_llm pho), cần --pho_model
  ca_hai        : chỉ đổi chữ khi PhoGPT và LLM chính chọn giống nhau (--norm_llm both)
Thêm cột 'giữ số': tỉ lệ câu giữ nguyên mọi chữ số (viết lại hay làm hỏng số).

    python scripts/eval_norm_llm.py --train /path/train --model /kaggle/input/qwen... --pho_model /kaggle/input/phogpt-4b --n 300
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datamate_pipeline as dp  # noqa: E402
from eval_normalize import corrupt, is_clean  # noqa: E402
from vn_normalize import Normalizer  # noqa: E402
from vn_context import ContextNormalizer, resolve_unsure, tokenize  # noqa: E402

REWRITE_SYS = ("Bạn là chuyên gia tiếng Việt. Viết lại câu hỏi của chủ shop thành tiếng Việt chuẩn có dấu: sửa teencode, viết tắt, "
               "gõ sai dấu. GIỮ NGUYÊN mọi con số, ngày tháng, tên hàng, tên người, mã. Không thêm, không bớt ý. Chỉ trả lại đúng một câu.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--llm', default='vllm', choices=['vllm', 'hf', 'dummy'])
    ap.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct-AWQ')
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--max_model_len', type=int, default=4096)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--kinds', default='sach,bo_dau,teen,tron,vung_mien,vni')
    ap.add_argument('--no_rewrite', action='store_true', help='bỏ cách llm_viet_lai (tiết kiệm thời gian)')
    ap.add_argument('--cache', default='/tmp/eval_norm_llm_cache')
    ap.add_argument('--pho_model', default=None, help='thư mục PhoGPT-4B offline: thêm cột pho_cham và ca_hai')
    ap.add_argument('--pho_margin', type=float, default=1.0)
    a = ap.parse_args()
    tr = Path(a.train)
    qs = [json.loads(l) for l in open(tr / 'questions.jsonl', encoding='utf-8')]
    rb = (tr / 'rulebook.md').read_text(encoding='utf-8')
    shops = sorted({q['business_id'] for q in qs})
    learn_shops = set(shops[0::2])
    learn = [q['question'] for q in qs if q['business_id'] in learn_shops]
    test = [q['question'] for q in qs if q['business_id'] not in learn_shops and is_clean(q['question'])]
    random.Random(0).shuffle(test)
    test = test[: a.n]
    cn = ContextNormalizer(Normalizer([], ['bill', 'voucher', 'top', 'cod', 'ok', 'sale'], [rb]), [rb] + learn)
    kinds = a.kinds.split(',')
    noisy_by = {}
    for kind in kinds:
        rng = random.Random(11)
        noisy_by[kind] = [corrupt(g, kind, rng, p=1.0 if kind == 'bo_dau' else 0.6) for g in test]
    pho = {}
    if a.pho_model:                      # PhoGPT chạy TRƯỚC, giải phóng GPU rồi mới nạp LLM chính (giống pipeline)
        import vn_lm_scorer as VS
        sc = VS.LMScorer(a.pho_model)
        for kind in kinds:
            pho[kind] = VS.resolve_unsure_lm(cn, noisy_by[kind], sc, margin=a.pho_margin)
        sc.close()
    llm = dp.LLM(a.llm, a.model, a.cache, max_model_len=a.max_model_len, tp=a.tp)

    def ok(pred, gold):
        return tokenize(pred) == tokenize(gold)

    def digits_kept(pred, raw):
        return re.findall(r'\d+', pred) == re.findall(r'\d+', raw)

    print(f"{'kiểu':10s} | {'ngu_canh':>8s} | {'llm_chon':>8s} | {'pho_cham':>8s} | {'ca_hai':>8s} | {'llm_viet_lai':>12s} | {'giữ số (viết lại)':>17s} | số chỗ hỏi LLM")
    for kind in kinds:
        noisy = noisy_by[kind]
        base = [cn.normalize(x)['text'] for x in noisy]
        chosen, log = resolve_unsure(cn, noisy, lambda msgs: llm.generate_short(msgs))
        p_ok = both_ok = None
        if kind in pho:
            import vn_lm_scorer as VS
            p_res, p_log = pho[kind]
            p_ok = 100 * sum(ok(r['text'], g) for r, g in zip(p_res, test)) / len(test)
            forced = VS.combine(p_log, log)
            both = [cn.normalize(x, forced=forced[i])['text'] if i in forced else cn.normalize(x)['text'] for i, x in enumerate(noisy)]
            both_ok = 100 * sum(ok(p, g) for p, g in zip(both, test)) / len(test)
        chosen = [r['text'] for r in chosen]
        r_ok = r_dig = None
        if not a.no_rewrite:
            msgs = [[{"role": "system", "content": REWRITE_SYS}, {"role": "user", "content": x}] for x in noisy]
            rew = [o.strip().split('\n')[0] for o in llm.generate_short(msgs, max_tokens=160)]
            r_ok = 100 * sum(ok(p, g) for p, g in zip(rew, test)) / len(test)
            r_dig = 100 * sum(digits_kept(p, x) for p, x in zip(rew, noisy)) / len(test)
        b_ok = 100 * sum(ok(p, g) for p, g in zip(base, test)) / len(test)
        c_ok = 100 * sum(ok(p, g) for p, g in zip(chosen, test)) / len(test)
        rs = f"{r_ok:12.1f}" if r_ok is not None else f"{'-':>12s}"
        ds = f"{r_dig:17.1f}" if r_dig is not None else f"{'-':>17s}"
        ps = f"{p_ok:8.1f}" if p_ok is not None else f"{'-':>8s}"
        bs = f"{both_ok:8.1f}" if both_ok is not None else f"{'-':>8s}"
        print(f"{kind:10s} | {b_ok:8.1f} | {c_ok:8.1f} | {ps} | {bs} | {rs} | {ds} | {len(log)}", flush=True)


if __name__ == '__main__':
    main()
