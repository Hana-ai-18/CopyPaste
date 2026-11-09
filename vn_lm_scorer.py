"""
PhoGPT (3,7B, tiếng Việt) làm BỘ CHẤM CÂU cho những chỗ sửa chữ còn phân vân.

Nguyên tắc an toàn: PhoGPT KHÔNG viết lại câu. Bộ sửa chữ (luật + mô hình 3-gram) sinh sẵn vài phương án cho chữ
chưa chắc; PhoGPT chỉ tính xác suất của TỪNG CÂU ỨNG VIÊN và chọn câu tự nhiên nhất. Vì vậy không thể bịa số,
đổi tên hàng hay thêm chữ lạ. Chênh điểm nhỏ hơn `margin` thì giữ lựa chọn cũ.

Chạy TRƯỚC khi bật vLLM (pha sửa chữ riêng), xong giải phóng GPU — hai model không phải chia bộ nhớ.
Tổng tham số: Qwen2.5-7B (7,6B) + PhoGPT-4B (3,7B) + bge-m3 (0,57B) ≈ 11,9B ≤ 15B.
"""
import math


class LMScorer:
    def __init__(self, path, device=None, dtype="bfloat16", batch=16):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if hasattr(cfg, "init_device"):
            cfg.init_device = self.device
        dt = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=dt, trust_remote_code=True).to(self.device).eval()
        self.batch = batch
        self.cache = {}

    def logprob(self, sentences):
        """Tổng log-xác suất của từng câu (nat). Có cache (cùng câu chấm một lần)."""
        torch = self.torch
        todo = [s for s in dict.fromkeys(sentences) if s not in self.cache]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
            for s, v in zip(chunk, tok_lp.sum(-1).tolist()):
                self.cache[s] = v
        return [self.cache[s] for s in sentences]

    def close(self):
        try:
            del self.model
            self.torch.cuda.empty_cache()
        except Exception:
            pass


def resolve_unsure_lm(cn, raws, scorer, max_per_sentence=3, margin=1.0):
    """Như vn_context.resolve_unsure nhưng chọn bằng PhoGPT: thử từng phương án của chữ chưa chắc (lần lượt, giữ các
    lựa chọn trước), chấm cả câu, lấy câu có log-xác suất cao nhất nếu hơn lựa chọn hiện tại ít nhất `margin` nat."""
    out, log = [], []
    for qi, raw in enumerate(raws):
        r = cn.normalize(raw)
        forced = {}
        for slot, word, opts in r["unsure"][:max_per_sentence]:
            if len(opts) < 2:
                continue
            cands = [cn.normalize(raw, forced={**forced, slot: o})["text"] for o in opts]
            sc = scorer.logprob(cands)
            best = max(range(len(opts)), key=lambda k: sc[k])
            cur = 0                                            # phương án 0 = lựa chọn hiện tại của bộ sửa chữ
            pick = best if sc[best] - sc[cur] >= margin else cur
            forced[slot] = opts[pick]
            log.append({"q": qi, "slot": slot, "chu": word, "phuong_an": opts, "diem": [round(x, 2) for x in sc], "chon": opts[pick], "nguon": "phogpt"})
        out.append(cn.normalize(raw, forced=forced) if forced else r)
    return out, log


def combine(pho_log, qwen_log):
    """Chế độ 'both': chỉ ĐỔI chữ khi PhoGPT và Qwen chọn GIỐNG nhau; khác nhau thì giữ lựa chọn của bộ sửa chữ.
    Trả {chỉ số câu: {slot: phương án}}."""
    q = {(x["q"], x["slot"]): x.get("chon") for x in qwen_log}
    forced = {}
    for x in pho_log:
        c = q.get((x["q"], x["slot"]))
        if c is not None and c == x["chon"]:
            forced.setdefault(x["q"], {})[x["slot"]] = c
    return forced


def nll_per_token(v, n):
    return -v / max(1, n) if n else math.inf
