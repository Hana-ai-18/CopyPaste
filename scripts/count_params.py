"""
Đếm TỔNG tham số mọi model trong image (luật 2: ≤ 15B, tính MỌI model dùng lúc suy luận, kể cả không nạp cùng lúc,
MoE tính đủ, encoder ảnh của model đa phương thức cũng tính). Đọc header safetensors — không cần GPU, không nạp trọng số.

    python scripts/count_params.py /models            # mỗi thư mục con là một model
    python scripts/count_params.py /models/Qwen3.5-9B /models/bge-m3

Thoát mã 1 nếu tổng > 15e9. Ghi kết quả vào README nộp bài.
"""
import json
import os
import struct
import sys

LIMIT = 15_000_000_000


def st_params(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    tot = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        c = 1
        for d in v["shape"]:
            c *= d
        tot += c
    return tot


def model_params(d):
    files = []
    for root, _, fs in os.walk(d):
        files += [os.path.join(root, f) for f in fs if f.endswith(".safetensors")]
    if not files:
        return None, 0
    return sum(st_params(f) for f in files), len(files)


def main(args):
    dirs = []
    for a in args or ["/models"]:
        subs = [os.path.join(a, x) for x in sorted(os.listdir(a)) if os.path.isdir(os.path.join(a, x))]
        has_st = any(f.endswith(".safetensors") for f in os.listdir(a))
        dirs += [a] if has_st or not subs else subs
    total = 0
    for d in dirs:
        n, nf = model_params(d)
        if n is None:
            print(f"  {d}: KHÔNG có .safetensors (đếm tay! .bin/.gguf không được đọc)")
            continue
        total += n
        print(f"  {d}: {n / 1e9:.3f}B tham số ({nf} file)")
    print(f"TỔNG: {total / 1e9:.3f}B / giới hạn 15B -> {'OK' if total <= LIMIT else 'VƯỢT'}")
    return 0 if total <= LIMIT else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
