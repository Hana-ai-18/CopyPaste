"""Merge shard outputs into one submission.jsonl in the question order (missing id -> abstain).

    python scripts/merge_shards.py --questions /data/questions.jsonl --out /result/submission.jsonl shard0.jsonl shard1.jsonl
"""
import argparse
import json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None, help="questions.jsonl defining order + completeness (missing -> abstain). Omit to just concatenate the shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args(argv)
    recs = {}
    for f in a.shards:
        try:
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    recs[r["id"]] = r
        except FileNotFoundError:
            print(f"WARNING: {f} missing - its questions become abstain")
    ids = [json.loads(l)["id"] for l in open(a.questions, encoding="utf-8") if l.strip()] if a.questions else list(recs)
    with open(a.out, "w", encoding="utf-8") as fo:
        for i in ids:
            fo.write(json.dumps(recs.get(i, {"id": i, "type": "abstain"}), ensure_ascii=False) + "\n")
    print(f"merged {len(recs)}/{len(ids)} answers -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
