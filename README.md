# DataMate pipeline (Zalo AI Challenge 2026)

Hỏi đáp số liệu kinh doanh trên dữ liệu shop: **1 LLM ~7–8B hiểu câu hỏi, code tính toán, kiểm chứng trước khi trả lời**.
Tổng tham số dưới 15B, chạy offline.

- Bản code hiện tại: **v0** (`datamate_pipeline.py`). Đã chạy được end-to-end và có selftest.
- Thiết kế mục tiêu: **v3** (`docs/DataMate_Pipeline_v3.md`), "đặc tả trước, kiểm bằng quan hệ", dựa trên các paper CHASE-SQL, CHESS, Arctic-Text2SQL-R1, OmniSQL, AMBROSIA, Chain-of-Table, LEVER, ReFoRCE.

## Cấu trúc

```
datamate_pipeline.py      # pipeline 1 file: đọc/làm sạch shop, thời gian, prompt, sandbox, 2 đường + đồng thuận, chấm điểm
retriever.py              # truy hồi câu train giống nhau (bge-m3 / TF-IDF) làm ví dụ mẫu cho LLM viết đặc tả
vn_lm_scorer.py           # PhoGPT chấm câu ứng viên ở chỗ sửa chữ chưa chắc (không viết lại câu)
rule_delta.py             # so rulebook đang chạy với dấu vân tay rulebook train: luật đổi/bỏ -> chặn compiler; thuật ngữ mới -> ép luật vào prompt
compiler.py               # bước 3 đặc tả (luật chữ / LLM viết JSON) + bước 4 đường 1 máy tính theo công thức
ledger.py                 # ~45 hàm nghiệp vụ trên bộ bảng chuẩn V (bán hàng, công nợ, kho, lô, giá nhập, lãi, nhân viên) — có sẵn trong solve()
vn_normalize.py           # sửa từng chữ: kéo dài (coasaaaa → có), Telex, teencode, bảo vệ thực thể
vn_context.py             # sửa chữ theo ngữ cảnh cả câu: ứng viên + mô hình 2-gram + Viterbi (khôi phục dấu)
lexicon/
  teencode.tsv            # ~625 viết tắt/teencode/tiếng Anh/từ địa phương, nhiều nghĩa (ngữ cảnh chọn); sửa file này, không sửa code
  teencode_variants.tsv   # 11.282 dòng dạng lệch -> từ chuẩn (sinh tự động; dạng an toàn được tra thẳng)
  gioitre_vungmien_lamtay.tsv  # ~370 teencode giới trẻ / MXH / từ Bắc-Trung-Nam viết tay + 32 cặp đồng nghĩa vùng miền (heo/lợn, ly/cốc…)
  teencode_gioitre_vungmien.tsv  # 11.722 dòng sinh tự động: kiểu viết giới trẻ, phát âm vùng miền, viết tắt, dính liền
  tu_thong_dung.txt       # ~1.200 từ giao tiếp/mua bán viết tay: nguồn từ chuẩn để sinh bộ trên
  error_scenarios.tsv     # 20.000 tình huống lỗi cụ thể (loại lỗi × số đo × kỳ × đối tượng × kiểu viết), câu ví dụ bịa
  gpl/                    # (GPL v2, tách riêng) mở rộng từ từ điển Viet22K: +20.000 dòng cho mỗi bộ sinh + 22.035 từ cho mô hình ngôn ngữ
  error_catalog.yaml      # 170 loại lỗi/nhầm lẫn: dấu hiệu, cách tái tạo, luật phân xử, cách chặn, đã code hay chưa
notebooks/
  datamate_kaggle.ipynb   # clone repo → selftest → đo nhanh → đánh giá TRAIN → chạy public
scripts/
  predict.sh              # điểm vào Docker: /data → /result (+ /result/logs)
  eval_normalize.py       # đo bộ sửa chữ trên câu train bị làm bẩn có kiểm soát
  audit_data.py           # soát làm sạch dữ liệu shop + sửa chữ câu hỏi (chỉ train) → báo cáo .md
  build_teencode_dataset.py # sinh lexicon/teencode_variants.tsv từ từ vựng chuẩn (train + cụm nghiệp vụ)
  build_gioitre_dataset.py  # sinh lexicon/teencode_gioitre_vungmien.tsv (thêm --with_shop_terms để lấy chữ trong tên hàng train)
  eval_norm_llm.py        # (GPU) đo LLM chọn trắc nghiệm / LLM viết lại / không LLM khi sửa chữ
  build_wordlist_variants.py # sinh lexicon/gpl/ từ Viet22K/Viet74K (git clone duyet/vietnamese-wordlist)
  build_error_scenarios.py   # sinh lexicon/error_scenarios.tsv từ error_catalog.yaml
  validate_semantics.py   # so lớp nghĩa (trạng thái, quy đổi, khách lẻ) với đáp án TRAIN
  validate_canonical.py   # so bộ bảng chuẩn V + ledger (doanh thu, bán hàng, nhập, công nợ, tồn kho, lô, lãi) với đáp án TRAIN
  build_spec_bank.py      # dựng lexicon/spec_bank.jsonl từ train + đo đoán dạng câu bằng shop khác
  eval_spec_noise.py      # độ bền đường máy tính khi câu viết bẩn (chỉ train)
  eval_llm_spec.py        # (GPU) đo LLM viết đặc tả, có / không ví dụ mẫu
  fit_abstain.py          # mô hình xác suất đúng (logistic) cho bỏ trống theo PA, từ log chạy train
  eval_ask_choice.py      # (GPU) đo câu hỏi trắc nghiệm A/B/C chọn đúng cách tính
  eval_scenarios.py       # soát nhận nhầm dạng câu trên 20.000 câu error_scenarios.tsv
  calibrate_decisions.py  # hiệu chỉnh bỏ trống theo PA từ log chạy train
  eval_compiler.py        # đo đường máy tính trên TRAIN (độ phủ, precision, PA) — không cần GPU; --rulebook <khác> mô phỏng Rule-Delta
  eval_period_variants.py # viết lại kỳ trong câu train sang dạng tắt (T5/26, 05/2026, Q2/25…) và so đáp án
  eval_vilexnorm.py       # đo bộ sửa chữ trên ViLexNorm (CC BY-NC-SA 4.0, chỉ để đo, không gộp vào lexicon)
  count_params.py         # tổng tham số mọi model trong thư mục (exit 1 nếu > 15B) — chạy trong docker build
  fuzz_shops.py           # shop giả biến đổi từ shop train (đảo cột, tách file, dòng tổng…) — kiểm độ bền, không cần đáp án
  check_compliance.py     # soát: không ID câu public, không logic theo ID, không file đáp án; nhắc giấy phép
tests/
  test_selftest.py        # pytest
docs/
  RISKS.md                    # SỔ RỦI RO: R1–R26, đã xử lý / còn mở (đọc trước khi nộp)
  TOM_TAT_PIPELINE.md         # tóm tắt 6 bước + trạng thái đã code / còn thiết kế (đọc trước)
  DataMate_Pipeline_v3.md     # thiết kế mới
  DataMate_Pipeline_v2.md     # thiết kế trước: luật, dữ liệu, 16 cách hiểu nhầm, 30 hàng rào
  DataMate_Pipeline_Design.md # bản chi tiết đầu tiên (phụ lục)
```

## Cài đặt

```bash
pip install -r requirements.txt          # máy có GPU
# chỉ CPU (selftest, chấm điểm, xem thẻ shop): pip install pandas numpy openpyxl python-calamine pytest
```

## Lệnh

```bash
# 1) Kiểm tra (không cần GPU)
python datamate_pipeline.py selftest
python datamate_pipeline.py selftest --split_dir /path/train        # thêm kiểm tra trên train, gold-vs-gold = 1.0
DATAMATE_TRAIN=/path/train pytest -q

# 2) Xem thẻ dữ liệu model sẽ đọc
python datamate_pipeline.py card --split_dir /path/train --biz biz_101

# 3) Chạy
python datamate_pipeline.py run --split_dir /path/train --out out_train \
  --llm vllm --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --n_paths 2 --batch 48 --cache out_train/cache \
  --biz biz_103,biz_108          # bỏ --biz để chạy hết; --limit N để đo nhanh

# 4) Chấm (CHỈ trên train, vì chỉ train có đáp án)
python datamate_pipeline.py score --split_dir /path/train --pred out_train/submission.jsonl --only_pred
```

Thêm `--lm_texts /path/train/questions.jsonl` để bộ sửa chữ học thêm từ câu hỏi train (chỉ train). `--no_ctx_norm` để tắt sửa theo ngữ cảnh. `--norm_llm off` để không hỏi LLM ở chỗ sửa chữ chưa chắc.

Đầu ra của `run`: `submission.jsonl`, `rulebook_rules.json` (luật đã tách + tham số rút từ rulebook), `logs/<id>.json` (câu đã chuẩn hoá, cảnh báo, code từng đường, quyết định), `run_summary.json`.
Chạy lại cùng lệnh thì **tiếp tục** từ câu chưa làm; thêm `--fresh` để chạy lại từ đầu.

## Kaggle (Run All → file nộp public)

1. Đẩy repo lên GitHub (`Hana-ai-18/CopyPaste`). Code ở gốc repo hay trong thư mục con đều được (notebook tự tìm `datamate_pipeline.py`).
2. Repo private: Kaggle *Add-ons → Secrets* → `GITHUB_TOKEN` (token chỉ đọc). Repo public: bỏ qua.
3. Notebook `notebooks/datamate_kaggle.ipynb`: *Accelerator = GPU T4 x2*, *Internet = On*, *Add Input* dataset `gmnguynhng/copypaste`.
4. **Run All** → `/kaggle/working/submission.jsonl` (+ `outputs.zip` log).

Notebook tự: giải nén dữ liệu, tìm train (có `answers.jsonl`) / public, clone code, chọn model + bản vLLM chạy được
(`MODEL` mặc định Qwen3.5-9B với vLLM 0.19.1/0.17.1 → lùi về Qwen2.5-Coder-7B-AWQ với vLLM 0.10.1.1/0.8.5 → backend `hf`
→ không LLM), nạp thử model thật bằng lớp `LLM` của pipeline và đòi trả lời đúng một phép cộng, dựng
`spec_bank` từ train, đo nhanh trên train, chạy public với ngân sách = thời gian còn lại của phiên, kiểm định dạng và ghi file nộp.
Sửa notebook: sửa `notebooks/build_kaggle_notebook.py` rồi chạy `python notebooks/build_kaggle_notebook.py`.

## Docker (bản nộp)

`scripts/predict.sh /data /result`. Trọng số model phải đóng sẵn trong image (không có mạng). Đường dẫn model đặt qua biến `MODEL`.

```bash
python scripts/build_spec_bank.py --train /path/train          # tạo lexicon/spec_bank.jsonl (câu train -> KHÔNG commit; build sẽ dừng nếu thiếu)
python scripts/count_params.py models                          # ≤ 15B tổng
docker build -f docker/Dockerfile --build-arg BASE_IMAGE=<image theo hướng dẫn BTC, CUDA ≥ 12.8, có sm_120> -t datamate .
```

Hành vi an toàn khi chạy (xem `docs/RISKS.md`):
- **Pha tạm** trước khi nạp model: đường máy tính trả lời các câu chắc chắn và ghi `submission.jsonl` đủ mọi id ngay. Model nạp lỗi / hết giờ / bị ngắt → vẫn còn file này.
- `submission.jsonl` luôn được ghi lại toàn bộ, nguyên tử (tmp + fsync + rename); `answers_final.jsonl` là nguồn chạy tiếp.
- Ngân sách `TIME_BUDGET` tính từ lúc tiến trình bắt đầu; `SHED=off` (không bỏ bước theo giờ đồng hồ → tái lập).

## Giấy phép dữ liệu phụ

- `lexicon/gpl/` phái sinh từ Viet22K/Viet74K (Hồ Ngọc Đức, https://github.com/duyet/vietnamese-wordlist), **GNU GPL v2**. Nếu nhóm không muốn ràng buộc GPL, xoá thư mục này: pipeline tự bỏ qua (mất phần mở rộng từ điển, các phần khác giữ nguyên). Nên hỏi ban tổ chức nếu chưa chắc.
- Phần còn lại của `lexicon/` do nhóm tự viết / sinh từ train.

## Tuân thủ (đọc trước khi commit)

- Chạy `python scripts/check_compliance.py --public /path/public_test` trước mỗi lần commit/nộp.

- **Không** đưa vào repo: dữ liệu cuộc thi, mọi file đáp án, file nộp. `.gitignore` đã chặn `*.jsonl`, `*.zip`, `data/`, `out*/`.
- Không dùng dữ liệu hay đáp án public/private test để huấn luyện hoặc chỉnh hệ thống (quy định 6). Lệnh `score` chỉ dùng cho train và sẽ cảnh báo nếu đường dẫn đáp án chứa `public` hoặc `private`.
- Không viết logic theo ID câu hỏi hay ID shop.
- Tổng tham số dưới 15B. Model phải phát hành trước 01/10/2026 và có giấy phép cho phép dùng; kiểm lại trước khi nộp.
- Tái lập: decode greedy, seed cố định, cache LLM, ghi commit đang chạy vào log.
