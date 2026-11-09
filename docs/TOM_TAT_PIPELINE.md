# Tóm tắt pipeline DataMate

## 6 bước

| # | Bước | Làm gì | Điểm hay |
|---|---|---|---|
| 1 | **Đọc luật** | Tách rulebook thành từng luật nhỏ | Mỗi luật thành một bài kiểm tra tự động, không sót dòng nào |
| 2 | **Làm sạch dữ liệu** | Mỗi shop đọc một lần; chuẩn hoá tiền, ngày, mã, đơn vị thành sổ chuẩn | Agent tự tìm hiểu file shop lạ → xử lý được ngành mới ở private |
| 3 | **Hiểu câu hỏi** | Sửa chữ sai theo ngữ cảnh cả câu, rồi model ghi ra câu hỏi đang hỏi gì | Model tự nêu chỗ có thể hiểu hai nghĩa; **rulebook chọn nghĩa đúng** |
| 4 | **Giải bằng hai cách độc lập** | Cách 1: máy tính theo công thức viết sẵn từ phần hiểu ở bước 3. Cách 2: model đọc câu gốc và tự viết code | Hai cách khác hẳn nhau nên ít khi cùng sai; cùng ra một số thì tin |
| 5 | **Kiểm tra và truy lỗi** | Soát theo luật (không tính đơn huỷ, đổi hộp → cái…). Hai cách lệch → tìm cách hiểu sai nào sinh ra con số kia | **Kiểm toán**: tự dò câu sai như cách người làm; hỏi model đúng một câu có/không ở chỗ phân vân |
| 6 | **Quyết định** | Chắc thì trả lời, không chắc thì bỏ trống | Ngưỡng tính theo công thức điểm (sai bị trừ); "không trả lời được" quyết định bằng luật |

Chung: 1 model ~8B, khoảng 2 lần gọi mỗi câu, dự kiến dưới 1,5 giờ. Mỗi ngày chạy kiểm toán trên train để ra bảng nguyên nhân sai và sửa loại lỗi nhiều nhất trước.

## Trạng thái (ngày 9/11, chiều)

| Phần | Trạng thái | Đã kiểm thế nào |
|---|---|---|
| Đọc luật: tách rulebook thành luật nguyên tử (train 83, public 105), chọn 6 luật liên quan nhất đưa vào prompt từng câu, rút tham số (khoảng Tết, ngày đầu tuần) từ rulebook được phát; ghi `rulebook_rules.json` | **Đã code** | Test: tách luật, chọn luật đúng mục/ngành, đổi tham số khi rulebook đổi |
| Lớp nghĩa: `__nhom_trang_thai` (4 nhóm + chưa rõ), `__sl_co_ban` (quy đổi theo bảng của shop), `__khach_le` | **Đã code** | `scripts/validate_semantics.py` so đáp án train: huỷ/tỉ lệ huỷ 170/170, số lượng quy đổi 10/10, khách ≥ N đơn 3/3 |
| **Bộ bảng chuẩn V** (cùng tên cột cho mọi shop): `don`, `dong`, `nhap`, `phieu_nhap`, `thu`, `hang` (có nhóm, hoạt chất, kê đơn, giá vốn), `ton`, `the_kho`, `tra`, `lo_nhap`, `ban_lo`, `quy_doi`, `chi_phi`; shop dịch vụ: `don` (mỗi dòng một phiếu), `viec`, `nhan_vien`. Mỗi cột ghi nguồn (`V_src`), cảnh báo (`V_notes`). Chọn cột theo **mô tả README trước** (đúng từng file), tên cột sau. Dòng voucher / dùng điểm / phí ship không vào doanh thu; phí phục vụ vào doanh thu nhưng không là mặt hàng; dòng số lượng âm = hàng trả; trạng thái là của ĐƠN (ưu tiên trả > huỷ); sổ không mã đơn dựng mã từ STT | **Đã code** | `scripts/validate_canonical.py` (chỉ train) — bảng dưới |
| **Lớp nghiệp vụ** `ledger.py` (~45 hàm, đã gắn V, có sẵn trong `solve()`): doanh thu kỳ / theo ngày / theo cột / theo mặt hàng – nhóm – hoạt chất – kê đơn – size; số lượng bán theo đơn vị hỏi; bán chạy; khách; số đơn, doanh thu bình quân đơn; tiền nhập, giá nhập bình quân, nhập trừ bán; công nợ (nợ từng đơn, tổng, số khách, top, tuổi nợ, quá N ngày, nợ thêm / trả nợ, thu nợ); tồn tại ngày X, tồn `[chẵn, lẻ]`, tồn mọi mặt hàng, **số mặt hàng hết hàng**, **giá trị hàng tồn kho**; lô, hạn dùng; lãi gộp (cả theo mặt hàng), chi phí, lãi ròng; doanh thu theo nhân viên, hoa hồng thợ | **Đã code** | Khớp đáp án train **1.039/1.040** câu kiểm (1 câu bán chạy biz_125 lệch 303/300, chưa rõ) |
| Quy đổi nhiều cấp (thùng → lốc → lon): bảng quy đổi + dòng quy đổi README, nhân dồn ≤ 5 cấp; không có thì **suy hệ số từ dòng bán** | **Đã code** | Test đơn vị; tồn `[chẵn, lẻ]`, giá nhập bình quân theo thùng khớp train |
| **Chống lỗi ngoài train/public**: (1) tự đối soát V không cần đáp án (mã đơn lặp, đơn không ngày, ngày sau ngày cuối dữ liệu, tổng dòng ≠ tiền hàng đầu đơn, tiền thu > tổng đơn, khoản thu không khớp mã đơn, lô ≠ tồn cuối, cột tên hàng là ngày/số) → ghi `KIỂM:` vào cảnh báo; (2) `scripts/fuzz_shops.py`: tạo shop giả từ shop train (đảo cột, đổi kiểu tên cột, tách file, chèn dòng tổng, khách họ Tống viết không dấu, bỏ mô tả cột README, xoá trạng thái) và kiểm doanh thu từng tháng không đổi | **Đã code** | Fuzz 8 shop × 7 biến đổi: đạt hết sau khi sửa các lỗi nó tìm ra |
| Đọc dữ liệu shop, làm sạch tiền/ngày/số lượng, thẻ shop | **Đã code** (v0, heuristic) | Selftest; chạy hết 30 shop train + 14 shop public không lỗi; `scripts/audit_data.py` soát trên train |
| Nhóm dòng (ô trống dòng tiếp nối), bỏ trùng cùng đơn, số lượng lẫn đơn vị | **Đã code** theo rulebook §8 | Đối chiếu đáp án train: số đơn huỷ, doanh thu quý, giá trị TB/đơn, tỉ lệ huỷ khớp |
| Sửa chữ theo ngữ cảnh (`vn_context.py`) | **Đã code** | Đo trên câu train bị làm bẩn (`scripts/eval_normalize.py`): câu sạch giữ nguyên 99,0%; 9 kiểu bẩn sửa đúng 92,7–97,7% câu |
| Bảng teencode mở rộng (`lexicon/teencode.tsv`, ~625 mục, nhiều nghĩa) + viết lệch ngoài bảng (teen j/z/w/f/k, VNI, vùng miền n/l s/x ch/tr d/gi/r hỏi/ngã) | **Đã code** | Như trên; 3.000 câu train: không đổi chữ số, không mất tên hàng |
| Bảng lỗi / nhầm lẫn (`lexicon/error_catalog.yaml`, 170 loại, 20 nhóm, có trường nguồn) + ma trận 20.000 tình huống (`lexicon/error_scenarios.tsv`) | **Đã lập**; kiểm toán / xưởng shop giả lập đọc các bảng này: thiết kế | Test định dạng, id không trùng; `check_compliance.py` |
| Mở rộng từ điển Viet22K (`lexicon/gpl/`, GPL v2): +20.000 dòng mỗi bộ sinh, 22.035 từ vào mô hình ngôn ngữ (đếm yếu) | **Đã code** | Câu sạch giữ 100%; vùng miền 94,5%; bỏ dấu 93,3%; teencode 96,0% |
| Bộ dữ liệu dạng lệch (`lexicon/teencode_variants.tsv`, 11.282 dòng, sinh bằng `scripts/build_teencode_dataset.py`) | **Đã sinh**; dạng ghép/viết tắt an toàn được tra thẳng | Đo từng dạng đứng một mình: ghép 96–97%, viết tắt an toàn 88% |
| Teencode giới trẻ + vùng miền: ~370 mục viết tay (`gioitre_vungmien_lamtay.tsv`) + 11.722 dòng sinh (`teencode_gioitre_vungmien.tsv`); đồng nghĩa vùng miền (heo/lợn…) chỉ thành gợi ý trong prompt | **Đã code** | Dạng an toàn đứng một mình: giới trẻ 94%, viết tắt 96,5%, dính liền 98%; câu sạch vẫn giữ 99,0% |
| LLM chọn trắc nghiệm ở chỗ sửa chữ chưa chắc (`--norm_llm main`) | **Đã code**, chưa đo với model thật | Chạy bằng model giả: không làm đổi kết quả; đo thật bằng `scripts/eval_norm_llm.py` trên Kaggle |
| Thời gian (tuần, tháng, quý, Tết…), cảnh báo dễ sai | **Đã code** (v0) | Selftest |
| Model viết code (đường A câu đã sửa chữ, đường B câu gốc), sandbox, sửa lỗi chạy | **Đã code** | Chỉ kiểm bằng model giả; **chưa chạy với model thật vì chưa có GPU** |
| **Bước 3 – đặc tả** (`compiler.py`): luật chữ nhận intent + kỳ (cả danh sách tháng, khoảng tháng) + mặt hàng / nhóm / hoạt chất / kê đơn + đơn vị + N; không nhận ra thì **LLM viết đặc tả JSON** theo danh mục 30 intent (kèm `doc_khac` = các cách hiểu khác), đặc tả được kiểm (ngày hợp lệ, mặt hàng có thật) | **Đã code** | Luật chữ: train 3.000 câu (có sửa chữ) nhận 60%; LLM viết đặc tả: chưa đo (cần GPU) |
| **Bước 4 – đường 1, máy tính theo công thức**: đặc tả → hàm `ledger` → đáp án đúng format §9 (làm tròn, thứ tự hoà theo mã, list/table); §7 "không trả lời được" khi kỳ hoàn toàn ngoài dữ liệu / shop không ghi loại thông tin (giới tính, địa chỉ, đánh giá chủ quan, lý do huỷ…) | **Đã code** | `scripts/eval_compiler.py`: intent tin cậy cao **1.762 đúng / 1 sai** (precision 99,94%), PA chỉ riêng đường này **0,587** |
| **Bước 5 – kiểm toán + hỏi đúng chỗ phân vân**: máy tính và code lệch → thử các CÁCH HIỂU SAI hay gặp (tính cả đơn trả/huỷ, mẫu số tỉ lệ huỷ, không quy đổi đơn vị, gồm cả N, đảo kê đơn) xem code model hiểu theo cách nào (ghi log); đặc tả tin cậy thấp / do LLM viết mà lệch → hỏi model **một câu trắc nghiệm A/B/C** theo luật liên quan | **Đã code** (5 cách hiểu sai) | Test đơn vị; hỏi A/B cần GPU để đo |
| **Bước 6 – quyết định** (`decide_all`): luật + tin cậy cao → lấy máy tính; tin cậy thấp → chỉ lấy khi khớp code hoặc model chọn A; không có máy tính → A/B đồng thuận; "không trả lời được" của máy tính chỉ nhận khi đặc tả bằng luật hoặc hai đường code cũng nói vậy | **Đã code** | Chạy model giả trên train: PA 0,587 (chỉ phần máy tính; phần còn lại bỏ trống) |
| **Agent tự soát kết quả** (vòng quan sát sau khi code chạy được): sai kiểu so với đặc tả, số đếm âm / lẻ, % vô lý, tỉ lệ ngoài 0–100, rỗng, ngày ngoài dữ liệu → cho model một vòng xem lại (không ép theo máy tính) | **Đã code** (`--sanity_rounds 1`) | Test đơn vị |
| Agent cấu hình shop (LLM đọc shop lạ) | **Thay bằng** bộ bảng chuẩn V đọc theo mô tả README + tự đối soát + fuzz (tất định, tái lập được) | Xem các dòng V ở trên |
| **Kiểm metamorphic** cho code model: chạy lại `solve()` trên MỌI bảng đã đảo thứ tự dòng; kết quả đổi = code dựa vào thứ tự dòng (iloc[0], head…) → agent cho model xem lại | **Đã code** (`--metamorphic on`) | Test: code `iloc[0]` bị bắt, code có sắp xếp không bị bắt |
| **Luật thành bài kiểm tra tự động** (lúc chạy, cho code model): chèn một đơn GIẢ chép từ đơn thật vào mọi bảng gốc có mã đơn + bảng chuẩn V — (1) đơn có ngày **sau hôm nay** (luật: chưa xảy ra), (2) đơn **huỷ** trong kỳ hỏi (luật §1: không tính) — chạy lại `solve()`; đáp án đổi = code vi phạm luật → agent nhận thông báo "VI PHẠM LUẬT …" và sửa | **Đã code** (`--rule_tests on`) | Shop thật biz_101/111/123: code quên lọc huỷ / quên giới hạn hôm nay bị bắt; code đúng và hàm ledger không bị báo nhầm |
| Kiểm hàm nghiệp vụ theo luật bằng đáp án train | `validate_canonical.py` (1.039/1.040) + `eval_compiler.py` | — |
| **Bỏ trống theo XÁC SUẤT ĐÚNG**: `scripts/fit_abstain.py` fit hồi quy logistic trên log chạy TRAIN (đặc trưng: nhãn quyết định, nguồn đặc tả, số đường khớp, đã sửa lỗi, cờ tự soát / vi phạm luật, kiểu đáp án, số chữ đã sửa), kiểm chéo theo shop để chọn ngưỡng; lúc chạy `--abstain_model` tính p(đúng) từng câu, p ≤ ngưỡng (≥ 0,5) thì bỏ trống | **Đã code** | Cần log chạy train với model thật |
| **Bỏ trống theo PA (bảng)**: `scripts/calibrate_decisions.py` đo tỉ lệ đúng của từng nhãn quyết định (và nhãn|intent) trên log chạy TRAIN; nhãn nào p ≤ 0,5 (đủ 15 mẫu) thì khi chạy test đổi thành bỏ trống (`--calibration`) | **Đã code** | Cần một lần chạy train với model thật để có số |
| **Đo & giảm lượt gọi model**: đếm lượt gọi dài / ngắn mỗi câu vào `run_summary.json`; **bỏ đường B** khi máy tính theo luật đã chắc (`--b_when_sure off`, ~60% câu train); vòng tự soát / bài kiểm luật chỉ gọi thêm khi phát hiện vấn đề | **Đã code** | Số lượt gọi thật: đọc `run_summary.json` sau lần chạy Kaggle |
| **Sửa lỗi có chẩn đoán**: vòng sửa gồm prompt gốc + code lỗi + thông báo lỗi **+ CHẨN ĐOÁN** (cột không có → cột thật + tên gần nhất; lỗi kiểu ngày; chia 0 → §7; quá giờ → bỏ vòng lặp; sai format → ví dụ đúng kiểu) | **Đã code** | Test |
| **Ngân sách thời gian thông minh**: dự báo theo tốc độ thật; vượt 90% ngân sách thì tắt dần bước phụ (tự soát → đảo dòng → hỏi A/B → đường B → sửa lỗi) thay vì bỏ trống cả loạt câu cuối | **Đã code** | Test đơn vị |
| **Bước 3 tổng quát hơn**: kho 1.567 đặc tả train đã kiểm (`lexicon/spec_bank.jsonl`) + truy hồi câu giống (bge-m3 nếu có, không thì TF-IDF ký tự) làm 4 ví dụ mẫu cho LLM viết đặc tả | **Đã code** | Đoán đúng dạng câu bằng câu train của SHOP KHÁC: 92,0% (TF-IDF); câu viết bẩn (teencode, bỏ dấu, telex, vùng miền…): đường máy tính vẫn đúng 97,6–100%, **sai 0%** (không nhận ra thì nhường đường code) |
| **Tận dụng hết các bảng tsv** khi sửa chữ: thêm (a) mục NHIỀU NGHĨA (an_toan=0, ~9.400 dạng) làm ứng viên cho chữ lạ, (b) ~5.000 CỤM nhiều chữ an toàn ('biên lợi nhuan' → 'biên lợi nhuận') ưu tiên khi cả cụm xuất hiện, (c) viết tắt có dấu chấm chưa có trong bảng ('d.thu' → ghép 'dthu' → doanh thu). Đo A/B, bỏ phần làm hại (nghĩa nhiều chữ từ mục mơ hồ: 'dh' → 'đơn huỷ') | **Đã code** | Câu teencode sửa đúng cả câu 95,2% → **99,0%**; các kiểu khác giữ hoặc tăng; câu sạch 100% |
| **Soát nhận nhầm dạng câu bằng 20.000 câu `error_scenarios.tsv`** (`eval_scenarios.py`, 5 kiểu viết) | **Đã code** | 4.000 câu: nhận nhầm 0,0–0,3% → sửa luật chặn câu hỏi hai đại lượng → 0 |
| **PhoGPT-4B chấm câu sửa chữ** (`vn_lm_scorer.py`): chỉ chọn trong phương án có sẵn, chênh < 1 nat giữ nguyên; chế độ `both` chỉ đổi khi PhoGPT và Qwen cùng chọn; chạy pha riêng trước vLLM | **Đã code** (`--norm_llm pho/both --pho_model`) | Test logic bằng model giả; **chưa đo thật** (máy này không tải được model) — đo bằng `eval_norm_llm.py --pho_model` |

**Không thể khẳng định "không còn lỗi"** cho phần chưa chạy với model thật. Việc đầu tiên: chạy notebook Kaggle để có PA mốc trên train, sau đó code các phần thiết kế theo thứ tự P0 trong v3 mục 9.

### Kết quả kiểm trên train (scripts/validate_canonical.py, 08/11 tối)

| Nhóm | Khớp | Nhóm | Khớp |
|---|---|---|---|
| Doanh thu tháng | 55/55 | Doanh thu theo mặt hàng | 42/42 |
| Doanh thu bình quân đơn | 32/32 | Doanh thu theo nhóm (cả tỉ trọng) | 78/78 |
| Ngày bán nhiều tiền nhất | 45/45 | Hoạt chất / thuốc kê đơn | 30/30 · 17/17 |
| Số lượng bán theo đơn vị hỏi | 133/133 | Bán chạy nhất | 18/19 |
| Khách mua nhiều tiền nhất | 6/6 | Tiền nhập / NCC | 30/30 · 2/2 |
| Giá nhập bình quân | 63/63 | Nhập trừ bán | 4/4 |
| Công nợ (6 dạng) | 116/116 | Tồn kho cuối ngày | 47/47 |
| Giá trị tồn kho / hết hàng | 6/6 · 1/1 | Lô, hạn dùng | 9/9 |
| Lãi gộp / theo mặt hàng / lãi ròng | 6/6 · 3/3 · 2/2 | Huỷ, quy đổi, khách (validate_semantics) | 170/170 · 10/10 · 3/3 |

### Lỗi thật tìm ra nhờ đối soát + fuzz (đã sửa)
- Dòng khách họ **Tống** viết không dấu ("Tong Huong Chau") bị xoá vì tưởng là dòng "Tổng" → mất đơn, lệch nợ/lãi.
- Dòng "Tổng ngày 01/03" nằm ở cột thứ 5 không bị bỏ (chỉ soát 3 cột đầu).
- File tách `_1`, `_2` có **thứ tự / kiểu tên cột khác nhau** không được gộp → mất nửa dữ liệu.
- pandas 2 và 3 xử lý ô trống khác nhau khi đổi sang chữ (`'nan'` vs NaN) → chuẩn hoá bằng `sid()` để kết quả như nhau trên Kaggle/Docker.
- Ghi chú trạng thái chỉ ở dòng đầu đơn → các dòng sau bị coi là hoàn thành; giờ lấy trạng thái theo ĐƠN.
- Cột mã hàng bị lấy làm tên hàng; cột "Giá bán theo đơn vị quy đổi" bị nhận là cột kê đơn; cột nhóm hằng "Hàng hóa".
