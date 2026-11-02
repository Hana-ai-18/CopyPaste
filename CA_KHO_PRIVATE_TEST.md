# Các ca khó có thể gặp ở private test và cách hệ thống xử lý

Trạng thái: **[T]** có test tự động · **[G]** gợi ý/quy tắc trong prompt (chưa đo được trên model thật) · **[D]** tự phát hiện từ dữ liệu · **[?]** chưa giải quyết được / chỉ giảm thiểu.
Chưa có ca nào ở đây được đo trên GPU: mọi cải thiện về PA là *kỳ vọng*, không phải kết quả.

## A. Định dạng dữ liệu của shop lạ
| Ca | Xử lý |
|---|---|
| CSV `;` / tab / BOM / UTF-16 / dòng trống | loader tự dò [T] |
| Excel có dòng tiêu đề/tiêu đề lệch dòng | tự tìm dòng header [T] |
| Dữ liệu tách theo sheet tháng | gộp bảng cùng schema [T] |
| Dòng "Tổng cộng", "Ghi chú:", header lặp giữa dữ liệu | loại bỏ [T] |
| Chữ Việt dạng NFD, khoảng trắng không ngắt/zero-width, thừa dấu cách cuối, ô lỗi `#N/A` | chuẩn hoá tên cột + từng ô [T] |
| JSON / JSONL / parquet | đọc được [T] |
| SQLite nhiều bảng (17 bảng) | đọc được; profile luôn liệt kê đủ bảng kể cả khi schema rất rộng [T] |
| Ngày: ISO, d/m/Y, `2026/3/4`, `1 thg 1 2025`, serial Excel (số hoặc chữ) | `parse_dates` [T] |
| So sánh cột ngày với `datetime.date` (pandas im lặng ra toàn False) | vá ở sandbox [T] |
| Số lưu dạng chữ: `37,000.0000`, `1.250.000đ`, `12,5`, `1 ly`, `50k`, `1,5tr` | `to_num`; profile báo cột nào là số-dạng-chữ [T][D] |
| Quy đổi đơn vị chỉ có trong README (`1 thùng = 24 lon`) | parse thành bảng `quy_doi_readme` [T] |
| Cột đơn hàng lặp trên mỗi dòng hàng (cộng nhầm nhiều lần) | profile liệt kê cột cấp đơn / cấp dòng tính từ dữ liệu [D] |
| Trạng thái viết ~50 kiểu | `norm_status` + profile in bảng ánh xạ thực tế của shop [T][D] |
| Khách lẻ viết nhiều kiểu (`kl`, `Khách lẻ`, rỗng…) | `anon_customer` + profile báo tỷ lệ [T][D] |
| Dòng trùng hệt, số lượng âm (trả hàng) | profile cảnh báo, không tự xoá [D] |
| File lạ hoàn toàn (xlsb, ods, ảnh, PDF) | **[?]** bỏ qua, ghi cảnh báo |

## B. Câu hỏi
| Ca | Xử lý |
|---|---|
| Viết tắt/telex/sai chính tả (`hnay`, `bnhiu`, `tawng/giamr`, `nượt`, `xửa`) | bảng viết tắt trong prompt [G]; LLM tự suy luận phần còn lại **[?]** |
| Mốc thời gian: hôm qua, tuần trước, tuần này đến nay, quý, cùng kỳ năm trước, Tết âm lịch, "7 ngày kể từ ngày X so với 7 ngày trước đó" | hàm helper có test [T] |
| "Thứ Tư", "ca sáng/chiều" | LLM tự lọc theo giờ/ca; profile báo cột giờ/ca có tồn tại không [D][G] |
| Tồn kho cuối ngày, giá trị tồn theo giá vốn, tuổi nợ 0-7/8-15/>15 ngày | cần suy từ nhập − bán hoặc sổ kho: **[?]** khó nhất, ~5% số câu train |
| Câu hỏi giả định ("tăng giá lên 23.000đ… so 7 ngày trước đó") | LLM tự hiểu là so sánh hai khoảng ngày **[?]** |
| Khách/mặt hàng tên gần giống, sai dấu | `fuzzy_find`, `contains_vi` [T] |
| Hoà điểm khi xếp hạng | `rank_top` tie-break theo mã [T] |

### B2. Viết tắt, gõ Telex, không dấu (module `datamate/qnorm.py`, chạy trước khi gọi model)
Đọc 4.000 câu hỏi thấy ba kiểu nhiễu chính: (1) viết tắt (`dthu`, `bnhiu`, `hnay`, `trc`, `mik`, `z`), (2) **gõ Telex chưa đổi** (`tieenf`=tiền, `dduwowcj`=được, `toongr`=tổng, `thangs`=tháng), (3) không dấu / sai dấu.
| Việc | Cách làm | Kết quả đo |
|---|---|---|
| Giải mã Telex | tách (nền, dấu thanh), kiểm tra bằng từ vựng có trong rulebook/README/câu hỏi/profile của shop, và nếu từ lạ thì kiểm tra bằng **luật âm tiết tiếng Việt** (phụ âm đầu + vần hợp lệ + dấu hợp lệ) | khôi phục đúng 99,9% từ trên câu train; **100% (368/368) trên từ bị giấu khỏi từ vựng** |
| Từ viết tắt | bảng viết tắt không mơ hồ (`dthu`→doanh thu, `bn`→bao nhiêu…). Bỏ những từ mơ hồ (`cn`, `km`, `dt`) | có test |
| `Th2/2026`, `T11/2025`, `q3` | đổi thành "tháng 2/2026", "quý 3"; `T2` đứng một mình giữ nguyên (thứ Hai) | có test |
| Số tiền `500k`, `1tr5`, `2 triệu`, `2 củ` | thêm dòng "Amounts: 500k = 500000" cho model | có test |
| Không đụng vào | tên hàng/thương hiệu (`Heineken max`, `coffee`), `50k`, `10ml`, `5kg` | có test |
| Dùng bản đã chuẩn hoá để | tìm ví dụ few-shot, chọn mục rulebook, và đưa thêm dòng "Reading: …" vào prompt (câu gốc vẫn là bản có thẩm quyền) | |
Đo độ chính xác tìm ví dụ few-shot (giao 6 ví dụ gần nhất với câu sạch) khi câu bị nhiễu: viết tắt 0,88→1,00; Telex 0,73→0,98; viết tắt + Telex 0,64→1,00; không dấu 1,00→1,00.
**Tiếng lóng / viết tắt Gen Z, kể cả kiểu chưa gặp (`datamate/slang.py`)**: bảng lóng đã biết (`ko/hok/hông`=không, `dc/đc`=được, `mik`=mình, `bn/bnhiu`, `ntn`, `trc`…) cộng 3 luật tổng quát chỉ chạy trên từ **không có trong tài liệu**: (1) kéo dài chữ (`nhieeeeu`, `khachhh`), (2) đổi chữ kiểu Gen Z (`zui`=vui, `fai`=phải, `wa`=quá), (3) viết tắt kiểu "giữ chữ đầu mỗi tiếng" (`dthu`=doanh thu, `tbinh`=trung bình). Phòng sửa nhầm: không đụng từ có dấu, tên riêng viết hoa, từ có thể là một âm tiết tiếng Việt (`Linh`, `Vinh`), từ có trong README/profile của shop, và không đoán khi có hai cách đọc ngang nhau.
Đo trên dữ liệu nhiễu tự tạo (không phải điểm thi): kéo dài chữ đúng 98% (sai 1%), đổi chữ 85% (sai 7%), viết tắt hai tiếng mới đúng 40% (sai 11%, phần còn lại để nguyên). Kiểm tra sửa nhầm trên 471 từ lạ lấy từ bảng dữ liệu (tên hàng, tên khách…): sửa 2 từ (`cty`, `hong`, cả hai đều là viết tắt thật) = 0,4%.

### B3. Sửa lỗi chính tả (module `datamate/spell.py`, chạy trong `normalize_question`, sau Telex/lóng)
Quét 4.000 câu: lỗi thật gồm sai/thiếu dấu thanh (`đươc`), thiếu chữ cuối (`tổg`, `cộg`, `bìh`, `đag`), thừa chữ (`lênn`, `nàoo`), chữ đầu kiểu Gen Z có dấu (`wán`, `júp`, `dzạ`, `zô`), dính chữ (`nhiềuđơn`, `caonhất`, `ngàynào`), nửa Telex (`đuowc`), phương ngữ (`kòn`, `ngừi`, `nhứt`).
| Lỗi | Cách sửa | Chốt chặn |
|---|---|---|
| Từ có dấu nhưng không phải âm tiết hợp lệ | tìm từ trong vốn từ cách ≤1 ký tự (bỏ dấu thanh), ưu tiên cùng dấu thanh và "thêm/bớt đúng 1 chữ" | từ ứng viên xuất hiện ≥3 lần, hơn ứng viên thứ hai ≥2 lần; hoà thì để nguyên |
| Chữ đầu `z j w f dz` có dấu | thử d/v/gi/r, qu, ph | chỉ khi duy nhất; bảng `COMMON` cho các ca rất phổ biến (`zô`→vô, `dzạ`→dạ, `kòn`→còn, `ngừi`→người) |
| Dính chữ | tách 2 âm tiết, mỗi nửa xuất hiện ≥10 lần | chỉ khi cách tách tốt nhất hơn hẳn cách thứ hai |
| Không dấu, gõ sai | chỉ sửa sang dạng không dấu của từ gần nhất (không đoán dấu) | ≥4 chữ, ứng viên là âm tiết hợp lệ, duy nhất |
| Không đụng | từ đã biết, từ trong README/profile của shop, tên riêng viết hoa giữa câu, âm tiết hợp lệ (`Linh`, `khát`) | |
Đo (tự tạo, không phải điểm thi): sửa nhầm trên 471 từ lấy từ bảng dữ liệu shop: **0** (trước khi thêm chốt chặn: 9%); lỗi giả lập thừa chữ cuối: sửa đúng 79%, sai 4%; dính hai từ thường gặp: đúng 46%, sai 0%; thiếu chữ cuối: đúng 16%, sai 10%; đổi sai dấu thanh sang một âm tiết khác *hợp lệ*: **không phát hiện được** (4% đúng) vì không thể phân biệt với từ thật nếu không có ngữ cảnh.
Giới hạn: sai dấu thanh tạo thành từ có thật (ví dụ `nhật` thay `nhất`) và lỗi chính tả ở tên riêng/tên hàng vẫn do model đọc; bản gốc luôn được giữ trong prompt, bản sửa chỉ là dòng "Reading: …".

Giới hạn chung: câu **không dấu và gõ Telex lẫn nhau**, hoặc kiểu viết tắt chưa có trong bảng, vẫn do model tự đọc.

## C. Unanswerable (ca dễ mất điểm nhất: sai bị −1)
- Train: câu hỏi tồn kho/công nợ/giá vốn/hạn dùng/thanh toán mà shop **không ghi nhận** thì đáp án gold là unanswerable.
- Hệ thống tự liệt kê "shop này có ghi nhận gì" bằng so khớp tên bảng/cột [D]. Đối chiếu với đáp án train (các câu tồn kho, công nợ, hạn dùng, thanh toán, giá vốn): **0 dương tính giả; 1 bỏ sót** trên khoảng 200 câu. Lưu ý: vài regex được chỉnh sau khi nhìn dữ liệu train nên đây không phải con số độc lập. Chỉ là *gợi ý*, model vẫn phải kiểm tra rulebook/dữ liệu.
- Ngưỡng unanswerable cao hơn (0.8) so với answer (0.6): thà abstain còn hơn trả lời sai.

## D. Lỗi "im lặng" (code chạy được nhưng sai)
| Ca | Xử lý |
|---|---|
| Lọc làm mất hết dòng → đáp án 0 / rỗng | vòng sửa riêng: in số dòng sau từng bước lọc [T] |
| 3 biến thể cùng sai giống nhau → đồng thuận giả (confidence 1.0) | **[?]** chưa có cách tin cậy; chỉ giảm bằng profile tốt hơn và few-shot |
| Hết thời gian / OOM | chia nhỏ prefill, giảm ngữ cảnh, abstain |

## E. Rủi ro lớn nhất còn lại (nói thẳng)
1. Chưa có số PA thật trên model thật. Việc cần làm tiếp: `smoke,preflight` rồi `build_exemplars,train_eval` với hold-out theo shop.
2. Chọn model (Qwen3-14B vs 8B vs ứng viên mới) chỉ nên quyết sau khi đo bằng cùng bộ hold-out.
3. Các câu tồn kho / tuổi nợ phụ thuộc mạnh vào khả năng suy luận của model; nên xem riêng PA của nhóm này sau khi chạy.
