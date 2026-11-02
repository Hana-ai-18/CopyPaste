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
