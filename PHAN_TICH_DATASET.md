# Phân tích dataset DataMate (train + public test)

Nguồn: `DATAMATE_1.zip` (giống hệt `DATAMATE.zip` đã nhận trước đó). Mọi con số dưới đây tôi tính từ file, trừ chỗ ghi "ước lượng".

## 1. Quy mô
| | Train | Public test |
|---|---|---|
| Số câu hỏi | 3.000 | 1.000 |
| Số shop | 30 (biz_101–130) | 14 (dev_001–014) |
| File đáp án | `answers.jsonl` (có) | **không có** |
| Rulebook | 18,5 nghìn ký tự | **23,8 nghìn ký tự (dài hơn)** |
| Khoảng `as_of_date` | 2025-02 → 2026-09 | 2025-02 → 2026-09 (204 ngày khác nhau) |

**Public test không có đáp án**, nên không ai (kể cả tôi) biết "đáp án đúng" của 1.000 câu này; ban tổ chức chấm khi bạn nộp. Điều tôi nói được là: nó gồm những gì, mỗi câu thuộc nhóm nào, và loại đáp án (number/percent/list…) dự kiến.

## 2. Đáp án của train (để biết cấu trúc đáp án)
| Loại | Số câu | % |
|---|---|---|
| number | 1.884 | 62,8 |
| percent | 578 | 19,3 |
| list | 193 | 6,4 |
| date | 145 | 4,8 |
| string | 88 | 2,9 |
| unanswerable | 70 | 2,3 |
| boolean | 23 | 0,8 |
| table | 19 | 0,6 |

- Đáp án `list`/`string`/`table` có thêm trường `alias` (mã sản phẩm, ví dụ `SP0003`): chấm nhận cả tên lẫn mã.
- Ước lượng cho public (đoán loại từ câu chữ, đúng 86% trên train): number ≈ 62%, percent ≈ 17%, string ≈ 11%, list ≈ 6%, date ≈ 3,5%, boolean < 1%. Đây là ước lượng, không phải đáp án.

## 3. 14 shop public gồm gì
| Shop | Số câu | Loại hình (suy từ README/câu hỏi) | Bảng / ghi chú dữ liệu |
|---|---|---|---|
| dev_001 | 87 | đồ uống đóng chai (Bia, nước…) | 3 bảng, ~91k dòng; sổ chi tiết bán hàng |
| dev_002 | 41 | quán trà/cà phê | file đơn hàng không có mã sản phẩm, nối bằng tên |
| dev_003 | 129 | quán cà phê | 10 bảng: bán hàng, phiếu thu, khách, giá, **báo cáo tiền mặt theo ca**, **sổ chi phí** (câu "lãi ròng") |
| dev_004 | 96 | vật liệu xây dựng | 7 bảng: sổ bán, **sổ giao hàng nhiều đợt**, **sổ nợ**, nhập hàng, tồn kho, bảng giá; mỗi tháng một sheet |
| dev_005 | 87 | nhà thuốc | 17 file theo tháng, lô hàng, danh sách thuốc |
| dev_006 | 111 | nhà thuốc (POS) | SQLite 17 bảng: đơn, lô, thẻ kho, thanh toán, trả hàng… |
| dev_007 | 60 | (README không nêu ngành; có "ca chiều") | 3 danh sách xuất từ phần mềm |
| dev_008 | 99 | (không nêu; câu hỏi dùng "món") | cùng cấu trúc dev_007, 153k dòng |
| dev_009 | 21 | (không nêu) | **một file duy nhất, mỗi dòng một đơn, không có chi tiết mặt hàng** |
| dev_010 | 59 | (không nêu; có "nhóm sỉ") | 5 file CSV nối bằng mã |
| dev_011 | 40 | cửa hàng điện thoại | sheet `Ghi tắt` (tên hàng viết tắt) + sổ bán; 6.043 dòng |
| dev_012 | 50 | **sửa xe** | 14 sheet tháng; 6.060 phiếu |
| dev_013 | 57 | **mẹ và bé** | header gộp 2 dòng; khuyến mãi, khách thành viên |
| dev_014 | 63 | cửa hàng điện thoại | sổ nợ, **sổ trả góp**, sổ bảo hành, sổ thu máy cũ |

Ngành **sửa xe, mẹ và bé, điện thoại** chỉ có ở public (≈ 8,6% số câu) và **không có trong train**: không có ví dụ đã giải để học theo, mô hình chỉ dựa vào rulebook.

## 4. Nhóm câu hỏi (tỉ lệ có thể chồng nhau)
| Nhóm | Train % | Public % |
|---|---|---|
| doanh thu / tiền bán | 48,7 | 44,8 |
| tỉ lệ / phần trăm | 20,0 | 19,3 |
| so sánh kỳ, tăng/giảm | 8,9 | 11,4 |
| xếp hạng (top, nhiều nhất) | 7,6 | 7,6 |
| số đơn | 7,2 | 6,8 |
| **khách hàng** | 9,7 | **16,2** |
| công nợ / trả góp | 3,9 | 5,1 |
| ca / giờ / thứ trong tuần | 4,8 | 4,1 |
| **lãi / lợi nhuận / chi phí / hoa hồng** | 1,3 | **4,1** |
| tồn kho / hạn dùng | 2,4 | 1,4 |
| nhập hàng / giá vốn | 3,4 | 1,1 |
| **bảo hành / sửa xe / thợ** | 0,0 | **3,5** |
| **máy cũ / điện thoại** | 0,6 | **3,0** |
| **mẹ và bé (voucher, điểm, bỉm)** | 0,0 | **2,1** |
| Tết / lịch âm | 0,2 | 0,8 |

Đáng chú ý: **chỉ 4,3% câu public** có "khung câu" trùng câu train (sau khi che số). Câu public hay dùng viết tắt/không dấu/gõ sai mới: `z?` (91 lần), `bnhiu`, `mik`, `dh`, `Th2/2026`, `3/25`, `11/25`, `xửa`, `nượt`…

## 5. Luật mới chỉ có ở rulebook public (cần đọc kỹ)
- Đơn trả hàng **sau kỳ, kể cả sau hôm nay**, vẫn bị loại khỏi doanh thu kỳ đặt.
- **Báo cáo tiền mặt theo ca**: chỉ tiền mặt, theo giờ thu tiền, gồm cả trả nợ; doanh thu/số đơn không tính từ báo cáo ca.
- "Đến hết ngày X" áp cả cho **chưa giao**.
- Trung bình mỗi thứ X: chia cho **số ngày thứ X trong kỳ, kể cả ngày không có đơn**.
- Định nghĩa: khách trung thành (nhiều **ngày** mua nhất), khách sộp (TB đơn cao nhất, chỉ xét ≥ 3 đơn), khách hay đổi ý (nhiều đơn huỷ nhất), mặt hàng ưa chuộng (có mặt trong nhiều **đơn** nhất), thợ chăm chỉ (nhiều **ngày làm** nhất), ngày đắt khách (nhiều đơn nhất), **giờ cao điểm** (khung 1 giờ tròn, đáp án là giờ bắt đầu, type `number`), nhà cung cấp chính (nhiều **phiếu** nhất, khác "nhập nhiều tiền nhất").
- Topping ghi chung ô với món: tính là mặt hàng riêng.
- Mẹ và bé: tiền voucher/điểm/ship trả số dương, theo ngày hoá đơn; "Số/Size" lấy theo cột danh mục; khách thành viên mới = có **Ngày đăng ký** trong tháng.
- Sửa xe: mỗi dòng sổ là một lượt xe; phiếu bảo hành (Tiền 0) không phải lượt làm việc; xe khách quen = ≥ 2 phiếu có thu tiền; tỉ lệ bảo hành, "nhớt" trơn = nhớt máy, "ruột" trơn = thay ruột…
- Điện thoại: trả góp là đơn ghi nợ; tiền thu máy cũ không trừ doanh thu; kỳ góp hạn **trước** ngày hỏi mà chưa đủ là quá hạn; dòng máy = phần tên trước `…GB`.
- Sổ không có cột trạng thái: trạng thái nằm trong cột Ghi chú.

## 6. Bẫy dữ liệu tìm thấy khi đọc kỹ (và việc đã làm)
| Bẫy | Ví dụ thật | Xử lý |
|---|---|---|
| Ngày dạng chữ "1 thg 6 2025" | biz_101 | **profile trước đây không nhận ra** → đã sửa, hiện báo "DATE-LIKE" kèm khoảng ngày |
| Một cột ngày trộn nhiều định dạng | dev_009, biz_126, dev_012 (`01/07/2025`, `1 thg 7 2025`, `46264` là số serial Excel) | profile báo "nhiều định dạng", `parse_dates` xử lý từng giá trị |
| Tên thợ viết 12 kiểu cho 4 người | dev_012: `A.Đức`, `Đức`, `a Đức` | khối "Detected structure" nhóm các cách viết; model phải đối chiếu với `DanhSachTho` |
| Tiền lẫn `160000` và `220.000đ` | dev_012 | `to_num` + cảnh báo trong profile |
| Số lượng có chữ kèm (`1 ly`) | biz_112 | cảnh báo "NUMERIC stored as text" |
| Shop nhiều bảng (17 bảng) làm profile bị cắt đuôi | biz_123, dev_006 | profile luôn liệt kê đủ bảng |
| **Rulebook public (18,8k) bị cắt GIỮA khi cấu hình T4 chỉ cho 13k** | cả 14 shop | **đã sửa**: chọn ngành theo README + tên cột + chính các câu hỏi của shop; nếu vẫn dài thì bỏ từng dòng luật ít liên quan, không cắt giữa; luôn giữ mục "không trả lời được", "cách ghi", "format đáp án" |
| Xác định ngành shop chỉ từ README hay sai (số **điện thoại** của khách làm nhầm thành "cửa hàng điện thoại") | dev_008, dev_010 | từ khoá cụ thể hơn; nếu không đủ chắc chắn thì giữ mọi ngành, không đoán |

## 7. Những gì chưa biết
- Đáp án public: không có file; chỉ biết sau khi nộp.
- Chưa chạy mô hình thật trên các câu này, nên chưa biết tỉ lệ đúng; các sửa ở trên giảm lỗi dữ liệu/luật nhưng **chưa được đo bằng điểm**.
- Ba ngành chỉ có ở public (sửa xe, mẹ và bé, điện thoại): không có ví dụ train để học. Cách duy nhất để biết chất lượng là nộp public (5 lượt/ngày) — chỉ để kiểm tra hợp lệ và cùng mức điểm, không để chỉnh theo bảng xếp hạng.
