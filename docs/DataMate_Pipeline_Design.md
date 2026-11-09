# DataMate: Thiết kế pipeline chặt (bản chốt)

> Mục tiêu: **dữ liệu sạch, có kiểm chứng**. **Lập luận đúng luật, tự bắt lỗi**. **Chịu được shop lạ ở private test**. **Không trả lời khi không chắc**.
> Phạm vi tài liệu: kiến trúc, hợp đồng của từng tầng, sổ rủi ro và cách vá, chiến lược dữ liệu hợp lệ, đánh giá, vận hành. Chưa có code.

---

## Mục lục
0. Nguyên tắc nền
1. Ràng buộc của cuộc thi và hệ quả với thiết kế
2. Kiến trúc tổng thể
3. Khối RULEBOOK
4. Khối DỮ LIỆU (onboarding mỗi business)
5. Khối LOGIC (mỗi câu hỏi)
6. Quyết định trả lời / abstain
7. Sổ rủi ro & lỗ hổng (đầy đủ)
8. Dữ liệu phát triển hợp lệ & huấn luyện
9. Đánh giá & kiểm thử
10. Vận hành: thời gian, tái lập, cô lập lỗi, log, Docker
11. Tuân thủ luật (checklist trước khi nộp)
12. Thứ tự triển khai
13. Chọn model và cách làm (model có sẵn + fine-tune LoRA)
14. Rà soát từng tầng: hợp đồng, kiểm tra, xử lý khi lỗi, rủi ro còn lại

---

## 0. Nguyên tắc nền

1. **LLM chỉ làm phần "hiểu", còn tính toán là code deterministic.** LLM đọc README, rulebook, câu hỏi và xuất **JSON theo schema**. Mọi con số do thư viện metric viết tay tính ra.
2. **Mỗi tầng có hợp đồng đầu ra, có kiểm tra, và báo lỗi to khi bất thường.** Cấm `except: pass`, cấm fallback âm thầm, cấm gán mặc định cho giá trị lạ.
3. **Không chắc thì không trả lời.** Sai bị −1, abstain được 0. Mỗi câu đều đi qua bước quyết định.
4. **Không có gì riêng cho từng shop trong code.** Phần riêng của shop nằm trong `ShopConfig` do máy sinh lúc chạy.
5. **Rulebook là đầu vào, không phải hằng số.** Mọi luật được tham số hóa từ rulebook *của split đang chạy*.
6. **Mỗi đáp án truy vết được**: log có câu gốc, câu chuẩn hóa, thực thể, kỳ, DSL, code, kết quả trung gian, giả định, quyết định, thời gian.
7. **Hai nguồn hiểu độc lập + kiểm tra không dùng LLM.** "Hai lần tính ra cùng số" không có nghĩa là đúng nếu cả hai cùng hiểu sai.

---

## 1. Ràng buộc của cuộc thi và hệ quả với thiết kế

| Ràng buộc | Hệ quả |
|---|---|
| Tổng tham số ≤ 15B (tính mọi model, kể cả MoE) | 1 LLM chính khoảng 7–8B (fine-tune LoRA rồi merge) + 1 embedding < 1B. Kiểm tra số tham số và ngày phát hành (trước 01/10/2026) của model cụ thể |
| Offline, không API | Mọi model và thư viện nằm sẵn trong image |
| 2×RTX 5090, 6 giờ cho toàn bộ private | Có ngân sách thời gian theo câu. Cache. Chế độ giảm tải khi gần hết giờ |
| Private có **loại cửa hàng mới**, README chất lượng thất thường | Onboarding tự động bằng agent + bất biến. Không có adapter viết tay |
| Rulebook có thể khác giữa các split | Đọc rulebook lúc chạy, sinh RuleConfig, diff với bản train |
| Tái lập: chênh PA ≤ 0,03 | Greedy, seed cố định, batch cố định, cache LLM theo hash prompt |
| Log bắt buộc trong `/result/logs/` | Logger có cấu trúc cho mỗi câu |
| Cấm logic theo ID, cấm dùng test để train hay thu đáp án | Xem mục 11 |
| `/data` chỉ đọc | File tạm ghi vào `/tmp/work` |
| Bị ngắt thì chỉ chấm phần đã ghi | Ghi `submission.jsonl` dần (append + flush) |

---

## 2. Kiến trúc tổng thể

```
PHA CHẠY (Docker)  /data ──▶ /result
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ ① RULEBOOK: đọc rulebook.md ▶ RuleConfig + Lexicon nghiệp vụ ▶ diff với train │
 └──────────────────────────────────────────────────────────────────────────────┘
 ┌──────────────────────── ② ONBOARDING mỗi business (cache) ───────────────────┐
 │ 2.1 Khám phá file/sheet/bảng ▶ 2.2 Nhận diện header & phân loại dòng          │
 │ 2.3 Profile cột ▶ 2.4 Ánh xạ ĐA GIẢ THUYẾT (agent + bằng chứng)               │
 │ 2.5 Parse có kiểu (quyết định định dạng theo CẢ CỘT)                          │
 │ 2.6 Ghép đơn (ffill, khử trùng, ngày đơn, status_class, trả hàng)             │
 │ 2.7 Giải thực thể (sp/khách/NV/NCC/nhóm/hoạt chất + aliases)                  │
 │ 2.8 Quy đổi đơn vị (bảng + README + suy từ giá)  2.9 Cờ dẫn xuất             │
 │ 2.10 BẤT BIẾN + PHÂN PHỐI + KIỂM CHỨNG TRẠNG THÁI ▶ vòng sửa ≤ 3             │
 │ ═▶ L2 (bảng chuẩn + bảng sự kiện + bảng thô dự phòng) + ShopProfile           │
 └──────────────────────────────────────────────────────────────────────────────┘
 ┌─────────────────────────── ③ MỖI CÂU HỎI ────────────────────────────────────┐
 │ 3.1 Chuẩn hoá (bảo vệ thực thể → Telex/kéo dài/teencode)                      │
 │ 3.2 Liên kết thực thể (trả MỌI ứng viên) ▶ 3.3 Giải thời gian (có unit test) │
 │ 3.4 LỚP HIỂU NGHĨA: loại câu ▶ khái niệm có nguồn rulebook ▶ phạm vi phủ định │
 │     ▶ kỳ ngầm định ▶ đầu ra nhiều phần ▶ DSL 7 ô (decode ràng buộc schema)    │
 │     ▶ NHẤT QUÁN QUA BIẾN THỂ (gốc / chuẩn hoá / bỏ dấu)                       │
 │ 3.5 Validator DSL (luật cứng) ▶ 3.6 Khả năng trả lời (chỉ xác định)           │
 │ 3.7a Đường A: metric  ║ 3.7b Đường B: code trên L2 từ câu GỐC ║ 3.7c Khứ hồi   │
 │ 3.8 Kiểm hợp lý ▶ 3.9 Sổ giả định ▶ ⑥ Quyết định ▶ 3.10 Định dạng + log        │
 └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Khối RULEBOOK

**Đầu vào:** `rulebook.md` của split, `calendar.csv`.
**Đầu ra:**
- `RuleConfig.json`: tham số luật, ví dụ:
  - `revenue.line_amount_after_line_discount = true`, `revenue.subtract_order_discount = false`, `revenue.include_shipping = false`, `revenue.include_vat = false`
  - `revenue.exclude_status = [cancelled, returned]`, `revenue.partial_return_excludes_whole_order = true`, `revenue.include_in_delivery = true`
  - `cancel_rate.denominator = all_orders_in_period`
  - `week.start = monday`, `last_week = full_prev_week`, `this_week = monday..today`
  - `month_to_date.compare = same_day_count_prev_month_clamped`
  - `shift.morning = 06:00–13:59`, `shift.afternoon = 14:00–21:59`, `shift.time_source = order_created`
  - `tet.window = [lunar 12/23 (Y-1), lunar 1/10 (Y)]`, `pre_tet = same_length_before`
  - `ranking.bestseller = qty_base`, `ranking.tie = code_asc`
  - `customer.walkin_counted = false`, `customer.identity = code_if_present_else_name_variant`
  - `debt.exclude = [cod, marketplace]`, `debt.age_from = order_date`, `debt.same_day_payment = paid_at_purchase`
  - `cost = weighted_avg_by_lot`, `gross_profit.cost_source = line_cost_per_sold_unit`
  - `inventory_value.price = catalog_cost`, `expiring_lot.capital = lot_import_price`
  - `rounding = half_up`, `money_decimals = 0`, `percent_decimals = 2`, `avg_decimals = 2`
  - Luật ngành: cà phê (size, topping), nhà thuốc, VLXD (đợt giao), mẹ & bé, sửa xe, điện thoại…
- `Lexicon.json`: cách nói → định nghĩa (bán chạy, ưa chuộng, lời nhất, trung thành, sộp, hay đổi ý, giỏi, chăm chỉ, đắt khách, giờ cao điểm, NCC chính và NCC nhập nhiều nhất, khách phải trả…), lấy từ rulebook.
- `rule_diff.log`: những dòng khác so với rulebook train.

**Cách làm:** chia rulebook theo bảng và dòng. LLM điền từng tham số kèm **trích dẫn dòng nguồn**. Tham số nào không tìm được dòng nguồn thì dùng mặc định của train *và* đánh dấu `uncertain`.

**Luật không ánh xạ được** (ngành mới, khái niệm mới): giữ nguyên văn trong `RuleConfig.free_text_rules[]`, đưa vào prompt của đường B. Metric bị luật đó chạm vào thì phải dùng ngưỡng tin cậy cao hơn.

---

## 4. Khối DỮ LIỆU (onboarding mỗi business)

### 4.1 Khám phá
- Duyệt mọi `.csv`, `.xlsx`/`.xls` (mọi sheet), `.sqlite` (mọi bảng và view). Đọc file xlsx lớn bằng engine nhanh (calamine), cache sang parquet ở `/tmp/work`.
- CSV: thử encoding (utf-8-sig, utf-8, cp1258, utf-16), đoán dấu phân cách, chấm điểm cách đọc hợp lý nhất.
- Mở merged cells. Header 2 tầng thì ghép thành "Nhóm/Tên cột".
- Gom file cùng cấu trúc (theo tháng, theo năm, _1/_2), giữ `src_file`, `sheet_name`, `sheet_year`.

### 4.2 Nhận diện header và phân loại dòng
- Chấm điểm từng dòng ứng viên header: tỉ lệ ô là chữ, độ khác nhau giữa các ô, độ khớp với từ khóa cột.
- Phân loại mọi dòng: **dữ liệu / header lặp / dòng tổng** (Tổng, Cộng, TỔNG NGÀY, Cộng tháng, Total, tổng dd/mm) **/ ghi chú / trống**. Log số lượng từng loại.
- Dòng tổng thường có ô mã đơn trống mà ô tiền rất lớn. Dùng làm bằng chứng, nhưng không chỉ dựa vào đó.

### 4.3 Profile cột (code thuần)
- Mẫu hình dạng giá trị (`str.replace(r'\d','9')`), tỉ lệ trống, số giá trị khác nhau, top 20 giá trị, khả năng parse (số, ngày, mã), min/max/trung vị.
- **Dấu hiệu nghi vấn:**
  - cột tăng dần 1, 2, 3… trong mỗi nhóm → số thứ tự dòng (không phải tiền)
  - cột lặp lại đúng một giá trị trên mọi dòng của đơn → thuộc tính đơn
  - cột toàn trống → bỏ
- **Quan hệ giữa các cột:** `qty × price ≈ amount`, `Σ amount theo đơn ≈ total`, `total − paid = debt`.

### 4.4 Ánh xạ đa giả thuyết (agent)
- **Công cụ của agent** (hẹp, chỉ đọc): `peek`, `value_counts`, `try_parse`, `check_relation`, `join_rate`, `sample_group(order_id)`.
- **Đầu ra:** `ShopConfig.json`, gồm:
  - `tables[]`: vai trò từng cột → trường chuẩn
  - `ffill`: cột cần fill và khóa nhóm (trong sheet hay trong file)
  - `status_map`: **mọi** giá trị → {completed, cancelled, returned, in_delivery}, kèm `evidence`
  - `return_detection`: status / negative_qty / lot_return / note / return_table
  - `walkin_tokens`, `has_customer_code`
  - `units`: nguồn quy đổi
  - `joins`
  - `debt_source`: payments_table / note_text / column / ledger
- **Đa giả thuyết:** với vai trò mơ hồ (cột tiền dòng hay tổng đơn, ngày đặt hay ngày giao…), giữ 2–3 giả thuyết và chấm điểm bằng **bằng chứng định lượng** (độ khớp quan hệ, độ khớp README, độ lớn giá trị). Hai giả thuyết đầu cách nhau ít hơn ngưỡng thì đánh dấu `ambiguous` để hạ trust của các metric liên quan.
- **Kiểm chứng trạng thái bằng dữ liệu:** đơn có trạng thái X có trong sổ giao không, có phiếu trả hoặc dòng âm không, tiền đã thu có bằng 0 không, có trong thẻ kho không. Ví dụ: "Không giao được" có 0/148 đơn được giao, nên là huỷ.
- **Kỷ luật:** agent chỉ được chọn trong số cột và giá trị có thật. Validator từ chối mọi tên cột hay giá trị không tồn tại.

### 4.5 Parse có kiểu: quyết định theo **cả cột**
| Kiểu | Xử lý | Quyết định theo cột |
|---|---|---|
| Tiền | `29k`, `43.000đ`, `99,999.9999`, `129 000`, `1,05tr`, `3,24tr`, `1tr380`, `1tr5`, `-4tr300`, `50000k`, `11,067tr` | `.` hay `,` là phân cách nghìn hay thập phân: chọn theo độ lớn điển hình và quan hệ `qty × price` |
| Ngày | ISO, `dd/mm/yyyy`, serial Excel, `1 thg 3 2025`, `d/m` (năm lấy từ sheet/file), có giờ hoặc không | dd/mm hay mm/dd: chọn cách cho ngày đơn điệu theo mã đơn và không có ngày > 12 ở vị trí tháng |
| Số lượng | `1 ly`, `2 hộp`, số âm | Đơn vị trong ô ưu tiên hơn cột đơn vị nếu mâu thuẫn, nhưng phải log |
| Mã | strip mọi khoảng trắng (đầu, cuối, giữa), chuẩn hóa hoa/thường | |
| Tên | NFC, gom khoảng trắng, tạo khóa bỏ dấu | |
| Ô ghép | topping "món + 2 X (14.000đ)"; IMEI "a / b"; ghi chú nợ dạng chữ | |

Tỉ lệ parse lỗi > 0,5% ở một cột quan trọng thì báo lỗi lên vòng sửa.

### 4.6 Ghép đơn
- Forward-fill **trong phạm vi khóa nhóm** (sheet/file), không fill tràn.
- **Parse trước, khử trùng sau.** Cùng đơn, giống nhau mọi ô (có số dòng, hoặc cả đơn giống hệt) thì tính 1 lần. Bảng không ghi theo đơn thì giữ nguyên.
- **Ngày đặt = ngày của dòng bán** (số lượng dương, hoặc dòng đầu đơn). Dòng trả mang ngày giờ trả, không dùng làm ngày đơn.
- `status_class` lấy theo đơn. Có bất kỳ dấu hiệu trả nào (kể cả trả một phần, kể cả phiếu trả sau hôm nay) thì là returned.
- Đơn mồ côi (có dòng mà không có đầu đơn, hoặc ngược lại): log ra.

### 4.7 Giải thực thể
- **Sản phẩm:** có mã thì theo mã. Không có mã thì gom tên theo khóa bỏ dấu, chọn tên chuẩn là dạng có dấu xuất hiện nhiều nhất. `aliases[]` gồm biến thể trong file, ghi tắt (sheet Ghi tắt), tên gọi khác trong danh mục và **từ điển vùng miền** (lạc/đậu phộng, lợn/heo, rán/chiên, phụ/hũ, ngao/nghêu, tràu/lóc, cốc/củ sạc, găng tay tất chân/bao tay bao chân, kệ/giá đỡ…).
- **Khách:**
  - Có mã thì theo mã (trùng tên vẫn là 2 khách).
  - Không có mã thì bỏ số điện thoại đính kèm (`- 0974 532 329`, `(09…)`, ` 09…`), rồi **chỉ gộp khi khác biệt là hoa/thường, khoảng trắng, hoặc một bên hoàn toàn không dấu.** Hai tên đều có dấu mà dấu khác nhau ("Lan" và "Lân") là **hai khách**.
  - Khách lẻ: mọi biến thể (kl, lẻ, khách lẻ…) thì không phải khách.
- **NV, NCC, nhóm hàng, hoạt chất, kênh, phương thức thanh toán**: lập danh sách giá trị chuẩn kèm biến thể.

### 4.8 Quy đổi đơn vị
- Nguồn theo thứ tự ưu tiên: bảng quy đổi > README > danh mục (cột "Quy đổi") > **suy từ giá** (giá đơn vị lớn ≈ k × giá đơn vị cơ bản, k nguyên, sai số < 5%).
- Giải hệ số lồng nhau (hộp = 15 vỉ, vỉ = 12 viên → 180). Áp cho **cả sổ bán và sổ nhập**.
- Đơn vị không có quy đổi nào: nếu trùng đơn vị cơ bản thì hệ số 1, nếu không thì đánh dấu `missing_conversion` (metric số lượng của mặt hàng đó hạ trust).

### 4.9 Cờ dẫn xuất (tính 1 lần)
`is_revenue_order`, `is_walkin`, `shift`, `hour`, `qty_base`, `is_service_fee`, `is_topping`, `size_M/L`, `is_rx`, `active_ingredient`, `item_kind` (máy mới/cũ/phụ kiện), `cogs_line`, `order_day`, `iso_week`, `tet_year`.

### 4.10 Bất biến, phân phối, vòng sửa
**Bất biến cứng:**

| # | Bất biến | Gợi ý nguyên nhân khi fail |
|---|---|---|
| I1 | Σ amount dòng ≈ tổng tiền hàng đơn (khi có cột) | sai vai trò cột / lọt dòng tổng / trùng |
| I2 | (order_id, line_no) duy nhất; mỗi đơn có 1 ngày, 1 trạng thái | fill sai / khử trùng sai |
| I3 | 100% trạng thái đã ánh xạ; ≥ 99% dòng join được sản phẩm | trạng thái lạ / tên lệch |
| I4 | Tồn theo lô = tồn thẻ kho; tồn ≥ 0 | quy đổi sai / hiểu trả hàng sai |
| I5 | Đã trả + còn nợ = phải trả | join phiếu thu sai (mã có khoảng trắng) |
| I6 | Mọi đơn vị bán đều có hệ số | thiếu quy đổi |
| I7 | Khoảng ngày dữ liệu khớp với README "dữ liệu có đến hết …" | parse ngày sai |
| I8 | `qty × price − line_discount ≈ amount` với ≥ 95% dòng | nhầm cột / topping chưa tách |

**Kiểm tra phân phối** (cho shop không có dư thừa để đối chiếu):
- doanh thu ngày không có ngoại lai ×2 so với trung vị (dấu hiệu dòng tổng)
- tỉ lệ huỷ trong khoảng 0–15%
- giá mỗi mặt hàng ổn định theo thời gian
- số đơn theo ngày trong tuần hợp lý

**Vòng sửa:** fail thì đưa thông báo lỗi có số liệu cụ thể cho agent sửa `ShopConfig`, tối đa 3 vòng. Vẫn fail thì ghi `trust[table] = low`.

### 4.11 Đầu ra của khối dữ liệu
- **L2 bảng chuẩn:** `orders, lines, payments, products, units, customers, staff, suppliers, imports, lots, lot_moves, stock_moves, expenses, deliveries, shifts_cash, warranty, installments, debt_ledger, price_history, promotions, members`. Bảng nào shop không có thì rỗng.
- **Bảng sự kiện tổng quát** `events(entity_type, entity_id, event_type, date, ts, qty, amount, attrs_json)`: chứa những khái niệm mới chưa có bảng riêng (ngành mới).
- **Bảng thô dự phòng:** mọi bảng đã parse kiểu nhưng không ánh xạ được, để đường B truy cập.
- **ShopProfile:** `capabilities` (có giờ? mã khách? nhóm khách? nhóm hàng? kênh? phương thức TT? giá vốn? lô? nhập? chi phí? công nợ? bảng giá? giao hàng? bảo hành?…), `data_range`, `trust[table]`, `ambiguities[]`, `vocab` (tên chuẩn + aliases + khóa bỏ dấu), `rule_overrides` (từ README, ví dụ "1 bó = 10 cây").

---

## 5. Khối LOGIC (mỗi câu hỏi)

### 5.1 Chuẩn hóa câu hỏi (module `vn_normalize.py`, code thuần, không dùng model)

**Thống kê thực tế trên 3.000 câu train:** viết tắt / teencode khoảng 1.460 chỗ, **Telex bị lọt** khoảng 370 chỗ (`thangs`, `tieenf`, `cuwar`, `ddown`, `soos`, `dduwowcj`), kéo dài gần như không có. Private test có thể có nhiều hơn, nên vẫn xử lý đủ cả ba loại **và trộn của chúng**.

**Các bước:**
1. NFC, bỏ emoji, gom dấu câu lặp (`???` → `?`).
2. **Bảo vệ thực thể TRƯỚC mọi bước sửa chữ:** tên hàng/khách/nhóm/NCC/NV trong `vocab` của *shop đang hỏi* (so khớp không dấu, không phân biệt hoa thường, theo ranh giới từ, ưu tiên chuỗi dài nhất), số, ngày (`9/2025`, `3/26`, `th10/2025`), tiền (`1tr5`, `3tr7`, `190k`, `10 củ`), mã (`SP0012`, `NV003`) → thay bằng placeholder `⟦E0⟧`. Lý do: tên hàng thật trông giống lỗi gõ (`Plusssz`, `Gas`, `Red Bull`, `Gold`, `Sino`, `Nurofen for Children`). Từ viết hoa ngắn (`AA`, `COD`, `VAT`) giữ nguyên.
3. **Mỗi token chưa hợp lệ đi qua bộ tìm ứng viên tổng quát** (`search_candidates`), xử lý được lỗi **trộn**:
   - Tách token thành các **cụm ký tự lặp** theo *chữ cái gốc* (bỏ dấu khi so, giữ ký tự có dấu khi co): `nhaatstttt` → `n|h|aa|t|s|tttt`.
   - Mỗi cụm lặp ≥ 2 thử **giữ 1 hoặc giữ 2** ký tự (vì Telex cần `aa ee oo dd`). Cụm ≥ 3 co miễn phí; co một cặp đôi tính chi phí 1.
   - Thử **bỏ 0–2 cụm cuối, chỉ khi cụm đó là chuỗi kéo dài** (gốc lặp ≥ 2), mỗi cụm chi phí 2. Người gõ giữ phím ở cuối thì xử lý được, còn `cafe` không bị cắt thành "cà".
   - Mỗi biến thể được đánh giá ở 2 dạng: nguyên dạng, và **giải mã Telex** (`dd→đ`, `aa→â`, `aw→ă`, `ee→ê`, `oo→ô`, `ow→ơ`, `uw/w→ư`, `uow→ươ`; phím thanh `s f r x j` đặt dấu theo quy tắc chính tả).
   - **Hợp lệ** khi: trúng teencode, hoặc là từ đã biết (vocab, rulebook, README), hoặc là **âm tiết tiếng Việt hợp lệ** (phụ âm đầu + vần trong bảng vần, vần tắc -p -t -c -ch chỉ đi với sắc/nặng). Ứng viên phải có nguyên âm, trừ teencode như `đc`.
   - **Chọn** theo thứ tự: (1) trúng teencode hoặc từ đã biết, (2) chi phí sửa thấp nhất, (3) ưu tiên dạng đã giải Telex. Không có ứng viên hợp lệ thì **giữ nguyên**.
4. Sửa lỗi gõ 1 ký tự (Damerau-Levenshtein ≤ 1) chỉ khi **duy nhất 1** ứng viên trong vocab miền và token dài ≥ 4. Token không dấu chỉ được sửa nếu ứng viên khác nó **ở phần dấu**.
5. **Không thêm dấu cho chữ không dấu** (`luot` có thể là lượt hoặc luật, `nhieu` có thể là nhiêu hoặc nhiều). Câu không dấu giữ nguyên, bước liên kết thực thể và DSL so bằng khóa bỏ dấu.
6. Teencode nhiều token (`xi ô đi` → cod, `o đờ` → đơn), trả thực thể về dạng chuẩn trong danh mục.
7. **Chạy liên kết thực thể lần 2** sau khi sửa chữ (ví dụ `cafe sữa (m)` → `cà phê sữa (m)` mới khớp được `Cà phê sữa (M)`).
8. Log mọi thay đổi `(token gốc, kết quả, cách sửa)`. Định kỳ đọc log các token bị giữ nguyên để bổ sung từ điển.

**Kết quả kiểm thử thực tế:**

| Đầu vào | Đầu ra | Cách sửa |
|---|---|---|
| `coasaaaa` | có | bỏ đuôi kéo dài `aaaa` → Telex `coas` = cóa → teencode → có |
| `nhaatstttt` | nhất | bỏ đuôi `tttt` → Telex `nhaats` |
| `tieeenffff` | tiền | `eee`→`ee`, `ffff`→`f` → Telex `tieenf` |
| `dduwowcjjjj` | được | co `jjjj` → Telex |
| `thangsss nayyy tooongrrr bnhiu ddownnnn` | tháng nay tổng bao nhiêu đơn | trộn Telex + kéo dài + teencode |
| `khônggggg bítttt` | không biết | kéo dài + teencode |
| `cafe sữa (m) bán đc bnhiuuu lyyyy` | cà phê sữa (m) bán được bao nhiêu ly | teencode + kéo dài, `cafe` không bị cắt |
| `xoong nồi`, `boong` | giữ nguyên | từ hợp lệ có chữ đôi |
| `tồn kho PLUSSSZ max …` | Plusssz Max … | thực thể được bảo vệ |

**Giới hạn được chấp nhận:** token sai mà có nhiều nghĩa (`ban` là bán hay bạn, `nhieeuuuu` là nhiêu hay nhiều khi không có phím thanh) thì để LLM hiểu theo ngữ cảnh ở tầng DSL. Nếu vẫn mơ hồ thì validator và bước quyết định cho **abstain**.

**Kiểm thử hồi quy:** chạy trên toàn bộ câu train sau mỗi lần sửa từ điển. Không được có thay đổi nào biến token thành chuỗi ≤ 1 ký tự hay đổi tên hàng.

### 5.2 Liên kết thực thể
- Trả về **mọi ứng viên** kèm điểm (khớp chính xác > alias > khóa bỏ dấu > fuzzy).
- Cấm `contains` với chuỗi ngắn (bài học "Costar", "Tiffy").
- Mơ hồ ("cà phê sữa" khớp cả (M) lẫn (L); "bia Tiger" khớp cả lon lẫn lon bạc): nếu câu không đủ phân biệt thì đánh dấu `ambiguous_entity`. Ở bước quyết định, chỉ cộng gộp khi câu rõ ràng hỏi theo nhóm, còn lại thì abstain.
- Tên không có trong dữ liệu (kể cả qua alias) thì `unknown_entity`. Không tự suy ra unanswerable trừ khi chắc chắn mặt hàng không tồn tại trong shop.

### 5.3 Giải thời gian (code thuần, unit test đầy đủ)
- Hỗ trợ: hôm nay/hôm qua/bữa nay/bữa qua; tuần này/tuần trước/tuần rồi; N ngày gần nhất/qua; tháng này (+ cùng kỳ tháng trước, chặn khi tháng trước thiếu ngày); từ đầu năm; tháng M/Y, `T2/26`, `2-2026`, `02/2026`, "tháng ba"; **quý: Qn, quý n, quý n/Y**; năm nay/năm ngoái/cùng kỳ năm trước; Tết năm Y, dịp Tết, trước Tết; khoảng "từ X đến Y"; "đến hết ngày X" (dạng *số còn lại*, tính từ đầu dữ liệu).
- Tháng/quý không ghi năm: lấy năm nay, trừ khi kỳ đó sau hôm nay thì lấy năm trước.
- Test ma trận: thứ trong tuần × đầu/cuối tháng × tháng 2 × năm nhuận × as_of nằm sau ngày cuối dữ liệu × Tết chưa tới.
- Đối chiếu `data_range`: kỳ nằm hoàn toàn ngoài thì unanswerable; nằm một phần thì tính trên phần có dữ liệu.

### 5.4 DSL ý định (7 ô + các trường hiểu nghĩa)
```json
{
  "question_type": "value|compare|rank|argmax|count_where|list_where|share|bool|state_at|multi_value|table",
  "measure": "revenue|order_count|qty|gross_profit|net_profit|debt|debt_new|debt_paid|stock|stock_value|cancel_rate|avg_order_value|import_amount|import_minus_sales|price|...",
  "object":  {"type": "shop|item|group|customer|staff|supplier|ingredient|lot|channel|payment|plate|model|brand", "ids": [...]},
  "filters": [{"field": "channel|pay_method|customer_group|shift|size|rx|item_kind|status|...",
               "op": "in", "values": [...], "negate": false, "span": "chữ trong câu sinh ra bộ lọc này"}],
  "concepts": [{"phrase": "khách sộp", "definition_id": "rank.big_spender", "rule_ref": "§3 dòng 'Khách sộp nhất'"}],
  "time":    {"target": {...}, "base": {...}|null, "kind": "flow|state",
              "implicit": false, "implicit_rule": null},
  "operation": "value|growth_pct|diff|argmax|argmin|top_k|count_where|list_where|share_pct|bool_all",
  "k": 3, "threshold": {"op": ">=|>|<=|<", "value": 0, "unit": "money|orders|qty|days"},
  "conditions": [ {"lhs": DSL, "op": ">", "rhs": DSL}, ... ],
  "outputs": [ {"label": "T7/2025", "sub": DSL}, ... ],
  "unit": "ly|lon|gói|viên|cái|kg|...|null",
  "overrides": [{"rule": "bestseller_by", "value": "amount", "span": "tính theo tiền"}],
  "output": {"type": "number|percent|date|date_month|string|list|table|boolean", "columns": [...]},
  "ambiguities": [{"what": "entity|time|concept|scope", "detail": "..."}]
}
```
- Sinh bằng **decode ràng buộc theo grammar/schema**, nên không bao giờ ra JSON sai định dạng.
- Prompt gồm: câu chuẩn hóa, thực thể đã liên kết (mọi ứng viên), kỳ đã giải, `capabilities`, Lexicon (từ RuleConfig) có **mã định nghĩa** (`definition_id`), khoảng 8 ví dụ few-shot truy hồi theo độ giống từ train.
- **`span`** là chữ trong câu sinh ra từng bộ lọc hoặc ghi đè. Trường này bắt model "chỉ tay" vào chữ làm căn cứ, để validator kiểm từng từ phủ định, từng cụm ghi đè đều có chỗ dùng.

**Ngữ pháp bắt buộc:**
- So sánh: "So với **B**, A", "A so (với) **B**", "A … so **B**" → `base = B`. "A hơn/kém B bao nhiêu" (con số) → A − B. "%" → (A − B)/B. "chiều so sáng" → base = sáng.
- Phủ định: không / ko / hổng / chẳng / ngoài / trừ / bỏ / không tính / không cần đơn bác sĩ → `negate` **trên đúng bộ lọc** mà nó bổ nghĩa.
- Câu đúng/sai: tách thành `conditions[]`, kết quả là **AND tất cả** ("và", "rồi… lại", "tăng liên tục", "luôn cao hơn").
- Câu nhiều phần ("lần lượt", "mỗi tháng", "từng kênh", "và … bao nhiêu"): `outputs[]` **có thứ tự đúng như câu nêu**.
- Ngưỡng: "từ N trở lên", "ít nhất N" → `>=`; "hơn/quá/trên N" → `>`; "dưới N" → `<`.

### 5.4A Lớp hiểu nghĩa: 7 cơ chế để hiểu đúng ý, không chỉ đúng chữ

**(1) Phân loại loại câu trước, điền ô sau.** Model xác định `question_type` trước, rồi mới điền các ô. Mỗi loại câu có **bộ ô bắt buộc riêng**:

| Loại câu | Ô bắt buộc | Ví dụ |
|---|---|---|
| value | measure, time.target | "doanh thu tháng 5" |
| compare | measure, target, **base**, operation ∈ {growth_pct, diff} | "so với tháng trước tăng mấy %" |
| rank / argmax | measure-xếp-hạng, object.type, k, time | "top 3 món bán chạy" |
| count_where | object.type, threshold, time | "mấy khách mua từ 5 đơn" |
| share | measure tử số, mẫu số (theo rulebook), time | "nhóm X chiếm mấy %" |
| bool | **conditions ≥ 1** (≥ 2 nếu có từ nối) | "T3 cao hơn T2 và T4 cao hơn T3?" |
| state_at | time.kind = state, mốc ngày | "còn nợ đến hết 30/6" |
| multi_value | outputs[] ≥ 2, đúng thứ tự | "doanh thu T7, T8, T9 lần lượt" |
| table | output.columns đúng thứ tự câu nêu | "5 khách nợ nhiều nhất kèm số nợ" |

**(2) Khái niệm phải có nguồn trong rulebook** (`concepts[]`). Mọi từ nghiệp vụ ("bán chạy", "sộp", "trung thành", "ế", "nhà cung cấp chính", "hàng lời nhất"…) phải gắn với một `definition_id` trong Lexicon, mà Lexicon thì sinh từ một dòng cụ thể của rulebook. Từ nào **không gắn được** ("bán chậm", "nên nhập thêm", "khách quen" khi rulebook ngành không định nghĩa) thì **không được tự chế định nghĩa**: đánh dấu `ambiguities` và abstain. Câu hỏi về đánh giá chủ quan (ngon, thân thiện, uy tín, chất lượng) thì unanswerable.

**(3) Phạm vi phủ định.** Mỗi từ phủ định trong câu phải được gắn vào **đúng một** bộ lọc (`span` chứa từ đó). "doanh thu **không tính** đơn giao hàng **của khách sỉ**": phủ định áp lên *(kênh = giao hàng ∧ nhóm = sỉ)*, hay chỉ lên kênh rồi lọc thêm khách sỉ? Hai cách cho hai con số khác nhau. Câu có từ nối trong phạm vi phủ định thì đánh dấu mơ hồ. Đường B tính độc lập; A và B lệch thì abstain.

**(4) Kỳ ngầm định.** Bảng luật suy kỳ (từ rulebook §2 + các mặc định đã kiểm trên train):
- tháng/quý không ghi năm → năm nay (sau hôm nay thì năm trước)
- "tháng trước" trong câu có kỳ tường minh → tháng trước **của kỳ được hỏi**
- "cùng kỳ" → cùng kỳ năm trước của kỳ được hỏi
- "tháng nào cao nhất" mà không có khoảng → **mơ hồ**, abstain trừ khi câu nêu rõ các tháng
- không nói kỳ, với câu dạng phát sinh → mơ hồ; với câu dạng số còn lại → đến hết hôm nay

Mỗi lần suy kỳ ngầm định, ghi `time.implicit = true` cùng tên luật vào sổ giả định.

**(5) Đầu ra nhiều phần.** `outputs[]` có thứ tự, và validator **đếm**: câu nêu 3 tháng thì đúng 3 phần tử, đúng thứ tự xuất hiện trong câu. "Lần lượt" và "theo thứ tự" giữ đúng thứ tự câu nêu. Bảng: cột đầu là tên đối tượng, các cột sau theo thứ tự số đo câu nêu.

**(6) Nhất quán qua biến thể.** Cùng một câu được phân tích từ **3 dạng**: câu gốc, câu đã chuẩn hóa, câu bỏ hết dấu. So 3 DSL (bỏ qua chênh lệch không ảnh hưởng kết quả, như thứ tự các bộ lọc):
- 3/3 giống nhau → tin cậy cao.
- 2/3 giống nhau → lấy dạng đa số, hạ tin cậy một bậc.
- Cả 3 khác nhau → abstain.

Lệch nhau nghĩa là model đang dựa vào **bề mặt chữ** chứ chưa hiểu nghĩa. Để tiết kiệm thời gian, chỉ chạy đủ 3 dạng khi tin cậy sau đường A/B ở mức trung bình, hoặc khi câu có từ phủ định, so sánh hay nhiều vế (**chạy thích ứng**).

**(7) Tự mở rộng cách nói từ train.** Gom câu train theo DSL đúng của chúng, rút ra các cụm khác nhau dùng cho **cùng một ý** ("cất hàng hết bao nhiêu" / "đánh hàng về" / "nhập vô" = nhập hàng; "chốt được bao nhiêu o đờ" = số đơn; "bán đắt nhất" = bán chạy). Kết quả được nạp vào Lexicon (dạng cụm → `definition_id`) và vào dữ liệu fine-tune T4. Không dùng câu public/private.

### 5.5 Validator DSL (luật cứng, không dùng LLM)
| Kiểm tra | Bắt lỗi |
|---|---|
| Câu có "so với X" thì `time.base` trùng kỳ X; target là kỳ còn lại | ngược chiều so sánh |
| Câu có "quý/Q\d" thì có kỳ loại quarter | nhầm quý thành tháng |
| Câu có "và / rồi / lại / liên tục / luôn" + dạng đúng/sai thì `len(conditions) ≥ 2` | bỏ sót vế |
| measure `import_minus_sales` hoặc so số lượng: `unit` không rỗng, và mọi vế quy được về unit | lệch đơn vị |
| Câu có từ phủ định thì có ít nhất 1 `negate` | lấy nhầm mệnh đề ngược |
| "ai" + (nhân viên / NV / bán giỏi / chăm chỉ) thì object = staff | nhầm khách với nhân viên |
| "tính theo / kể cả / không tính / chỉ tính" thì `overrides` không rỗng | bỏ qua chỗ câu ghi đè mặc định |
| Kiểu output khớp kiểu câu ("tháng nào" → date_month; "ngày nào" → date; "mấy %" → percent; "có… không/phải không" → boolean; top k/liệt kê → list; "kèm số" → table) | sai format |
| Mọi id trong DSL tồn tại trong vocab | hỏi thực thể không có |
| `time.kind = state` cho các câu "còn nợ / còn tồn / đến hết ngày" | nhầm phát sinh với số còn lại |
| Ô bắt buộc theo `question_type` đều có (bảng 5.4A-1) | phiếu thiếu ý |
| Mỗi từ phủ định trong câu nằm trong `span` của đúng 1 bộ lọc có `negate=true`; không có bộ lọc negate nào thiếu từ phủ định tương ứng | sai phạm vi phủ định |
| Mỗi cụm ghi đè ("tính theo", "kể cả"…) nằm trong `span` của một `overrides` | bỏ sót ghi đè |
| Mọi `concepts[].definition_id` tồn tại trong Lexicon của split | tự chế định nghĩa |
| Số phần tử `outputs[]` = số kỳ/đối tượng câu liệt kê; thứ tự trùng thứ tự xuất hiện | thiếu phần hoặc đảo thứ tự |
| `time.implicit = true` thì `implicit_rule` thuộc bảng luật suy kỳ | suy kỳ tùy tiện |
| Kỳ được hỏi nằm sau hôm nay ("tháng sau", "tuần tới") | câu hỏi về tương lai → unanswerable |
| `ambiguities` không rỗng | chuyển thẳng sang quyết định với tin cậy thấp |

Fail thì LLM sinh lại 1 lần kèm thông báo lỗi. Vẫn fail thì **abstain**.

### 5.6 Khả năng trả lời (chỉ quyết định xác định)
- DSL cần capability shop không có (giới tính, địa chỉ khi không có, lý do huỷ, đánh giá chủ quan "ngon/thân thiện/uy tín/chất lượng", nhóm hàng khi không có cột nhóm, giờ khi không có giờ…) → **unanswerable**.
- Kỳ nằm hoàn toàn ngoài `data_range` → **unanswerable**.
- Có dữ liệu nhưng không có phát sinh → **0**, không phải unanswerable.
- LLM "không chắc" → **abstain**, không phải unanswerable (vì trả unanswerable cho câu trả lời được cũng bị −1).

### 5.7 Thực thi và kiểm chứng độc lập
- **Đường A:** DSL chạy qua thư viện metric (tham số hóa theo RuleConfig) trên L2. Dùng số nguyên, Decimal ở bước cuối.
- **Đường B:** LLM viết pandas ngắn trên L2, **từ câu gốc** (không dùng câu chuẩn hóa hay DSL), dùng prompt khác và ví dụ few-shot khác, có RuleConfig + free_text_rules. Chạy trong subprocess có timeout và giới hạn RAM, không mạng, chỉ đọc.
- **Khứ hồi:** dịch DSL ngược thành câu chuẩn bằng **template code thuần**, rồi LLM chấm (có/không) xem câu đó có cùng nghĩa với câu gốc không. Template không thể hiểu sai cùng kiểu với LLM, nên đây là tín hiệu độc lập.
- Câu ngoài DSL thì chỉ có đường B, bắt buộc qua khứ hồi mô tả (LLM mô tả code bằng lời, so với câu gốc) và dùng ngưỡng cao.

### 5.8 Kiểm tính hợp lý
- Dấu của tăng trưởng khớp phép so trực tiếp A > B.
- Số đơn quý > số đơn mỗi tháng trong quý. Doanh thu nhóm ≤ doanh thu shop. Tỉ trọng trong [0, 100].
- Tuổi nợ ≥ 0, tồn ≥ 0. Top-k đủ phần tử, không trùng, giảm dần. Đáp án tên phải có trong vocab.
- Đồng hạng ở biên k: xếp theo mã, ghi log (gold đã loại các câu đồng hạng ở biên, nên đồng hạng ở đây thường báo hiệu ta tính lệch).

### 5.9 Sổ giả định
Mỗi đáp án kèm danh sách mặc định đã áp (ví dụ: "bán chạy = qty_base", "bỏ đơn trả", "khách lẻ không tính", "tuần từ thứ Hai", "năm = năm nay"). Câu có cụm mâu thuẫn với một giả định thì hạ điểm tin cậy và log.

### 5.10 Định dạng và log
- Làm tròn HALF_UP. Tiền: số nguyên đồng. %, trung bình: 2 chữ số. Ngày `YYYY-MM-DD`, tháng `YYYY-MM`. List giữ thứ tự. Table: cột đầu là tên đối tượng, các cột số theo đúng thứ tự câu hỏi nêu.
- Tên: lấy dạng chuẩn có trong dữ liệu.
- Log JSON mỗi câu: `id, question, normalized, changes, entities, time, dsl, validator, pathA, pathB, roundtrip, sanity, assumptions, confidence, decision, answer, elapsed_ms`.

---

## 6. Quyết định trả lời / abstain

- **Đặc trưng:** DSL qua validator ngay lần đầu hay phải sinh lại; **mức nhất quán qua biến thể (3/3, 2/3, lệch)**; số `ambiguities`; có suy kỳ ngầm định không; shop thuộc ngành chưa gặp ở train không; A khớp B; khứ hồi đạt; `trust` của các bảng dùng tới; độ chắc của liên kết thực thể; metric đã gặp trong train hay chưa; có free_text_rule liên quan không; sanity đạt; số giả định bị mâu thuẫn.
- **Mô hình:** logistic regression nhỏ (hoặc luật bậc thang) dự đoán P(đúng). **Trả lời khi P > 0,5 + biên an toàn.**
- **Hiệu chỉnh:** k-fold **theo shop** trên 30 shop train + shop làm bẩn. Chọn ngưỡng ổn định qua các fold, không chọn ngưỡng tối ưu riêng một fold.
- **Ưu tiên** khi có xung đột: unanswerable xác định (5.6) > abstain do validator fail > mô hình tin cậy.

---

## 7. Sổ rủi ro và lỗ hổng (đầy đủ)

### 7.1 Dữ liệu
| ID | Rủi ro | Mức | Vá |
|---|---|---|---|
| D1 | Ánh xạ sai nhưng vẫn qua bất biến (shop có ít cột đối chiếu) | Cao | Đa giả thuyết + bằng chứng, đánh dấu ambiguous, hạ trust |
| D2 | Shop không có dư thừa để đối chiếu | TB | Kiểm tra phân phối, hạ trust có chủ đích |
| D3 | dd/mm hay mm/dd mơ hồ | Cao | Quyết định theo cột, kiểm đơn điệu theo mã đơn |
| D4 | `.` `,` mơ hồ | Cao | Quyết định theo cột, kiểm qty × price |
| D5 | Gộp nhầm khách khác dấu | TB | Chỉ gộp khi khác hoa/thường, khoảng trắng, hoặc một bên không dấu |
| D6 | Khử trùng nhầm giao dịch thật | TB | Chỉ khử khi có khóa đơn và số dòng; log |
| D7 | Trạng thái mơ hồ hoặc hiếm (chỉ xuất hiện vài tháng) | **Rất cao** | Ánh xạ *mọi* giá trị, kiểm chứng bằng dữ liệu, fail khi gặp giá trị lạ |
| D8 | Quy đổi không ghi ở đâu | Cao | Suy từ giá, đánh dấu missing_conversion |
| D9 | Khái niệm ngành mới | Cao | Bảng sự kiện + bảng thô + đường B |
| D10 | Dòng tổng / header lặp lọt vào | **Rất cao** | Phân loại dòng + bất biến I1 + ngoại lai doanh thu ngày |
| D11 | Mã có khoảng trắng làm join hụt | Cao | strip mọi mã; bất biến I5, join_rate |
| D12 | Ngày đơn lấy từ dòng trả | TB | Ngày đơn = ngày dòng bán |
| D13 | Fill tràn qua sheet | TB | ffill theo khóa nhóm |
| D14 | Topping, IMEI, ghi chú nợ ghép trong ô | TB | Parser ô ghép + bất biến I8 |
| D15 | Encoding, merged cells, header 2 tầng | TB | Đọc thử nhiều cách, chấm điểm |
| D16 | Hiệu năng file lớn | TB | calamine, parquet cache, song song |
| D17 | README sai hoặc thiếu | TB | README là bằng chứng *yếu*; dữ liệu thắng khi mâu thuẫn, log |
| D18 | Một khái niệm có nhiều nguồn mâu thuẫn (ví dụ tồn trong danh mục và tồn trong thẻ kho) | TB | Thứ tự ưu tiên theo RuleConfig; log chênh lệch |

### 7.2 Logic
| ID | Rủi ro | Mức | Vá |
|---|---|---|---|
| L1 | Đường A và B cùng hiểu sai | **Rất cao** | B đọc câu gốc + prompt khác; khứ hồi bằng template; validator không dùng LLM |
| L2 | Câu ghi đè mặc định | Cao | `overrides[]` + validator |
| L3 | Thực thể mơ hồ | Cao | Mọi ứng viên; abstain khi không phân biệt được |
| L4 | Unanswerable dương tính giả | Cao | Chỉ trả unanswerable khi có lý do xác định |
| L5 | Câu ngoài DSL | Cao | Đường B + khứ hồi mô tả + ngưỡng cao |
| L6 | Biên thời gian | Cao | Unit test ma trận |
| L7 | Làm tròn sai do số thực | TB | Số nguyên + Decimal HALF_UP |
| L8 | Validator regex dễ vỡ | TB | Kiểm trên DSL là chính, regex chỉ cảnh báo |
| L9 | Hiệu chỉnh ngưỡng overfit | TB | k-fold theo shop + shop làm bẩn |
| L10 | Rulebook có luật mới | Cao | RuleConfig có trích dẫn + free_text_rules + ngưỡng cao |
| L11 | Chiều so sánh, quý, câu nhiều vế, đơn vị (4 loại lỗi đã gặp) | **Rất cao** | Ngữ pháp bắt buộc trong DSL + validator 5.5 + dữ liệu train tổng hợp nhắm đúng chỗ |
| L12 | Nhầm "phát sinh" với "số còn lại" | Cao | `time.kind` + validator |
| L13 | "Bán được bao nhiêu <đơn vị>" cộng nhầm mặt hàng khác đơn vị cơ bản | TB | Lọc theo base_unit trong metric |
| L14 | Mẫu số tỉ lệ sai (tỉ trọng nhóm phải chia doanh thu cả shop, gồm đơn vãng lai và phí phục vụ) | Cao | Định nghĩa mẫu số trong metric, không để LLM tự chọn |
| L15 | Nhầm tên đầu ra (sai dấu, sai biến thể) | TB | Xuất tên chuẩn từ vocab |
| L16 | Prompt quá dài (vocab hàng trăm mặt hàng) | TB | Chỉ đưa ứng viên đã liên kết + top-N truy hồi |

### 7.3 Hệ thống và tuân thủ
| ID | Rủi ro | Vá |
|---|---|---|
| S1 | Không tái lập | Greedy, seed, batch cố định, cache LLM theo hash; chạy 2 lần và so |
| S2 | Code B treo hoặc tràn RAM | subprocess + timeout + giới hạn RAM; lỗi thì abstain |
| S3 | Crash lan rộng | Cô lập theo câu và theo shop; ghi đáp án dần |
| S4 | Vượt 6 giờ | Ngân sách theo câu; chế độ giảm tải (bỏ B, chỉ trả lời câu tin cậy cao) |
| S5 | CUDA/Blackwell | Base image theo hướng dẫn; test trên GPU tương tự |
| S6 | Luật 4: logic theo ID | Không `if business_id`; config sinh lúc chạy |
| S7 | Luật 6: dữ liệu test | Không train hay tinh chỉnh bằng public/private; không giữ file đáp án public trong repo |
| S8 | Log thiếu | Logger bắt buộc, test có log đủ trường |
| S9 | Phụ thuộc phiên bản thư viện | Pin version, lockfile, build image tái lập |
| S10 | Ghi vào `/data` | Mọi file tạm vào `/tmp/work` |

---

## 8. Dữ liệu phát triển hợp lệ và huấn luyện

1. **Train (3.000 câu, 30 shop):** gán nhãn DSL, chỉ giữ cặp mà đường A tái tạo đúng gold.
2. **Câu tổng hợp sinh từ DSL** trên shop train: lấy mẫu DSL (metric × thực thể × kỳ × so sánh × phủ định × câu nhiều vế × đơn vị), render bằng template, rồi viết lại cho tự nhiên (teencode, Telex lọt, không dấu, ba miền, đảo trật tự "so với"). **Nhãn đúng tuyệt đối theo cách xây dựng.** Dùng LLM bên ngoài để viết lại trong lúc phát triển: kiểm tra quy định chung hoặc hỏi ban tổ chức trước.
3. **Shop làm bẩn và shop tổng hợp:** biến đổi shop train (đổi tên cột, trộn định dạng, chèn header và dòng tổng, tách/gộp file, chuyển sqlite, thêm khoảng trắng vào mã, đổi từ đồng nghĩa trạng thái, xóa README). Đáp án giữ nguyên. Dùng để huấn luyện và kiểm agent onboarding.
4. **Ca khó viết mới trên shop train:** "So với X thì Y", Qn, "A và B phải không", "nhập − bán theo đơn vị nhỏ", phủ định, "kể cả / tính theo".
5. **Fine-tune:** LoRA cho (a) câu → DSL, (b) README + profile → ShopConfig, (c) câu → code L2. Merge vào một model. Đánh giá theo k-fold shop.
6. **Không dùng** dữ liệu public/private, không dùng đáp án dò được, không dò leaderboard.

---

## 9. Đánh giá và kiểm thử

- **Chỉ số chính:** PA theo k-fold shop (30 shop train).
- **Độ bền:** PA trên shop làm bẩn (mục tiêu sụt < 3 điểm).
- **Precision khi trả lời** và **tỉ lệ abstain**, xem theo nhóm metric và theo ngành.
- **Ma trận lỗi theo tầng:** dữ liệu / thời gian / ý định / metric / định dạng / tin cậy.
- **Diff hồi quy:** mỗi lần sửa, chạy lại toàn bộ và liệt kê câu đổi đáp án (đúng → sai, sai → đúng).
- **Unit test:** parser (mọi định dạng đã gặp), bộ giải thời gian (ma trận), metric (so với kết quả tính tay trên shop nhỏ), validator (các ca lỗi đã biết).
- **Kiểm tái lập:** chạy toàn bộ 2 lần, chênh lệch phải bằng 0.

---

## 10. Vận hành

- **Thời gian:** onboarding khoảng 2–5 phút/shop (song song, cache). Câu hỏi khoảng 5–15 giây (≤ 3 lần gọi LLM + thực thi). Đo trên GPU thật, đặt ngân sách cứng mỗi câu.
- **Chế độ giảm tải:** còn < 15% thời gian thì tắt đường B và khứ hồi, chỉ trả lời khi validator đạt và trust cao.
- **Cô lập lỗi:** try/except **theo đơn vị công việc** (câu, shop), không bao quanh logic bên trong. Lỗi thì abstain kèm log stacktrace.
- **Ghi dần:** append + flush `submission.jsonl` sau mỗi câu. Câu chưa xử lý không ghi (tính như abstain).
- **Log:** `/result/logs/<id>.json` + `run_summary.json` (thời gian, số câu theo quyết định, số shop trust thấp, rule_diff).
- **Docker:** `predict.sh` đọc `/data`, ghi `/result`. Không mạng. README khai báo model, số tham số, giấy phép.

---

## 11. Tuân thủ luật: checklist trước khi nộp

- [ ] Không có file đáp án public hay dữ liệu public nào trong repo hoặc image.
- [ ] Không có điều kiện nào theo `id` câu hỏi hay `business_id`.
- [ ] Mọi `ShopConfig` của test được sinh lúc chạy; config viết tay chỉ của shop train (làm ví dụ).
- [ ] Model công bố trước 01/10/2026, giấy phép cho phép dùng, tổng ≤ 15B, đã khai trong README.
- [ ] Chạy offline (đã test với mạng tắt).
- [ ] Hai lần chạy cho kết quả giống nhau.
- [ ] Log đủ: id, đáp án, thời gian, và cách suy ra (DSL, code, kết quả).
- [ ] Toàn bộ private chạy dưới 6 giờ trên cấu hình tương đương.

---

## 12. Thứ tự triển khai

| Bước | Module | Tiêu chí xong |
|---|---|---|
| 1 | `parsers/` (tiền, ngày, SL, mã, tên, ô ghép) | 100% unit test các định dạng đã gặp |
| 2 | `schema/` L2 + `invariants/` + kiểm tra phân phối | Chạy trên config viết tay 3–4 shop train, bất biến đạt |
| 3 | `discovery/` + `profiling/` (header, phân loại dòng, profile) | 30 shop train không sót dòng tổng hay header |
| 4 | `rulebook/` → RuleConfig + Lexicon + diff | Diff train ↔ public đúng các dòng khác |
| 5 | `onboarding_agent/` (đa giả thuyết, kiểm chứng trạng thái, vòng sửa) | Tự sinh config cho 30 shop; bất biến đạt ≥ 90% |
| 6 | `time_resolver/` | Ma trận unit test đạt |
| 7 | `normalize/` + `entity_link/` | Không sửa nhầm tên hàng trên train |
| 8 | `dsl/` schema + `dsl_parser/` (LLM) + `dsl_validator/` | Validator bắt được cả 4 loại lỗi đã biết |
| 9 | `metrics/` (tham số hóa theo RuleConfig) | Khớp gold trên phần lớn câu train |
| 10 | `pathB/` sandbox + `roundtrip/` + `sanity/` | Tín hiệu độc lập hoạt động |
| 11 | `decision/` (hiệu chỉnh k-fold) + `formatter/` + `logger/` | PA k-fold ổn định |
| 12 | `eval/` (shop làm bẩn, diff hồi quy) + Docker + `predict.sh` | Chạy offline, tái lập, trong ngân sách thời gian |
| song song | `finetune/`: sinh T1–T6, LoRA, merge, chọn checkpoint theo k-fold (mục 13) | PA k-fold tăng so với model gốc + few-shot |

---

## 13. Chọn model và cách làm (dùng model có sẵn + fine-tune, KHÔNG build từ đầu)

### 13.1 Vì sao không tự build
Train LLM từ đầu cần hàng nghìn GPU và hàng tỉ token, không khả thi và không cần. Cách đúng là **lấy model mã nguồn mở đã pretrain**, rồi **fine-tune LoRA** cho đúng 3 việc của pipeline.

### 13.2 Ngân sách tham số (≤ 15B, tính tất cả model dùng lúc inference)

| Vai trò | Model đề xuất (chọn 1) | Tham số | Giấy phép | Ghi chú |
|---|---|---|---|---|
| **LLM chính** (DSL, onboarding, code đường B, chấm khứ hồi) | **Qwen3-8B** (phương án A) | ~8,2B | Apache-2.0 | Tiếng Việt khá, code tốt, có chế độ thinking, **tắt khi chạy** để nhanh và tái lập |
| | **Qwen2.5-7B-Instruct** / **Qwen2.5-Coder-7B-Instruct** (phương án B) | ~7,6B | Apache-2.0 | Ổn định, nhiều công cụ hỗ trợ fine-tune |
| **Embedding** (truy hồi few-shot, so tên gần đúng) | **BAAI/bge-m3** | ~0,57B | MIT | Đa ngôn ngữ, có tiếng Việt |
| | hoặc **intfloat/multilingual-e5-small** | ~0,12B | MIT | Nhẹ hơn |
| Chuẩn hóa chữ, tính toán, kiểm tra | **Không dùng model** | 0 | | Code thuần |
| **Tổng** | | **≈ 8,8B** | | Dư ~6B, đủ chỗ đổi sang model lớn hơn nếu đo thấy lợi |

> **Bắt buộc tự kiểm trước khi chốt:** (1) **ngày công bố trọng số** trên trang model (phải trước 01/10/2026); (2) **giấy phép**; (3) **số tham số thực tế** (đọc `config.json` hoặc đếm khi load, kể cả embedding và `lm_head`). Ghi đủ 3 thông tin này vào README nộp kèm.
> **Không nên** dùng model 14B làm LLM chính (ví dụ Qwen3-14B khoảng 14,8B): cộng embedding là vượt 15B, mà chạy 2 đường cho mỗi câu cũng khó vừa 6 giờ.
> **Không cần PhoGPT** hay model "dịch teencode" riêng: tốn ngân sách tham số, có nguy cơ tự sửa số và tên hàng, mà module luật đã xử lý được.

### 13.3 Chạy model (inference)
- **vLLM** bản hỗ trợ Blackwell (CUDA ≥ 12.8, theo base image của ban tổ chức). Trọng số bf16 khoảng 16GB vừa trong 1 GPU 32GB.
- **2 GPU = 2 instance song song** (data parallel), không cần tensor parallel. GPU 1 chạy đường A, DSL và onboarding; GPU 2 chạy đường B và khứ hồi, hoặc chia đều theo câu.
- **Decode ràng buộc** (guided JSON / grammar, ví dụ xgrammar trong vLLM) cho DSL và ShopConfig, nên luôn ra JSON đúng schema.
- **Tái lập:** `temperature=0`, `top_p=1`, seed cố định, `max_num_seqs` cố định, tắt thinking (`enable_thinking=False` với Qwen3), **cache mọi lần gọi LLM** theo `sha256(model_version + prompt)` vào `/tmp/work/llm_cache`.
- **Giới hạn độ dài:** prompt ≤ 8k token (chỉ đưa ứng viên thực thể đã liên kết và top-N ví dụ), đầu ra DSL ≤ 512 token, code đường B ≤ 1.024 token.
- **Lượng tử hóa:** không cần (đủ VRAM). Nếu muốn tăng tốc, thử FP8 và đo lại PA trước khi dùng.

### 13.4 Ba việc của LLM chính và mẫu dữ liệu fine-tune

**Việc 1: Câu hỏi → DSL** (quan trọng nhất)
```json
{"messages": [
 {"role": "system", "content": "Chuyển câu hỏi thành DSL theo schema. Lexicon: ... RuleConfig tóm tắt: ..."},
 {"role": "user", "content": "Câu: so voi th2/2025, doanh thu th2/2026 tang truong may %?\nas_of: 2026-09-19\nThực thể: []\nKỳ đã giải: [2025-02-01..2025-02-28], [2026-02-01..2026-02-28]\nCapabilities: [...]"},
 {"role": "assistant", "content": "{\"measure\":\"revenue\",\"time\":{\"target\":{\"month\":\"2026-02\"},\"base\":{\"month\":\"2025-02\"}},\"operation\":\"growth_pct\",\"output\":{\"type\":\"percent\"}}"}
]}
```
**Việc 2: README + profile → ShopConfig** (agent onboarding, nhiều lượt gọi công cụ)
```json
{"messages": [
 {"role": "system", "content": "Bạn là agent ánh xạ dữ liệu. Công cụ: peek, value_counts, try_parse, check_relation, join_rate. Đầu ra: ShopConfig theo schema."},
 {"role": "user", "content": "README: ...\nProfile cột: ...\nRulebook tóm tắt: ..."},
 {"role": "assistant", "tool_calls": [{"name": "value_counts", "arguments": {"table": "SoBanHang", "col": "Trạng thái xử lý"}}]},
 {"role": "tool", "content": "{\"Hoàn thành\": 4809, \"Không giao được\": 148, ...}"},
 {"role": "assistant", "tool_calls": [{"name": "join_rate", "arguments": {"left": "orders[status='Không giao được']", "right": "SoGiaoHang"}}]},
 {"role": "tool", "content": "{\"rate\": 0.0}"},
 {"role": "assistant", "content": "{...ShopConfig với status_map['Không giao được']='cancelled', evidence='0/148 có trong sổ giao'...}"}
]}
```
**Việc 3: Câu hỏi → code pandas trên L2** (đường B, đọc câu GỐC)
```json
{"messages": [
 {"role": "system", "content": "Viết hàm answer(L2, rules) trả về giá trị. Schema L2: ... Luật: ..."},
 {"role": "user", "content": "Câu gốc: Q1/2026 tiem co tat ca bao nhieu don?\nas_of: 2026-07-31"},
 {"role": "assistant", "content": "def answer(L2, rules):\n    o = L2.orders\n    m = o.is_revenue_order & o.order_date.between('2026-01-01','2026-03-31')\n    return int(m.sum())"}
]}
```

### 13.5 Nguồn dữ liệu fine-tune (chỉ dùng train và dữ liệu tự tổng hợp)

| Tập | Cách tạo | Số lượng mục tiêu | Kiểm nhãn |
|---|---|---|---|
| T1: câu train → DSL | Gán DSL (bán tự động), **chỉ giữ cặp mà đường A ra đúng gold** | ~2.500–3.000 | Gold của train |
| T2: câu tổng hợp → DSL | Lấy mẫu DSL (metric × thực thể × kỳ × so sánh × phủ định × câu nhiều vế × đơn vị), render bằng template, rồi viết lại cho tự nhiên | 20.000–40.000 | **Nhãn là DSL gốc, đúng theo cách xây dựng** |
| T3: làm bẩn ngôn ngữ | Áp lên T1+T2: bỏ dấu, Telex lọt, kéo dài, teencode, gõ nhầm phím, dính chữ, giọng ba miền | ×2–3 lần | Như trên |
| T4: ca khó có chủ đích | "So với X thì Y", Qn, "A và B phải không", nhập − bán theo đơn vị nhỏ, phủ định, "kể cả / tính theo" | 3.000–5.000 | Như trên |
| T5: onboarding | 30 shop train + **shop làm bẩn** (đổi tên cột, trộn định dạng, chèn header/dòng tổng, tách/gộp file, sqlite, khoảng trắng trong mã, đổi từ đồng nghĩa trạng thái, xóa README) → quỹ đạo gọi công cụ + ShopConfig đúng | 300–600 quỹ đạo | Bất biến đạt + đáp án train khớp |
| T6: câu → code | Từ T1/T4, chỉ giữ code chạy ra đúng gold hoặc đúng kết quả đường A | 3.000–6.000 | Thực thi |

> Dùng LLM bên ngoài để *viết lại câu* hay *sinh quỹ đạo* trong lúc phát triển: kiểm tra quy định chung hoặc hỏi ban tổ chức trước. **Tuyệt đối không** dùng public/private test hay đáp án dò được.

### 13.6 Cấu hình fine-tune (LoRA)
- Công cụ: **LLaMA-Factory**, **Unsloth** hoặc **PEFT + TRL (SFT)**.
- Cài đặt khởi điểm: LoRA `r=32`, `alpha=64`, `dropout=0.05`, target mọi lớp linear (q,k,v,o,gate,up,down); `lr=1e-4`, cosine, warmup 3%; 2–3 epoch; `max_len=4096`; bf16; batch hiệu dụng 64. Chỉ tính loss trên phần trả lời của assistant.
- **Trộn dữ liệu** trong một lần train: T1+T2+T3+T4 khoảng 60%, T5 khoảng 20%, T6 khoảng 20%. Mỗi việc có system prompt riêng để model phân biệt.
- **Chống quên:** trộn thêm khoảng 5% dữ liệu instruct tiếng Việt tổng quát (công khai, giấy phép cho phép).
- Thời gian ước tính: vài giờ trên 1–2 GPU 32GB.
- **Merge LoRA vào trọng số gốc** để vLLM chạy như model thường. Ghi version và checksum vào README.
- **Chọn checkpoint** theo PA k-fold shop (mục 9), không theo loss.

### 13.7 Embedding
- Index: câu train (đã chuẩn hóa) + DSL của chúng, dùng để truy hồi 8 ví dụ few-shot gần nhất cho việc 1 và việc 3.
- So tên gần đúng: tín hiệu **phụ** sau khớp chính xác / alias / khóa bỏ dấu, không bao giờ là tín hiệu duy nhất.
- Không cần fine-tune. Nếu muốn, fine-tune nhẹ trên cặp (biến thể tên ↔ tên chuẩn) từ shop train.

### 13.8 Phương án dự phòng khi thời gian hoặc chất lượng không đạt
- LLM không đủ tốt ở việc 2 (onboarding): dùng **ShopConfig suy bằng luật** (heuristic theo profile cột + từ khóa) làm khởi tạo, agent chỉ sửa phần fail bất biến.
- Thiếu thời gian lúc chạy: bỏ đường B và khứ hồi, chỉ trả lời câu có validator đạt và trust cao (chế độ giảm tải, mục 10).
- Model A và B cho PA tương đương: chọn model **nhỏ hơn và nhanh hơn**.

---

## 14. Rà soát từng tầng: hợp đồng, kiểm tra, xử lý khi lỗi, rủi ro còn lại

> Mỗi tầng được rà theo 4 câu hỏi: **(a)** đầu vào/đầu ra cam kết gì; **(b)** kiểm tra nào chứng minh đầu ra đúng; **(c)** sai thì làm gì; **(d)** còn lỗ hổng nào. Những lỗ hổng **mới phát hiện trong lần rà này** được đánh dấu 🆕 và đã có cách vá ngay tại chỗ.

### 14.1 Tầng R: Rulebook → RuleConfig + Lexicon
- **(a)** Vào `rulebook.md`. Ra RuleConfig (tham số), Lexicon (cụm → `definition_id`), free_text_rules, rule_diff.
- **(b)** Mỗi tham số kèm **trích dẫn dòng nguồn**. 🆕 **Kiểm bằng train:** chạy pipeline trên train với RuleConfig *trích tự động từ rulebook train*; PA phải bằng PA khi dùng RuleConfig viết tay. Đây là cách chứng minh bộ trích luật đúng.
- **(c)** Tham số không có dòng nguồn: dùng mặc định train và gắn `uncertain`. Metric nào phụ thuộc tham số `uncertain` thì nâng ngưỡng.
- **(d)** 🆕 Luật ngành mới chỉ có trong private: đưa vào free_text_rules, đường B đọc nguyên văn; nếu đường A không biểu diễn được luật đó thì chỉ cho trả lời khi B và khứ hồi đều đạt. 🆕 Rulebook có **câu bẫy phân biệt** ("Khác nhà cung cấp nhập nhiều tiền nhất"): Lexicon phải tạo **hai** definition_id riêng, và validator kiểm câu hỏi dùng đúng id.

### 14.2 Tầng D1: Khám phá file
- **(a)** Vào thư mục business. Ra danh sách bảng (file, sheet, vùng ô).
- **(b)** Mọi file đều được mở; số sheet và bảng log đầy đủ; tổng số dòng đọc = tổng số dòng file (trừ header).
- **(c)** File không đọc được: log và hạ trust các capability liên quan, không dừng cả shop.
- **(d)** 🆕 **Nhiều bảng trong một sheet** (xếp chồng, ngăn bằng dòng trống, hoặc đặt cạnh nhau): tách theo khối dòng/cột trống. 🆕 **Sheet ẩn, sheet báo cáo tổng hợp** (pivot, "Báo cáo", "Tổng hợp", báo cáo ca tiền mặt) **không phải giao dịch**: phân loại `transactional` hoặc `aggregate`; bảng aggregate **không được gộp** vào dữ liệu đơn, chỉ dùng cho câu hỏi đúng về nó (ví dụ "tiền mặt thu theo ca"). 🆕 Các file theo tháng có **thứ tự cột khác nhau**: ghép theo tên cột đã chuẩn hóa, không theo vị trí.

### 14.3 Tầng D2: Header và phân loại dòng
- **(a)** Ra nhãn cho từng dòng: data / header lặp / tổng / ghi chú / trống.
- **(b)** Bất biến I1 và ngoại lai doanh thu ngày (dòng tổng lọt vào sẽ làm doanh thu ngày gấp đôi).
- **(c)** Ngoại lai thì tìm lại dòng tổng bằng luật "ô mã trống + ô tiền ≈ tổng các dòng phía trên".
- **(d)** 🆕 Dòng tổng **có mã giả** (ví dụ "tổng 05/01" ở cột mã đơn): luật nhận mã hợp lệ phải theo **mẫu mã chiếm đa số** của cột, không chỉ "không trống".

### 14.4 Tầng D3: Profile cột
- **(a)** Ra thống kê và mẫu hình dạng của từng cột, cùng các quan hệ giữa cột.
- **(b)** Tự kiểm: mọi cột đều có profile; quan hệ `qty × price ≈ amount` được thử trên mọi bộ ba cột số hợp lý.
- **(d)** 🆕 **Cột kiểu trộn** (số lẫn chữ, ví dụ "1 ly" và "2"): profile theo từng mẫu hình dạng, không theo kiểu pandas đoán. 🆕 **Ô trống mang nghĩa 0** (ô Bớt, Thu máy cũ để trống là "không có") khác **ô trống do nhóm dòng** (cần forward-fill): phân biệt bằng việc ô có trống trùng đúng các dòng không phải dòng đầu nhóm hay không.

### 14.5 Tầng D4: Ánh xạ đa giả thuyết (agent)
- **(a)** Ra ShopConfig + `evidence` cho từng quyết định + `ambiguous[]`.
- **(b)** Bất biến tầng D10; điểm bằng chứng của giả thuyết thắng phải cách giả thuyết nhì một khoảng tối thiểu.
- **(c)** Fail thì vòng sửa ≤ 3; vẫn fail thì dùng config heuristic và trust thấp.
- **(d)** 🆕 **Cùng tên cột, khác nghĩa giữa các bảng** ("Thành tiền" trong sổ nhập khác "Thành tiền" trong sổ bán): ánh xạ theo cặp (bảng, cột), không theo tên cột toàn cục. 🆕 **Nhiều nguồn cùng một sự thật** (tổng đơn ở bảng đơn và tổng dòng ở bảng chi tiết; tồn ở danh mục và ở thẻ kho): ghi **thứ tự nguồn chuẩn** theo rulebook (doanh thu = tổng dòng), nguồn còn lại chỉ để đối chiếu. 🆕 **Ngày đặt, ngày giao, ngày thu tiền, ngày HĐ điện tử** cùng có trong file: agent phải chọn đúng **ngày đặt** cho kỳ của đơn, với bằng chứng (README, tên cột, ngày đặt ≤ ngày giao).

### 14.6 Tầng D5: Parse có kiểu
- **(a)** Ra cột đã đúng kiểu + tỉ lệ parse lỗi theo cột.
- **(b)** Tỉ lệ lỗi < 0,5%; quan hệ I8 đạt; khoảng ngày khớp README (I7).
- **(d)** 🆕 Số âm dạng **ngoặc đơn** `(50.000)`, **phần trăm** `10%`, hậu tố `VND`/`vnđ`/`đồng`, **dấu cách không ngắt** (`\xa0`), chữ số toàn chiều rộng. 🆕 Năm 2 chữ số (`3/26`). 🆕 Ngày dạng `T2 03/02` (có thứ). 🆕 Giờ 12h có SA/CH/AM/PM. Tất cả có test trong `parsers/`.

### 14.7 Tầng D6: Ghép đơn
- **(a)** Ra `orders`, `lines` với order_date, status_class, cờ trả hàng.
- **(b)** I2 (mỗi đơn 1 ngày, 1 trạng thái), số đơn mồ côi = 0 hoặc đã log.
- **(d)** 🆕 **Đơn qua nửa đêm**: ngày đơn lấy theo thời điểm tạo, không theo ngày thanh toán. 🆕 **Mã đơn tái sử dụng** giữa các năm hoặc các file (số thứ tự đơn bắt đầu lại từ 1 mỗi sổ): khóa đơn = (nguồn sổ, năm, mã) khi phát hiện trùng mã nhưng khác ngày/khách. 🆕 **Đơn đổi hàng** (trả món A, lấy món B cùng mã đơn): theo rulebook vẫn là đơn có trả hàng, nên bị loại khỏi doanh thu.

### 14.8 Tầng D7: Giải thực thể
- **(a)** Ra product_id, customer_id… chuẩn, kèm aliases.
- **(b)** ≥ 99% dòng join được; không gộp hai mã khác nhau; không gộp tên có dấu khác nhau.
- **(d)** 🆕 **Cùng mã nhưng đổi tên theo thời gian**: mã thắng, tên hiển thị là tên trong danh mục hiện tại. 🆕 **Hai mặt hàng khác nhau trùng tên** ở hai nhóm: tách theo mã/nhóm, khi liên kết câu hỏi thì là mơ hồ. 🆕 **Khách lẻ ghi kèm số điện thoại** vẫn là khách lẻ.

### 14.9 Tầng D8: Quy đổi đơn vị
- **(b)** I6 (mọi đơn vị có hệ số); kiểm chéo bằng giá.
- **(d)** 🆕 Tên đơn vị khác cách viết (`Thùng`/`thùng`/`th`), có kèm số (`hộp 30`, `bao 25kg`, `cuộn 10m`) — chuẩn hóa nhưng **giữ phần số** vì nó là đơn vị khác. 🆕 Hệ số trong dữ liệu (cột "SL quy đổi") mâu thuẫn với bảng quy đổi: tin cột dữ liệu theo từng dòng, log chênh lệch.

### 14.10 Tầng D10: Bất biến, phân phối, vòng sửa
- **(d)** 🆕 **Ngưỡng sai số**: các bất biến về tiền dùng sai số tuyệt đối ≤ 1 đồng × số dòng (do làm tròn), không dùng phần trăm. 🆕 **Bất biến chỉ kiểm được khi có cột đối chiếu**: shop không có cột thì đánh dấu `unverifiable`, khác với `pass`, để bước quyết định phân biệt.

### 14.11 Tầng Q1: Chuẩn hóa câu hỏi
- **(b)** Hồi quy trên câu train: không thay đổi nào biến token thành ≤ 1 ký tự, không đổi tên hàng.
- **(d)** 🆕 Bảo vệ thực thể **2 lượt** (trước và sau khi sửa chữ). 🆕 Từ viết hoa ngắn (AA, COD, VAT) giữ nguyên. 🆕 Câu trộn tiếng Anh (`revenue`, `top`, `best seller`, `order`) đưa vào từ đã biết.

### 14.12 Tầng Q2: Liên kết thực thể
- **(a)** Ra mọi ứng viên kèm điểm và lý do khớp.
- **(d)** 🆕 Thực thể **có trong danh mục nhưng không có giao dịch** trong kỳ: trả **0**, không phải unanswerable. 🆕 Thực thể **không có trong shop** (kể cả qua alias): unanswerable chỉ khi chắc chắn (khớp không dấu không ra gì và không có ứng viên gần), còn lại thì abstain. 🆕 Tên nhóm hàng phải khớp **đúng tên nhóm**, không gộp nhóm gần tên (rulebook §3).

### 14.13 Tầng Q3: Giải thời gian
- **(b)** Ma trận unit test (thứ trong tuần × đầu/cuối tháng × tháng 2 × năm nhuận × as_of sau ngày cuối dữ liệu × Tết chưa tới × Tết năm trước).
- **(d)** 🆕 "hôm qua" khi as_of là ngày đầu dữ liệu: kỳ nằm ngoài dữ liệu → unanswerable. 🆕 Câu hỏi về **tương lai** → unanswerable. 🆕 **Bản ghi sau hôm nay** coi như chưa xảy ra, **trừ trạng thái huỷ/trả** lấy theo đơn (rulebook đầu §).

### 14.14 Tầng Q4: Lớp hiểu nghĩa và DSL (mục 5.4, 5.4A)
- **(b)** Validator 5.5; nhất quán qua biến thể; khứ hồi.
- **(d)** 🆕 Model thêm bộ lọc **không có trong câu** (bịa điều kiện): mỗi bộ lọc phải có `span` thật sự xuất hiện trong câu, nếu không thì loại bộ lọc đó và log.

### 14.15 Tầng Q5–Q6: Validator và khả năng trả lời
- **(d)** 🆕 Unanswerable chỉ được sinh từ **3 nguồn xác định**: capability thiếu, kỳ ngoài dữ liệu hoặc tương lai, khái niệm chủ quan. Mọi trường hợp khác không chắc thì abstain.

### 14.16 Tầng Q7: Đường A (thư viện metric)
- **(b)** Unit test từng metric trên shop nhỏ tính tay; đối chiếu gold train theo nhóm metric.
- **(d)** 🆕 Mẫu số tỉ lệ cố định trong code theo RuleConfig (tỉ trọng nhóm chia doanh thu cả shop, gồm đơn vãng lai và phí phục vụ; tỉ lệ huỷ chia mọi đơn). 🆕 Mọi metric xử lý **tập rỗng** rõ ràng (không chia cho 0; trả 0 hay unanswerable theo luật).

### 14.17 Tầng Q8: Đường B (code)
- **(d)** 🆕 Code chỉ được đọc **L2 và bảng thô đã parse**, không đọc file gốc, để tránh lặp lại lỗi parse. 🆕 Code trả về kiểu sai → coi như B không có kết quả.

### 14.18 Tầng Q9: Khứ hồi
- **(d)** 🆕 Template dịch DSL ngược phải phủ **mọi** `question_type`. Loại câu chưa có template thì khứ hồi để trống (không tính là đạt).

### 14.19 Tầng Q10–Q11: Kiểm hợp lý và quyết định
- **(d)** 🆕 **Ngành mới** (shop không giống ngành nào ở train theo profile): tăng biên an toàn của ngưỡng. 🆕 Quyết định không được dựa vào một tín hiệu duy nhất; tối thiểu cần validator đạt **và** (A khớp B **hoặc** khứ hồi đạt).

### 14.20 Tầng Q12: Định dạng
- **(d)** 🆕 Kiểm cuối bằng **schema đáp án** cho từng type (number không phải chuỗi; percent không kèm %; list kể cả 1 phần tử; date đúng mẫu; table ô số là number). 🆕 Tên đáp án lấy **đúng một cách viết có trong dữ liệu** của shop.

### 14.21 Tầng vận hành
- **(d)** 🆕 **Kiểm tra sức khỏe trước khi chạy**: load model, chạy 1 câu mẫu, kiểm GPU, kiểm dung lượng `/tmp`. 🆕 **Watchdog** theo câu (quá thời gian thì abstain câu đó, chạy tiếp). 🆕 `run_summary.json` liệt kê số câu ở mỗi nhánh (trả lời / abstain / unanswerable / lỗi), để phát hiện lỗi âm thầm (bài học A2: fallback che lỗi).

### 14.22 Bổ sung vào sổ rủi ro (mục 7)
| ID | Rủi ro | Vá tại |
|---|---|---|
| D19 | Sheet tổng hợp/pivot bị gộp vào giao dịch | 14.2 |
| D20 | Nhiều bảng trong một sheet | 14.2 |
| D21 | Ô trống mang nghĩa 0 bị forward-fill | 14.4 |
| D22 | Cùng tên cột khác nghĩa giữa các bảng | 14.5 |
| D23 | Chọn nhầm loại ngày (giao/thu thay vì đặt) | 14.5 |
| D24 | Mã đơn tái sử dụng | 14.7 |
| L17 | Khái niệm không có trong rulebook bị tự chế định nghĩa | 5.4A(2) |
| L18 | Sai phạm vi phủ định | 5.4A(3) |
| L19 | Suy kỳ ngầm định tùy tiện | 5.4A(4) |
| L20 | Câu nhiều phần thiếu hoặc đảo thứ tự | 5.4A(5) |
| L21 | Hiểu theo bề mặt chữ (lệch qua biến thể) | 5.4A(6) |
| L22 | Model bịa bộ lọc không có trong câu | 14.14 |
| L23 | Thực thể không có giao dịch bị trả unanswerable thay vì 0 | 14.12 |
| L24 | Câu hỏi về tương lai | 14.13 |
