# DataMate Pipeline v2: chặt, nhẹ, dựa trên cách tư duy "tìm cách hiểu nhầm"

> Bản này **thay thế** bản trước (`DataMate_Pipeline_Design.md` giữ lại làm phụ lục chi tiết).
> Mục tiêu: **đúng tối đa, sai ít nhất, chạy nhanh, 1 model ~8B, phần còn lại là code.**

---

## 0. Một trang tóm tắt

- Model chỉ làm phần **hiểu** (hiểu cột, hiểu câu). Mọi phép **tính** là code viết sẵn nên đúng tuyệt đối.
- Mỗi shop được **đọc 1 lần**: soi hình dạng cột, làm sạch thành **sổ chuẩn**, tự kiểm tổng. Có cache.
- Rulebook được đọc thành **bảng tham số**, không viết cứng trong code.
- Câu hỏi: sửa chữ bằng luật, model ra **phiếu yêu cầu**, luật cứng soát phiếu.
- **Đọc rules như bộ luật đánh số**: mỗi dòng là một luật có mã, phải được ánh xạ hết (báo cáo độ phủ), kiểm bằng train.
- **Lớp hiểu nghĩa**: đánh dấu 10 loại từ mang nghĩa; mọi chữ quan trọng phải "có việc" trong phiếu; khái niệm phải có nguồn rulebook.
- ★ **Tìm cách hiểu nhầm:** từ phiếu, code tự sinh 3–8 cách hiểu dễ nhầm (trong 16 loại), tính hết, luật cứng loại các cách trái câu chữ. Còn **một đáp án** thì trả lời, **mơ hồ thật** thì bỏ trống.
- **Hàng rào chống lỗi ngớ ngẩn:** 30 kiểm tra tự động, mỗi lỗi đã gặp đều có một hàng rào riêng.
- Chạy khoảng **1–2 giờ** cho ~1.000 câu (dư xa giới hạn 6 giờ), đúng 1 model.

```
① LUẬT     rulebook.md ─▶ RuleConfig + Lexicon (có trích dòng nguồn)
② DỮ LIỆU  file ─▶ soi hình dạng ─▶ cấu hình (model đề xuất + bằng chứng) ─▶ làm sạch ─▶ SỔ CHUẨN ─▶ tự kiểm
③ CÂU HỎI  câu ─▶ sửa chữ ─▶ phiếu yêu cầu ─▶ soát phiếu ─▶ ★ TÌM CÁCH HIỂU NHẦM ─▶ quyết định ─▶ đáp án + log
```

---

## 1. Cách tư duy được chuyển thành pipeline

Đây là 5 bước tôi đã làm bằng tay để tìm ra 6 câu sai. Mỗi bước trở thành một module tự động.

| Bước tư duy | Tôi đã làm | Module tự động |
|---|---|---|
| 1. Đọc luật trước | Đọc rulebook + README mọi shop trước khi tính | `rules/` → RuleConfig, Lexicon |
| 2. Soi hình dạng dữ liệu | Thay chữ số bằng `9`, đếm mẫu → lộ định dạng tiền/ngày, dòng tổng, header lặp, mã có khoảng trắng | `profile/` (mặt nạ chữ số) |
| 3. Tính độc lập rồi so | Tự tính mọi câu, so với đáp án cũ | Đường A (metric) + đường B (code) |
| 4. **Truy ngược con số lệch** | "con số = công thức ngược chiều", "con số = số đơn tháng 1 thay vì quý 1", "con số = số hộp trừ số cái" | ★ `interpret/`: sinh và tính các cách hiểu nhầm |
| 5. Chỗ mơ hồ thì tìm bằng chứng | Giá 90k/hộp = 6 × 15k/cái; đối chiếu bộ train | Kiểm chứng bằng dữ liệu (trạng thái, quy đổi) + tham số có nguồn |

**Nguyên tắc rút ra:** *Một con số chỉ đáng tin khi mọi cách hiểu hợp lý khác đều đã bị loại bằng luật hoặc bằng chứng.*

---

## 2. Ràng buộc và ngân sách

| Hạng mục | Giới hạn của cuộc thi | Thiết kế |
|---|---|---|
| Tham số | ≤ 15B (tính tất cả) | 1 LLM ~8B + 1 embedding ≤ 0,6B ≈ **8,8B** |
| Thời gian | 6 giờ cho toàn bộ private | Ước tính **1–2 giờ** (mục 9) |
| Phần cứng | 2×RTX 5090 32GB | 1 bản model mỗi GPU, chia câu theo shop |
| Mạng | Không | Mọi thứ nằm sẵn trong image |
| Tái lập | Chênh PA ≤ 0,03 | Greedy, seed cố định, cache LLM, mặc định chạy batch = 1 |
| Log | Bắt buộc | 1 file JSON mỗi câu |

---

## 3. Khối ① LUẬT: đọc rulebook kỹ, không sót dòng nào

Rulebook là **"đề bài thật"**: phần lớn câu hỏi chỉ đúng khi áp đúng một dòng luật cụ thể. Vì vậy rulebook được đọc như **một bộ luật có đánh số**, không đọc lướt như văn bản.

### 3.1 Bước 1: Tách thành luật nguyên tử (code thuần)
- Phân tích cấu trúc markdown: mục, bảng, gạch đầu dòng. **Mỗi dòng bảng hoặc mỗi gạch đầu dòng là một luật nguyên tử**, có mã riêng (ví dụ `R2.07` = mục 2, dòng 7).
- Giữ nguyên văn, kèm ngữ cảnh (tiêu đề mục, tên cột bảng).
- Đánh dấu các **từ ngoại lệ** trong mỗi luật: "riêng", "trừ khi", "kể cả", "không tính", "chỉ", "khác", "nếu … thì". Các luật ngoại lệ ghi đè luật chung, và hay là bẫy (ví dụ "khác nhà cung cấp nhập nhiều tiền nhất").

### 3.2 Bước 2: Phân loại từng luật (model, mỗi luật 1 nhãn)
| Loại | Ví dụ | Đi vào |
|---|---|---|
| Định nghĩa từ ngữ | "bán chạy = theo số lượng", "khách sộp = …" | `Lexicon` |
| Công thức | doanh thu, lãi gộp, tỉ lệ huỷ, tuổi nợ | `RuleConfig.metric.*` |
| Thời gian | tuần trước, tháng này, Tết, ca | `RuleConfig.time.*` + unit test của bộ giải thời gian |
| Xếp hạng, đồng hạng | theo mã tăng dần | `RuleConfig.rank.*` |
| Xử lý dữ liệu | dòng trùng, khách lẻ, tên khác cách viết, trạng thái | `RuleConfig.data.*` (khối dữ liệu dùng) |
| Ngành | cà phê (size, topping), nhà thuốc (lô, hạn dùng), VLXD (đợt giao)… | `RuleConfig.industry.<ngành>.*` |
| Định dạng đáp án | làm tròn, "tháng nào" → YYYY-MM | `RuleConfig.format.*` |
| Không trả lời được | thông tin không có trong dữ liệu | `RuleConfig.unanswerable.*` |

### 3.3 Bước 3: Trích tham số kèm nguồn (model, JSON ràng buộc)
- Mỗi tham số bắt buộc có `source: "R2.07"` và một **trích dẫn ngắn** nguyên văn. Không có nguồn thì không được điền (dùng mặc định train, gắn `uncertain`).
- Luật không biểu diễn được thành tham số thì đưa vào `free_text_rules` (đường B đọc nguyên văn).

### 3.4 Bước 4: Kiểm độ phủ, không sót dòng nào (code thuần)
- **Mọi luật nguyên tử phải được ánh xạ** vào đúng một trong: tham số / Lexicon / free_text / format / unanswerable. Luật nào chưa được ánh xạ thì `coverage_report` báo ra.
- Mọi tham số được metric dùng tới phải có nguồn hoặc mặc định đã kiểm.

### 3.5 Bước 5: Thứ tự ưu tiên khi luật chồng nhau
```
câu hỏi nói rõ  >  README của shop  >  luật ngành  >  luật chung  >  mặc định train
```
Ví dụ: câu "bán chạy **tính theo tiền**" ghi đè luật "bán chạy = số lượng". README "1 bó = 10 cây" bổ sung cho bảng quy đổi.

### 3.6 Bước 6: Kiểm luật bằng dữ liệu train (lúc phát triển)
- **Kiểm tương đương:** RuleConfig trích tự động phải cho PA bằng RuleConfig viết tay.
- **Kiểm độ nhạy:** lật từng tham số (ví dụ tính đơn đang giao → không tính), chạy lại train. Tham số nào lật mà PA **không đổi** thì kiểm lại xem metric có thật sự dùng nó không. Tham số nào lật mà PA **tụt mạnh** thì đó là luật quan trọng, cần unit test riêng.
- **Unit test theo luật:** mỗi luật thời gian, xếp hạng, công thức có ít nhất 1 test (ví dụ R2: "tuần trước khi hôm nay là thứ Hai").

### 3.7 Bước 7: Rulebook mới ở private (lúc chạy)
- Diff theo **luật nguyên tử** với rulebook train: thêm / bớt / sửa.
- Luật bị sửa mà ánh xạ được vào tham số thì dùng giá trị mới và log.
- Luật mới không ánh xạ được thì đưa vào free_text, và metric bị luật đó chạm vào **nâng ngưỡng tin cậy**.

**Đầu ra khối luật:** `RuleConfig` (có nguồn từng tham số), `Lexicon` (cụm → `definition_id` → luật nguồn), `free_text_rules`, `coverage_report`, `rule_diff`.

---

## 4. Khối ② DỮ LIỆU (mỗi shop 1 lần, có cache)

### 4.1 Khám phá
- Mở mọi `.csv` (thử encoding utf-8-sig, utf-8, cp1258, utf-16), `.xlsx` (mọi sheet, đọc bằng calamine cho nhanh), `.sqlite` (mọi bảng và view). Cache sang parquet ở `/tmp/work`.
- Tách **nhiều bảng trong một sheet** (theo khối dòng/cột trống). Mở merged cells. Ghép header 2 tầng.
- Phân loại bảng: **giao dịch** hay **tổng hợp** (pivot, "Báo cáo", báo cáo ca). Bảng tổng hợp **không bao giờ** gộp vào dữ liệu đơn.
- Gộp các file cùng cấu trúc (theo tháng/năm, _1/_2) **theo tên cột**, không theo vị trí.

### 4.2 Soi hình dạng cột (code thuần, lõi của khối dữ liệu)
- **Mặt nạ chữ số:** `re.sub(r'\d','9', value)`, đếm mẫu theo cột. Ví dụ `9.999đ`, `9tr999`, `99/99/9999`, `99999` (serial Excel), `9 thg 9 9999`, `ORD999999`, `" ORD999999"`.
- Mỗi cột có: tỉ lệ trống, số giá trị khác nhau, top 20 giá trị, khả năng parse, min/max.
- Phân loại dòng: dữ liệu / header lặp / **dòng tổng** (mã trống hoặc mã không theo mẫu đa số, ô tiền lớn, chữ "Tổng/Cộng/TỔNG NGÀY") / ghi chú / trống.
- Quan hệ giữa các cột: `qty × price ≈ amount`, `Σ amount theo đơn ≈ total`, `total − paid = debt`.
- Phân biệt **ô trống do nhóm dòng** (cần forward-fill) với **ô trống mang nghĩa 0** (ô Bớt, Thu máy cũ để trống).

### 4.3 Agent cấu hình shop (chi tiết)

**Vai trò:** gặp shop lạ, agent tự "làm quen" như một người phân tích: xem dữ liệu, thử giả thuyết, kiểm bằng số liệu, rồi ghi ra `ShopConfig`. **Agent không tự tính đáp án và không sửa dữ liệu.** Nó chỉ quyết định *cách đọc* dữ liệu, còn code làm sạch và kiểm tra.

#### 4.3.1 Nguyên tắc thiết kế agent
1. **Code làm trước, agent chỉ lo phần code không quyết được.** Heuristic (từ khóa tên cột, mẫu hình dạng, quan hệ số) tạo **cấu hình nháp** cho khoảng 70–80% trường. Agent chỉ xử lý các trường `unknown` hoặc `ambiguous`. Cách này nhanh hơn, ít lỗi hơn, và vẫn chạy được khi model yếu.
2. **Công cụ hẹp, chỉ đọc, trả về số liệu.** Agent không chạy code tùy ý ở bước này.
3. **Mọi quyết định có bằng chứng định lượng** lưu trong `evidence`. Quyết định nào không có bằng chứng thì validator từ chối.
4. **Có ngân sách cứng:** tối đa 25 lần gọi công cụ và 3 vòng sửa mỗi shop.
5. **Tái lập:** greedy, cache theo (shop, bước), thứ tự bảng và cột cố định.

#### 4.3.2 Bộ công cụ (đầu vào và đầu ra đều có schema)
| Công cụ | Đầu vào | Trả về |
|---|---|---|
| `list_tables()` | | danh sách bảng: tên, số dòng, số cột, loại (giao dịch/tổng hợp) |
| `profile(table, col)` | | mẫu hình dạng, tỉ lệ trống, top giá trị, min/max, tỉ lệ parse được |
| `peek(table, n, where?)` | ≤ 20 dòng | các dòng mẫu (cắt ngắn ô dài) |
| `value_counts(table, col, top=50)` | | giá trị và số lần xuất hiện |
| `try_parse(table, col, kind, option?)` | kind ∈ money/date/qty/code | tỉ lệ parse thành công, ví dụ lỗi, khoảng giá trị |
| `check_relation(expr)` | ví dụ `qty*price≈amount`, `sum(amount) by order≈total` | tỉ lệ dòng thỏa, ví dụ dòng không thỏa |
| `join_rate(left_key, right_key)` | | tỉ lệ khớp hai chiều, ví dụ không khớp |
| `cross_status(table, status_col, value, other_table)` | | đơn có trạng thái X: bao nhiêu có trong sổ giao, có phiếu trả, có dòng âm, đã thu tiền |
| `unit_price_ratio(product, unit_a, unit_b)` | | tỉ lệ giá trung vị giữa 2 đơn vị (để suy hệ số quy đổi) |
| `group_shape(table, key)` | | trong mỗi nhóm theo key: cột nào hằng số, cột nào trống ở các dòng sau (để tìm cột cần forward-fill) |
| `run_invariants(config)` | ShopConfig nháp | kết quả 8 bất biến kèm số liệu |
| `submit(config)` | ShopConfig | ok, hoặc danh sách lỗi validator |

#### 4.3.3 Vòng làm việc (máy trạng thái, không để agent tự do)
```
S0  Code: khám phá + profile + heuristic → cấu hình nháp + danh sách trường unknown/ambiguous
S1  Agent: xác định bảng chính (đơn / dòng / thanh toán / nhập / lô …) và khóa join   [≤ 5 lần gọi công cụ]
S2  Agent: gán vai trò cột còn unknown/ambiguous, mỗi trường kèm bằng chứng          [≤ 8 lần gọi]
S3  Agent: ánh xạ MỌI giá trị trạng thái; giá trị mơ hồ thì dùng cross_status          [≤ 5 lần gọi]
S4  Agent: trả hàng, khách lẻ, quy đổi (unit_price_ratio cho hệ số thiếu), nguồn công nợ [≤ 5 lần gọi]
S5  Code: validator cấu hình + run_invariants
      đạt                 → XONG
      fail (≤ 3 vòng)     → gửi lỗi có số liệu cho agent, quay lại bước gây lỗi
      fail sau 3 vòng     → giữ cấu hình tốt nhất, trust thấp, ghi ambiguous
```
Mỗi bước có **prompt riêng, ngắn**: chỉ đưa phần dữ liệu liên quan, không đưa cả profile. Cách này giữ prompt dưới 4k token và giúp model ~8B làm tốt.

#### 4.3.4 Validator cấu hình (code, chặn agent sai)
- Mọi tên bảng, tên cột, giá trị trạng thái trong config **phải tồn tại thật**.
- Mỗi bảng giao dịch phải có đủ trường bắt buộc: mã đơn (hoặc nhóm dòng), ngày, số tiền dòng hoặc (số lượng + đơn giá).
- **100% giá trị trạng thái** đã được ánh xạ; mỗi ánh xạ có `evidence`.
- Cột ngày của đơn phải là ngày **đặt**: không muộn hơn ngày giao hay ngày thu khi các cột đó có mặt.
- Hệ số quy đổi là số nguyên dương; hệ số suy từ giá phải lệch < 5%.
- Không có hai trường chuẩn trỏ cùng một cột (trừ trường hợp được phép).

#### 4.3.5 Đa giả thuyết và chấm điểm bằng chứng
Với trường mơ hồ, agent đề xuất 2–3 phương án; **code chấm điểm**, không để model tự chấm:
- Điểm = Σ trọng số × (tỉ lệ thỏa quan hệ số, tỉ lệ parse, độ khớp từ khóa tên cột, độ khớp README, độ hợp lý độ lớn giá trị).
- Chọn phương án điểm cao nhất. Chênh lệch giữa hai phương án đầu < ngưỡng thì ghi `ambiguous[field]`, và mọi metric dùng trường đó bị hạ tin cậy.

#### 4.3.6 Trường hợp đặc biệt agent phải xử lý được
- **Shop chỉ có 1 bảng một dòng mỗi đơn** (không có chi tiết hàng): capability về mặt hàng là `không có`.
- **Sổ tay ghi tay**: dòng tổng ngày xen kẽ, ô trống theo nhóm, ghi chú trong ô.
- **SQLite nhiều bảng**: suy khóa join từ tên cột và `join_rate`.
- **Ngành lạ**: bảng không ánh xạ được vào sổ chuẩn thì đưa vào `events` hoặc giữ làm bảng thô đã parse, và ghi mô tả để đường B dùng.
- **README mâu thuẫn với dữ liệu**: dữ liệu thắng, log mâu thuẫn.

#### 4.3.7 Huấn luyện và đánh giá agent
- **Quỹ đạo mẫu:** viết tay ShopConfig đúng cho 30 shop train, rồi dựng quỹ đạo gọi công cụ dẫn tới config đó. Nhân lên bằng **shop làm bẩn** (đổi tên cột, trộn định dạng, chèn dòng tổng, đổi từ đồng nghĩa trạng thái, xóa README, đổi sang sqlite…).
- **Chỉ số:**
  - % trường đúng so với config chuẩn
  - % shop đạt bất biến
  - PA của câu hỏi khi dùng config do agent sinh, so với khi dùng config chuẩn (đây là chỉ số quan trọng nhất)
  - số lần gọi công cụ trung bình
- **Phương án dự phòng:** agent lỗi hoặc hết ngân sách thì dùng cấu hình nháp heuristic, trust thấp.

### 4.4 Làm sạch (thư viện chung, có unit test)
| Loại | Xử lý | Quyết định theo **cả cột** |
|---|---|---|
| Tiền | `29k`, `43.000đ`, `99,999.9999`, `129 000`, `1,05tr`, `3,24tr`, `1tr380`, `1tr5`, `-4tr300`, `(50.000)`, `VND`, dấu cách không ngắt | `.` và `,` là phân cách nghìn hay thập phân: chọn theo độ lớn và quan hệ `qty × price` |
| Ngày | ISO, `dd/mm/yyyy`, serial, `1 thg 3 2025`, `d/m` (năm lấy từ sheet), năm 2 chữ số, SA/CH | dd/mm hay mm/dd: chọn cách cho ngày đơn điệu theo mã đơn |
| Số lượng | `1 ly`, `2 hộp`, số âm | |
| Mã | strip mọi khoảng trắng | |
| Tên | NFC, gom khoảng trắng, khóa bỏ dấu để so | |
| Ô ghép | topping, IMEI "a / b", ghi chú nợ dạng chữ | |

### 4.5 Ghép đơn
- Forward-fill **trong phạm vi sheet/file**.
- **Parse trước, khử trùng sau.** Cùng đơn và giống nhau mọi ô thì tính 1 lần. Bảng không ghi theo đơn thì giữ cả hai.
- **Ngày đơn = ngày của dòng bán**, không lấy ngày của dòng trả.
- Có bất kỳ dấu hiệu trả nào (kể cả trả một phần, kể cả phiếu trả ghi sau hôm nay) thì đơn là trả hàng.
- Mã đơn bị dùng lại giữa các sổ hoặc năm: khóa đơn = (sổ, năm, mã).

### 4.6 Thực thể và quy đổi
- **Sản phẩm:** có mã thì theo mã. Tên chuẩn là dạng có dấu xuất hiện nhiều nhất. `aliases` gồm biến thể trong file, ghi tắt, và **từ điển vùng miền** (lạc/đậu phộng, lợn/heo, rán/chiên, phụ/hũ, ngao/nghêu, tràu/lóc, cốc/củ sạc, găng tay tất chân/bao tay bao chân…).
- **Khách:** có mã thì theo mã. Không có mã thì bỏ số điện thoại đính kèm, rồi **chỉ gộp khi khác ở hoa/thường, khoảng trắng, hoặc một bên không dấu**. "Lan" và "Lân" là hai khách.
- **Quy đổi:** bảng quy đổi > README > danh mục > **suy từ giá** (k nguyên, sai số < 5%). Giải hệ số lồng nhau. Áp cho **cả sổ bán lẫn sổ nhập**.

### 4.7 Sổ chuẩn (L2) và tự kiểm
- Bảng: `orders, lines, payments, products, units, customers, staff, suppliers, imports, lots, stock_moves, expenses, deliveries, …`. Thêm bảng `events` tổng quát cho ngành mới, và các bảng thô đã parse để dự phòng.
- Cờ tính sẵn: `is_revenue_order, is_walkin, shift, qty_base, is_service_fee, is_topping, size, is_rx, cogs_line`.
- **Bất biến:**
  - I1: tổng dòng ≈ tổng đơn
  - I2: (đơn, dòng) duy nhất; 1 ngày và 1 trạng thái mỗi đơn
  - I3: 100% trạng thái đã ánh xạ; ≥ 99% dòng join được sản phẩm
  - I4: tồn theo lô = tồn thẻ kho, không âm
  - I5: đã trả + còn nợ = phải trả
  - I6: mọi đơn vị có hệ số
  - I7: khoảng ngày khớp README
  - I8: `qty × price ≈ amount`
  - Cộng thêm kiểm phân phối: doanh thu ngày không vọt ×2, tỉ lệ huỷ hợp lý.
- Fail thì đưa lỗi có số liệu cho model sửa cấu hình, tối đa 3 vòng. Vẫn fail thì `trust` thấp. Bất biến không kiểm được (shop thiếu cột đối chiếu) ghi `unverifiable`, khác với `pass`.
- **ShopProfile:** `capabilities` (shop có ghi gì), `data_range`, `trust` theo bảng, `vocab`.

---

### 4.8 Cổng chất lượng dữ liệu (quyết định có tin shop này đến đâu)
Mỗi shop có một **báo cáo kiểm định** (`data_audit.json` + bản tóm tắt dễ đọc) gồm:
- **Đếm dòng theo từng loại:** đọc vào / dữ liệu / header lặp / dòng tổng / ghi chú / trùng bị khử / parse lỗi. Tổng các loại phải bằng số dòng đọc vào.
- **Bảng trạng thái:** mỗi giá trị → loại, số đơn, bằng chứng.
- **Bảng đơn vị:** mỗi (sản phẩm, đơn vị) → hệ số, nguồn hệ số.
- **Kết quả 8 bất biến:** pass / fail / unverifiable, kèm số liệu.
- **Kiểm chéo 3 chiều** khi có đủ dữ liệu: doanh thu tính từ dòng = từ tổng đơn = từ phiếu thu (trừ nợ); tồn từ lô = từ thẻ kho = nhập − bán.
- **Mức tin cậy theo bảng:** cao / trung bình / thấp. Metric dùng bảng tin cậy thấp thì nâng ngưỡng khi trả lời.

Khi phát triển, đọc báo cáo này của 30 shop train là cách nhanh nhất để phát hiện lỗi làm sạch, trước cả khi chạy câu hỏi.

---

## 5. Khối ③ CÂU HỎI

### 5.1 Sửa chữ (code thuần, `vn_normalize.py`)
1. **Bảo vệ thực thể trước** (tên trong vocab của shop, số, ngày, tiền, mã, chữ viết hoa ngắn như AA/COD) → placeholder.
2. Từng token chưa hợp lệ: tách cụm ký tự lặp → mỗi cụm thử giữ 1 hoặc 2 → bỏ đuôi **chỉ khi đuôi là ký tự bị kéo dài** → thử giải mã Telex → chọn ứng viên trúng teencode/từ đã biết, ít sửa nhất, là âm tiết hợp lệ. Không có ứng viên thì giữ nguyên.
   - Ví dụ: `coasaaaa` → có, `nhaatstttt` → nhất, `tieeenffff` → tiền, `thangsss nayyy tooongrrr bnhiu ddownnnn` → tháng nay tổng bao nhiêu đơn.
3. **Không thêm dấu cho chữ không dấu** (`luot` có thể là lượt hoặc luật).
4. Bảo vệ thực thể lượt 2 sau khi sửa (`cafe sữa (m)` → khớp `Cà phê sữa (M)`).

### 5.2 Liên kết thực thể và thời gian (code thuần)
- Trả về **mọi ứng viên** kèm điểm. Cấm `contains` với chuỗi ngắn.
- Thực thể có trong danh mục nhưng không bán trong kỳ thì đáp án là **0**.
- Bộ giải thời gian có ma trận unit test: hôm nay/qua; tuần này/trước (thứ Hai đến Chủ nhật); N ngày gần nhất; tháng này và cùng kỳ tháng trước (chặn khi tháng trước thiếu ngày); **quý**; tháng viết bằng chữ; tháng không ghi năm; Tết và trước Tết (từ `calendar.csv`); "đến hết ngày X"; tương lai → unanswerable.

### 5.3 Phiếu yêu cầu (model ~8B, 1 lần gọi, JSON ràng buộc theo schema)
```json
{
  "question_type": "value|compare|rank|argmax|count_where|list_where|share|bool|state_at|multi_value|table",
  "measure": "...", "object": {"type": "...", "ids": []},
  "filters": [{"field": "...", "values": [], "negate": false, "span": "chữ trong câu"}],
  "concepts": [{"phrase": "khách sộp", "definition_id": "...", "rule_ref": "..."}],
  "time": {"target": {}, "base": null, "kind": "flow|state", "implicit": false},
  "operation": "...", "k": 3, "threshold": {"op": ">=", "value": 0},
  "conditions": [], "outputs": [], "unit": null,
  "overrides": [{"rule": "...", "value": "...", "span": "..."}],
  "output": {"type": "number|percent|date|date_month|string|list|table|boolean"},
  "ambiguities": []
}
```
- **`span` bắt model chỉ ra chữ trong câu làm căn cứ** cho mỗi bộ lọc và mỗi ghi đè. Bộ lọc không có `span` thật thì bị loại (model không được bịa điều kiện).
- Prompt gồm: câu đã sửa, ứng viên thực thể, kỳ đã giải, capabilities, Lexicon, 6–8 ví dụ gần nhất từ train.

### 5.3A Lớp hiểu nghĩa: đọc câu chữ kỹ như người phân tích

Sửa chữ (5.1) chỉ làm sạch **mặt chữ**. **Nghĩa** được hiểu nhờ 8 cơ chế sau. Phiếu yêu cầu chỉ là chỗ ghi kết quả của chúng.

**(1) Đọc câu theo 10 loại "từ mang nghĩa".** Trước khi điền phiếu, code đánh dấu sẵn trong câu, model chỉ việc dùng:

| Loại từ | Ví dụ | Ảnh hưởng |
|---|---|---|
| Hỏi gì | bao nhiêu, mấy, nào, ai, khi nào, có … không, phải không, mấy % | `question_type`, kiểu đáp án |
| Thời gian | hôm qua, tuần rồi, T3/26, Q1, quý 2, Tết, đầu năm tới giờ, đến hết ngày | `time` |
| So sánh | so với, so, hơn, kém, tăng, giảm, chênh | `base`, `operation` |
| Phủ định | không, ko, hổng, ngoài, trừ, bỏ, không tính | `negate` |
| Lượng từ | mỗi, từng, tất cả, tổng, trung bình, bình quân, cả | phép gộp và **mẫu số** |
| Thứ hạng | nhất, nhì, thứ hai, top 3, 5 … nhiều nhất | `k`, thứ hạng cụ thể |
| Ngưỡng | từ … trở lên, ít nhất, hơn, quá, dưới, không quá | `threshold` |
| Đơn vị | ly, lon, gói, viên, hộp, thùng, cây, cuộn, đồng, triệu | `unit`, quy đổi |
| Đối tượng | món, hàng, khách, nhân viên, NCC, nhóm, lô, kênh | `object` |
| Ghi đè | tính theo, kể cả, chỉ tính, không tính, riêng | `overrides` |

**(2) Xác định loại câu trước, điền ô sau.** Mỗi loại câu có bộ ô bắt buộc riêng (so sánh phải có base, đúng/sai phải có điều kiện, nhiều phần phải đủ phần…).

**(3) Mọi từ nghiệp vụ phải có nguồn trong rulebook.** Mỗi từ gắn với một `definition_id` → mã luật. Từ không gắn được ("bán chậm", "nên nhập thêm") thì **không tự chế định nghĩa**: đánh dấu mơ hồ và bỏ trống.

**(4) Chữ nào cũng phải "có việc".** Mỗi từ mang nghĩa đã đánh dấu ở (1) phải được dùng trong phiếu (`span`). Có từ mang nghĩa không được dùng, ví dụ câu có "không" mà phiếu không có phủ định, thì phiếu thiếu ý, phải sinh lại. Ngược lại, mỗi bộ lọc trong phiếu phải có chữ trong câu làm căn cứ (không bịa điều kiện).

**(5) Phạm vi phủ định và phạm vi bộ lọc.** Phủ định gắn đúng vào bộ lọc nó bổ nghĩa. Câu có phủ định kèm từ nối ("không tính đơn giao hàng của khách sỉ") thì sinh cả hai cách hiểu (P9) và để bước 6 quyết định.

**(6) Kỳ ngầm định theo bảng luật**, không tùy tiện. Mỗi lần suy kỳ ngầm định đều ghi vào sổ giả định.

**(7) Câu nhiều phần:** đếm số phần, giữ đúng thứ tự câu nêu ("lần lượt", "mỗi tháng", "từng kênh", "kèm số").

**(8) Nhất quán qua biến thể:** với câu khó, phân tích lại từ câu gốc và câu bỏ dấu. Ba phiếu phải giống nhau; lệch nhau thì model đang hiểu theo bề mặt chữ, hạ tin cậy.

**Ví dụ đi qua lớp hiểu nghĩa:**
> `so voi th2/2025, doanh thu th2/2026 tang truong may %?` (câu minh hoạ)
> - Từ mang nghĩa: *so với* (so sánh), *th2/2025* và *th2/2026* (thời gian), *doanh thu* (đại lượng), *tăng trưởng mấy %* (hỏi gì → percent)
> - Loại câu: compare → bắt buộc có base
> - "so với **th2/2025**" → base = 02/2025, target = 02/2026
> - Mọi từ mang nghĩa đều có chỗ dùng trong phiếu → đạt
> - Tìm cách hiểu nhầm: P1 (đổi chiều) bị luật "vế sau *so với* là base" loại → còn 1 đáp án

### 5.4 Soát phiếu (luật cứng)
- **Bộ thông tin bắt buộc theo loại câu:** câu so sánh phải có base, câu đúng/sai có ≥ 1 điều kiện (≥ 2 nếu câu có từ nối), câu nhiều phần đủ số phần đúng thứ tự, câu ở dạng "số còn lại" có `kind = state`.
- Mỗi **từ phủ định**, mỗi **cụm ghi đè** ("tính theo", "kể cả") phải nằm trong `span` của đúng một bộ lọc hoặc một ghi đè.
- Mọi `definition_id` phải tồn tại trong Lexicon. Khái niệm không có nguồn trong rulebook thì **không tự chế định nghĩa**.
- Kiểu đáp án khớp kiểu câu ("tháng nào" → YYYY-MM, "mấy %" → percent, "phải không" → boolean…).
- Fail thì gọi model sinh lại 1 lần kèm thông báo lỗi. Vẫn fail thì bỏ trống.

### 5.5 Khả năng trả lời (chỉ quyết định xác định)
Trả **unanswerable** chỉ trong 3 trường hợp: (1) shop không ghi thông tin câu hỏi cần (giới tính, địa chỉ, lý do huỷ, nhóm hàng khi không có cột nhóm…); (2) kỳ hỏi nằm ngoài dữ liệu hoặc ở tương lai; (3) câu hỏi về đánh giá chủ quan (ngon, thân thiện, uy tín…). Mọi trường hợp không chắc khác thì **bỏ trống**.

---

## 6. ★ TÌM CÁCH HIỂU NHẦM (điểm đột phá)

### 6.1 Ý tưởng
Không đặt cược vào một cách hiểu duy nhất. Từ phiếu chính, **code** (không phải model) sinh ra các cách hiểu dễ nhầm (16 loại), tính tất cả trên sổ sạch (mỗi cách vài mili giây), rồi dùng **luật câu chữ** và **bằng chứng dữ liệu** để loại. Pipeline làm *trước khi trả lời* đúng việc tôi đã làm bằng tay khi truy ra 6 câu sai.

### 6.2 Danh mục cách hiểu nhầm (mỗi cách kèm luật loại)
| # | Cách hiểu nhầm | Sinh ra khi | Luật để loại / chọn |
|---|---|---|---|
| P1 | **Đổi chiều so sánh** (target ↔ base) | Có `base` | Vế đi sau "so với / so" là base. Câu "So với X, Y" thì X là base |
| P2 | **Quý ↔ tháng** | Có số 1–4 cạnh "Q/quý/T/th" | Có chữ "Q"/"quý" thì là quý; có "T"/"th"/"tháng" thì là tháng |
| P3 | **Bỏ một vế** của câu đúng/sai | `conditions ≥ 2` hoặc câu có từ nối | Có "và / rồi / lại / liên tục / luôn" thì mọi vế bắt buộc |
| P4 | **Đơn vị** (giữ nguyên hay quy về đơn vị cơ bản / đơn vị câu hỏi) | Câu hỏi số lượng, nhập − bán | Câu nêu đơn vị thì quy mọi vế về đơn vị đó; không nêu thì dùng đơn vị cơ bản |
| P5 | **Tính / không tính đơn đang giao, đơn trả một phần** | Mọi metric doanh thu, số đơn | Theo RuleConfig, trừ khi câu có `override` |
| P6 | **Phạm vi tên**: khớp hẹp (đúng 1 SKU) hay rộng (mọi SKU cùng tên gốc) | Liên kết thực thể có > 1 ứng viên | Câu nêu biến thể (size, quy cách) thì khớp hẹp; câu nêu chung chung thì là mơ hồ |
| P7 | **Năm**: năm nay hay năm trước | Kỳ không ghi năm | Kỳ sau hôm nay thì lùi năm; còn lại là năm nay |
| P8 | **"Tháng trước"** so với hôm nay hay so với kỳ được hỏi | Câu có kỳ tường minh + "tháng trước" | Có kỳ tường minh thì tính theo kỳ được hỏi |
| P9 | **Phạm vi phủ định** (áp lên cả cụm hay một phần) | Câu có từ phủ định + từ nối | Không quyết được bằng luật thì đánh dấu mơ hồ |
| P10 | **Tiêu chí xếp hạng** (số lượng, tiền, số đơn, số ngày) | Câu xếp hạng | Theo Lexicon/RuleConfig, trừ khi câu có "tính theo …" |
| P11 | **Biên ngưỡng** (≥ hay >) | Có `threshold` | "từ … trở lên", "ít nhất" → ≥; "hơn/quá/trên" → > |
| P12 | **Phát sinh trong kỳ hay số còn lại** | Câu về nợ, tồn | "còn / đến hết ngày" thì là số còn lại |
| P13 | **Mẫu số của trung bình** (mỗi đơn / mỗi ngày / mỗi khách / mỗi ngày có bán) | Câu có "trung bình", "bình quân", "mỗi" | Chữ sau "mỗi" quyết định mẫu số; không có thì theo rulebook |
| P14 | **Thứ hạng cụ thể** (nhất hay nhì; top-k hay phần tử thứ k) | Câu có "nhì / thứ hai / thứ 3" | "thứ k" → một phần tử ở hạng k; "top k" → danh sách k |
| P15 | **Mẫu số tỉ trọng** (cả shop hay trong nhóm) | Câu "chiếm mấy %" | Theo rulebook (doanh thu cả shop, gồm vãng lai và phí phục vụ), trừ khi câu nói "trong nhóm …" |
| P16 | **Đối tượng "ai"**: khách hay nhân viên | Câu hỏi "ai" | Có "nhân viên / NV / bán giỏi / chăm chỉ / đứng ca" → nhân viên; còn lại → khách |

### 6.3 Quy tắc quyết định
```
cách_hợp_lệ = [phiếu chính] + [mọi cách hiểu nhầm KHÔNG bị luật loại]
đáp_án = {tính(c) cho c trong cách_hợp_lệ}

nếu tất cả cách hợp lệ cho CÙNG một đáp án      → trả lời (câu không mơ hồ)
nếu luật chọn được đúng 1 cách                  → trả lời theo cách đó
nếu còn ≥ 2 cách hợp lệ với đáp án KHÁC nhau    → bỏ trống (mơ hồ thật)
```
- **Độ mơ hồ** (số cách hợp lệ, số đáp án khác nhau) trở thành một đặc trưng của bước quyết định.
- **Ví dụ đúng 6 lỗi cũ:** P1 chặn 2 câu so sánh ngược chiều; P2 chặn câu "Q1/2026"; P3 chặn câu "T3 > T2 và T4 > T3"; P4 chặn 2 câu nhập − bán. Cả 6 đều bị **luật câu chữ** chọn đúng, không phụ thuộc model.

### 6.4 Kiểm chứng độc lập (chỉ chạy khi cần)
- **Đường B (agent viết code, chi tiết):**
  - **Khi nào chạy:** câu nằm ngoài phiếu; còn ≥ 2 cách hiểu hợp lệ cho ra đáp án khác nhau; trust shop thấp; ngành chưa gặp; hoặc rulebook có luật mới chạm vào metric.
  - **Đầu vào:** **câu gốc** (không phải câu đã sửa, không phải phiếu, để giữ tính độc lập), as_of, schema sổ chuẩn (tên bảng, cột, kiểu, 3 dòng mẫu), RuleConfig tóm tắt + free_text_rules, thư viện hàm tiện ích (`period(...)`, `revenue_orders(...)`, `qty_base(...)`) để code ngắn và ít sai.
  - **Đầu ra:** hàm `answer(L2, rules) -> value`, kèm một dòng mô tả bằng lời "đã tính gì".
  - **Sandbox:** subprocess, timeout 10 giây, giới hạn RAM 4GB, không mạng, chỉ đọc parquet của sổ chuẩn. Cấm `import os/sys/subprocess/requests`, cấm ghi file (kiểm bằng phân tích AST trước khi chạy).
  - **Vòng sửa:** lỗi chạy (exception, sai kiểu trả về) thì gửi lỗi lại, tối đa 2 lần. Vẫn lỗi thì B không có kết quả.
  - **Kiểm:** mô tả bằng lời của B được so nghĩa với câu gốc (khứ hồi cho B). Kết quả B được so với các cách hiểu ở mục 6.3: B khớp đúng một cách hiểu hợp lệ nào thì cách hiểu đó được cộng điểm.
- **Nhất quán qua biến thể:** phân tích lại câu ở dạng gốc và dạng bỏ dấu. Chỉ chạy cho câu có phủ định, so sánh hoặc nhiều vế khi độ tin cậy ở mức trung bình.
- **Khứ hồi:** dịch phiếu ngược thành câu bằng template code thuần, so nghĩa với câu gốc. Rẻ, nên chạy cho mọi câu nằm trong phiếu.

### 6.5 Quyết định cuối
Trả lời khi **đủ cả hai điều kiện**:
1. Phiếu qua soát.
2. **Một trong hai:** tìm cách hiểu nhầm cho ra một đáp án duy nhất, **hoặc** đường B khớp đường A.

Thêm vào đó, điểm tin cậy tổng hợp (từ trust shop, độ chắc liên kết thực thể, độ mơ hồ, ngành mới hay không…) phải vượt ngưỡng. Ngưỡng được hiệu chỉnh bằng k-fold theo shop trên train.

---

## 7. Hàng rào chống lỗi ngớ ngẩn (30 kiểm tra tự động)

Mỗi lỗi từng gặp (của file cũ, của chính tôi lúc dò, hoặc trong checklist của bạn) có **một hàng rào chạy tự động**. Hàng rào nào fail thì câu bị bỏ trống hoặc shop bị hạ trust.

| # | Lỗi ngớ ngẩn | Hàng rào |
|---|---|---|
| G1 | Dòng tổng / header lặp lọt vào doanh thu | Phân loại dòng + I1 + ngoại lai doanh thu ngày |
| G2 | Mã có khoảng trắng làm join hụt | strip mọi mã + `join_rate` ≥ 99% |
| G3 | Sót trạng thái hiếm (chỉ xuất hiện vài tháng) | Ánh xạ **mọi** giá trị; giá trị lạ thì báo lỗi |
| G4 | "Không giao được" hiểu sai | Kiểm chứng trạng thái bằng sổ giao / phiếu thu |
| G5 | Khử trùng trước khi parse | Thứ tự cố định: parse → khử trùng |
| G6 | Fill tràn qua sheet | Fill theo khóa nhóm |
| G7 | Ô trống mang nghĩa 0 bị fill | Phân biệt ô trống do nhóm dòng và ô trống mang nghĩa 0 |
| G8 | Lấy nhầm ngày giao hoặc ngày thu làm ngày đơn | Cấu hình ghi rõ loại ngày, kèm bằng chứng |
| G9 | Ngày đơn lấy từ dòng trả | Ngày đơn = ngày của dòng bán |
| G10 | Gộp bảng tổng hợp vào giao dịch | Phân loại bảng giao dịch / tổng hợp |
| G11 | Dấu `.` `,` hoặc dd/mm hiểu sai | Quyết định theo cả cột + I8 + I7 |
| G12 | Quên quy đổi hộp → cái, thùng → lon | I6 + P4 |
| G13 | Topping bị đếm là size M | Tách topping; size theo hậu tố của món |
| G14 | Phí phục vụ được xếp hạng như món | Cờ `is_service_fee` |
| G15 | Gộp nhầm khách "Lan"/"Lân" | Luật gộp khách |
| G16 | Khách lẻ bị xếp hạng | Cờ `is_walkin` |
| G17 | Lọc tên bằng `contains` | Cấm `contains`; liên kết trả mọi ứng viên |
| G18 | Câu dùng tên gọi khác với file (lạc/đậu phộng) | Từ điển vùng miền + aliases |
| G19 | So sánh ngược chiều | P1 + soát phiếu |
| G20 | Quý đọc thành tháng | P2 + kiểm hợp lý: số đơn quý > số đơn mỗi tháng |
| G21 | Câu "A và B" bỏ sót vế | P3 + soát phiếu |
| G22 | Mẫu số tỉ lệ sai | Mẫu số cố định trong code metric |
| G23 | ≥ và > lẫn nhau | P11 |
| G24 | Làm tròn sai do số thực | Số nguyên + Decimal HALF_UP |
| G25 | Kiểu đáp án sai (số dạng chuỗi, list 1 phần tử thành chuỗi) | Schema đáp án ở bước cuối |
| G26 | Trả unanswerable cho câu trả lời được | Chỉ 3 nguồn xác định (mục 5.5) |
| G27 | Mặt hàng không có giao dịch bị trả unanswerable | Trả 0 |
| G28 | Model bịa điều kiện lọc | Mỗi bộ lọc phải có `span` thật |
| G29 | `except: pass` che lỗi | Cấm trong code review; `run_summary` đếm số câu theo từng nhánh |
| G30 | Dấu của tăng trưởng sai | Kiểm hợp lý: dấu khớp phép so A > B |

---

## 8. Model (dùng sẵn + fine-tune, không tự build)

| Vai trò | Model đề xuất | Tham số | Giấy phép |
|---|---|---|---|
| LLM chính | **Qwen3-8B** (hoặc Qwen2.5-7B-Instruct / Coder-7B) | ~8,2B (~7,6B) | Apache-2.0 |
| Embedding | **bge-m3** (hoặc multilingual-e5-small) | ~0,57B (~0,12B) | MIT |

> Phải tự kiểm **ngày công bố trước 01/10/2026**, giấy phép, và số tham số thực tế, rồi ghi vào README.

**Fine-tune LoRA** (r = 32, lr = 1e-4, 2–3 epoch, merge vào trọng số gốc) cho 3 việc: câu → phiếu; README + profile → cấu hình shop; câu → code.

**Dữ liệu fine-tune** (chỉ train và dữ liệu tự tổng hợp):
- Câu train kèm phiếu đã được kiểm (đường A ra đúng gold).
- Câu sinh từ phiếu mẫu, nhãn đúng theo cách xây dựng.
- Bản "làm bẩn" ngôn ngữ (Telex, kéo dài, teencode, không dấu).
- **Ca khó nhắm đúng 16 cách hiểu nhầm** ở mục 6.2.
- Shop làm bẩn cho việc cấu hình.

**Chạy:** vLLM, greedy, tắt chế độ thinking, JSON ràng buộc theo schema, cache theo hash prompt.

---

## 9. Thời gian chạy: có lâu không?

**Không.** Phần nặng duy nhất là gọi model, và mỗi câu chỉ gọi **1 lần là chính**. Các phần còn lại là pandas trên sổ đã sạch.

Ước tính cho ~15 shop và ~1.000 câu (cần đo lại trên máy thật):

| Bước | Mỗi đơn vị | Tổng |
|---|---|---|
| Load model, khởi động | | 2–3 phút |
| Đọc + soi + làm sạch mỗi shop (code) | 10–90 giây | ~10 phút |
| Cấu hình shop (model, 5–15 lần gọi) | 30–90 giây | ~15 phút (2 GPU song song: ~8 phút) |
| Sửa chữ + liên kết + thời gian (code) | < 50 ms/câu | ~1 phút |
| **Phiếu yêu cầu (1 lần gọi model)** | 1,5–3 giây/câu | 25–50 phút |
| Tìm cách hiểu nhầm (3–8 cách × vài ms) | < 0,5 giây/câu | ~5 phút |
| Khứ hồi template (code) | ~ms | ~0 |
| Đường B / sinh lại phiếu / biến thể (chỉ ~20–30% câu) | 3–5 giây/câu | 10–25 phút |
| **Tổng, chia 2 GPU** | | **~45–90 phút** |

**Mẹo giữ nhanh và nhẹ:**
- Mặc định **batch = 1** để tái lập tuyệt đối; thời gian vẫn dư nhiều. Batch lớn chỉ dùng nếu đo thấy thật sự cần.
- Prompt ngắn: chỉ đưa ứng viên thực thể và vài ví dụ gần nhất (~2–3k token). Phiếu ≤ 400 token đầu ra.
- Cache: sổ chuẩn mỗi shop, kết quả mọi lần gọi model, kết quả metric theo (shop, phiếu).
- Các bước kiểm tra tốn thêm (đường B, biến thể) **chỉ chạy khi cần**.
- **Chế độ giảm tải:** thời gian còn < 15% thì tắt mọi bước tùy chọn, chỉ trả lời câu tin cậy cao. Đáp án được ghi dần vào file sau mỗi câu.

---

## 10. Đánh giá: tự truy lỗi trên train (giống cách tôi làm, nhưng hợp lệ)

- Chạy pipeline trên **30 shop train**, chia k-fold theo shop.
- Mỗi câu sai được **tự truy ngược**: thử 16 cách hiểu nhầm, cộng các biến thể dữ liệu (bỏ hay tính đơn đang giao, không quy đổi, khớp tên rộng), xem cách nào tái tạo được **đáp án gold**. Kết quả là ngay **loại lỗi**, ví dụ "lỗi P1 ở shop X", "lỗi G3 trạng thái hiếm".
- Bảng tổng hợp lỗi theo loại → sửa đúng loại có nhiều lỗi nhất → **chạy lại toàn bộ, diff** số câu đổi từ đúng sang sai và ngược lại.
- Đo thêm trên **shop làm bẩn** (đổi tên cột, trộn định dạng, chèn dòng tổng, đổi sang sqlite…) để chắc pipeline chịu được shop lạ.
- **Không dùng đáp án public/private** để chỉnh hay train. Điểm cuối tính trên private test có loại shop mới, nên chỉnh theo public vừa vô ích vừa phạm luật 6.

---

## 11. Tuân thủ (kiểm trước khi nộp)
- [ ] Không có dữ liệu hay đáp án public/private trong repo hoặc image
- [ ] Không có logic nào theo `id` câu hỏi hay `business_id`; mọi cấu hình shop sinh lúc chạy
- [ ] Model công bố trước 01/10/2026, giấy phép hợp lệ, tổng ≤ 15B, đã khai trong README
- [ ] Chạy được với mạng tắt; 2 lần chạy cho kết quả giống nhau
- [ ] Log đủ: id, đáp án, thời gian, phiếu, các cách hiểu đã thử, quyết định
- [ ] Chạy toàn bộ dưới 6 giờ trên cấu hình tương đương

---

## 12. Kế hoạch 10 ngày (hạn Docker 16/11)

| Ngày | Việc | Xong khi |
|---|---|---|
| 1–2 | Thư viện làm sạch + soi hình dạng + sổ chuẩn + bất biến (cấu hình viết tay cho 3–4 shop train mẫu) | Bất biến đạt trên các shop mẫu |
| 2–3 | Bộ giải thời gian + thư viện metric (~30 metric phổ biến) | Unit test đạt; khớp gold phần lớn câu của shop mẫu |
| 3–4 | Phiếu yêu cầu (model gốc + few-shot) + soát phiếu + **tìm cách hiểu nhầm** | PA k-fold có số đầu tiên |
| 4–6 | Cấu hình shop tự động (model + bằng chứng + vòng sửa) cho cả 30 shop | ≥ 90% shop đạt bất biến |
| 5–7 | Sinh dữ liệu và fine-tune LoRA (song song) | PA k-fold tăng so với few-shot |
| 7–8 | Đường B, khứ hồi, quyết định, hiệu chỉnh ngưỡng | Precision khi trả lời ≥ mục tiêu |
| 8–9 | Tự truy lỗi trên train, sửa theo loại lỗi, test trên shop làm bẩn | Không còn lỗi nhóm G nào lặp lại |
| 9–10 | Docker, `predict.sh`, đo thời gian, kiểm tái lập, README | Checklist mục 11 đạt hết |


---

## 13. Xử lý trường hợp ngoài train/public và làm pipeline mạnh thật

### 13.1 Năm cơ chế giúp pipeline xử lý dữ liệu và câu hỏi chưa từng gặp
1. **Dữ liệu: dựa trên năng lực, không dựa trên tên file.** Pipeline không hỏi "đây có phải shop cà phê không". Nó hỏi "shop này **có** bảng đơn, dòng hàng, lô, công nợ, giao hàng… không". Ngành mới có thêm khái niệm thì đưa vào `events` hoặc giữ thành bảng thô đã parse, đường B vẫn truy cập được.
2. **Phép tính ghép từ khối nhỏ, không chỉ là danh sách metric cố định.** Phiếu yêu cầu là một **kế hoạch truy vấn** gồm các khối:
   `chọn bảng → lọc (kỳ, đối tượng, điều kiện, phủ định) → nhóm theo → gộp (sum/count/count_distinct/avg/min/max) → so sánh (diff/growth/share) → xếp hạng / ngưỡng → định dạng`.
   Câu hỏi mới chỉ là **tổ hợp mới của các khối cũ** nên vẫn tính được mà không cần viết thêm code. Metric đặc thù (công nợ, lô, Tết…) là các khối được đóng gói sẵn. Việc này được kiểm bằng cách: lấy câu train **ẩn đi metric tương ứng** xem bộ ghép khối có tự tính đúng không.
3. **Đường B là lưới an toàn** cho những gì bộ khối không biểu diễn được, có hàm tiện ích để code ngắn và ít sai.
4. **Phát hiện "lạ"** để tự thận trọng: shop có profile khác xa các shop train (khoảng cách embedding của profile), câu có phiếu ít gặp, rulebook có luật mới. Càng lạ thì ngưỡng trả lời càng cao. Sai bị trừ điểm, nên thận trọng có chọn lọc là cách mạnh nhất để không mất điểm ở vùng lạ.
5. **Không cứng tên, không cứng cấu trúc:** cấu hình shop sinh lúc chạy, Lexicon sinh từ rulebook, alias sinh từ dữ liệu. Không có gì gắn với 30 shop train.

### 13.2 Bài kiểm tra quan trọng nhất: ngành giả lập hoàn toàn mới
Đây là cách duy nhất để **đo** sức mạnh với trường hợp ngoài, trước khi gặp private:
- Viết **bộ sinh dữ liệu tổng hợp** cho 6–10 ngành **không có** trong train/public: spa, phòng gym, nhà sách, cửa hàng thú cưng, tiệm in, quán karaoke, cửa hàng nông sản, phòng khám nha… Mỗi ngành có cấu trúc file riêng (sổ tay, POS, sqlite), định dạng bẩn có chủ đích, và khái niệm riêng (gói tập, buổi, giờ phòng, hội viên…).
- Bộ sinh **biết đáp án đúng**, vì nó tạo ra dữ liệu. Câu hỏi được sinh từ phiếu mẫu rồi viết lại cho tự nhiên.
- Kèm một rulebook "ngành mới" với vài luật mới, để kiểm khả năng đọc luật lạ.
- **Không ai trong đội nhìn dữ liệu này khi phát triển**; chỉ dùng để đo.

### 13.3 "Mạnh" nghĩa là đạt các con số này (mục tiêu đo, không phải lời hứa)
| Bài đo | Mục tiêu | Ý nghĩa |
|---|---|---|
| PA k-fold theo shop trên train | ≥ 0,90 | Đúng trên dữ liệu quen |
| **Precision khi trả lời** | ≥ 0,97 | Đã trả lời thì gần như luôn đúng |
| PA trên shop làm bẩn | sụt < 3 điểm so với shop gốc | Chịu được định dạng lạ |
| PA trên **ngành giả lập mới** | ≥ 0,75, precision ≥ 0,95 | Chịu được ngành lạ; không đoán bừa |
| Tỉ lệ câu bị unanswerable nhầm | < 1% | Không mất điểm vì từ chối sai |
| Hai lần chạy | giống nhau 100% | Tái lập |
| Thời gian toàn bộ | < 2 giờ | Dư xa giới hạn |

### 13.4 Vòng tăng sức mạnh (lặp mỗi ngày từ ngày 3)
```
chạy toàn bộ bài đo → tự truy lỗi (16 cách hiểu nhầm + biến thể dữ liệu) → bảng lỗi theo loại
→ sửa ĐÚNG loại nhiều lỗi nhất (luật, hàng rào, khối tính, dữ liệu fine-tune) → chạy lại + diff
```
Mỗi lỗi mới đều phải thành **một hàng rào hoặc một test**, để không bao giờ lặp lại.

### 13.5 Điểm yếu còn lại (nói thẳng)
- **Model ~8B hiểu câu phức tạp chưa từng gặp** vẫn kém model lớn. Bù bằng: phiếu có cấu trúc, đánh dấu từ mang nghĩa, tìm cách hiểu nhầm, và **bỏ trống khi mơ hồ**.
- **Định dạng file cực lạ** (báo cáo dạng văn bản, bảng xoay ngang theo ngày) có thể vượt khả năng agent. Bù bằng: trust thấp và thận trọng.
- **Rủi ro lớn nhất là thời gian triển khai (10 ngày), không phải thiết kế.** Ưu tiên có **phiên bản chạy đầu-cuối trước ngày 3** (dù đơn giản), rồi mới tăng sức mạnh theo bảng lỗi. Một pipeline đơn giản nhưng đo được và sửa được hằng ngày sẽ mạnh hơn một pipeline hoàn hảo trên giấy.

---

## Phụ lục A. Đặc tả sổ chuẩn (L2)

| Bảng | Cột bắt buộc | Cột tùy chọn | Cờ dẫn xuất |
|---|---|---|---|
| `orders` | order_key, order_id, order_date, status_raw, status_class | order_ts, customer_key, customer_name, channel, pay_method, staff_id, total_payable, order_discount, ship_fee, vat, note, src | is_revenue_order, is_walkin, shift, hour, has_return |
| `lines` | order_key, line_no, product_key, qty, unit, line_amount | unit_price, line_discount, cost_per_unit_sold, topping_of | qty_base, is_service_fee, is_topping, size, cogs_line |
| `payments` | order_key, pay_ts, amount | method, note | is_same_day |
| `products` | product_key, name | code, aliases[], group, base_unit, catalog_cost, list_price, rx_flag, active_ingredient, brand, supplier | |
| `units` | product_key, unit, factor | source (table/readme/catalog/price) | |
| `customers` | customer_key, name | code, phone, group, reg_date | |
| `staff` | staff_id | name | |
| `imports` | receipt_id, date, product_key, qty, unit, line_amount | supplier, status, note | qty_base, cancelled |
| `lots` | lot_id, product_key, import_date, qty_in_base, unit_cost | expiry_date | |
| `lot_moves` | lot_id, ts, qty_base, kind (sale/return/adjust) | order_key | |
| `stock_moves` | product_key, ts, qty_base, kind | ref | balance_after |
| `expenses` | date, amount | category, note | |
| `deliveries` | order_key, batch_no, delivery_ts, product_key, qty | | |
| `price_history` | product_key, effective_date, new_price | old_price | |
| `events` | entity_type, entity_id, event_type, date, qty, amount | ts, attrs_json | (cho ngành mới) |

Kèm theo: `ShopProfile` (capabilities, data_range, trust theo bảng, ambiguous[], vocab) và `data_audit.json`.

## Phụ lục B. Danh mục metric (đường A)

| Nhóm | Metric | Tham số RuleConfig dùng |
|---|---|---|
| Doanh số | revenue, order_count, avg_order_value, revenue_by(day/hour/shift/channel/pay/staff/group/customer_group), share_pct, growth_pct, diff | metric.revenue.*, time.* |
| Mặt hàng | qty_sold(base/unit), item_revenue, n_orders_with_item, top_k(by qty/amount/orders/profit), slow_movers, never_sold_in_window | rank.*, unit.* |
| Khách | n_customers, orders_per_customer, top_customers(by orders/amount/days/avg), count_customers_where, cancel_by_customer | customer.* |
| Nhân viên | staff_revenue, staff_days | staff.* |
| Huỷ/trả | cancel_count, cancel_rate, return_count | metric.cancel_rate.* |
| Lợi nhuận | gross_profit, net_profit (− chi phí), item_profit | cost.* |
| Công nợ | debt_at(date), debt_by_customer, aging_buckets, new_debt, debt_paid | debt.* |
| Kho | stock_at, stock_value, lots_expiring(window), capital_in_expiring, import_amount, import_minus_sales, main_supplier, top_supplier_by_amount | inventory.* |
| Giá | price_effect(item, date, N) | price.* |
| Thời gian đặc biệt | tet_revenue, pre_tet, peak_hour, busiest_day | time.tet.*, time.shift.* |
| Ngành | delivery_batches, warranty_*, installment_*, size_count, topping_count… | industry.* |

Mỗi metric: hàm thuần, nhận (L2, RuleConfig, tham số đã giải), trả giá trị kèm `assumptions[]`. Có unit test trên shop nhỏ tính tay.

## Phụ lục C. Cấu trúc thư mục và giao diện

```
datamate/
  predict.sh                  # /data → /result
  run.py                      # điều phối: luật → mỗi shop → mỗi câu; ghi dần; watchdog
  rules/       parse_rulebook.py  extract_params.py  coverage.py  diff.py
  data/        discover.py  profile.py  heuristics.py  agent_config/ (tools.py, states.py, prompts/)
               clean/ (money.py, date.py, qty.py, code.py, name.py, composite.py)
               assemble.py  entities.py  units.py  flags.py  invariants.py  audit.py
  question/    normalize.py (vn_normalize)  meaning_marks.py  entity_link.py  time_resolver.py
               dsl_schema.json  dsl_parse.py  dsl_validate.py  answerability.py
  interpret/   perturb.py (16 loại)  eliminate.py  decide.py
  metrics/     sales.py items.py customers.py staff.py profit.py debt.py inventory.py price.py special.py industry/
  pathB/       prompt.py  sandbox.py  ast_guard.py
  verify/      roundtrip_templates.py  sanity.py  variants.py
  output/      format.py  answer_schema.py  logger.py
  models/      llm.py (vLLM, cache)  embed.py
  eval/        kfold.py  dirty_shops.py  forensic.py (tự truy lỗi)  regression_diff.py
  tests/       test_clean_*.py  test_time_matrix.py  test_metrics_*.py  test_dsl_validate.py  test_perturb.py
```

**Giao diện chính (hợp đồng giữa các module):**
- `load_rules(path) -> RuleConfig, Lexicon, coverage_report`
- `onboard(shop_dir, rules) -> L2, ShopProfile, ShopConfig, audit`
- `parse_question(q, as_of, profile, rules) -> Normalized, Marks, Entities, Periods`
- `to_dsl(...) -> DSL` · `validate(dsl, marks) -> ok | errors`
- `interpretations(dsl, marks) -> [DSL]` · `eliminate(cands, marks, rules) -> [DSL]`
- `compute(dsl, L2, rules) -> Value, assumptions`
- `decide(features) -> answer | abstain | unanswerable` · `format(value, dsl.output) -> submission_line`

## Phụ lục D. Mức độ hoàn thiện của thiết kế (tự đánh giá)

| Phần | Mức chi tiết | Ghi chú |
|---|---|---|
| Luật (rulebook) | Đủ để code | |
| Dữ liệu: khám phá, soi, làm sạch, ghép, thực thể, quy đổi, bất biến, kiểm định | Đủ để code | |
| Agent cấu hình shop | Đủ để code | Cần viết prompt cho từng bước S1–S4 khi code |
| Câu hỏi: sửa chữ (đã có code), hiểu nghĩa, phiếu, soát | Đủ để code | Template khứ hồi viết dần theo `question_type` |
| Tìm cách hiểu nhầm (16 loại) | Đủ để code | |
| Đường B | Đủ để code | |
| Metric | Danh mục đủ; công thức chi tiết lấy theo rulebook | |
| Model, fine-tune, thời gian, tuân thủ | Đủ | Số đo thời gian thực phải đo trên máy |
| **Chưa có, chỉ có khi chạy thực tế** | Ngưỡng quyết định, trọng số chấm điểm bằng chứng, danh sách teencode đầy đủ | Hiệu chỉnh trên train (k-fold) |
