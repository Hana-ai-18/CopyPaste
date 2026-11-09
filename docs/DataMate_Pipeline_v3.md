# DataMate Pipeline v3: "Đặc tả trước, kiểm bằng quan hệ"

> Bản này xây lại pipeline từ v2 (`DataMate_Pipeline_v2.md`) và dựa trên các paper text-to-SQL / table QA / kiểm thử chương trình mới nhất.
> Mục tiêu: **khác biệt so với các đội khác**, mạnh, nhưng vẫn chỉ **1 model ~7–8B**, khoảng **2 lần gọi model mỗi câu**.

---

## 0. Một trang tóm tắt

**Ý tưởng nền mà đội nào cũng có:** đưa bảng và câu hỏi cho LLM, LLM viết pandas/SQL, chạy ra số, có thể bỏ phiếu vài lần.
Cách này hỏng ở đúng những chỗ đã làm mình sai 6 câu public: hiểu ngược chiều so sánh, hiểu "Q1" thành tháng 1, kiểm thiếu vế của câu điều kiện, quên quy đổi hộp → cái. **Bỏ phiếu không cứu được**, vì cả 5 lần sinh đều mắc cùng một cách hiểu sai.

**v3 khác ở 5 điểm (đều chạy offline, rẻ):**

| # | Đột phá | Bắt được lỗi gì | Học từ paper |
|---|---|---|---|
| ★1 | **Đặc tả có các cách đọc khác** (Spec + Readings): model viết *đặc tả* câu hỏi, kèm mọi cách đọc khác ở từng chỗ mơ hồ. Code chọn cách đọc bằng rulebook. | Hiểu ngược chiều, Q1/tháng 1, phạm vi "và/hoặc" | AMBROSIA (các loại mơ hồ) |
| ★2 | **Hai bộ sinh khác bản chất**: một *trình biên dịch* đặc tả → chuỗi phép toán bảng (không dùng LLM), và một LLM viết code tự do từ câu gốc. Hai đường khớp nhau thì tin. | Lỗi do một đường đơn lẻ | CHASE-SQL (đa dạng ứng viên), Chain-of-Table (chuỗi phép bảng) |
| ★3 | **Kiểm bằng quan hệ (metamorphic) + luật thành test**: chạy lại chương trình trên dữ liệu đã *biến đổi có kiểm soát* (tiêm đơn huỷ, đổi đơn vị, nhân bản dòng…), kiểm kết quả có đổi đúng như luật không. Mỗi dòng rulebook thành một assert trên bảng trung gian. | **Chỉ lỗi cài đặt**: quên lọc huỷ/trả, quên quy đổi, cộng trùng, biên kỳ. Không bắt lỗi hiểu câu. | Metamorphic testing, CHESS (unit test chọn ứng viên), LEVER (kiểm bằng kết quả chạy) |
| ★4 | **Hỏi đúng chỗ phân vân + dịch ngược bằng code**: không bỏ phiếu cả câu; hỏi model câu có/không về đúng trường khác nhau giữa các cách đọc; dịch đặc tả ngược thành câu chuẩn để model xác nhận cùng nghĩa. | Lỗi hiểu câu còn sót sau ★1 | AMBROSIA, CHASE-SQL (chọn bằng so sánh cặp) |
| ★5 | **Kiểm toán**: tự dò lại câu sai như cách làm bằng tay. Hai đường lệch → tìm phép hiểu nhầm nào tái tạo đúng con số kia → biết bên nào sai và vì sao. Trên train, tự gắn nhãn nguyên nhân cho mọi câu sai. | Mọi lỗi có trong danh mục, kể cả khi hai đường cùng đồng ý | LEVER (gộp theo kết quả chạy), ReFoRCE (hoãn và sửa có mục tiêu) |

Hỗ trợ (không phải điểm nhấn): **xưởng shop giả lập nhỏ** để hiệu chỉnh ngưỡng và đo OOD (theo OmniSQL, Auto-Tables). **Không dùng dữ liệu test.**

Cộng thêm phần quyết định: một **bộ chấm nhẹ** (logistic regression trên các tín hiệu kiểm chứng) được hiệu chỉnh để quyết định trả lời hay bỏ trống. Theo nghiên cứu selective prediction, tự-nhất-quán (bỏ phiếu) hiệu chỉnh kém; tín hiệu từ kiểm chứng tốt hơn nhiều.

```
① LUẬT     rulebook ─▶ luật nguyên tử ─▶ RuleConfig + RuleTests (mỗi luật = 1 assert)
② DỮ LIỆU  file ─▶ soi hình dạng ─▶ chuỗi biến đổi về dạng quan hệ ─▶ SỔ CHUẨN L2 ─▶ chỉ mục giá trị + thẻ nén
③ CÂU HỎI  câu ─▶ sửa chữ ─▶ [gọi 1] ĐẶC TẢ + CÁC CÁCH ĐỌC ─▶ chọn cách đọc bằng luật
                 ├─▶ Trình biên dịch: đặc tả ─▶ chuỗi phép bảng ─▶ kết quả A (+ bảng trung gian)
                 └─▶ [gọi 2] LLM viết code tự do từ câu gốc ─▶ kết quả B
④ KIỂM     RuleTests + metamorphic + hợp lý + dịch ngược ─▶ KIỂM TOÁN (giải thích con số, nghi ngờ câu đồng thuận, soi dòng)
           ─▶ hỏi đúng chỗ phân vân (nếu cần) ─▶ bộ chấm ─▶ trả lời | sửa 1 lần | bỏ trống
```

**Chi phí:** 2 lần gọi model mỗi câu, cộng 1 lần gọi 1-token cho dịch ngược; khoảng 10–20% câu thêm câu hỏi có/không; khoảng 15% câu thêm 1 lần sửa. Phần kiểm metamorphic là chạy lại pandas trên dữ liệu đã có sẵn trong RAM, mỗi câu tốn khoảng 0,1–1 giây. Tổng thời gian dự kiến cho ~1.000 câu trên 2×5090 là **40–80 phút** (cần đo thật).

---

## 1. Paper đã đọc và điều lấy về

| Paper | Ý chính (diễn giải) | Lấy gì | Không lấy gì (và vì sao) |
|---|---|---|---|
| **CHASE-SQL** (ICLR 2025) | Sinh nhiều ứng viên SQL bằng các chiến lược khác nhau (chia để trị, CoT theo kế hoạch truy vấn, ví dụ tổng hợp theo từng DB), rồi một agent đã fine-tune chọn ứng viên bằng so sánh từng cặp. | Các ứng viên phải **khác bản chất** để lỗi không trùng nhau → trình biên dịch + LLM tự do. Chọn bằng so sánh kết quả. | Dùng 3 chiến lược sinh và model chọn riêng: quá nặng. |
| **CHESS** | Bốn agent: truy xuất thông tin (chỉ mục LSH cho giá trị), chọn schema, sinh ứng viên có sửa, và unit tester (LLM viết unit test bằng ngôn ngữ tự nhiên để chọn ứng viên). | **Chỉ mục giá trị** để nối thực thể. **Test để chọn ứng viên**, nhưng test lấy **từ rulebook** (xác định, không tốn gọi model). | Unit test do LLM viết: tốn gọi và có thể sai. |
| **Arctic-Text2SQL-R1** | GRPO chỉ dùng phần thưởng là chạy đúng hay sai; model 7B vượt các model 70B; lúc suy luận dùng bỏ phiếu và truy xuất giá trị. | Phần thưởng chỉ cần **chạy ra đúng** → làm được trên shop giả lập. Bản nhẹ: rejection-sampling fine-tune (RFT). | GRPO đầy đủ: thời gian 9 ngày không đủ, để làm mục mở rộng. |
| **OmniSQL / SynSQL-2.5M** | Tổng hợp DB → SQL → câu hỏi có văn phong → CoT, ở quy mô lớn; model 7B/14B/32B trên nền Qwen2.5-Coder. | **Xưởng shop giả lập** theo đúng thứ tự: sổ → truy vấn → câu hỏi → lời giải. | Quy mô hàng triệu mẫu: không cần, chỉ cần vài nghìn mẫu sát rulebook. |
| **AMBROSIA** (NeurIPS 2024) | Câu hỏi DB có 3 loại mơ hồ: phạm vi, gắn kết, mơ hồ từ ngữ; LLM kém trong việc nhận ra chúng. | Bắt model **liệt kê cách đọc khác** cho từng chỗ mơ hồ, rồi **luật** chọn. Không tin model tự nhận ra mơ hồ. | Trả nhiều đáp án: cuộc thi chỉ nhận 1. |
| **Chain-of-Table** (ICLR 2024) | Suy luận bằng chuỗi phép toán trên bảng, bảng trung gian được cập nhật qua từng bước. | Đặc tả được **biên dịch thành chuỗi phép bảng**; mỗi bảng trung gian kiểm được bằng luật. | Để LLM chọn từng phép một: tốn nhiều lần gọi. |
| **LEVER** (ICML 2023) | Huấn luyện bộ kiểm để đánh giá chương trình dựa trên câu hỏi, chương trình và **kết quả chạy**; gộp với xác suất sinh và gộp các chương trình cho cùng kết quả. | Bộ chấm dùng **đặc trưng của kết quả chạy** (kiểu, dấu, độ lớn so với dữ liệu). Gộp ứng viên theo kết quả. | Bộ kiểm là LM lớn: thay bằng logistic regression. |
| **ReFoRCE** | Agent text-to-SQL: nén thông tin DB bằng gom bảng theo mẫu, tự sửa, bỏ phiếu đồng thuận và **hoãn** các câu chưa đồng thuận để khám phá cột thêm. | Câu không đồng thuận → **hoãn và sửa có mục tiêu** (không sửa mọi câu). Gom bảng cùng cấu trúc (đã có `merge_same_structure`). | |
| **Metamorphic prompt testing** | Kiểm chương trình LLM sinh ra mà không cần đáp án, bằng cách so các chương trình sinh từ các câu diễn đạt lại. | **Quan hệ metamorphic theo nghiệp vụ** (cộng dồn theo kỳ, tỉ lệ theo số lượng…) → kiểm mà không cần đáp án. | Diễn đạt lại câu: tốn gọi; quan hệ nghiệp vụ mạnh hơn. |
| **Auto-Tables** (VLDB 2023) | Tự tổng hợp chuỗi biến đổi (stack, wide→long, transpose, pivot) để đưa bảng "ngoài đời" về dạng quan hệ; tự tạo dữ liệu huấn luyện bằng cách áp phép biến đổi ngược lên bảng chuẩn. | Bộ phép **chuẩn hoá bảng** cho shop lạ. Mẹo **phép biến đổi ngược** để sinh shop bẩn từ sổ sạch. | |
| **SpreadsheetLLM** (EMNLP 2024) | Nén bảng tính cho LLM bằng điểm neo cấu trúc, chỉ mục ngược và gom theo định dạng số. | **Thẻ shop nén**: chỉ giữ hàng/cột "khác biệt", giá trị dạng chỉ mục ngược, cột số mô tả bằng định dạng + khoảng. | |
| **Selective prediction cho text-to-SQL** (2025–2026) | Tự-nhất-quán hiệu chỉnh kém; tín hiệu dựa trên kiểm chứng dự đoán đúng/sai tốt hơn; cần hiệu chỉnh ngưỡng trên phần dữ liệu tách riêng. | Ngưỡng trả lời/bỏ trống **hiệu chỉnh trên shop giả lập + train k-fold**, không đoán tay. | |

---

## 2. ★1 Đặc tả có các cách đọc khác

### 2.1 Vì sao không dùng "phiếu" đơn như v2
Phiếu đơn chỉ ghi **một** cách hiểu. Khi model hiểu sai, mọi bước sau tính đúng một điều sai. Ở v2, phần "tìm cách hiểu nhầm" do code tự sinh nhiễu (16 loại P1–P16). v3 giữ 16 loại đó, nhưng **bắt chính model nói ra chỗ nó phân vân**, vì model biết câu nào có chỗ hai nghĩa tốt hơn danh sách cứng.

### 2.2 Đặc tả (JSON, ràng buộc bằng schema khi sinh)
```json
{
  "measure":   {"name": "doanh_thu", "src": "doanh thu"},
  "agg":       "sum",
  "entity":    {"type": "san_pham", "value": "Croissant bơ", "src": "bánh sừng bò bơ"},
  "periods":   [{"slot": "A", "ref": "quy_1_2026", "src": "Q1"}, {"slot": "B", "ref": "quy_4_2025", "src": "quý trước"}],
  "compare":   {"op": "pct_change", "base": "B", "target": "A", "src": "tăng bao nhiêu % so với"},
  "filters":   [{"field": "kenh", "op": "=", "value": "online", "src": "bán online"}],
  "logic":     null,
  "unit":      {"want": "cai", "src": "bao nhiêu cái"},
  "output":    {"type": "number", "round": 2},
  "readings":  [
     {"slot": "periods.A", "alt": "thang_1_2026", "why": "Q1 có thể là tháng 1"},
     {"slot": "compare",   "alt": {"base": "A", "target": "B"}, "why": "chiều so sánh"}
  ]
}
```
- Mọi trường có `src`: **đoạn chữ gốc** sinh ra trường đó. Code kiểm `src` có thật trong câu (đã chuẩn hoá). Không có `src` → trường bị nghi ngờ.
- `readings` liệt kê cách đọc khác. Code **luôn bổ sung** thêm các cách đọc bắt buộc theo 16 loại của v2 (P1–P16), kể cả khi model không nêu.

### 2.3 Chọn cách đọc bằng luật (không bằng model)
Thứ tự áp dụng, dừng ở bước đầu tiên quyết định được:
1. **Luật rulebook có số dòng**: ví dụ "Qn = quý n" → loại `thang_1`. "X so với Y" → Y là gốc.
2. **Từ khoá cứng trong câu** (bảng `meaning_marks` ở v2): "giảm" + kết quả âm → giữ dấu theo rulebook.
3. **Bằng chứng dữ liệu**: cách đọc nào làm thực thể không tồn tại, kỳ không có dữ liệu, hay đơn vị không khớp → loại.
4. **Kết quả trùng**: các cách đọc còn lại cho **cùng con số** → không cần chọn.
5. Còn ≥ 2 con số khác nhau → **mơ hồ thật** → sang bộ chấm với cờ `ambiguous=1` (thường bỏ trống).

---

## 3. ★2 Hai bộ sinh khác bản chất

### 3.1 Đường A: Trình biên dịch đặc tả → chuỗi phép bảng (không LLM)
Đặc tả được dịch **xác định** thành một chuỗi phép trên sổ chuẩn L2. Mỗi bước giữ bảng trung gian:
```
SELECT(orders) → APPLY_RULE(R3.1 loại huỷ) → APPLY_RULE(R3.2 loại có trả) → FILTER(kenh=online)
→ FILTER(period=A) → JOIN(lines) → CONVERT_UNIT(hộp→cái) → AGG(sum line_amount_after_disc) → COMPARE(pct_change, B)
→ ROUND(HALF_UP, 2)
```
- Các phép `APPLY_RULE` được chèn **tự động** theo `measure` (ví dụ doanh thu luôn kèm R3.1, R3.2). Model không thể "quên" chúng.
- Thư viện phép (khoảng 20 phép): SELECT, FILTER, PERIOD, JOIN, APPLY_RULE, CONVERT_UNIT, GROUP, AGG, RANK(tie=code asc), TOPK, COMPARE, RATIO, EXISTS, ALL/ANY (logic nhiều vế), DEDUP, ROUND, FORMAT.
- Đặc tả nằm ngoài khả năng biên dịch → đường A trả `UNSUPPORTED` (không phải lỗi), quyết định dựa vào đường B và các kiểm tra.

### 3.2 Đường B: LLM viết code tự do (đã có trong v0)
- Prompt dùng **câu gốc đã sửa chữ** + rulebook + thẻ shop, **không đưa đặc tả** để hai đường độc lập thật sự.
- Code phải trả về `(value, trace)`, trong đó `trace` là các DataFrame trung gian có tên chuẩn (`base_orders`, `filtered`, `lines`, …) để kiểm luật được.

### 3.3 Gộp kết quả (theo LEVER: gộp theo kết quả)
- A = B (sau chuẩn hoá) → điểm đồng thuận cao.
- A ≠ B → so A với **các cách đọc khác** của đặc tả: nếu B trùng một cách đọc khác, đó là tín hiệu mơ hồ thật hoặc model hiểu sai, cần ③ kiểm thêm.
- Chỉ có một đường → không bao giờ tự tin cao, bộ chấm phải dựa vào ④.

---

## 4. ★3 Kiểm bằng quan hệ và luật thành test

### 4.1 Luật thành test (RuleTests)
Mỗi luật nguyên tử ở khối ① sinh ra một **assert** chạy trên bảng trung gian (`trace`) của cả hai đường:

| Luật (ví dụ) | Assert |
|---|---|
| Doanh thu loại đơn huỷ | `not trace.filtered.status.isin(HUY).any()` |
| Doanh thu loại đơn có trả hàng | `trace.filtered.order_id.isin(RETURNED).sum() == 0` |
| Đơn đang giao vẫn tính | số đơn `DANG_GIAO` trong kỳ đều có mặt |
| Khách vãng lai không tính là khách | `KHACH_LE not in trace.customers` |
| Công nợ loại COD và sàn | `not trace.debt.channel.isin(['COD','san']).any()` |
| Tuần là Thứ Hai → Chủ nhật | `trace.period.start.weekday() == 0` |
| Hoà thì xếp theo mã tăng dần | so thứ tự kết quả với `sort_values([metric, code], ascending=[False, True])` |

Assert được sinh **một lần cho mỗi rulebook** (có template cho các loại luật phổ biến; luật lạ dùng model sinh 1 lần, có người duyệt trên train). Không tốn gọi model mỗi câu.

### 4.2 Kiểm metamorphic (không cần đáp án)
Chạy lại **cùng chương trình** trên dữ liệu đã biến đổi, kiểm quan hệ phải đúng:

| Mã | Biến đổi dữ liệu / câu | Quan hệ phải đúng | Bắt lỗi |
|---|---|---|---|
| M1 | Tiêm 1 đơn **huỷ** giá trị lớn vào kỳ | Doanh thu không đổi; tỉ lệ huỷ tăng đúng 1/N | Quên lọc huỷ, sai mẫu số |
| M2 | Tiêm 1 đơn **có trả hàng** | Doanh thu không đổi | Quên luật trả hàng |
| M3 | Đổi 1 dòng bán từ "cái" sang "hộp" (số lượng ÷ hệ số) | Số cái không đổi | Quên quy đổi đơn vị |
| M4 | Nhân đôi một dòng y hệt (bản trùng) | Nếu luật nói khử trùng → không đổi | Cộng trùng |
| M5 | Tiêm 1 đơn **ngay ngoài biên kỳ** (ngày trước/sau kỳ 1 ngày) | Kết quả không đổi | Biên kỳ lệch 1 ngày, tuần sai ngày đầu |
| M6 | Tiêm 1 đơn của **khách vãng lai** / kênh COD-sàn | Số khách / công nợ không đổi | Quên luật loại trừ |
| M7 | Nhân đôi mọi số lượng | Doanh thu/số lượng ×2; tỉ lệ không đổi | Dùng sai cột, cộng hằng số |
| M8 | Xáo thứ tự hàng | Kết quả không đổi | Lấy dòng đầu/cuối ngẫu nhiên |

- Chỉ chạy các quan hệ **áp dụng được** cho `measure` và `agg` của đặc tả (bảng áp dụng viết sẵn). Mọi biến đổi tác động lên **bản sao L2** trong RAM, không gọi model.
- Một vi phạm cho biết **có lỗi**, chưa cho biết lỗi gì. Vì vậy mỗi vi phạm được gắn với **một gợi ý sửa** cụ thể cho bước sửa (mục 5.2).

> **Giới hạn quan trọng (sửa so với bản trước):** kiểm metamorphic chỉ bắt **lỗi cài đặt** (quên lọc, quên quy đổi, cộng trùng, biên kỳ). Nó **không bắt lỗi hiểu câu**: nếu chương trình đã hiểu "Q1" là tháng 1, hay đảo chiều so sánh, thì mọi quan hệ trên vẫn đúng. Lỗi hiểu câu được bắt ở ★1 (cách đọc + luật), mục 4.3 (dịch ngược), mục 5.2 (hỏi đúng chỗ phân vân) và mục 5A (kiểm toán).

### 4.3 Dịch ngược bằng code (bắt lỗi hiểu câu)
Đặc tả được dịch ngược thành **một câu tiếng Việt chuẩn bằng template** (không dùng LLM), ví dụ: "Doanh thu quý 1/2026 thay đổi bao nhiêu % so với quý 4/2025 (mốc là quý 4/2025)?". Model chỉ trả lời một token: câu chuẩn này **có cùng nghĩa** với câu gốc không. Lấy xác suất token "có" làm đặc trưng `roundtrip_p` cho bộ chấm. Câu khác nghĩa → sinh đặc tả lại một lần, kèm câu chuẩn sai để model thấy chỗ lệch.
- Chi phí: 1 lần gọi rất ngắn (chỉ sinh 1 token, prefix cache dùng chung), chạy cho mọi câu hoặc chỉ câu có dấu hiệu rủi ro.
- Template viết theo từng `measure` × `compare` × `period`; trường không có template thì bỏ qua bước này.

### 4.4 Kiểm hợp lý (từ G1–G30 của v2)
Kiểu đúng với `output.type`; dấu đúng; độ lớn nằm trong khoảng dữ liệu (ví dụ doanh thu 1 ngày ≤ max ngày × 1,5); tỉ lệ trong [0, 100]; danh sách không rỗng khi dữ liệu có; ngày nằm trong khoảng dữ liệu.

---

## 5. ④ Quyết định: chấm, sửa có mục tiêu, bỏ trống có hiệu chỉnh

### 5.1 Bộ chấm (nhẹ, giải thích được)
Logistic regression trên khoảng 15 đặc trưng:
`agree_AB`, `A_supported`, `n_readings_alive`, `ambiguous`, `ruletests_failed_A`, `ruletests_failed_B`, `metamorphic_failed_A/B`, `sanity_failed`, `roundtrip_p` (dịch ngược), `audit_explained` (K2 giải thích được lệch), `audit_conflict` (K4), `data_suspect` (K5), `targeted_p` (5.2), `src_coverage` (tỉ lệ chữ quan trọng có trường), `entity_link_score`, `period_has_data`, `shop_quality_gate`, `question_type`, `logprob_B` (nếu có).
- Huấn luyện trên **câu hỏi train (k-fold theo shop)** và **câu hỏi shop giả lập**. Nhãn là đúng/sai của từng ứng viên.
- Hiệu chỉnh bằng Platt hoặc isotonic trên phần tách riêng. Ngưỡng chọn để **tối đa PA = (đúng − sai)/N**: trả lời khi `P(đúng) > 0,5` là tối ưu về kỳ vọng khi sai bị trừ 1 và bỏ trống được 0. Ngưỡng thực tế đặt hơi cao hơn 0,5 để bù sai lệch hiệu chỉnh khi chuyển sang ngành mới.

### 5.2 Hỏi đúng chỗ phân vân (thay cho bỏ phiếu cả câu)
Khi còn ≥ 2 cách đọc cho ra số khác nhau, **không** hỏi lại cả câu. Code tìm **trường duy nhất** làm hai cách đọc khác nhau, rồi hỏi model một câu có/không về đúng trường đó:
- "Trong câu: «…», 'Q1' là **quý 1** phải không?"
- "Trong câu: «…», mốc để so sánh là **tháng 2/2025** phải không?"
- "Trong câu: «…», điều kiện «… và …» yêu cầu **cả hai** vế đúng phải không?"

Đọc xác suất token trả lời (1 token, rất rẻ). Xác suất > 0,8 → chọn cách đọc đó; nằm giữa → giữ cờ `ambiguous`, bộ chấm quyết định. Câu hỏi hẹp như vậy model 7B trả lời đúng hơn nhiều so với hiểu lại cả câu. Chỉ chạy cho khoảng 10–20% số câu.

### 5.3 Sửa có mục tiêu (theo ReFoRCE: chỉ sửa câu bị hoãn)
Chỉ câu có `P(đúng)` trong vùng xám (ví dụ 0,3–0,7) mới được **1 lần sửa**. Prompt sửa đưa cho model **đúng thông tin lỗi**: "assert R3.1 thất bại: có 3 đơn huỷ trong bảng filtered", "M3 thất bại: số cái đổi khi đổi đơn vị", "kiểm toán: B = A nhưng không quy đổi đơn vị". Sau khi sửa, chấm lại; không lặp lần hai.

### 5.4 Đầu ra
`answer` (ghi theo định dạng rulebook §9) | `null` (bỏ trống) | `"không trả lời được"`.

**"Không trả lời được" quyết định bằng luật, không bằng cảm giác của model.** Train có 70/3.000 câu loại này (2,3%), là nguồn điểm chắc nếu nhận đúng. Các điều kiện xác định:
- kỳ hỏi nằm sau `as_of_date` hoặc ngoài khoảng dữ liệu của shop;
- trường cần (ví dụ giá vốn, nhân viên) không tồn tại trong bất kỳ bảng nào của shop;
- thực thể không khớp được với giá trị nào trong chỉ mục (điểm khớp dưới ngưỡng ở cả 4 mức);
- rulebook quy định rõ trường hợp phải trả "không trả lời được".

Model chỉ được đề xuất; code kiểm điều kiện rồi mới ghi.

---

## 5A. ★5 Kiểm toán: tự dò lại và phát hiện câu sai (cách tôi đã làm bằng tay)

### 5A.1 Cách làm bằng tay đã tìm ra 6 câu sai
1. **Giải lại từng câu độc lập** từ dữ liệu gốc, so với đáp án đang có.
2. Câu lệch → **truy ngược con số lệch**: thử xem cách hiểu sai nào cho ra đúng con số đó (đảo chiều, tháng thay quý, bỏ một vế, không quy đổi).
3. **Phân xử** bằng dòng rulebook hoặc bằng chứng dữ liệu (giá hộp = 6 × giá cái; train trả lời các câu cùng dạng thế nào).
4. **Soi các dòng dữ liệu đứng sau con số**: mã có khoảng trắng, ngày trống, đơn trùng chỉ khác định dạng.

Mấu chốt là bước 2: con số lệch chỉ có ích khi biết **lệch vì đâu**. Biết vì đâu thì biết bên nào sai và sửa thế nào.

### 5A.2 Lúc chạy thật (không có đáp án)
| Bước | Làm gì | Ra gì |
|---|---|---|
| K1. Giải lại khác góc nhìn | Đường A chạy trên **sổ chuẩn L2**; đường B nhận **bảng gốc đã đọc nhưng chưa làm sạch sâu** (chỉ chuẩn hoá tiền/ngày). Lỗi làm sạch ở L2 sẽ làm hai đường lệch nhau. | A, B |
| K2. Giải thích con số | A ≠ B → áp lần lượt các phép hiểu nhầm P1–P16 (và tổ hợp 2 phép) lên đặc tả của A, biên dịch, xem phép nào **tái tạo đúng B** (sai số ≤ 1e-6 hoặc khớp sau làm tròn). Làm ngược lại với B nếu B có `trace`. | Nhãn `B = A ∘ Pk` hoặc "không giải thích được" |
| K3. Phân xử | Có nhãn Pk → tra luật liên quan Pk (ví dụ P1 đảo chiều ↔ luật "vế sau *so với* là mốc"). Luật nói rõ → chọn bên đúng. Luật không nói → hỏi đúng chỗ phân vân (5.2). | Đáp án đã chọn hoặc `ambiguous` |
| K4. Nghi ngờ câu đã đồng thuận | Câu có dấu hiệu rủi ro (so sánh, quý/tuần/Tết, "và/hoặc", hộp/cái, "còn lại/trừ", xếp hạng có hoà) → tính mọi cách đọc liên quan. Đáp án hiện tại **trùng một cách đọc bị luật cấm**, hoặc cách đọc **đúng luật** cho số khác → gắn cờ sai. | Cờ `audit_conflict` |
| K5. Soi dòng tạo đáp án | Từ `trace` lấy các dòng góp vào kết quả: kiểm mã có khoảng trắng/khác hoa thường, ngày trống, dòng trùng sau chuẩn hoá, đơn vị lẫn lộn, dòng Tổng/Cộng, giá trị ngoại lai (> p99 × 10). | Cờ `data_suspect` + mô tả |
| K6. Quyết định | Một cách đọc hợp luật duy nhất → sửa sang đáp án đó. Còn nhiều → bộ chấm (có thêm đặc trưng K2–K5). "Không giải thích được" → hạ mạnh `P(đúng)`. | Trả lời / sửa / bỏ trống |

Chi phí: K1–K5 gần như toàn bộ là code chạy pandas trên dữ liệu đã nạp. Chỉ K3 có thể gọi thêm câu có/không của 5.2.

### 5A.3 Lúc phát triển (trên train có đáp án): tự gắn nhãn nguyên nhân sai
Chạy K2 nhưng so với **đáp án train** thay vì đường kia:
- Câu sai mà tái tạo được bằng Pk → nhãn "sai vì Pk".
- Câu sai mà đáp án train tái tạo được từ **dữ liệu chưa làm sạch** → nhãn "lỗi làm sạch" kèm cột nghi ngờ.
- Không tái tạo được → nhãn "chưa rõ", đưa vào danh sách người xem.

Đầu ra mỗi lần chạy là bảng **phân bố nguyên nhân sai**, ví dụ: "38 câu sai do đơn vị, 21 do chiều so sánh, 15 chưa rõ…". Mỗi ngày sửa loại lớn nhất, chạy lại, chỉ giữ thay đổi nếu PA k-fold tăng. Loại "chưa rõ" sau khi người xem được thêm thành **phép hiểu nhầm mới** vào danh mục, nên danh mục lớn dần qua từng vòng. Cách này chỉ dùng đáp án train nên hợp lệ với quy định 6.

### 5A.4 Bản đồ độ nhạy luật
Với mỗi câu train, tính lại đáp án khi **lật từng luật** (tính cả đơn huỷ, không quy đổi đơn vị, tuần bắt đầu Chủ nhật, không loại khách vãng lai, làm tròn HALF_EVEN…). Ghi lại câu nào đổi đáp án khi luật nào bị lật.
- Biết **loại câu nào phụ thuộc luật nào** → K4 chỉ cần tính các cách đọc liên quan, nhanh hơn.
- Khi **rulebook private khác public**: so hai rulebook → biết ngay loại câu nào bị ảnh hưởng, kiểm lại đúng chỗ đó.
- Các phép lật luật cũng chính là các phép hiểu nhầm để K2 dùng.

### 5A.5 Giới hạn
Kiểm toán chỉ giải thích được lỗi **có trong danh mục**. Lỗi kiểu mới rơi vào "không giải thích được" → lúc chạy thật thì bỏ trống, lúc phát triển thì được người xem và bổ sung danh mục. Nếu cả hai đường cùng sai theo một cách **không có trong danh mục**, kiểm toán sẽ không thấy.

---

## 6. Xưởng shop giả lập, bản thu nhỏ (hiệu chỉnh và đo OOD)

> **Thu nhỏ so với bản trước:** chỉ khoảng 30–50 shop, ~2.000 câu. Mục đích chính là **hiệu chỉnh bộ chấm** và **đo OOD** trên ngành chưa gặp. Fine-tune từ dữ liệu này là tuỳ chọn (P2) và chỉ giữ nếu PA train k-fold tăng. **GRPO bị bỏ** vì không đủ thời gian để làm đúng và đo được trong 9 ngày.

### 6.1 Ba tầng sinh (theo OmniSQL và mẹo phép ngược của Auto-Tables)
1. **Sổ sạch**: bộ mô phỏng sinh shop theo một trong khoảng 12 ngành (bánh, cafe, mỹ phẩm, điện thoại, vật tư y tế, nông sản, thời trang, phụ tùng xe, nhà sách, spa, sửa chữa, đồ chơi), cộng **ngành ngẫu nhiên** (tên hàng và đơn vị sinh ngẫu nhiên) để kiểm OOD. Có đủ đơn, dòng, trả hàng, huỷ, nhập hàng, thu nợ, khách lẻ, mùa Tết theo `calendar.csv`.
2. **Làm bẩn có kiểm soát** (phép ngược của bước làm sạch): tiền kiểu "1tr380", "(250k)", dấu chấm/phẩy lẫn lộn; ngày dmy/mdy/serial Excel; mã có khoảng trắng; dòng Tổng/Cộng; header lặp; bảng theo tháng tách sheet; bảng dạng rộng (mỗi tháng một cột); đơn vị hộp/cái lẫn lộn; tên hàng viết tắt/teencode. Vì biết phép làm bẩn, ta biết **đáp án làm sạch đúng** → huấn luyện và chấm được bước làm sạch.
3. **Câu hỏi + đáp án**: lấy mẫu đặc tả từ phân phối loại câu ở train → engine tham chiếu (đường A trên sổ sạch) tính đáp án → model diễn đạt thành câu tiếng Việt với nhiều văn phong (trang trọng, nói tắt, teencode, kéo dài chữ). Kiểm khứ hồi: model đọc lại câu ra đặc tả, phải trùng.

### 6.2 Dùng vào đâu
| Dùng cho | Cách |
|---|---|
| Fine-tune **gọi 1** (câu → đặc tả + cách đọc) | SFT bằng LoRA trên khoảng 6–10k cặp; cách đọc khác lấy từ chính phép sinh câu mơ hồ có chủ đích |
| Fine-tune **gọi 2** (câu → code) | **RFT**: model gốc sinh 4 code mỗi câu, giữ code chạy ra đúng, SFT trên phần đó (bản nhẹ của RL chỉ dùng kết quả chạy) |
| Hiệu chỉnh bộ chấm | Nhãn đúng/sai có sẵn, số lượng lớn, có cả ngành lạ |
| Đo OOD | Tập "ngành ngẫu nhiên" không bao giờ dùng để train |

**Tuân thủ quy định 6:** không dùng public/private test để huấn luyện, không dùng đáp án public. Shop giả lập sinh từ mô phỏng và rulebook; train chính thức được dùng.

---

## 7. ② Dữ liệu: các điểm nâng cấp so với v2

- **Chuẩn hoá bảng kiểu Auto-Tables**: thêm 4 phép `stack`, `wide_to_long`, `transpose`, `pivot`. Agent cấu hình shop (v2 mục 4.3) chọn chuỗi phép dựa trên dấu hiệu: tên cột là tháng/ngày → `wide_to_long`; cột đầu là tên chỉ số → `transpose`. Mỗi chuỗi được kiểm bằng bất biến I1–I8.
- **Thẻ shop nén kiểu SpreadsheetLLM**: hiện thẻ ~14k ký tự. Nén bằng (a) chỉ giữ hàng mẫu "khác biệt" theo mặt nạ chữ số, (b) giá trị phân loại ghi dạng chỉ mục ngược `giá trị → số lần, cột`, (c) cột số chỉ ghi định dạng, khoảng, số dòng. Mục tiêu: dưới 4k token, giúp prefix cache hiệu quả và còn chỗ cho rulebook.
- **Chỉ mục giá trị kiểu CHESS**: n-gram ký tự + bge-m3 trên mọi tên hàng/khách/nhân viên. Nối thực thể: khớp chính xác → khớp sau bỏ dấu → n-gram → embedding; điểm khớp đưa vào bộ chấm.

---

### 7.1 Sửa chữ theo ngữ cảnh cả câu (`vn_context.py`, đã code và đo)
Sửa từng chữ riêng lẻ không biết chữ đó có hợp với câu không ("teen" là "tên" hay tên hàng?). Cách làm:
1. Mỗi chữ có **danh sách ứng viên do luật sinh ra**: giữ nguyên, gộp chữ kéo dài, giải Telex, Telex thiếu mũ/móc ("nhieuf" → "nhiều"), teencode, thêm dấu cho chữ không dấu. Mỗi ứng viên có chi phí sửa.
2. **Mô hình ngôn ngữ 2-gram** học từ rulebook, README, tên hàng và câu hỏi **train** có dấu (bỏ ~26% câu train không dấu để không học sai).
3. **Viterbi** chọn tổ hợp ứng viên làm cả câu "giống câu thật" nhất. Kết quả luôn nằm trong danh sách ứng viên (không bịa chữ); số, ngày, tiền, tên hàng đã khoá trước.
4. Chỗ hai ứng viên sát điểm → `unsure` → có thể hỏi LLM trắc nghiệm (`mc_prompt`).
5. Câu đã viết có dấu (≥ 50% chữ có dấu) thì **không thêm dấu**, chỉ sửa chữ hỏng, tránh "hôm qua" → "hôm quả".

**Kết quả đo** (`scripts/eval_normalize.py`): câu hỏi train sạch bị làm bẩn có kiểm soát; mô hình ngôn ngữ (3-gram) học trên nửa số shop, đo trên nửa còn lại. Tỉ lệ chữ đúng / câu đúng hoàn toàn:

| Kiểu bẩn | Không sửa | Sửa từng chữ | Theo ngữ cảnh |
|---|---|---|---|
| **Câu sạch (không được sửa nhầm)** | 100 / 100 | 100 / 100 | 99,9 / 99,0 |
| Bỏ dấu cả câu | 20,5 / 0,0 | 20,5 / 0,0 | 98,9 / 92,3 |
| Telex sót | 52,6 / 0,5 | 99,7 / 96,5 | 99,6 / 96,2 |
| Telex thiếu mũ | 52,3 / 0,0 | 90,2 / 25,7 | 99,6 / 96,3 |
| Kéo dài chữ | 40,7 / 0,0 | 99,7 / 96,3 | 99,8 / 98,3 |
| Trộn các kiểu | 61,3 / 1,0 | 87,3 / 22,0 | 99,5 / 96,0 |
| Teencode + thay chữ kiểu teen | 18,9 / 1,0 | 92,6 / 54,8 | 99,6 / 96,0 |
| Gõ VNI | 2,6 / 0,3 | 2,7 / 0,3 | 99,2 / 96,2 |
| Lẫn chính tả vùng miền | 90,3 / 31,5 | 90,3 / 31,5 | 99,3 / 93,0 |

Lưu ý: kiểu "teencode" sinh viết tắt từ chính bảng `lexicon/teencode.tsv` nên phần viết tắt là trong-bảng; phần thay chữ (gi→j, qu→w, ph→f, c→k, v→z), VNI và vùng miền là quy tắc tổng quát, không cần có trong bảng.

**Chế độ cẩn thận:** câu có ≥ 50% chữ có dấu thì chữ hợp lệ được giữ, chỉ đổi khi ngữ cảnh ủng hộ rất rõ (chi phí chọn bằng dò lưới trên train). Không có chế độ này, ~7% câu sạch bị sửa nhầm.

**Các hàng rào đã thêm sau khi soát lỗi trên câu train thật:**
- Chữ gõ hỏng trong câu train ('hangf', 'ddown') không được học thành từ vựng chuẩn.
- Chữ tiếng Anh hay gặp (bill, best, seller, combo…) giữ nguyên, không thêm dấu ('bill' từng bị thành 'bị').
- Teencode nhiều nghĩa để ngữ cảnh chọn ('nhiu nhất' → nhiều, 'bao nhiu' → nhiêu).
- Dạng viết sai hiếm cạnh dạng rất phổ biến không được dùng để thêm dấu; ứng viên chưa gặp trong văn bản sạch bị phạt.
- Tên hàng chỉ được khớp kiểu bỏ dấu khi người hỏi không gõ dấu và tên đủ dài ('hôm qua' từng bị khoá thành hàng 'quả').
- Câu phần lớn có dấu vẫn được thêm dấu cho chữ sót, nhưng với chi phí cao hơn.

**Lỗi còn lại:** chữ mơ hồ thật (đắt/đạt, ngoài/ngoái, chưa/chua) và tên riêng hiếm. Chỗ sát điểm được đánh dấu `unsure` để hỏi LLM trắc nghiệm; mô hình chính luôn nhận cả câu gốc lẫn câu đã sửa.

### 7.1b Teencode giới trẻ và vùng miền
- `lexicon/gioitre_vungmien_lamtay.tsv` (~370 mục viết tay): teencode giới trẻ, viết tắt chat, mạng xã hội, từ Bắc/Trung/Nam. Không đưa từ thật hay gặp (chi, qua, biểu, kêu, ảnh, hè, bổ…).
- Nhóm `dong_nghia_*` (heo/lợn, ly/cốc, lạc/đậu phộng…): **không sửa câu**, chỉ thêm gợi ý vào prompt khi câu dùng một từ mà dữ liệu shop dùng từ kia (`alias_hints`, so khớp giữ dấu để 'lợn' không trùng 'lon').
- `lexicon/teencode_gioitre_vungmien.tsv` (11.722 dòng, `scripts/build_gioitre_dataset.py`): từ ~1.560 từ chuẩn (`tu_thong_dung.txt` viết tay + chữ trong câu hỏi/README/tên hàng train), áp quy tắc viết của giới trẻ (c→k, qu→w, ph→f, gi→j, d/v→z, ch→k/ck, ng→g/q, i→j, ôi→ui, ươ→uo, kéo dài; chồng hai quy tắc; không dấu), quy tắc phát âm vùng miền, viết tắt, dính liền. Dạng an toàn (không phải âm tiết hợp lệ, một nghĩa) được tra thẳng.
- Từ/cụm chuẩn của lexicon được đưa vào mô hình ngôn ngữ với 1 lần đếm, để cụm bung ra từ viết tắt không bị phạt; chữ chỉ coi là "đã gặp" khi đếm ≥ 2.

### 7.2 LLM chọn ở chỗ sửa chữ chưa chắc (đã code, `--norm_llm main`)
- Luật + bảng (`teencode.tsv` ~620 mục, `teencode_variants.tsv` 11.282 dòng) sinh phương án; mô hình 3-gram chọn trước.
- Chỗ hai phương án sát điểm (`unsure`, ~8% câu train) → hỏi **LLM chính** một câu trắc nghiệm (A = lựa chọn hiện tại, B–D = phương án khác), LLM chỉ trả 1 chữ cái → **không bịa chữ, không đụng số / tên hàng**, cache, greedy.
- LLM lỗi hoặc trả không đọc được → giữ kết quả không có LLM. Log từng lựa chọn ở `logs/<id>.json` (`normalize_llm`).
- Không dùng LLM viết lại cả câu (dễ đổi tên hàng, con số). `scripts/eval_norm_llm.py` đo cả 3 cách trên GPU; nếu `llm_chon` không hơn `ngu_canh` thì đặt `--norm_llm off`.
- Model tiếng Việt riêng (PhoGPT-4B-Chat 3,7B) vừa ngân sách 15B nhưng chưa thêm vào pipeline: chỉ thêm nếu `eval_norm_llm.py` cho thấy hơn Qwen rõ rệt.

## 8. Model và ngân sách

| Thành phần | Model | Tham số |
|---|---|---|
| Gọi 1 (đặc tả) + gọi 2 (code) + sửa | **Qwen2.5-Coder-7B-Instruct** (hoặc Qwen3-8B, chọn theo train eval), một model, hai LoRA hoặc một LoRA đa nhiệm | ~7,6–8,2B |
| Nối thực thể | bge-m3 | ~0,57B |
| Bộ chấm | Logistic regression | ~20 tham số |
| **Tổng** | | **≈ 8,2–8,8B (< 15B)** |

Cần kiểm lại ngày phát hành và giấy phép của từng model trước khi nộp (quy định trọng số phát hành trước 01/10/2026, giấy phép cho phép dùng).

**Thời gian chạy (ước tính, cần đo):** ~1.000 câu × 2 lần gọi, prefix cache rulebook + thẻ shop, vLLM batch trên 2×5090 → 20–40 phút sinh; kiểm metamorphic + chạy code ~10–20 phút; nạp shop ~5 phút. Tổng **40–80 phút**, còn dư xa giới hạn 6 giờ.

---

## 9. Kế hoạch 9 ngày (7/11 → 16/11) và thứ tự ưu tiên

Thứ tự theo **lợi ích / công sức**; mỗi bước phải đo được bằng PA trên train k-fold trước khi sang bước sau.

| Ngày | Việc | Mức |
|---|---|---|
| 7–8/11 | Chạy v0 trên Kaggle, lấy **PA train mốc**. Thêm `trace` vào hợp đồng code của đường B. **Kiểm toán trên train (5A.3)** để có bảng phân bố nguyên nhân sai ngay từ đầu. | **P0** |
| 8–10/11 | Đặc tả + cách đọc (gọi 1). Trình biên dịch đường A cho ~10 metric hay gặp nhất (doanh thu, số đơn, số lượng, tỉ lệ huỷ, top-k, so sánh kỳ, công nợ, tồn kho, nhập hàng, khách). Luật chọn cách đọc (2.3). | **P0** |
| 9–10/11 | RuleTests cho ~15 luật phổ biến; metamorphic M1–M6. Luật "không trả lời được" (5.4). | **P0** |
| 10–11/11 | Kiểm toán lúc chạy (K1–K6), hỏi đúng chỗ phân vân (5.2), dịch ngược (4.3) cho các template phổ biến. | **P0** |
| 11–12/11 | Bản đồ độ nhạy luật (5A.4). Bộ chấm logistic, hiệu chỉnh trên train k-fold. | P1 |
| 11–13/11 | Xưởng shop giả lập thu nhỏ; đo OOD; hiệu chỉnh lại ngưỡng. | P1 |
| 12–14/11 | LoRA SFT gọi 1 / RFT gọi 2 nếu có GPU. Chỉ giữ nếu PA k-fold tăng. | P2 |
| 14–15/11 | Docker offline, đo thời gian thật, chạy 2 lần kiểm tái lập (\|ΔPA\| ≤ 0,03). Log đầy đủ. | **P0** |
| 16/11 | Nộp. | |

**Bỏ hẳn:** GRPO. **Chỉ làm nếu còn thời gian:** M7–M8, chuẩn hoá bảng 4 phép, thẻ shop nén.

---

## 10. Điểm yếu còn lại (nói thẳng)

- **Đường A chỉ phủ các metric đã viết**. Câu lạ ở private sẽ chỉ có đường B, nên độ tin thấp hơn và bỏ trống nhiều hơn. Đây là đánh đổi có chủ đích, vì sai bị trừ điểm còn bỏ trống thì không.
- **Metamorphic chỉ bắt lỗi cài đặt**, không bắt lỗi hiểu câu, và không chỉ ra lỗi là gì. Lỗi hiểu câu dựa vào cách đọc + luật, dịch ngược, hỏi đúng chỗ, kiểm toán.
- **Kiểm toán chỉ thấy lỗi có trong danh mục.** Hai đường cùng sai theo kiểu mới thì không phát hiện được.
- **Dịch ngược và hỏi đúng chỗ phụ thuộc chính model 7B.** Model có thể đồng ý với câu chuẩn sai. Vì vậy chúng chỉ là đặc trưng cho bộ chấm, không tự quyết.
- **Shop giả lập có thể lệch với dữ liệu thật**. Mọi quyết định giữ hay bỏ một tính năng phải dựa trên **train thật k-fold**.
- **Các con số thời gian và độ chính xác ở đây là ước tính**, chưa đo.
- **Rulebook private có thể khác**: RuleTests, bảng áp dụng metamorphic và bản đồ độ nhạy phải sinh lại từ rulebook mới.
- **Khối lượng code lớn so với 9 ngày.** Nếu trễ, giữ theo thứ tự P0; mỗi phần P0 đều có ích độc lập.

---

## 11. Tự đánh giá: pipeline này đã nổi bật và đủ hay chưa?

**Phần có khả năng khác các đội khác:**
- Phần lớn đội sẽ dùng: LLM viết code + bỏ phiếu + có thể fine-tune. v3 khác ở chỗ **coi lỗi hiểu câu là bài toán tìm kiếm có luật phân xử** (cách đọc → luật → hỏi đúng chỗ → kiểm toán giải thích con số), thay vì hy vọng model hiểu đúng. Đây là điểm khác thật, vì nó nhắm đúng loại lỗi chiếm 4/6 câu sai đã phân tích.
- **Kiểm toán + nhãn nguyên nhân sai trên train** biến việc cải tiến thành vòng lặp có số đo, không dựa vào cảm giác. Đội nào không có vòng này sẽ cải tiến chậm hơn trong 9 ngày còn lại.
- **Bỏ trống theo PA**, hiệu chỉnh trên dữ liệu tách riêng, và **"không trả lời được" bằng luật**: ít đội làm kỹ phần này, trong khi với điểm (đúng − sai)/N nó ảnh hưởng trực tiếp.

**Phần không mới:** hai đường + đồng thuận, sandbox, fine-tune LoRA, thẻ shop. Đội mạnh nào cũng có thể có.

**Chưa thể khẳng định "hay nhất":**
- Chưa có **số PA nào đo thật**. Một ý chỉ hay khi tăng PA trên train k-fold và trên shop giả lập ngành lạ.
- Các paper làm nguồn đo trên text-to-SQL tiếng Anh; mức lợi ở bài này phải tự đo.
- Giá trị lớn nhất có thể đến từ **độ phủ của đường A** (10 metric chiếm phần lớn câu train) hơn là từ ý mới. Đây là việc "nặng tay", không "đột phá", nhưng có thể quyết định điểm.

**Tiêu chí để biết đã đủ hay:** sau mỗi phần P0, so với mốc v0 trên cùng 6 shop mù:
1. PA tăng; 2. tỉ lệ trả lời sai (precision lỗi) giảm; 3. trên shop giả lập ngành lạ, PA không giảm quá 0,05 so với train; 4. bảng nguyên nhân sai không còn loại nào chiếm hơn 10% số câu sai.

## Nguồn
- CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL (ICLR 2025). https://arxiv.org/abs/2410.01943
- CHESS: Contextual Harnessing for Efficient SQL Synthesis. https://arxiv.org/abs/2405.16755
- Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL. https://arxiv.org/abs/2505.20315
- OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale. https://arxiv.org/abs/2503.02240
- AMBROSIA: A Benchmark for Parsing Ambiguous Questions into Database Queries (NeurIPS 2024). https://arxiv.org/abs/2406.19073
- Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding (ICLR 2024). https://arxiv.org/abs/2401.04398
- LEVER: Learning to Verify Language-to-Code Generation with Execution (ICML 2023). https://proceedings.mlr.press/v202/ni23b.html
- ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Consensus Enforcement, and Column Exploration. https://arxiv.org/abs/2502.00675
- Validating LLM-Generated Programs with Metamorphic Prompt Testing. https://arxiv.org/abs/2406.06864
- Auto-Tables: Synthesizing Multi-Step Transformations to Relationalize Tables without Using Examples (VLDB 2023). https://arxiv.org/abs/2307.14565
- SpreadsheetLLM: Encoding Spreadsheets for Large Language Models (EMNLP 2024). https://www.microsoft.com/en-us/research/publication/encoding-spreadsheets-for-large-language-models/
- Confidence Estimation for Error Detection in Text-to-SQL Systems. https://arxiv.org/abs/2501.09527
- What Predicts Correctness in Text-to-SQL? A Selective-Prediction Study (2026). https://arxiv.org/abs/2607.06799
