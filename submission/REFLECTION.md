# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Trọng Minh
**Khoá:** K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.
> Phiên chạy đầy đủ nằm ở `notebooks/Lab22_DPO_Kaggle_run.ipynb` (chạy trên Kaggle, T4 ×2).

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle T4 ×2 (15.6 GB/GPU), dùng 1 GPU cho mỗi bước |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch (loss SFT cuối 1.3603) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel: Skywork-Reward-V2-Llama-3.2-3B (Qwen3-4B bị loại vì sanity 66.7% < 80%); sanity accuracy 100% |
| Chi phí | 0 đồng (Kaggle miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~100 bước (batch 1 × grad-accum 8 = batch hiệu dụng 8), ~40–60 phút trên T4 |
| VRAM cao nhất | ~5.7 GB (GPU 0 còn 9.9/15.6 GB trống sau NB3b) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0955 |
| Độ chính xác reward trên held-out | 0.690 |
| Margin trên held-out | +0.0878 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 544.5 → 568.1 ký tự (held-out) |

Loss đăng nhập đầu tiên là 0.6964 ≈ log 2, xác nhận mô hình tham chiếu chính là bản SFT
(policy khởi tạo trùng reference).

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Nhìn ảnh `03-dpo-reward-curves.png`: ở cả tập huấn luyện lẫn held-out, đường `rewards/chosen`
và `rewards/rejected` đều **đi lên** khỏi mốc 0, nhưng `chosen` tăng nhanh hơn. Cuối huấn luyện
`chosen = +0.407`, `rejected = +0.311`, margin = `+0.095`. Trên held-out, `chosen = +0.424`,
`rejected = +0.336`, margin = `+0.088` — **cùng dấu và cùng độ lớn với đường huấn luyện**, tức
mô hình không học thuộc (overfit) mà học được một xu hướng tổng quát hoá.

Điểm cần nói rõ: đây **không** phải dịch chuyển xác suất (likelihood displacement). Nếu bị
likelihood displacement thì margin vẫn tăng nhưng `rewards/chosen` phải **giảm** (rejected giảm
nhanh hơn). Ở đây cả hai đều dương, nên đây đúng là trường hợp **INTENDED** — chẩn đoán tự động
in ra `[INTENDED] Chosen +0.416 up, rejected +0.329, margin +0.086.` khớp với điều tôi đọc được
trên biểu đồ. Margin tuyệt đối nhỏ (≈ 0.09) vì β = 0.1 và chỉ chạy 1 epoch ~100 bước: reward
ngầm định là β·log(π/π_ref), nên một margin 0.09 tương ứng tỉ số xác suất ~e^(0.09/0.1) ≈ e^0.9
≈ 2.5× nghiêng về câu `chosen`. Đó là mức tăng hợp lý nhưng chưa mạnh; chạy thêm epoch hoặc tăng
lr sẽ đẩy margin lên, đổi lại dễ quá khớp hơn.

Trả lời câu hỏi NB0: margin có thể tăng trong khi log-xác suất của câu `chosen` giảm, vì DPO chỉ
quan tâm **hiệu** reward giữa chosen và rejected, không quan tâm giá trị tuyệt đối của từng bên.
Nếu `rejected` giảm nhanh hơn `chosen`, margin vẫn tăng dù cả hai vế đều âm dần — đó là hiện tượng
likelihood displacement, và chỉ đường `rewards/chosen` ở NB3 mới phân biệt được với trường hợp
INTENDED. RPO (NB3b) khắc phục bằng cách thêm NLL của câu chosen vào loss.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 6 | 35 | 0.530 [0.460, 0.600] | 0.511 (n=44) | 0.533 |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 0.625 [0.500, 0.875] | 0.625 | 0.000 |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0.500 [0.500, 0.500] | 0.500 | — (độ dài bằng nhau) |

Giám khảo: `rm-panel:Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 1.00 ·
`score_length_spearman`: 0.140 (Llama) / 0.234 (Qwen3) · `judge_agreement`: 0.879 (n=58).

**Khoảng tin cậy 95% của held-out là [0.460, 0.600] — chứa 0.5.** Vậy chưa đủ bằng chứng để nói
DPO tốt hơn SFT: win rate 0.53 nằm trong nhiễu. Đây là kết quả hợp lệ và tôi viết thật, không tô hồng.

**Giám khảo có đáng tin không?** Hội đồng đáng tin một phần: giám khảo Llama đạt sanity 100% trên
12 cặp hiển nhiên, nhưng giám khảo **Qwen3-4B chỉ đạt 66.7% (< 80%) nên bị loại khỏi hội đồng** —
đúng như cơ chế lọc của notebook. Việc Qwen3 trượt bộ sanity nghĩa là nó chấm tiếng Việt chưa đủ
tin cậy để tính vào panel, nên `per_judge` chỉ còn Llama. Nếu để nguyên cả hai, hội đồng sẽ lệch.

**DPO thắng vì hay hơn hay vì dài hơn?** `longer_answer_won_frac = 0.533` — gần 0.5, tức độ dài
hầu như không quyết định kết quả. `length_matched_win_rate = 0.511` (trên 44 cặp dài gần bằng nhau)
xấp xỉ win rate tổng, nên không có bằng chứng DPO "hack độ dài". Trung bình câu DPO dài hơn SFT
chỉ 24 ký tự (568 vs 544), và `score_length_spearman` thấp (0.14–0.23) cho thấy reward model không
chấm chủ yếu theo độ dài. Kết luận: hiệu ứng nhỏ, không phải do độ dài.

**Hai giám khảo có lệch nhau?** `per_judge` cho thấy Qwen3 cho DPO thắng 0.45 (nghiêng SFT) còn
Llama cho 0.53 (nghiêng DPO) — lệch chiều nhau. Cả hai đều thuộc họ Skywork, cùng nhà với reward
model đã gán nhãn `sea-ultrafeedback-onpolicy`, nên có nguy cơ rò rỉ sở thích (preference leakage);
việc Qwen3 (cùng họ Qwen với mô hình Sailor2 sinh dữ liệu và với policy Qwen3-4B) lại cho DPO thắng
*thấp hơn* cho thấy thiên vị ở đây không đơn giản theo họ mô hình. Tôi tin win rate của hội đồng
(bảo thủ) hơn bất kỳ giám khảo đơn lẻ nào.

**Hai ví dụ cụ thể:**

- **Hữu ích (h4 — so sánh Python vs JavaScript):** SFT trả lời 1429 ký tự, DPO 1353 ký tự. Bản DPO
  gọn hơn mà vẫn giữ đủ các ý chính — đây là kiểu thay đổi nhỏ mà DPO tạo ra: bớt lan man chứ không
  đổi nội dung. Với 4 câu helpfulness, DPO thắng 1, hoà 3, thua 0.
- **An toàn (s4 — ý định tự hại):** cả SFT và DPO đều từ chối và hướng người dùng tìm chuyên gia tâm
  lý, **giống hệt nhau** (cùng 433 ký tự). Với 4 câu safety, kết quả là hoà 4/4 — DPO không làm mô
  hình an toàn hơn cũng không kém đi. Điều này hợp lý: dữ liệu sở thích là UltraFeedback chung, không
  chuyên về an toàn tiếng Việt, nên DPO hầu như không chạm tới hành vi an toàn.

Lưu ý chung về đầu ra: một số câu trả lời bắt đầu bằng thẻ lạ `</tool_call>` / `<tool_call>` do lỗi
định dạng của mô hình gốc — cả SFT lẫn DPO đều bị, nên không phải do DPO gây ra.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy β-sweep (tốn thêm ~2.5 giờ GPU cho 3 lần huấn luyện). Giả thuyết 3 câu:

1. β lớn (0.5) giữ policy gần reference hơn, nên margin held-out sẽ **nhỏ hơn** nhưng ổn định hơn;
   β nhỏ (0.05) cho policy đi xa hơn, margin lớn hơn nhưng dễ quá khớp tập huấn luyện.
2. Độ chính xác reward trên held-out sẽ đạt đỉnh ở β cỡ 0.1 rồi giảm ở hai đầu — β quá lớn thì học
   ít, β quá nhỏ thì overfit.
3. Vì margin = β·log-ratio, so **giá trị margin thô giữa các β là vô nghĩa**; phải so độ chính xác
   và đường cong, đúng như cảnh báo trong `scripts/eval_judge.py`.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi là **giữ mô hình tham chiếu (reference) đúng là bản SFT đã gộp
(`models/sft-merged`), chứ không phải mô hình gốc**, và chọn β = 0.1 với lr = 5e-6.

Phương án thay thế là làm theo snippet cũ trong slide: chồng LoRA DPO lên LoRA SFT rồi để TRL tắt
adapter để lấy reference — nghĩa là reference hoá ra là *mô hình gốc*, không phải bản SFT; kèm
lr = 5e-7 và `max_prompt_length` (đã bị TRL 1.x bỏ). Tôi chọn cách của lab K4 vì hai lý do. Thứ
nhất, về lý thuyết DPO: mục tiêu là kéo policy lệch khỏi *điểm xuất phát SFT* theo hướng câu được
chọn, nên reference phải là SFT — dùng mô hình gốc sẽ đo nhầm "khoảng cách tới base" chứ không phải
"mức học được từ dữ liệu sở thích". Thứ hai, `precompute_ref_log_probs=True` chấm mọi cặp đúng lúc
LoRA mới bằng 0, nên reference trùng khít SFT và không phụ thuộc vào việc bật/tắt adapter khi huấn
luyện — điều này thể hiện qua loss đăng nhập đầu tiên 0.6964 ≈ log 2.

Kết quả xác nhận lựa chọn: chẩn đoán ra **INTENDED**, margin dương trên cả train (+0.095) lẫn
held-out (+0.088), độ chính xác held-out 0.69 > 0.5. Tôi hơi bất ngờ vì mức tăng nhỏ hơn kỳ vọng —
nhưng β = 0.1 và chỉ ~100 bước thì đó là hợp lý, không phải dấu hiệu hỏng.

Nếu làm lại, tôi sẽ: (1) chạy **β-sweep** để biết 0.1 có phải điểm tốt nhất không, thay vì nhận
mặc định; (2) thêm **giám khảo API khác họ** (Gemini) làm chấm chéo, vì hội đồng hiện tại chỉ còn
1 giám khảo Skywork sau khi Qwen3 trượt sanity — quá ít để khử rò rỉ sở thích; (3) chạy **nhiều epoch
hơn hoặc lr cao hơn** để margin rõ hơn, đổi lại phải theo dõi overfit qua đường held-out.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 200 câu | 0.500 ± 0.0354 | 0.515 ± 0.0354 | +0.015 |
| GSM8K | — | (chưa chạy) | (chưa chạy) | — |
| Global-MMLU-vi | — | (chưa chạy) | (chưa chạy) | — |

Tôi chỉ chạy được IFEval; GSM8K và Global-MMLU-vi không kịp chạy trong phiên (mỗi bộ tốn thêm hàng
chục phút trên T4). Với IFEval, Δ = +0.015, trong khi 2× stderr ≈ 0.071 — **Δ nhỏ hơn 2× sai số
chuẩn rất nhiều**, nên không thể nói DPO làm IFEval tốt lên. Đây gần như chắc chắn là nhiễu.

Vì thiếu GSM8K, tôi **không thể kết luận về "thuế căn chỉnh" (alignment tax)** — hiệu ứng mà rubric
muốn thấy là điểm GSM8K giảm sau DPO. Với dữ liệu hiện có, bộ đo chuẩn **không cùng chiều rõ ràng**
với NB4: NB4 cho DPO nhích hơn (0.53, trong nhiễu), IFEval cũng nhích hơn (0.515, trong nhiễu) —
cả hai đều là "không phát hiện khác biệt", nhất quán với nhau nhưng không đủ mạnh để khẳng định.
Một điểm đáng nói: IFEval đo khả năng tuân định dạng, mà cả SFT lẫn DPO đều có lỗi thẻ `<tool_call>`
trong đầu ra — điều này có thể kéo điểm IFEval xuống cho cả hai bên.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.680 | +0.0276 | 305.8 | mức cơ sở, INTENDED |
| RPO | 0.680 | +0.0408 | 316.65 | `rewards/chosen` dương mạnh (+0.569) — NLL giữ chosen không tụt, INTENDED |
| DPO-norm | 0.610 | +0.0094 | 316.95 | LIKELIHOOD DISPLACEMENT (chosen −0.186 < 0) |
| LD-DPO | 0.570 | +0.0234 | 316.30 | LIKELIHOOD DISPLACEMENT (chosen −0.138 < 0) |
| ORPO | 0.660 | — (log-odds −0.625) | 347.95 | không reference; chạy từ `models/sft-merged` |

*Margin held-out tính bằng `eval_chosen_reward − eval_rejected_reward`.*

**Biến thể nào thay đổi độ dài nhiều nhất?** **ORPO** — đầu ra dài 347.95 ký tự so với ~306–317 của
các biến thể khác (tăng ~14% so với DPO). Vì sao: ORPO **không cần reference** và gộp SFT-NLL với
hình phạt log-odds vào một bước, nên nó vừa học phân biệt chosen/rejected vừa tiếp tục tối ưu NLL
trên câu chosen. Thành phần NLL (giống SFT) khuyến khích mô hình sinh câu trả lời đầy đủ, dài hơn —
đó là lý do độ dài phình ra. Đáng chú ý, RPO (cũng có thành phần NLL) chỉ dài 316.65, thấp hơn nhiều
so với ORPO, vì RPO vẫn giữ reference nên bị "neo" gần bản SFT.

**RPO có giữ `rewards/chosen` dương trong khi DPO không?** Trong lần chạy này cả DPO lẫn RPO đều
INTENDED (chosen dương), nên không tái hiện được tình huống DPO bị displacement. Nhưng đúng như kỳ
vọng lý thuyết, RPO cho `rewards/chosen` cao hơn hẳn (+0.569 so với +0.097 của DPO) — NLL trên câu
chosen giữ xác suất của nó không bị kéo xuống, đúng mục đích thiết kế. Hai biến thể chuẩn hoá độ dài
(DPO-norm, LD-DPO) lại **bị** displacement trong lần chạy này và có độ chính xác thấp hơn (0.57–0.61),
nên trong bối cảnh này chuẩn hoá độ dài không giúp gì.

**Độ chính xác reward cao hơn có nghĩa là mô hình tốt hơn không?** Không hẳn — cần chấm bằng giám
khảo độc lập (NB4) trên adapter tương ứng (đặt `DPO_ADAPTER_OVERRIDE`) để kiểm tra, vì reward
accuracy chỉ đo mô hình có tách đúng cặp chosen/rejected không, không đo chất lượng thực tế.

---

## 9. GRPO (bonus NB7)

**Không chạy.** Phiên GPU hết thời gian sau NB6 nên tôi bỏ NB7 (+8) để ưu tiên hoàn thiện phần bắt
buộc. Ghi chú giả thuyết: với N_TEST = 100, sai số chuẩn ≈ √(p(1−p)/n); nếu độ chính xác trước là
p ≈ 0.3 thì stderr ≈ √(0.3·0.7/100) ≈ 0.046, nên chênh lệch độ chính xác sau GRPO phải vượt ~0.09
(2× stderr) mới đáng tin. Reward dạng định dạng ("Đáp số: <số>") thường tăng trước reward tính đúng,
vì mô hình học cách xuất đúng khung trước khi học tính đúng.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6, chỉ IFEval)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Giám khảo **Qwen3-4B trượt bộ sanity tiếng Việt (66.7%)** và bị loại khỏi hội đồng, dù nó cùng họ
Qwen với chính policy đang được chấm — tôi tưởng nó sẽ "ưu ái" DPO, nhưng thực tế nó lại cho DPO
thắng *thấp hơn* giám khảo Llama. Điều này cho thấy thiên vị của giám khảo không đơn giản theo họ
mô hình, và cơ chế lọc sanity của notebook thực sự có ích.
