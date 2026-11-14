# LifeMate: bàn giao fine-tune Qwen3.5-4B (chuyển từ Kaggle sang Colab)

Tài liệu này tóm tắt toàn bộ những gì đã làm, kết quả khảo sát, các lỗi đã gặp và việc còn lại,
để mở phiên mới làm tiếp trên Google Colab. Mọi đường dẫn tính từ `llm-test/lifemate/`.

> Dữ liệu cuộc thi **không được chia sẻ/phát tán**. Mọi dataset/notebook chứa dữ liệu phải để **Private**.

---

## 1. Mục tiêu

Fine-tune `Qwen/Qwen3.5-4B` (LoRA/QLoRA) bằng `train/train.jsonl` để agent chọn tool call chính xác hơn.
Agent chạy theo vòng lặp (`main/agent.py`): mỗi lượt model trả đúng một JSON
`{"action":"call","name":...,"arguments":{...}}` hoặc `{"action":"done"}`, tool chạy qua `ToolEnv`, kết quả đưa lại vào
`EXECUTION_TRACE`. Điểm chấm theo `tool_calls` + `state_hash`.

## 2. Dữ liệu (đã khảo sát)

- `train/train.jsonl`: 1545 mẫu. Mỗi mẫu: `audio_text`, `context`, `history`, `label{tool_calls, tool_results, final_state}`.
- 0 đến 6 tool call mỗi mẫu; 146 tool; 111 mẫu không cần gọi tool (`[]`); 217 call là `sys.ask_missing_info`; 1018 mẫu có `history`.
- `sys.ask_missing_info` có dạng `{"tool_name": "phone.call", "params": ["contact_id"]}`, là bước cuối (sau đó dừng).
- Replay `label.tool_calls` qua `ToolEnv` cho `final_state` khớp 300/300 mẫu đã thử, nên chấm điểm cục bộ được.
- `README.md` mô tả đầy đủ format; `train/zaic_tools` là thư viện chạy tool (chỉ dùng stdlib).

## 3. Những gì đã làm

### 3.1 Sửa prompt: `main/agent.py`
(bản gốc đã lưu ở scratchpad của phiên cũ, không nằm trong repo)
- `conversation_summary()`: **sửa lỗi** trước đây bỏ mất `tool_calls/tool_results` của lượt assistant trong history. Giờ giữ kèm `tools` (kết quả cắt 600 ký tự).
- Thêm `time_hints()`: giờ hiện tại + lịch 10 ngày tới có thứ ("hôm nay/ngày mai/ngày kia"), thêm vào `USER_CONTEXT` dưới khoá `time_hints`.
- `decision_prompt()`: viết lại luật (8 luật: không đoán ID, yêu cầu có điều kiện phải đọc trước, nhiều đối tượng thì gọi từng cái,
  tham chiếu lấy từ history, đổi số tiền/ngày giờ, khi nào `ask_missing_info`, khi nào `done`, hành động bảo thủ).
  Thứ tự mới: luật, `TOOL_SCHEMAS`, rồi `USER_CONTEXT`, `CONVERSATION_HISTORY`, `LATEST_REQUEST`, `EXECUTION_TRACE`, dòng chốt `QUYẾT ĐỊNH TIẾP THEO (JSON):`.
- **Chưa đo** được prompt mới có tốt hơn bản cũ hay không (chưa có bộ đánh giá).

### 3.2 Dữ liệu SFT: `finetune/build_sft.py`
- Tách mỗi mẫu thành các example từng bước: `prompt` = đúng output của `decision_prompt()`, `completion` = JSON quyết định kế tiếp.
  Sau `ask_missing_info` không có bước `done`. Mẫu không gọi tool chỉ có một example `done`.
- Tách train/val theo `id` (10% mẫu làm val, seed 0) → `finetune/out/val_ids.json`.
- Kết quả: `finetune/out/sft_train.jsonl` (4377 example, ~78 MB), `finetune/out/sft_val.jsonl` (492 example).
  Prompt trung bình ~16,7 nghìn ký tự, dài nhất ~28 nghìn ký tự (≈ 5–10 nghìn token, ước lượng, chưa đo bằng tokenizer).
- Chạy lại khi đổi prompt: cần `pydantic` (`python -I finetune/build_sft.py` từ thư mục `lifemate/`).
- **Phải dựng lại + tải lên lại mỗi khi sửa `decision_prompt()`**, vì model phải thấy đúng format lúc train như lúc chạy.

### 3.3 Script train: `finetune/train_kaggle.py`
QLoRA 4-bit, batch 1, accumulation 16, loss chỉ tính trên `completion`. Bản hiện tại đã gồm các sửa sau (rút từ lỗi thực tế):

| Vấn đề gặp | Cách sửa đã áp dụng |
|---|---|
| `TrainingArguments(warmup_ratio=...)` lỗi (transformers 5 bỏ tham số) | dùng `warmup_steps=max(1, int(0.03 * TOTAL_STEPS))` |
| OOM 4,46 GiB do bảng logits `[seq × vocab]` toàn chuỗi | `CompletionTrainer.compute_loss` gọi `model(..., logits_to_keep=n_comp+1)`, chỉ tính loss trên token completion; chuẩn hoá theo `num_items_in_batch` |
| `AssertionError` batch ≠ 1 (T4×2: Trainer nhân batch theo số GPU) | `os.environ["CUDA_VISIBLE_DEVICES"]="0"` ở đầu cell 2, **trước khi import torch** |
| OOM 1,87 GiB ở backward (ma trận attention bình phương, nghi attention kiểu eager) | `attn_implementation="sdpa"` khi nạp model; `MAX_LEN` giảm 10240, 8192, 6144; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**Trạng thái:** sau các sửa trên, **chưa xác nhận** là train chạy thành công với `sdpa` + `MAX_LEN=6144`.
Lần chạy gần nhất mới báo OOM ở backward (trước khi áp dụng `sdpa`). Chưa có log tốc độ (giây/step) hay đường cong loss.

Tham số hiện tại: `MODEL_ID="Qwen/Qwen3.5-4B"`, `MAX_LEN=6144`, `MAX_EXAMPLES=300` (chạy thử; đổi `None` để train đủ),
`EPOCHS=1`, `LR=1e-4`, `ACCUM=16`, LoRA `r=16, alpha=32, dropout=0.05, target_modules="all-linear"`,
`fp16=True` (T4 không có bf16), gradient checkpointing, lưu checkpoint mỗi 100 step, tự dừng sau 10,5 giờ.

## 4. Ước lượng thời gian (chưa đo)
- Chạy thử 300 mẫu: ~20–40 phút (gồm tải model).
- Train đủ 4377 example, 1 epoch: ~4–8 giờ trên T4 (≈ 274 step; thời gian ≈ 274 × giây/step). Hai epoch gần như không kịp.
- Số thật chỉ có sau khi chạy thử và đọc log.

## 5. Chuyển sang Colab: điều cần đổi

1. **Đường dẫn dữ liệu**: không có `/kaggle/input`. Tải `sft_train.jsonl`, `sft_val.jsonl` lên Google Drive riêng tư (hoặc upload thẳng vào runtime),
   rồi sửa `DATA_DIR` ở cell 2 (ví dụ `/content/drive/MyDrive/lifemate/`). `OUT` nên trỏ vào Drive để checkpoint không mất khi runtime bị ngắt.
2. **Không có "Save & Run All" chạy ngầm** như Kaggle. Runtime Colab miễn phí bị ngắt khi idle/đóng tab và có giới hạn giờ ngắn hơn.
   Vì vậy: lưu checkpoint vào Drive (đã có `save_steps=100`), và khi bị ngắt dùng `trainer.train(resume_from_checkpoint=True)`.
   Chạy nền thật sự cần gói trả phí. Giảm `TIME_LIMIT_H` cho phù hợp với runtime của bạn.
3. **GPU**: T4 (~15 GB) có cùng giới hạn như Kaggle, nên giữ `fp16`. Nếu có **L4/A100** thì bf16 dùng được (đổi `bnb_4bit_compute_dtype=torch.bfloat16`, `fp16=False, bf16=True`), VRAM lớn hơn nên có thể tăng `MAX_LEN` và train nhanh hơn.
   Colab chỉ có 1 GPU nên `CUDA_VISIBLE_DEVICES` giữ nguyên cũng không sao.
4. Cell 1 (`pip install`): giữ `transformers>=5.0.0 peft accelerate bitsandbytes`, restart runtime nếu được yêu cầu. Cần đăng nhập/mạng để tải weights từ HuggingFace.
5. Việc cần kiểm tra khi chạy thật: có cảnh báo về "fast path"/thiếu `flash-linear-attention` hay `causal-conv1d` khi nạp model không (Qwen3.5 có lớp attention tuyến tính;
   thiếu thư viện này thì có thể chậm/tốn bộ nhớ). Nếu có, thử cài hai thư viện đó (có thể phải build, mình chưa kiểm chứng).

## 6. Việc còn lại (theo thứ tự mình khuyên)

1. **Chạy thử 300 mẫu** trên Colab, ghi lại: dòng `max len` + số mẫu bị bỏ ở cell 3 (nếu mất nhiều mẫu thì rút gọn prompt/schema thay vì cứ hạ `MAX_LEN`),
   giây/step, loss giảm không (không `nan`; nếu `nan` thì `LR=5e-5`).
2. **Viết bộ đánh giá** `finetune/eval_agent.py` (chưa có): chạy agent trên các id trong `val_ids.json`, replay `label.tool_calls` qua `ToolEnv` để lấy `state_hash` vàng,
   so `tool_calls` + `state_hash`. Chạy cho model gốc (làm điểm nền) và model đã fine-tune.
3. **Train bằng transcript ASR nhiễu** (quan trọng): lúc thi model nhận chữ từ PhoWhisper, ví dụ `artifacts/public_test_asr.jsonl` có câu bị sai như
   "chạy bốn sáng mai mốt hai tiếng". Train hiện chỉ dùng `audio_text` sạch.
   Cách làm: chạy `main/asr.py --split train --include-history --model <đúng model dùng khi thi> --device cuda --output train_asr.jsonl`
   (5036 file wav, ~6,9 giờ âm thanh, ~760 MB; cần upload audio vào Drive/Colab riêng tư), rồi sửa `build_sft.py`
   dùng chữ ASR cho khoảng một nửa số mẫu (`attach_asr_transcripts()` đã có trong `agent.py`), giữ nguyên `label`. Dựng lại `sft_*.jsonl`.
4. **(Tuỳ chọn) rút ngắn `TOOL_SCHEMAS`**: `allowed_tool_specs()` đã không đưa `returns/app` vào prompt. Chỉ còn bỏ `description` của tham số
   (giảm thêm khoảng 25% schema nhưng có thể mất thông tin), chỉ làm sau khi có số đo từ bước 2.
5. Sau khi train: gộp LoRA vào model gốc (cell 6 trong script, đang để comment), đặt `MODEL_ID` trong `main/qwen_local.py` trỏ tới model đã gộp, chạy eval.
6. **Kiểm tra thể lệ**: fine-tune và mang weights riêng vào Docker có được phép không (README không nói rõ; docstring `qwen_local.py` ghi "no fine-tuning" cho bản gốc).
   Transcript `public_test` phải do cùng model ASR ở bước 3 tạo ra.

## 7. Bản đầy đủ các cell hiện tại
Xem `finetune/train_kaggle.py` (các khối `# %%`), bản này đã gồm mọi sửa ở mục 3.3. Chỉ cần đổi `DATA_DIR`/`OUT` cho Colab.
