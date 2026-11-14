# Kaggle notebook (dán từng khối "# %%" thành 1 cell). Accelerator: GPU T4 x2 hoặc P100. Internet: ON.
# Input dataset: thư mục chứa sft_train.jsonl, sft_val.jsonl  (mount tại /kaggle/input/<tên-dataset>/)

# %% 1. Cài thư viện (qwen3_5 cần transformers mới)
# !pip install -q -U "transformers>=5.0.0" peft accelerate bitsandbytes

# %% 2. Cấu hình
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # chỉ dùng 1 GPU, nếu không Trainer nhân batch theo số GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # phải đặt trước khi dùng CUDA
import glob, json, math, time, torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, TrainerCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen3.5-4B"
DATA_DIR = os.path.dirname(glob.glob("/kaggle/input/**/sft_train.jsonl", recursive=True)[0])
OUT = "/kaggle/working/lora"
MAX_LEN = 6144          # bỏ mẫu dài hơn (không cắt, vì phần cuối prompt là trace quan trọng). Xem "max len" in ở cell 3 rồi chỉnh
MAX_EXAMPLES = None     # đặt ví dụ 2000 để chạy thử nhanh / nếu 1 epoch đầy đủ quá lâu
EPOCHS, LR, ACCUM = 1, 1e-4, 16
TIME_LIMIT_H = 10.5     # tự dừng + lưu trước giới hạn 12h của Kaggle

# %% 3. Token hoá: loss chỉ tính trên phần completion
tok = AutoTokenizer.from_pretrained(MODEL_ID)
def encode(row):
    p = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    c_ids = tok(row["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids}

def load(name):
    rows = [json.loads(l) for l in open(f"{DATA_DIR}/{name}.jsonl", encoding="utf-8")]
    enc = [encode(r) for r in rows]
    kept = [e for e in enc if len(e["input_ids"]) <= MAX_LEN]
    print(name, len(rows), "->", len(kept), "| max len", max(len(e["input_ids"]) for e in enc))
    return kept
train_ds, val_ds = load("sft_train"), load("sft_val")
if MAX_EXAMPLES:
    import random; random.seed(0); random.shuffle(train_ds); train_ds = train_ds[:MAX_EXAMPLES]

def collate(batch):  # batch size 1 nên không cần pad; vẫn viết tổng quát
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": (torch.arange(n) < torch.tensor([[len(b["input_ids"])] for b in batch])).long()}

# %% 4. Model 4-bit + LoRA
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map={"": 0},
                                             attn_implementation="sdpa")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# %% 5. Train
class TimeLimit(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if time.time() - T0 > TIME_LIMIT_H * 3600:
            control.should_save = True; control.should_training_stop = True
T0 = time.time()
TOTAL_STEPS = math.ceil(len(train_ds) / ACCUM) * EPOCHS
args = TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=ACCUM,
    num_train_epochs=EPOCHS, learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * TOTAL_STEPS)),
    fp16=True, gradient_checkpointing=True, logging_steps=10,
    save_steps=100, save_total_limit=2, eval_strategy="no",   # eval chậm với prompt dài; đánh giá bằng agent ở bước sau
    report_to="none", remove_unused_columns=False, dataloader_pin_memory=False)
class CompletionTrainer(Trainer):
    """Chỉ tính logits cho các token completion (cuối chuỗi) -> tránh bảng logits [seq x vocab] khổng lồ."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        assert labels.shape[0] == 1, "script giả định batch size 1"
        n_comp = int((labels[0] != -100).sum())          # completion nằm liền ở cuối chuỗi
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    logits_to_keep=n_comp + 1)           # +1: vị trí dự đoán token completion đầu tiên
        logits = out.logits[:, :-1].float()
        tgt = labels[:, -n_comp:]
        if num_items_in_batch is not None:               # Trainer tự chuẩn hoá theo tổng token của cả accumulation
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum") / num_items_in_batch
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return (loss, out) if return_outputs else loss

trainer = CompletionTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collate,
                            callbacks=[TimeLimit()])
trainer.train()      # nếu bị ngắt: trainer.train(resume_from_checkpoint=True)
model.save_pretrained(f"{OUT}/final"); tok.save_pretrained(f"{OUT}/final")

# %% 6. (Cell riêng, sau khi train xong) Gộp LoRA vào model gốc để chạy offline trong Docker
# import torch; from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map={"": "cpu"})
# merged = PeftModel.from_pretrained(base, "/kaggle/working/lora/final").merge_and_unload()
# merged.save_pretrained("/kaggle/working/merged"); tok.save_pretrained("/kaggle/working/merged")
