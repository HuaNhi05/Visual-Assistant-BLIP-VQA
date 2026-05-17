import os
import json
import re
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForQuestionAnswering, BlipProcessor
from torch.optim import AdamW
from tqdm import tqdm

# ========================================
# 1. CÀI ĐẶT MÔI TRƯỜNG VÀ THAM SỐ HUẤN LUYỆN
# ========================================
DATA_PATH = "/kaggle/working/vqav2_data/data"
BASE_OUTPUT = "/kaggle/working/vizwiz_blip_results"  
MODEL_DIR = os.path.join(BASE_OUTPUT, "model")
PROCESSOR_DIR = os.path.join(BASE_OUTPUT, "processor")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSOR_DIR, exist_ok=True)

NUM_SAMPLES = 10000   
BATCH_SIZE = 8      
EPOCHS = 3        

# ========================================
# 2. CHUẨN HÓA TEXT & HÀM TÍNH ACCURACY
# ========================================
def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def vqa_accuracy(prediction, answers):
    prediction = normalize_text(prediction)
    if isinstance(answers, str): 
        answers = [answers]
    match_count = sum(1 for a in answers if normalize_text(a) == prediction)
    return min(match_count / 1.0, 1.0)

# ========================================
# 3. DATASET THU NHỎ
# ========================================
class VQAv2MiniDataset(Dataset):
    def __init__(self, parquet_dir, processor, num_samples=1000):
        all_parquet_files = [os.path.join(r, f) for r, _, fs in os.walk(parquet_dir) for f in fs if f.endswith(".parquet")]
        if not all_parquet_files:
            raise FileNotFoundError("Không thấy file .parquet dữ liệu!")
        
        full_df = pd.concat([pd.read_parquet(f) for f in all_parquet_files]).reset_index(drop=True)
        self.df = full_df.head(num_samples)
        self.processor = processor
        print(f" Đã bóp nhỏ dữ liệu thành công! Đang sử dụng {len(self.df)} mẫu.")

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: 
            image = row["image"].convert("RGB")
        except: 
            image = Image.new("RGB", (384, 384), color=(128, 128, 128))
        
        encoding = self.processor(
            images=image, 
            text=str(row["question"]), 
            padding="max_length", 
            truncation=True, 
            max_length=64, 
            size={"height": 384, "width": 384}, 
            return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            text=str(row["multiple_choice_answer"]),
            padding="max_length",
            truncation=True,
            max_length=8,
            return_tensors="pt"
        ).input_ids

        return {
            "pixel_values": encoding.pixel_values.squeeze(0), 
            "input_ids": encoding.input_ids.squeeze(0), 
            "attention_mask": encoding.attention_mask.squeeze(0), 
            "labels": labels.squeeze(0),
            "raw_answers": str(row["multiple_choice_answer"]),
            "question_text": str(row["question"])
        }

# ========================================
# 4. KHỞI TẠO MÔ HÌNH VÀ BỘ TỐI ƯU HÓA
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(device)

train_dataset = VQAv2MiniDataset(DATA_PATH, processor, num_samples=NUM_SAMPLES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

optimizer = AdamW(model.parameters(), lr=5e-6)

# ========================================
# 5. TIẾN TRÌNH HUẤN LUYỆN SIÊU TỐC
# ========================================
print("\n Bắt đầu tiến trình Mini-Train tạo trọng số...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"🎬 Epoch {epoch+1}/{EPOCHS}")
    
    for batch in pbar:
        optimizer.zero_grad()
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

# ========================================
#  6. ĐÁNH GIÁ VÀ IN ACCURACY 
# ========================================
print("\n Đang quét dữ liệu để tính toán điểm Accuracy...")
model.eval()
score, total, demo = 0, 0, []
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

with torch.no_grad():
    for batch in tqdm(val_loader, desc=" Evaluating Accuracy"):
        generated_ids = model.generate(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_length=8, num_beams=1
        )
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for pred, gt, q in zip(preds, batch["raw_answers"], batch["question_text"]):
            acc = vqa_accuracy(pred, gt)
            score += acc
            total += 1
            if len(demo) < 30:
                demo.append({"q": q, "a": pred, "gt": gt, "acc": acc})

final_acc = (score / total) * 100
print(f"\n Accuracy: {final_acc:.2f}%")

# ========================================
# 7. LƯU THƯ MỤC 
# ========================================
model.save_pretrained(MODEL_DIR)          
processor.save_pretrained(PROCESSOR_DIR) 

with open(os.path.join(BASE_OUTPUT, "accuracy.txt"), "w") as f:
    f.write(f"Final Evaluation Accuracy: {final_acc:.2f}%\n")

with open(os.path.join(BASE_OUTPUT, "result_zero_shot.json"), "w", encoding="utf-8") as f:
    json.dump(demo, f, ensure_ascii=False, indent=4)

print(f"Toàn bộ hệ thống file nằm tại: {BASE_OUTPUT}")