import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# =========================
# SETTINGS
# =========================
DATA_PATH = r"C:\ASTD\Tweets.txt"
MODEL_NAME = "UBC-NLP/MARBERT"

MAX_LEN = 256          # جرّب 128 أو 256
BATCH_SIZE = 8         # لو RAM يسمح: 16
EPOCHS = 10            # مع Early stopping
LR = 1e-5              # غالبًا أهدى وأفضل من 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.3

SEED = 42
TEST_SIZE = 0.20       # 80/20 ثم 10/10 من 20%
PATIENCE = 2           # Early stopping patience

# لو بياناتك 4-classes فعلاً خليها كما هي
LABELS = {"POS", "NEG", "NEU", "MIX"}

# =========================
# SEED
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# READ DATA
# =========================
def read_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            a, b = parts[0].strip(), parts[1].strip()

            if a.upper() in LABELS:
                label, text = a.upper(), b
            elif b.upper() in LABELS:
                label, text = b.upper(), a
            else:
                continue

            text = str(text).strip()
            if text:
                rows.append((text, label))

    df = pd.DataFrame(rows, columns=["text", "label"])
    return df

df = read_dataset(DATA_PATH)
print("Total samples:", len(df))
print(df.head())

if len(df) == 0:
    raise ValueError("Dataset is empty after parsing. Check Tweets.txt format (TAB between text and label).")

print("\nLabel distribution (all):")
print(df["label"].value_counts())

# =========================
# SPLIT DATA (stratified)
# =========================
train_df, temp_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=SEED
)

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

print("\nLabel distribution (train/val/test):")
print("Train:\n", train_df["label"].value_counts())
print("Val:\n", val_df["label"].value_counts())
print("Test:\n", test_df["label"].value_counts())

# =========================
# LABEL ENCODING
# =========================
labels = sorted(train_df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

train_df["label_id"] = train_df["label"].map(label2id)
val_df["label_id"]   = val_df["label"].map(label2id)
test_df["label_id"]  = test_df["label"].map(label2id)

print("\nLabel map:", label2id)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# DATASET
# =========================
class ASTDDataset(Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# إعدادات DataLoader (آمنة في Windows)
num_workers = 0
pin_memory = True if device.type == "cuda" else False

train_loader = DataLoader(ASTDDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
val_loader   = DataLoader(ASTDDataset(val_df), batch_size=BATCH_SIZE,
                          num_workers=num_workers, pin_memory=pin_memory)
test_loader  = DataLoader(ASTDDataset(test_df), batch_size=BATCH_SIZE,
                          num_workers=num_workers, pin_memory=pin_memory)

# =========================
# MODEL
# =========================
class MARBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.fc(cls)

model = MARBERTClassifier(len(label2id)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()

# =========================
# EVALUATE (returns metrics + preds)
# =========================
def evaluate(loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, mask)
            p = torch.argmax(outputs, dim=1)

            preds.extend(p.cpu().numpy().tolist())
            true.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(true, preds)
    p, r, f1, _ = precision_recall_fscore_support(true, preds, average="macro", zero_division=0)
    return acc, p, r, f1, true, preds

# =========================
# TRAIN LOOP + EARLY STOPPING
# =========================
best_f1 = -1
best_epoch = -1
bad_epochs = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))
    val_acc, val_p, val_r, val_f1, _, _ = evaluate(val_loader)

    print(f"Epoch {epoch:02d} | AvgLoss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val MacroF1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        bad_epochs = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best Val_F1={best_f1:.4f})")
            break

print(f"\nBest Val_F1={best_f1:.4f} at epoch={best_epoch}")

# =========================
# TEST (best checkpoint)
# =========================
model.load_state_dict(torch.load("best_model.pt", map_location=device))

test_acc, test_p, test_r, test_f1, y_true, y_pred = evaluate(test_loader)

print("\n=== FINAL TEST RESULT (best checkpoint) ===")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro P  : {test_p:.4f}")
print(f"Test Macro R  : {test_r:.4f}")
print(f"Test Macro F1 : {test_f1:.4f}")

# تقرير تفصيلي
target_names = [id2label[i] for i in range(len(id2label))]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))