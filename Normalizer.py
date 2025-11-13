import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# ========================
# 1. Carga y normalización
# ========================
df = pd.read_csv("dataset.csv")  # <-- cambia al nombre real de tu dataset
df["text"] = df["text"].astype(str).str.strip()

# ========================
# 2. Codificación de la etiqueta
# ========================

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["grade"])

# ========================
# 3. División en train/test
# ========================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ========================
# 4. Tokenización
# ========================
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text", "grade"])
test_dataset = test_dataset.remove_columns(["text", "grade"])
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ========================
# 5. Configuración del dispositivo CUDA
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# ========================
# 6. Modelo
# ========================
num_labels = len(encoder.classes_)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.to(device)  # Mover el modelo al dispositivo CUDA

# ========================
# 7. Métricas
# ========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ========================
# 8. Configuración entrenamiento
# ========================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
)

# ========================
# 9. Trainer
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ========================
# 10. Entrenamiento
# ========================
trainer.train()

# ========================
# 11. Evaluación final
# ========================
results = trainer.evaluate()
print("Resultados de evaluación:", results)