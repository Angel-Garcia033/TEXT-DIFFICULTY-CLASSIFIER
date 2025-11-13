# =========================================
# 📦 Importar librerías
# =========================================
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# 1️⃣ Cargar tokenizer y modelo
# =========================================
checkpoint_path = r"C:\Proyectos Programación\New folder\New folder\results\checkpoint-3165"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# =========================================
# 2️⃣ Cargar dataset CSV
# =========================================
dataset = load_dataset("csv", data_files={"test": "dataset.csv"})

# =========================================
# 3️⃣ Validar columnas requeridas
# =========================================
required_columns = {"text", "grade"}
if not required_columns.issubset(dataset["test"].column_names):
    print(f"❌ El dataset no contiene las columnas requeridas: {required_columns}")
    print(f"✅ Columnas encontradas: {dataset['test'].column_names}")
    exit()

# =========================================
# 4️⃣ Codificar etiquetas una sola vez
# =========================================
dataset["test"] = dataset["test"].rename_column("grade", "label")

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(dataset["test"]["label"])

dataset["test"] = dataset["test"].add_column("label_encoded", labels_encoded)
dataset["test"] = dataset["test"].remove_columns(["label"])
dataset["test"] = dataset["test"].rename_column("label_encoded", "label")

# =========================================
# 5️⃣ Tokenización
# =========================================
def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    tokens["labels"] = batch["label"]
    return tokens

test_dataset = dataset["test"].map(tokenize, batched=True)
test_dataset = test_dataset.remove_columns(["text", "label"])
test_dataset.set_format("torch")

print("✅ Dataset después de la preparación:")
print(test_dataset)

# =========================================
# 6️⃣ Predicciones
# =========================================
trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = np.array(dataset["test"]["label"])  # ← etiquetas reales numéricas

# =========================================
# 7️⃣ Métricas y matriz de confusión
# =========================================
print("\n📈 Accuracy:", accuracy_score(y_true, y_pred))
print("\n📋 Reporte de clasificación:\n", classification_report(y_true, y_pred, target_names=encoder.classes_))

# =========================================
# 🔟 Matriz de confusión (normalizada)
# =========================================
# Normaliza por fila → cada fila (clase real) suma 1.0
cm = confusion_matrix(y_true, y_pred, normalize='true')

# Redondear a 2 decimales
cm_percent = np.round(cm * 100, 2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=encoder.classes_)

# Graficar
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')

# Añadir valores dentro de las celdas
for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        value = cm_percent[i, j]
        ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                color="white" if value > 50 else "black", fontsize=10, fontweight="bold")

# Etiquetas y formato
ax.set(
    xticks=np.arange(len(encoder.classes_)),
    yticks=np.arange(len(encoder.classes_)),
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    ylabel="Etiqueta real",
    xlabel="Etiqueta predicha",
    title="📊 Matriz de Confusión (normalizada en %)"
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% de ejemplos")
plt.tight_layout()
plt.show()

