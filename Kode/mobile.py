import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

folder = r"C:\RISMA\KULIAH\semester 5\ML"
df = pd.read_csv(folder + r"\train.csv")

print("Jumlah baris & kolom:", df.shape)
print("Jumlah tiap kelas harga:")
print(df["price_range"].value_counts().sort_index())

nama_kelas = ["Murah", "Menengah", "Mahal", "Sangat Mahal"]

# ---------- 1. Grafik distribusi kelas harga ----------
fig, ax = plt.subplots(figsize=(6, 4))
df["price_range"].value_counts().sort_index().plot(
    kind="bar", color=["#4C9BE8", "#5FBF8F", "#F2B84B", "#E8684C"], ax=ax)
plt.title("Distribusi Kelas Harga HP")
plt.xlabel("Kelas harga (price_range)")
plt.ylabel("Jumlah HP")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(folder + r"\mobile_1_distribusi_kelas.png", dpi=150)
plt.show()
plt.close("all")

# ---------- 2. Siapkan data ----------
X = df.drop(columns="price_range")
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 3. Latih 3 model ----------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

hasil = {}
for nama, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    hasil[nama] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro"),
        "Recall": recall_score(y_test, pred, average="macro"),
        "F1-Score": f1_score(y_test, pred, average="macro"),
    }

hasil_df = pd.DataFrame(hasil).T
print(hasil_df.round(3))

# ---------- 4. Grafik perbandingan model ----------
fig, ax = plt.subplots(figsize=(9, 5))
hasil_df.plot(kind="bar", ax=ax)
plt.title("Perbandingan Performa Model")
plt.ylabel("Skor")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(folder + r"\mobile_2_perbandingan_model.png", dpi=150)
plt.show()
plt.close("all")

# ---------- 5. Confusion matrix model terbaik (berdasarkan F1) ----------
terbaik = hasil_df["F1-Score"].idxmax()
print("Model terbaik:", terbaik)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_estimator(
    models[terbaik], X_test, y_test,
    display_labels=nama_kelas, cmap="Blues", ax=ax)
plt.title(f"Confusion Matrix - {terbaik}")
plt.tight_layout()
plt.savefig(folder + r"\mobile_3_confusion_matrix.png", dpi=150)
plt.show()
plt.close("all")

# ---------- 6. Fitur paling berpengaruh (Random Forest) ----------
rf = models["Random Forest"]
penting = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
penting.tail(10).plot(kind="barh", color="#4C9BE8", ax=ax)
plt.title("10 Fitur Paling Berpengaruh terhadap Harga HP")
plt.xlabel("Tingkat kepentingan")
plt.tight_layout()
plt.savefig(folder + r"\mobile_4_fitur_penting.png", dpi=150)
plt.show()
plt.close("all")