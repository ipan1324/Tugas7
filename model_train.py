"""
model_train.py
==============
Tugas 7 – Backpropagation: Prediksi Penyakit Ginjal Kronis (CKD)

Dataset : data/kidney_disease.csv  (UCI CKD Dataset)
Sumber  : https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

Fitur yang digunakan (14 fitur numerik):
  age  – usia (tahun)
  bp   – tekanan darah (mm/Hg)
  sg   – berat jenis urin
  al   – albumin (0-5)
  su   – gula (0-5)
  bgr  – glukosa darah acak (mg/dL)
  bu   – urea darah (mg/dL)
  sc   – kreatinin serum (mg/dL)
  sod  – natrium (mEq/L)
  pot  – kalium (mEq/L)
  hemo – hemoglobin (g/dL)
  pcv  – volume sel terkemas (%)
  wc   – jumlah sel darah putih (cells/cmm)
  rc   – jumlah sel darah merah (millions/cmm)
Target:
  classification – 'ckd'=1, 'notckd'=0
"""

import os, sys, json
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')           # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(__file__)
DATASET     = os.path.join(BASE, 'data', 'kidney_disease.csv')
MODEL_PATH  = os.path.join(BASE, 'ckd_model.h5')
SCALER_PATH = os.path.join(BASE, 'scaler.pkl')
PLOTS_DIR   = os.path.join(BASE, 'static', 'plots')
METRICS_F   = os.path.join(BASE, 'static', 'metrics.json')

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
FEATURE_COLS = ['age','bp','sg','al','su','bgr','bu','sc',
                'sod','pot','hemo','pcv','wc','rc']
TARGET_COL   = 'classification'
RANDOM_STATE = 42
TEST_SIZE    = 0.2
EPOCHS       = 150
BATCH_SIZE   = 32

# ─── 1. Load & Bersihkan Dataset ──────────────────────────────────────────────
if not os.path.exists(DATASET):
    print(f"[ERROR] File tidak ditemukan: {DATASET}")
    sys.exit(1)

print("[INFO] Memuat dataset...")
df = pd.read_csv(DATASET)

# Bersihkan spasi pada semua nilai string
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Encode target: ckd=1, notckd=0
df[TARGET_COL] = df[TARGET_COL].map({'ckd': 1, 'notckd': 0})

# Konversi kolom fitur ke numerik (paksa; non-numerik → NaN)
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Isi NaN dengan median tiap kolom
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

# Hapus baris di mana target masih NaN
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"[INFO] Total sampel: {len(df)}")
print(f"[INFO] Distribusi kelas:\n  CKD (1): {y.sum()}  |  Tidak CKD (0): {(y==0).sum()}\n")

# ─── 2. Split & Normalisasi ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"[INFO] Training set : {len(X_train)} sampel")
print(f"[INFO] Test set     : {len(X_test)} sampel\n")

# ─── 3. Arsitektur Model ANN (Backpropagation) ───────────────────────────────
print("[INFO] Membangun model ANN...")
model = Sequential([
    # Input + Hidden Layer 1
    Dense(64, activation='relu', input_shape=(len(FEATURE_COLS),),
          name='hidden_1'),
    BatchNormalization(),
    Dropout(0.3),
    # Hidden Layer 2
    Dense(32, activation='relu', name='hidden_2'),
    BatchNormalization(),
    Dropout(0.2),
    # Hidden Layer 3
    Dense(16, activation='relu', name='hidden_3'),
    # Output Layer – Sigmoid untuk klasifikasi biner
    Dense(1, activation='sigmoid', name='output'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ─── 4. Training ──────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss', patience=15,
    restore_best_weights=True, verbose=1
)

print("\n[INFO] Memulai training...\n")
history = model.fit(
    X_train_sc, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ─── 5. Evaluasi Model ────────────────────────────────────────────────────────
print("\n[INFO] Evaluasi pada data test...")
loss, acc = model.evaluate(X_test_sc, y_test, verbose=0)

y_pred      = (model.predict(X_test_sc) >= 0.5).astype(int).flatten()
precision   = precision_score(y_test, y_pred, zero_division=0)
recall      = recall_score(y_test, y_pred, zero_division=0)
f1          = f1_score(y_test, y_pred, zero_division=0)
cm          = confusion_matrix(y_test, y_pred).tolist()

print(f"\n  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['Tidak CKD','CKD'])}")

# ─── 6. Simpan Metrik ke JSON ─────────────────────────────────────────────────
metrics = {
    "accuracy":         round(float(acc)       * 100, 2),
    "precision":        round(float(precision) * 100, 2),
    "recall":           round(float(recall)    * 100, 2),
    "f1_score":         round(float(f1)        * 100, 2),
    "loss":             round(float(loss), 4),
    "confusion_matrix": cm,           # [[TN,FP],[FN,TP]]
    "epochs_run":       len(history.history['loss']),
    "train_samples":    int(len(X_train)),
    "test_samples":     int(len(X_test)),
    "features":         FEATURE_COLS,
}
with open(METRICS_F, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n[SUKSES] Metrics disimpan: {METRICS_F}")

# ─── 7. Plot Grafik Training ──────────────────────────────────────────────────
plt.style.use('dark_background')

# 7a. Loss Curve
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history.history['loss'],     label='Training Loss',   color='#e05252', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', color='#f97316', linewidth=2, linestyle='--')
ax.set_title('Kurva Loss per Epoch', fontsize=14, fontweight='bold', color='white', pad=12)
ax.set_xlabel('Epoch', color='#a0aec0')
ax.set_ylabel('Loss', color='#a0aec0')
ax.legend(fontsize=10)
ax.tick_params(colors='#a0aec0')
ax.set_facecolor('#161b22')
fig.patch.set_facecolor('#0d1117')
ax.spines[:].set_color('#2d3748')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png'), dpi=120, bbox_inches='tight')
plt.close()

# 7b. Accuracy Curve
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history.history['accuracy'],     label='Training Accuracy',   color='#34d399', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#6ee7b7', linewidth=2, linestyle='--')
ax.set_title('Kurva Akurasi per Epoch', fontsize=14, fontweight='bold', color='white', pad=12)
ax.set_xlabel('Epoch', color='#a0aec0')
ax.set_ylabel('Accuracy', color='#a0aec0')
ax.set_ylim([0, 1])
ax.legend(fontsize=10)
ax.tick_params(colors='#a0aec0')
ax.set_facecolor('#161b22')
fig.patch.set_facecolor('#0d1117')
ax.spines[:].set_color('#2d3748')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_curve.png'), dpi=120, bbox_inches='tight')
plt.close()

print(f"[SUKSES] Plot disimpan: {PLOTS_DIR}")

# ─── 8. Simpan Model & Scaler ─────────────────────────────────────────────────
model.save(MODEL_PATH)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"[SUKSES] Model  : {MODEL_PATH}")
print(f"[SUKSES] Scaler : {SCALER_PATH}")
print("\n[SELESAI] Training sukses! Jalankan app.py untuk membuka aplikasi web.\n")