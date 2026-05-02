from flask import Flask, render_template, request
import numpy as np
import pickle
import json
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE, 'ckd_model.h5')
SCALER_PATH = os.path.join(BASE, 'scaler.pkl')
METRICS_F   = os.path.join(BASE, 'static', 'metrics.json')

# ─── Load Model & Scaler ──────────────────────────────────────────────────────
model  = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Urutan fitur HARUS sama persis dengan saat training
FEATURE_NAMES = ['age','bp','sg','al','su','bgr','bu','sc',
                 'sod','pot','hemo','pcv','wc','rc']

FEATURE_LABELS = {
    'age':  'Usia (tahun)',
    'bp':   'Tekanan Darah (mm/Hg)',
    'sg':   'Berat Jenis Urin',
    'al':   'Albumin (0–5)',
    'su':   'Gula (0–5)',
    'bgr':  'Glukosa Darah Acak (mg/dL)',
    'bu':   'Urea Darah (mg/dL)',
    'sc':   'Kreatinin Serum (mg/dL)',
    'sod':  'Natrium (mEq/L)',
    'pot':  'Kalium (mEq/L)',
    'hemo': 'Hemoglobin (g/dL)',
    'pcv':  'Volume Sel Terkemas (%)',
    'wc':   'Sel Darah Putih (cells/cmm)',
    'rc':   'Sel Darah Merah (millions/cmm)',
}


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_values = []
        for name in FEATURE_NAMES:
            val = request.form.get(name, '').strip()
            if val == '':
                raise ValueError(f"Kolom '{FEATURE_LABELS[name]}' tidak boleh kosong.")
            raw_values.append(float(val))

        features_arr    = np.array(raw_values).reshape(1, -1)
        scaled_features = scaler.transform(features_arr)

        prob_raw    = float(model.predict(scaled_features)[0][0])
        probability = round(prob_raw * 100, 2)

        if prob_raw >= 0.5:
            result       = 'TERDETEKSI CKD'
            result_class = 'danger'
            description  = ('Model ANN mendeteksi indikasi Penyakit Ginjal Kronis (CKD) '
                            'berdasarkan data klinis yang dimasukkan. Segera konsultasikan '
                            'dengan dokter spesialis ginjal untuk pemeriksaan lebih lanjut.')
        else:
            result       = 'TIDAK TERDETEKSI CKD'
            result_class = 'success'
            description  = ('Model ANN tidak mendeteksi indikasi Penyakit Ginjal Kronis (CKD). '
                            'Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.')

        return render_template(
            'index.html',
            result       = result,
            result_class = result_class,
            probability  = probability,
            description  = description,
            form_data    = request.form,
        )

    except ValueError as e:
        return render_template('index.html', error=str(e), form_data=request.form)
    except Exception as e:
        return render_template('index.html',
                               error=f'Terjadi kesalahan sistem: {str(e)}',
                               form_data=request.form)


@app.route('/evaluasi')
def evaluasi():
    metrics = None
    error   = None
    if os.path.exists(METRICS_F):
        with open(METRICS_F, 'r') as f:
            metrics = json.load(f)
    else:
        error = ('File metrics.json belum ditemukan. '
                 'Jalankan model_train.py terlebih dahulu untuk melatih model.')
    return render_template('evaluasi.html', metrics=metrics, error=error)


if __name__ == '__main__':
    app.run(debug=True)