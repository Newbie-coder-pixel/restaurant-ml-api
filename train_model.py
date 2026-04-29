"""
Prep Time Prediction - Generate Synthetic Data & Train Model
Fitur: total_quantity, total_item_types, weighted_prep_time,
       special_request_count, hour_of_day, queue_length
Target: actual_prep_minutes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ── 1. Data menu asli dari Supabase ──────────────────────────────────────────
MENU_ITEMS = [
    {"name": "Puding Coklat",       "prep_time": 2},
    {"name": "Es Krim Vanilla",     "prep_time": 3},
    {"name": "Es Teh Manis",        "prep_time": 3},
    {"name": "Es Jeruk",            "prep_time": 4},
    {"name": "Kopi Hitam",          "prep_time": 5},
    {"name": "Klepon",              "prep_time": 5},
    {"name": "Jus Alpukat",         "prep_time": 5},
    {"name": "Milkshake Coklat",    "prep_time": 7},
    {"name": "Gado-Gado",           "prep_time": 8},
    {"name": "Kentang Goreng",      "prep_time": 8},
    {"name": "Nasi Goreng Spesial", "prep_time": 10},
    {"name": "Pisang Goreng Keju",  "prep_time": 10},
    {"name": "Lumpia Goreng",       "prep_time": 12},
    {"name": "Mie Ayam Bakso",      "prep_time": 12},
    {"name": "Soto Ayam",           "prep_time": 15},
    {"name": "Ayam Bakar",          "prep_time": 20},
]

# ── 2. Generate Synthetic Training Data ──────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 2000

records = []
for _ in range(N_SAMPLES):
    # Simulasi jam ramai (jam makan siang & malam lebih ramai)
    hour = np.random.choice(
        list(range(10, 22)),
        p=[0.04, 0.08, 0.12, 0.12, 0.06, 0.04, 0.08, 0.12, 0.12, 0.10, 0.08, 0.04]
    )

    # Jumlah jenis menu yang dipesan (1-5 jenis)
    n_item_types = np.random.randint(1, 6)

    # Pilih menu secara random
    selected = np.random.choice(MENU_ITEMS, size=n_item_types, replace=False)

    # Quantity tiap menu (1-4 porsi)
    quantities = np.random.randint(1, 5, size=n_item_types)
    total_quantity = int(quantities.sum())

    # Weighted prep time = sum(prep_time * qty)
    weighted_prep = sum(
        m["prep_time"] * q for m, q in zip(selected, quantities)
    )

    # Special request (30% chance ada special request)
    special_request_count = int(np.random.binomial(n_item_types, 0.3))

    # Panjang antrian dapur — lebih ramai saat jam peak
    is_peak = hour in [12, 13, 18, 19, 20]
    queue_length = np.random.randint(3, 10) if is_peak else np.random.randint(0, 5)

    # ── Hitung actual_prep_minutes dengan logika realistis ──
    # Base: ambil prep_time menu terlama (paralel cooking)
    base_time = max(m["prep_time"] for m in selected)

    # Tambahan karena quantity banyak
    qty_factor = total_quantity * 0.5

    # Tambahan karena special request
    special_factor = special_request_count * 1.5

    # Tambahan karena antrian dapur
    queue_factor = queue_length * 0.8

    # Tambahan saat jam ramai
    peak_factor = 3.0 if is_peak else 0.0

    # Noise realistis
    noise = np.random.normal(0, 1.5)

    actual_prep = base_time + qty_factor + special_factor + queue_factor + peak_factor + noise
    actual_prep = max(2.0, round(actual_prep, 1))  # minimal 2 menit

    records.append({
        "total_quantity":       total_quantity,
        "total_item_types":     n_item_types,
        "weighted_prep_time":   weighted_prep,
        "special_request_count": special_request_count,
        "hour_of_day":          hour,
        "queue_length":         queue_length,
        "actual_prep_minutes":  actual_prep,
    })

df = pd.DataFrame(records)
print(f"✅ Generated {len(df)} training samples")
print(f"   Rata-rata prep time: {df['actual_prep_minutes'].mean():.1f} menit")
print(f"   Min: {df['actual_prep_minutes'].min()} | Max: {df['actual_prep_minutes'].max()}")
print()

# ── 3. Train Model ────────────────────────────────────────────────────────────
FEATURES = [
    "total_quantity",
    "total_item_types",
    "weighted_prep_time",
    "special_request_count",
    "hour_of_day",
    "queue_length",
]

X = df[FEATURES]
y = df["actual_prep_minutes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    random_state=42,
)
model.fit(X_train, y_train)

# ── 4. Evaluasi ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"📊 Evaluasi Model:")
print(f"   MAE  : {mae:.2f} menit (rata-rata selisih prediksi vs aktual)")
print(f"   R²   : {r2:.4f} ({r2*100:.1f}% variance explained)")
print()

# ── 5. Feature Importance ────────────────────────────────────────────────────
print("🔍 Feature Importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"   {feat:<25} {bar} {imp:.4f}")
print()

# ── 6. Simpan Model ───────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/prep_time_model.pkl")
joblib.dump(FEATURES, "model/features.pkl")
print("💾 Model disimpan ke /home/claude/ml/model/")

# ── 7. Test Prediksi Manual ──────────────────────────────────────────────────
print("\n🧪 Test Prediksi:")
test_cases = [
    {"total_quantity": 2, "total_item_types": 1, "weighted_prep_time": 6,  "special_request_count": 0, "hour_of_day": 10, "queue_length": 1},
    {"total_quantity": 5, "total_item_types": 3, "weighted_prep_time": 35, "special_request_count": 1, "hour_of_day": 13, "queue_length": 7},
    {"total_quantity": 8, "total_item_types": 4, "weighted_prep_time": 60, "special_request_count": 2, "hour_of_day": 19, "queue_length": 9},
]
labels = ["Order ringan (1 menu, 2 qty)", "Order sedang (3 menu, jam makan siang)", "Order berat (4 menu, jam makan malam)"]

for label, tc in zip(labels, test_cases):
    pred = model.predict(pd.DataFrame([tc]))[0]
    print(f"   {label}")
    print(f"   → Estimasi: {pred:.1f} menit\n")