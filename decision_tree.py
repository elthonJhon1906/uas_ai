import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('heart.csv')
    df.index = range(1, len(df) + 1)
    print("Dataset berhasil dimuat.")
    print("Ukuran dataset:", df.shape)
    print("\n5 baris pertama dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: File 'heart.csv' tidak ditemukan.")
    print("Pastikan Anda telah mengunduh dataset dan menempatkannya di direktori yang benar.")
    exit()

X = df.drop('target', axis=1) # Fitur
y = df['target']              # Target

print("\nFitur (X) memiliki", X.shape[1], "kolom.")
print("Target (y) memiliki", y.shape[0], "data.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nUkuran data pelatihan (X_train, y_train):", X_train.shape, y_train.shape)
print("Ukuran data pengujian (X_test, y_test):", X_test.shape, y_test.shape)

model = DecisionTreeClassifier(random_state=42)

print("\nMelatih model Decision Tree...")
model.fit(X_train, y_train)
print("Model selesai dilatih.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAkurasi Model: {accuracy:.2f}")
print("\nLaporan Klasifikasi:")
print(report)

plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=X.columns.tolist(),
          class_names=['Tidak Sakit Jantung', 'Sakit Jantung'],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title("Pohon Keputusan untuk Prediksi Penyakit Jantung")
plt.show()

print("\n--- Contoh Prediksi untuk Data Pasien Baru (Dummy) ---")

#contoh inputan diambil dari dataset
new_patient_data = pd.DataFrame([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]],
                                  columns=X.columns)

prediction = model.predict(new_patient_data)
if prediction[0] == 1:
    print("Pasien ini diprediksi memiliki risiko PENYAKIT JANTUNG.")
else:
    print("Pasien ini diprediksi TIDAK memiliki risiko penyakit jantung.")