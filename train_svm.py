# train_svm.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)

train_path = "data/SyntheticControl/SyntheticControl_TRAIN.txt"
test_path = "data/SyntheticControl/SyntheticControl_TEST.txt"

train = pd.read_csv(train_path, sep=r"\s+", header=None)
test = pd.read_csv(test_path, sep=r"\s+", header=None)

X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0].astype(int)

X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0].astype(int)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM 7 kelas
model = SVC(
    kernel="rbf",
    C=10,
    gamma=0.01,
    probability=True,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

# Evaluasi
pred = model.predict(X_test_scaled)
print(classification_report(y_test, pred))

# Simpan model & scaler
pickle.dump(model, open("models/svm_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Training selesai! Model tersimpan.")
