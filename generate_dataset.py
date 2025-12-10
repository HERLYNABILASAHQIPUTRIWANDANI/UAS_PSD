# generate_dataset.py
import numpy as np
import os

os.makedirs("data/SyntheticControl", exist_ok=True)

def generate_class_data(class_type, n_samples=60, length=60, noise=0.05):
    X = []
    for _ in range(n_samples):
        if class_type == "Normal":
            ts = np.ones(length) + np.random.normal(0, noise, length)
        elif class_type == "Cyclic":
            ts = np.sin(np.linspace(0, 4*np.pi, length)) + np.random.normal(0, noise, length)
        elif class_type == "Increasing":
            ts = np.linspace(0, 1, length) + np.random.normal(0, noise, length)
        elif class_type == "Decreasing":
            ts = np.linspace(1, 0, length) + np.random.normal(0, noise, length)
        elif class_type == "Upward":
            ts = np.concatenate([np.zeros(length//2), np.ones(length//2)]) + np.random.normal(0, noise, length)
        elif class_type == "Downward":
            ts = np.concatenate([np.ones(length//2), np.zeros(length//2)]) + np.random.normal(0, noise, length)
        elif class_type == "Valley":
            ts = np.concatenate([np.linspace(1,0,length//2), np.linspace(0,1,length//2)]) + np.random.normal(0, noise, length)
        else:
            raise ValueError("Unknown class_type")
        X.append(ts)
    return np.array(X)

classes = ["Normal", "Cyclic", "Increasing", "Decreasing", "Upward", "Downward", "Valley"]

train_rows = []
test_rows = []

for idx, cls in enumerate(classes, 1):
    X = generate_class_data(cls, n_samples=60)
    labels = np.full((60, 1), idx)
    data = np.hstack([labels, X])
    train_rows.append(data[:45])  # 45 sample train
    test_rows.append(data[45:])   # 15 sample test

train_data = np.vstack(train_rows)
test_data = np.vstack(test_rows)

np.savetxt("data/SyntheticControl/SyntheticControl_TRAIN.txt", train_data, fmt="%.4f")
np.savetxt("data/SyntheticControl/SyntheticControl_TEST.txt", test_data, fmt="%.4f")

print("Dataset SyntheticControl 7 kelas berhasil dibuat!")
