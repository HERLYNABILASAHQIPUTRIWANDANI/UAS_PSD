import pickle
import numpy as np

def load_svm_model():
    return pickle.load(open("models/svm_model.pkl", "rb"))

def load_scaler():
    return pickle.load(open("models/scaler.pkl", "rb"))

def preprocess(data, scaler):
    data = np.array(data).reshape(1, -1)
    return scaler.transform(data)
