import numpy as np

def preprocess_images(X):
    X = X.reshape(-1, 28, 28, 1)
    X = X.astype("float32") / 255.0
    return X

def label_to_letter(label):
    return chr(label + 65)
