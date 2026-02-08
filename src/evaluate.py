from tensorflow.keras.models import load_model
from data_loader import load_data
from preprocess import preprocess_images
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def main():
    model = load_model("models/sign_mnist_cnn.h5")

    _, _, X_test, y_test = load_data(
        "data/raw/sign_mnist_train.csv",
        "data/raw/sign_mnist_test.csv"
    )

    X_test = preprocess_images(X_test)

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
