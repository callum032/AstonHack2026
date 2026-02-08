from data_loader import load_data
from preprocess import preprocess_images
from model import build_model

def main():
    X_train, y_train, X_test, y_test = load_data(
        "data/raw/sign_mnist_train.csv",
        "data/raw/sign_mnist_test.csv"
    )

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    model = build_model()

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )

    model.save("models/sign_mnist_cnn.h5")
    print("Model saved to models/sign_mnist_cnn.h5")

if __name__ == "__main__":
    main()
