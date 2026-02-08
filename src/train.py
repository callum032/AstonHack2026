from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import load_data
from preprocess import preprocess_images
from model import build_model

def main():
    X_train, y_train, X_test, y_test = load_data(
        "AstonHack2026/data/raw/sign_mnist_train.csv",
        "AstonHack2026/data/raw/sign_mnist_test.csv"
    )

    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    datagen.fit(X_train)

    model = build_model()

    model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=15,
        validation_data=(X_test, y_test)
    )

    model.save("AstonHack2026/models/sign_mnist_cnn.h5")

if __name__ == "__main__":
    main()