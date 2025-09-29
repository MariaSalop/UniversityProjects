from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_model(input_shape=(100,), num_classes=2):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(
        Dense(num_classes, activation="softmax")
    )  # Use 'sigmoid' for binary classification

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=(
            "categorical_crossentropy"
            if num_classes > 2
            else "binary_crossentropy"
        ),
        metrics=["accuracy"],
    )
    return model
