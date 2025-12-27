import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     Add, GlobalAveragePooling2D, Dense, Dropout,
                                     MaxPooling2D, Input)
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

train_path = "/content/drive/MyDrive/Cataract dataset/train"
test_path  = "/content/drive/MyDrive/Cataract dataset/test"

print("Train folders:", os.listdir(train_path))
print("Test folders:", os.listdir(test_path))

train = keras.utils.image_dataset_from_directory(
    directory=train_path,
    labels="inferred",
    label_mode="int",
    batch_size=10,
    image_size=(256,256)
)

validation = keras.utils.image_dataset_from_directory(
    directory=test_path,
    labels="inferred",
    label_mode="int",
    batch_size=10,
    image_size=(256,256)
)

class_names = train.class_names
print("Classes:", class_names)

plt.figure(figsize=(10, 8))
for images, labels in train.take(1):
    for i in range(len(class_names)):
        plt.subplot(1, len(class_names), i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    break
plt.show()

def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

train = train.map(process)
validation = validation.map(process)

def residual_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding="same")(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x

def build_mini_resnet(input_shape=(256,256,3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = residual_block(x, 32)
    x = MaxPooling2D((2,2))(x)

    x = residual_block(x, 64)
    x = MaxPooling2D((2,2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D((2,2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D((2,2))(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model

model = build_mini_resnet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train,
    epochs=20,
    validation_data=validation
)

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

model.save("/content/drive/MyDrive/mini_resnet_model.h5")
print("Model saved!")

test_loss, test_acc = model.evaluate(validation)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

y_true = []
y_pred = []

for images, labels in validation:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
accuracy = correct / len(y_true)

print("\nManual Correct Predictions:", correct)
print("Total Images:", len(y_true))
print("Manual Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256,256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Cataract" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else (1 - pred)
    confidence = round(float(confidence) * 100, 2)

    print("\nImage:", img_path)
    print("Prediction:", label)
    print("Confidence:", confidence, "%")
    return label, confidence