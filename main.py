import sys
import os
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


import numpy as np


def categorize_age(age):
    if age <= 12:
        return 'child'
    elif age <= 25:
        return 'young'
    elif age <= 50:
        return 'adult'
    else:
        return 'senior'


def load_dataset(path):
    dataset = []

    for filename in os.listdir(path):
        age, gender = filename.split('.')[0].split('_')[:2]

        age = int(filename.split('_')[0])
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))

        dataset.append({
            "image": img,
            "age": int(age),
            "gender": "male" if gender == '0' else 'female',
        })

    return dataset


if __name__ == '__main__':
    path = sys.argv[1]

    dataset = load_dataset(path)

    df = pd.DataFrame({'age': [x["age"] for x in dataset]})
    df['age_group'] = df['age'].apply(categorize_age)

    x = np.array([x["image"] for x in dataset]) / 255.0
    y = pd.get_dummies(df['age_group']).values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y)

    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Congelar capas base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Evolución de la precisión')
    plt.show()

    # Ejemplo de predicción
    pred = model.predict(x_test[:5])
