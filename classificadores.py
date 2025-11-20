import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.flatten()
y_test = y_test.flatten()

def build_lenet():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (5, 5), activation='relu', padding="same"),
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Conv2D(48, (5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



lenet = build_lenet()
lenet.summary()

history_lenet = lenet.fit(x_train, y_train, epochs=15, validation_split=0.2)

def build_minivgg():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

vgg = build_minivgg()
vgg.summary()

history_vgg = vgg.fit(x_train, y_train, epochs=20, validation_split=0.2)

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnetlite():
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)

    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)

    x = residual_block(x, 32)  # mantém 32 filtros aqui também
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

resnet = build_resnetlite()
resnet.summary()

history_resnet = resnet.fit(x_train, y_train, epochs=20, validation_split=0.2)

def avaliar_modelo(model, nome):
    y_pred = np.argmax(model.predict(x_test), axis=1)

    print(f"\n===== {nome} =====")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.show()

avaliar_modelo(lenet, "LeNet")
avaliar_modelo(vgg, "MiniVGGNet")
avaliar_modelo(resnet, "ResNetLite")

