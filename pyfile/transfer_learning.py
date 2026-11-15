import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
base_model = VGG16(weights='imagenet', include_top=False,
input_shape=(32, 32, 3))
for layer in base_model. layers:
    layer.trainable = False
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train,
epochs=5,
batch_size=64,
validation_data=(x_test, y_test))
for layer in base_model. layers[-4:]: # Last 4 conv Layers
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers. Adam(1e-5), # Lower LR
loss='categorical_crossentropy',
metrics=['accuracy'])
history_fine = model.fit(x_train, y_train,
epochs=3,
batch_size=64,
validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy after fine-tuning:", test_acc)
