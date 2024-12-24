import tensorflow as tf
import numpy as np
  
#  mnist = tf.keras.datasets.fashion_mnist
#  (x_train, y_train), (x_test, y_test) = mnist.load_data()
#  x_train, x_test = x_train / 255.0, x_test / 255.0
#  
#  model = tf.keras.models.Sequential([
    #  tf.keras.layers.Flatten(input_shape=(28, 28)),
    #  tf.keras.layers.Dense(512, activation=tf.nn.relu),
    #  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#  ])
#  
#  model.compile(optimizer=tf.optimizers.Adam(),
              #  loss='sparse_categorical_crossentropy',
              #  metrics=['accuracy'])
#  
#  model.fit(x_train, y_train, epochs=10)
#  
#  model.save('fashion_mnist_model.h5')

model = tf.keras.models.load_model('fashion_mnist_model.h5')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255.0

prediction = model.predict(x_test)

predicted_classes = np.argmax(prediction, axis=1)

for i in range(5):
    print(f"Image {i}: Predicted class: {predicted_classes[i]}, True class: {y_test[i]}")
