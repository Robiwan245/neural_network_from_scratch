import tensorflow as tf
import matplotlib.pyplot as plt

def load_data():
    return tf.keras.datasets.mnist.load_data(path="mnist.npz")

def show_data_example(x_train):
    plt.figure(figsize=(18, 18))

    for idx in range(16):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(x_train[idx])

    plt.show()


(x_train, y_train), (x_test, y_test) = load_data()
# show_data_example(x_train)
