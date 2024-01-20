from keras.datasets import mnist
from autoencoder import Autoencoder

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


if __name__=="__main__":
    x_train, y_train, x_test, y_test =  load_mnist()
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)