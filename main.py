import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from anogan.anogan import AnoGAN

if __name__ == '__main__':
    tf.reset_default_graph()
    model = AnoGAN()
    training_result = model.train(epochs=300, print_interval=10)

    model.anomaly_detector()
    errors = model.train_anomaly_detector(epochs=3, print_interval=1)

    plt.scatter(errors[0,:,0], errors[0,:,1])
    plt.show()
