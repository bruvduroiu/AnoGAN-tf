import click

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from anogan.anogan import AnoGAN

@click.command()
@click.option('--display/--no-display', default=False)
def main(display):
    tf.reset_default_graph()
    model = AnoGAN()
    training_result = model.train(epochs=300, print_interval=10)

    model.anomaly_detector()
    generated, test_data = model.train_anomaly_detector(epochs=3, print_interval=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(generated[0,:,0], generated[0,:,1], c='b', alpha=0.7)
    ax.scatter(test_data[0,:,0], test_data[0,:,1], c='g', alpha=0.7)

    if display:
        plt.show()
    else:
        fig.savefig('manifold.png')


if __name__ == '__main__':
    main()