import click

import numpy as np
import tensorflow as tf

from anogan.anogan import AnoGAN

@click.command()
@click.option('--display/--no-display', default=False)
@click.option('--epochs', '-e', default=300, help='Num. epochs to train for.')
@click.option('--print-interval', '-p', default=10, help='Print output every x epochs.')
def main(display, epochs, print_interval):
    tf.reset_default_graph()
    model = AnoGAN()
    training_result = model.train(epochs=epochs, print_interval=print_interval)

    model.anomaly_detector()
    generated, test_data = model.train_anomaly_detector(epochs=3, print_interval=1)
    
    if display:
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
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