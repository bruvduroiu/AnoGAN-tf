import click

import numpy as np
import tensorflow as tf

from anogan.anogan import AnoGAN

@click.command()
@click.option('--display/--no-display', default=False)
@click.option('--g-epochs', '-e', default=300, help='Num. epochs to train for.')
@click.option('--g-print-interval', '-p', default=10, help='Print output every x epochs.')
@click.option('--a-epochs', default=300, help='Num. epochs to train anomaly detector for.')
@click.option('--a-print-interval', default=10, help='Print output every x epochs.')
@click.option('--outlier/--no-outlier', default=False)
def main(display, g_epochs, g_print_interval, a_epochs, a_print_interval, outlier):
    tf.reset_default_graph()
    model = AnoGAN()
    training_result = model.train(epochs=g_epochs, print_interval=g_print_interval)

    model.anomaly_detector()
    generated, test_data = model.train_anomaly_detector(epochs=a_epochs, print_interval=a_print_interval, outlier=outlier)
    
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