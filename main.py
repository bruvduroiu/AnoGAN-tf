from datetime import datetime

import click
import numpy as np
import tensorflow as tf

from anogan.anogan import AnoGAN
from utils import s3_utils
from utils.gaussian_data import (
    multivariate_normal_sampler,
)

@click.command()
@click.option('--display/--no-display', default=False)
@click.option('--g-epochs', '-e', default=300, help='Num. epochs to train for.')
@click.option('--g-print-interval', '-p', default=10, help='Print output every x epochs.')
@click.option('--a-epochs', default=300, help='Num. epochs to train anomaly detector for.')
@click.option('--a-print-interval', default=10, help='Print output every x epochs.')
@click.option('--outlier/--no-outlier', default=False)
@click.option('--lambda-ano', default=0.1)
@click.option('--s3-bucket', default='dissertation-backups')
@click.option('--s3-path', default='results/anogan')
def main(display, g_epochs, g_print_interval, a_epochs, a_print_interval, outlier, lambda_ano, s3_bucket, s3_path):
    tf.reset_default_graph()

    model = AnoGAN()
    training_result = model.train(epochs=g_epochs, print_interval=g_print_interval)

    model.construct_anomaly_detector(lambda_ano=lambda_ano)
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
    for point in test_data[0]:
        if np.linalg.norm(point-np.mean(generated[0], axis=0)) > 2 * np.std(generated[0]):
            ax.scatter(point[0], point[1], c='r', marker='+', s=100)

    if display:
        plt.show()
    else:
        s3_file_path = '{base}/{timestamp}/{fname}'.format(base=s3_path,
                                                           timestamp=datetime.now().__str__().replace(' ', '_'),
                                                           fname='manifold.png')
        s3_utils.savefig(fig,
                         file_path=s3_file_path,
                         bucket_name=s3_bucket)
        print('Saved manifold at: s3://{bucket_name}/{file_path}'.format(bucket_name=s3_bucket, file_path=s3_file_path))

    # Testing
    sampler = multivariate_normal_sampler(np.array([2.,3.]), np.array([[1.,0.],[0.,1.]]))
    healthy = sampler(1)
    anomalous = sampler(1)
    anomalous[0][0] = [10, 10]

    healthy_score = model.evaluate(healthy)
    anom_score = model.evaluate(anomalous)

    print(healthy_score, anom_score)

if __name__ == '__main__':
    main()