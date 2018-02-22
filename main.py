import numpy as np

from anogan.anogan import AnoGAN

if __name__ == '__main__':
    model = AnoGAN()

    eval = model.train(epochs=300)

    print(eval)
    print(np.mean(eval), np.std(eval))
