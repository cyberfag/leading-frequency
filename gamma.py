import neo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from scipy.signal import welch
import os

def load_data(fnames=[]):
    n1s = []
    for fname in fnames:
        if os.path.exists(fname[:-4] + '.npy'):
            n1 = np.load(fname[:-4] + '.npy')
            print('from .npy ', n1.shape)
        else:
            segm = neo.PlexonIO(fname).read_segment()
            n1 = segm.analogsignals[0][::10]
            print('resampled from .plx ', n1.shape)
            np.save(fname[:-4], n1)
        n1s.append(n1)
    n1 = np.vstack(tuple(n1s))
    print('vstacked ', n1.shape)
    print('division residual ', n1.shape[0] % 60000)
    n1 = n1[:-1 * (n1.shape[0] % 60000)]
    print('trimmed ', n1.shape)
    n1 = np.array(n1) * 1000
    chunks = np.empty((60000, 0))
    for i in range(n1.shape[0] // 60000):
        chunks = np.hstack((chunks, n1[(60000 * i):(60000 * (i + 1))]))
    print('chunks ', chunks.shape)
    return chunks

def calc_lf(chunks, fmin=55, fmax=170):
    fs, specs = welch(chunks, fs=1000., axis=0, nperseg=1024)
    print('freqs and specs', fs.shape, specs.shape)
    freqs, gamma = fs[(fs > fmin) & (fs < fmax)], specs[(fs > fmin) & (fs < fmax)]
    print('gamma freqs and specs', freqs.shape, gamma.shape)
    lfs = gamma.argmax(axis=0)
    print('max idxs', lfs.shape)
    return np.array([freqs[i] for i in lfs])

def plot_lr(dots, trash=0, calc_lr=True, lr_to=None):
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.Figure()
    ax = plt.subplot(111)
    ax.scatter(np.arange(len(dots) - trash), dots[trash:])
    if calc_lr:
        if lr_to is not None:
            reg = lr().fit(np.arange(len(dots) - trash - lr_to)[:, np.newaxis], dots[trash:-1 * lr_to][:, np.newaxis])
        else:
            reg = lr().fit(np.arange(len(dots) - trash)[:, np.newaxis], dots[trash:][:, np.newaxis])
        print('lr results (R^2, slope, intercept) ',
              reg.score(np.arange(len(dots) - trash)[:, np.newaxis], dots[trash:][:, np.newaxis]),
              reg.coef_, reg.intercept_)
        k, b = reg.coef_[0][0], reg.intercept_[0]
        model = k * np.arange(len(dots) - trash) + b
        ax.plot(model, c='r', lw=1)
        ax.legend(['LF = {:.2f} * time + {:.2f}'.format(k, b), 'LFs'], fontsize=15)
    ax.set_ylabel('Leading frequency (LF), Hz', fontsize=15)
    ax.set_xlabel('Time sice anesthesia administration, mins', fontsize=15)
    plt.show()