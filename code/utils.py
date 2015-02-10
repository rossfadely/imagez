import numpy as np

def zero_mean(x):
    """
    Return the mean subtracted matrix.
    """
    return x - np.mean(x, axis=0)

def zca_whiten(x, epsilon=1.e-10):
    """
    Return the ZCA whitening matrix.
    """
    assert x.shape[0] >= x.shape[1], 'Not enough samples.'
    cov = np.dot(x, x.T) / (data.shape[0])
    u, s, v = np.linalg.svd(cov)
    evl, evc = s / (x.shape[0]), v.T
    pt1 = np.dot(np.diag(1. / np.sqrt(evl + epsilon)), np.dot(evc.T, x))
    xwzca = np.dot(evc, pt1) / np.sqrt(x.shape[0])
    return xwzca

if __name__ == '__main__':

    # Test ZCA
    if True:
        import os, glob
        import pyfits as pf 

        datadir = os.environ['IMGZDATA']
        patchdir = datadir + 'patches/dr7/orig/'
        os.chdir(patchdir)
        files = glob.glob('*fits')

        N = 625
        data = np.zeros((N, 625))
        for i in range(N):
            f = pf.open(files[i])
            data[i] = f[0].data[:, 0].flatten()
            f.close()

        data = zero_mean(data[:, :10])
        data_whitened = zca_whiten(data)
        print np.std(data_whitened, axis=0)
        new_cov = np.dot(data_whitened.T, data_whitened) / N
        print np.diag(new_cov)
