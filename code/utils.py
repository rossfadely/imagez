import numpy as np

def zero_mean(x):
    """
    Return the mean subtracted matrix.
    """
    return x - np.mean(x, axis=0)

def zca_whiten(data, epsilon=1.e-10, K=None):
    """
    Return the ZCA whitening matrix. Seems silly to switch input from NxD to
    DxN but seems to be faster in most cases.

    See http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    # returned matrix y does not give yyT = I unless true
    assert data.shape[0] > data.shape[1], 'Nsamples should be > Ndim'

    x = data.T # assumes NxD input
    if K is not None:
        raise Exception('Dimensionality reduction not implemented.')
    cov = np.dot(x, x.T) / x.shape[1]
    u, s, v = np.linalg.svd(cov)
    xrot = np.dot(u.T, x)
    xpca = np.dot(np.diag(1. / np.sqrt(s + epsilon)), xrot)
    xzca = np.dot(u, xpca)
    return xzca.T

if __name__ == '__main__':

    # Test ZCA
    if True:
        import os, glob
        import pyfits as pf 

        datadir = os.environ['IMGZDATA']
        patchdir = datadir + 'patches/dr7/orig/'
        os.chdir(patchdir)
        files = glob.glob('*fits')

        N = 700
        data = np.zeros((N, 625))
        for i in range(N):
            f = pf.open(files[i])
            data[i] = f[0].data[:, 0].flatten()
            f.close()

        data = zero_mean(data[:, :10])
        data_whitened = zca_whiten(data, epsilon=0.)
        print 'STD along dim axis:\n', np.std(data_whitened, axis=0)
        new_cov = np.dot(data_whitened.T, data_whitened) / N
        print 'Element of Cov diagonal:\n', np.diag(new_cov)
