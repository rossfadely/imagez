import numpy as np

def process(x, logscale=True, seed=8675309, test_frac=0.2, zm=True, zca=True,
            divide=False, r=2, flatten=False):
    """
    Zero mean and ZCA whiten the data.  Optionally convert to log space
    Finally, split into train/test.
    """
    np.random.seed(seed)
    if divide:
        for i in range(x.shape[1]):
            if i != r:
                x[:, i] /= x[:, r]
    if logscale:
        x = np.log10(x)
    trn, tst, trn_ind, tst_ind = split_data(x, test_frac)
    if zm:
        trn, mean = zero_mean(trn)
        tst, _ = zero_mean(tst, mean)
    if zca:
        trnshp = trn.shape
        tstshp = tst.shape
        newtrnshape = (trnshp[0], trnshp[1] * trnshp[2] * trnshp[3]) 
        newtstshape = (tstshp[0], tstshp[1] * tstshp[2] * tstshp[3]) 
        trn, u, s = zca_whiten(trn.reshape(newtrnshape))
        tst, _, _ = zca_whiten(tst.reshape(newtstshape), u=u, s=s)
        if not flatten:
            trn = trn.reshape(trnshp)
            tst = tst.reshape(tstshp)
    return trn, tst, trn_ind, tst_ind

def split_data(x, test_frac):
    """
    Split the data into train and test.
    """
    N = x.shape[0]
    Ntst = np.round(test_frac * N).astype(np.int)
    ind = np.random.permutation(N)
    trn_ind = ind[Ntst:]
    tst_ind = ind[:Ntst]
    trn = x[trn_ind]
    tst = x[tst_ind]
    return trn, tst, trn_ind, tst_ind

def zero_mean(x, mean=None):
    """
    Return the mean subtracted matrix.
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    return x - np.mean(x, axis=0), mean

def zca_whiten(data, epsilon=1.e-10, K=None, u=None, s=None):
    """
    Return the ZCA whitened matrix. Seems silly to switch input from NxD to
    DxN but seems to be faster in most cases.

    See http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """
    # returned matrix y does not give yyT = I unless true
    assert data.shape[0] > data.shape[1], 'Nsamples should be > Ndim'

    x = data.T # assumes NxD input
    if K is not None:
        raise Exception('Dimensionality reduction not implemented.')
    cov = np.dot(x, x.T) / x.shape[1]
    if u is None:
        u, s, v = np.linalg.svd(cov)
    xrot = np.dot(u.T, x)
    xpca = np.dot(np.diag(1. / np.sqrt(s + epsilon)), xrot)
    xzca = np.dot(u, xpca)
    return xzca.T, u, s

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
