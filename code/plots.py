from matplotlib import use; use('Agg')

import os
import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

def gallery_plot(data_file, N=30, seed=8675309):
    """
    Plot N random examples from the data, for visual checking.
    """
    np.random.seed(seed)

    filts = 'ugriz'
    data = {}
    f = pf.open(data_file)
    for i, filt in enumerate(filts):
        data[filt] = f[i].data
    f.close()

    size = np.sqrt(data['u'][0].size) # we are working with square patches
    inds = np.random.permutation(data['u'].shape[0])[:N]

    fs = 3.5
    fig = pl.figure(figsize=(5 * fs, N * fs))
    for i in range(N):
        for j in range(5):
            ax = pl.subplot(N, 5, i * 5 + j + 1)
            img = ax.imshow(data[filts[j]][inds[i]].reshape(size, size),
                            origin='lower', interpolation='nearest')
            pl.colorbar(img, shrink=0.7)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    pl.tight_layout()
    fig.savefig('../plots/' + data_file.split('/')[-1].split('.')[0] + '.png')

def plot_results(specz, results, names, plot_name):
    """
    Plot the results of various regressors as an 'x vs x' plot, top panel is 
    binned box plots.
    """
    N = specz.shape[0]
    buff = 0.05
    lm, rm = 2 * buff,  buff
    Ncols = len(names)
    col_width = (1. - lm - rm - (Ncols - 1) * buff) / Ncols
    b = buff * 1.5
    h = 1. - b - 2 * buff
    f = 0.25
    hs = f * h - buff
    sub_axes = [[lm + i * (buff + col_width), b, col_width, hs]
                for i in range(Ncols)]
    main_axes = [[lm + i * (buff + col_width), b + hs + buff, col_width,
                  (1. - f) * h - buff] for i in range(Ncols)]

    fs = 10
    fig = pl.figure(figsize=(Ncols * fs, fs))
    for i in range(Ncols):
        mn, mx = 0.0, np.sort(specz)[np.round(0.95 * N).astype(np.int)]
        ax = pl.axes(main_axes[i])
        ax.plot(specz, results[i], 'k.', alpha=0.5)
        ax.plot([mn, mx], [mn, mx], 'r', lw=2)
        ax.set_ylabel('Estimated z', fontsize=20)
        ax.set_xticklabels([])
        pl.xlim(mn, mx)
        pl.ylim(mn, mx)

        err = (results[i] - specz)
        # SDSS photoz's have nans!!
        tmp = specz[np.isfinite(err)]
        err = err[np.isfinite(err)]
        ax2 = pl.axes(sub_axes[i])
        ax2.plot(tmp, err, 'k.', alpha=0.5)
        rng = np.sqrt(np.sort(err ** 2.)[np.int(np.round(0.95 * N))])
        ax2.plot([mn, mx], [0., 0.], 'r', lw=2)
        pl.xlim(mn, mx)
        pl.ylim(-rng, rng)
        ax2.set_xlabel('Spec. z', fontsize=20)
        ax2.set_ylabel('Residuals', fontsize=20)
        ax.set_title(names[i] + ', RMSE: %0.2e' % np.sqrt(err ** 2.).sum(),
                     fontsize=20)

    fig.savefig(plot_name)

if __name__ == '__main__':
    
    if False:
        data_file = os.environ['IMGZDATA'] + '/patches/dr12/shifted/'
        data_file += 's_r_5x5.fits'
        gallery_plot(data_file)

    if True:
        N = 8000
        s = np.random.rand(N)
        r = [s + np.random.randn(N) * 0.03, s + np.random.randn(N) * 0.1]
        plot_results(s, r, ['SDSS Photoz', 'Random Forest'],
                     '../plots/foo.png')
