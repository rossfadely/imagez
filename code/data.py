from matplotlib import use; use('Agg')
from skimage.transform import downscale_local_mean, rotate

import os
import glob
import time
import warnings
import multiprocessing
import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl


def query_sdss(save_dir, run, rerun, camcol, field, context='dr12'):
    """
    Retrieve a calibrated images from sdss for the given parameters.
    """
    nz = 6 - len(run)
    filled_run = '0' * nz + run
    nz = 4 - len(field)
    filled_field = '0' * nz + field
    frames = ['frame-%s-%s-%s-%s.fits.bz2' % (f, filled_run, camcol,
                                              filled_field)
              for f in 'ugriz']

    cwd = os.getcwd()
    save_dir += run + '/'
    try:
        os.mkdir(save_dir)
    except:
        pass
    os.chdir(save_dir)

    cmd = 'wget http://data.sdss3.org/sas/%s/boss/photoObj/frames/' % context
    cmd += '%s/%s/%s/' % (rerun, run, camcol)
    for i in range(5):
        file_path = save_dir + frames[i]
        if os.path.exists(file_path):
            pass
        else:
            os.system(cmd + frames[i])
            while True:
                time.sleep(2)
                if os.path.exists(file_path):
                    break

    os.chdir(cwd)
    return frames

def query_sdss_dr7(save_dir, run, rerun, camcol, field):
    """
    Retrieve a calibrated images from sdss for the given parameters.
    """
    run = str(run)
    field = str(field)
    rerun = str(rerun)
    camcol = str(camcol)

    cwd = os.getcwd()
    save_dir += run + '/'
    try:
        os.mkdir(save_dir)
    except:
        pass
    os.chdir(save_dir)

    nz = 6 - len(run)
    filled_run = '0' * nz + run
    nz = 4 - len(field)
    filled_field = '0' * nz + field
    frames = ['fpC-%s-%s-%s.fit.gz' % (filled_run, f + camcol,
                                       filled_field) for f in 'ugriz']
    cmd = 'wget http://das.sdss.org/imaging/'
    cmd += '%s/%s/corr/%s/' % (run, rerun, camcol)
    for i in range(5):
        file_path = save_dir + frames[i]
        if os.path.exists(file_path):
            pass
        else:
            os.system(cmd + frames[i])
            while True:
                time.sleep(2)
                if os.path.exists(file_path):
                    break
    os.chdir(cwd)
    return frames

def get_images(photfile, datadir, context='dr12', start=None, end=None):
    """
    Take photometry file (with list of run, camcol, etc) and get all the SDSS
    images needed.
    """
    f = pf.open(photfile)
    data = f[1].data
    f.close()

    if start is not None:
        if end is None:
            end = -1
        data = data[start:end]

    if context == 'dr7':
        fetcher = query_sdss_dr7
    else:
        fetcher = query_sdss

    message = '#' * 80 + '\n' * 3 + 'Now on %d' + '\n' * 3 + '#' * 80

    for i in range(len(data)):
        if i % 300 == 0:
            print message % i
        fetcher(datadir, str(data['run'][i]), str(data['rerun'][i]),
                str(data['camcol'][i]), str(data['field'][i]))

def make_img_list(photfile, data_dir, listfile, context='dr7'):
    """
    Make a list of the images need to be downloaded from SDSS.
    """
    f = pf.open(photfile)
    data = f[1].data
    f.close()

    if context == 'dr7':
        frame = data_dir + '%s/fpC-%s-%s-%s.fit.gz\n'
    else:
        assert False, 'need to specify format for other contexts'

    uniques = []
    f = open(listfile, 'w')
    for i in range(len(data)):
        if i % 500 == 0:
            print iscr
        run = data['run'][i].astype(np.str)
        rerun = data['rerun'][i].astype(np.str)
        field = data['field'][i].astype(np.str)
        camcol = data['camcol'][i].astype(np.str)
        nz = 6 - len(run)
        filled_run = '0' * nz + str(run)
        nz = 4 - len(field)
        filled_field = '0' * nz + str(field)
        for filt in 'ugriz':
            line = frame % (run, filled_run, filt + camcol, filled_field)
            if line in uniques:
                pass
            else:
                f.write(line)
                uniques.append(line)
    f.close()

def get_orig_patch(data, patch_size, row, col):
    """
    Extract a patch from the SDSS calibrated image.
    """
    row = np.round(row)
    col = np.round(col)
    assert patch_size % 2 == 1, 'Patch size should be odd'
    dlt = (patch_size - 1) / 2
    patch = data[row - dlt - 1:row + dlt, col - dlt - 1:col + dlt]
    return patch

def downsample(patch, factor, save_path=None):
    """
    Downsample the original patch.
    """
    new = downscale_local_mean(patch, (factor, factor))
    if save_path is not None:
        hdu = pf.PrimaryHDU(new)
        hdu.writeto(save_path)
    return new

def make_centered_rotated_patches(photfile, patch_dir, ind_file, PA_final=45.,
                                  pshape=(25, 25), start=0, end=None,
                                  do_shift=True, do_rotation=True,
                                  floor=1.e-5):
    """
    Patchify original image, use skimage to shift to common center, rotate to
    common angle, and save patch.
    """
    assert end != -1, 'Use actual end, not -1'
    f = pf.open(photfile)
    data = f[1].data
    f.close()

    # only gri get to vote on rotation angle
    angles = data['fracdev_g'] * data['devphi_g']
    angles += (1. - data['fracdev_g']) * data['expphi_g']
    for f in 'ri':
        angles += data['fracdev_' + f] * data['devphi_' + f]
        angles += (1. - data['fracdev_' + f]) * data['expphi_' + f]
    angles /= 3.

    # read spec z file with indicies in first column
    info = np.loadtxt(ind_file)
    if end is None:
        end = len(info[:, 0])
    inds = info[start:end, 0].astype(np.int)

    for i in inds:
        patch_file = patch_dir + 'orig_%s.fits' % str(data['specobjid'][i])
        if os.path.exists(patch_file):
            continue

        run = str(data['run'][i])
        field = str(data['field'][i])
        camcol = str(data['camcol'][i])

        nz = 6 - len(run)
        filled_run = '0' * nz + run
        nz = 4 - len(field)
        filled_field = '0' * nz + field

        filts = 'ugriz'
        frames = ['./%s/frame-%s-%s-%s-%s.fits.bz2' % (run, f, filled_run,
                                                       camcol, filled_field)
                  for f in 'ugriz']

        out_patch_data = np.zeros((patch_size ** 2., 5))
        for j in range(5):
            # check that image exists
            if not os.path.exists(frames[j]):
                print frames[j]
                assert False, 'Image frame has not been downloaded.'

            # unpack data and read image
            img_file = frames[j]
            os.system('bzip2 -d %s' % frames[j])
            f = pf.open(frames[j][:-4])
            os.system('bzip2 %s' % frames[j][:-4])
            img = f[0].data
            f.close()

            # floor the row and col centers
            flrr = np.floor(data['rowc_' + filts[j]][i])
            flrc = np.floor(data['colc_' + filts[j]][i])

            # get patch centered on the floored centers
            patch = get_orig_patch(img, patch_size, flrr, flrc)
            if floor is not None:
                patch = np.maximum(floor, patch)

            if do_shift | do_rotation:
                pmn, pmx = patch.min(), patch.max()
                rng = pmx - pmn
                patch = (patch - pmn) / rng
                shift, rotation = None
                if do_shift:
                    # find subpixel shift
                    dltr = data['rowc_' + filts[j]][i] - flrr - 0.5
                    dltc = data['colc_' + filts[j]][i] - flrc - 0.5
                    shift = -1. * np.array([dltr, dltc])
                if do_rotation:
                    rotation = np.deg2rad(45. - angles[i])
                tform = AffineTransform(rotation=rotation, translation=shift)
                patch = warp(patch, tform)
                patch = patch * rng + pmn

            assert False, 'Check the above is cool.'

def generate_zspec_file(photfile, zspec_file):
    """
    Run through the photo data and generate a list of spec z's and indicies on
    the file.  Necessary since same obj has multiple spectra in a bunch of
    cases.
    """
    f = pf.open(photfile)
    photdata = f[1].data
    f.close()
    N = len(photdata)

    i = 0
    inds = np.zeros(N, dtype=np.int) - 1
    speczs = np.zeros(N)
    speczerrs = np.zeros(N)
    while i < N:
        inds[i] = i
        speczs[i] = photdata['specz'][i]
        speczerrs[i] = photdata['specz_err'][i]
        if i != (N - 1):
            if photdata['objid'][i] == photdata['objid'][i + 1]:
                if photdata['objid'][i] == photdata['objid'][i + 2]:
                    print i
                    assert 0
                w = 1. / photdata['specz_err'][i:i + 2] ** 2.
                norm = w.sum()
                speczs[i] = photdata['specz'][i] * w[0]
                speczs[i] += photdata['specz'][i + 1] * w[1]
                speczs[i] /= norm
                speczerrs[i] = 1. / np.sqrt(norm)
                i += 1
        i += 1

    ind = np.where(inds >= 0)[0]
    print ind.shape

if __name__ == '__main__':
    # Run the script
    img_dir = os.environ['IMGZDATA']
    photfile = img_dir + 'dr12_main_50k_rfadely.fit'
    patch_dir = img_dir + 'patches/dr12/orig/'
    data_dir = img_dir + 'patches/dr12/'
    img_dir += 'dr12/'

    if False:
        get_images(photfile, img_dir, start=32500, end=35000)

    if True:
        zspec_file = img_dir + 'dr12_main_50k_z-ind.txt'
        generate_zspec_file(photfile, zspec_file)

    if False:
        make_rotated_patches(photfile, patch_dir)

