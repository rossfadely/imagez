import os
import glob
import time
import warnings
import multiprocessing
import numpy as np
import pyfits as pf

from skimage.transform import downscale_local_mean

def query_sdss_dr10(save_dir, run, rerun, camcol, field):
    """
    Retrieve a calibrated images from sdss for the given parameters.
    """
    nz = 6 - len(run)
    filled_run = '0' * nz + run
    nz = 6 - len(field)
    filled_field = '0' * nz + field
    frames = ['frame-%s-%s-%s-%s.fits.bz2' % (f, filled_run, camcol,
                                              filled_field)
              for f in 'ugriz']
    cmd = 'wget http://data.sdss3.org/sas/dr10/boss/photoObj/frames/'
    cmd += '%s/%s/%s/' % (rerun, run, camcol)
    for i in range(5):
        file_path = save_dir + frames[i]
        if os.path.exists(file_path):
            pass
        else:
            os.system(cmd + frames[i])
            while True:
                time.sleep(10)
                if os.path.exists(file_path):
                    continue
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

def get_images(photfile, datadir, context='dr7'):
    """
    Take photometry file (with list of run, camcol, etc) and get all the SDSS
    images needed.
    """
    f = pf.open(photfile)
    data = f[1].data
    f.close()

    if context == 'dr7':
        fetcher = query_sdss_dr7
    else:
        fetcher = query_sdss_dr10

    message = '#' * 80 + '\n' * 3 + 'Now on %d' + '\n' * 3 + '#' * 80

    for i in range(len(data)):
        if i % 300 == 0:
            print message % i
        fetcher(datadir, data['run'][i], data['rerun'][i], data['camcol'][i],
                data['field'][i])

def get_orig_patch(data, patch_size, row, col):
    """
    Extract a patch from the SDSS calibrated image.
    """
    row = np.round(row)
    col = np.round(col)
    assert patch_size % 2 == 1, 'Patch size should be odd'
    dlt = (patch_size - 1) / 2
    patch = data[row - dlt - 1:row + dlt, col - dlt - 1:col + dlt]
    return patch.ravel()

def block_patchify((data, spec, patch_dir, patch_size, tol, thread)):
    """
    Take a chunk of the data and make patches.
    """

def make_orig_patches(photfile, specfile, img_dir, patch_dir, patch_size=25,
                      tol=1.e-3, start=0, end=None):
    """
    Cut out patches in the images and save to disk.
    """
    os.chdir(img_dir)
    f = pf.open(photfile)
    data = f[1].data
    f.close()

    f = pf.open(specfile)
    spec = f[1].data
    f.close()

    if end == None:
        end = len(data)

    for i in range(start, end):
        if i % 5 == 0:
            print 'Processing patch', i

        # match to spectral info, reject if necessary
        ind = np.where(data['specobjid'][i] == spec['specobjid'])[0]
        if len(ind) > 1:
            ind = ind[0] # some repeat images of same object!!
        specinfo = spec[ind]
        if ((specinfo['specclass'] != 2) | (specinfo['zwarning'] != 0)):
            continue

        run = str(data['run'][i])
        field = str(data['field'][i])
        camcol = str(data['camcol'][i])

        nz = 6 - len(run)
        filled_run = '0' * nz + run
        nz = 4 - len(field)
        filled_field = '0' * nz + field

        filts = 'ugriz'
        frames = ['./%s/fpC-%s-%s-%s.fit.gz' % (run, filled_run, f + camcol,
                                                filled_field) for f in filts]

        patch = np.zeros((patch_size ** 2., 5))
        patch_file = patch_dir + 'orig_%s.fits' % str(data['specobjid'][i])
        if os.path.exists(patch_file):
            continue

        for j in range(5):
            if not os.path.exists(frames[j]):
                print glob.glob('ls -l')
                print os.getcwd()
                print i, j, data[i]
                print frames[j]
                assert False, 'Image frame has not been downloaded.'

            # unpack data
            img_file = frames[j]
            os.system('gunzip %s' % frames[j])

            # read image
            f = pf.open(frames[j][:-3])
            img = f[0].data
            f.close()
            os.system('gzip %s' % frames[j][:-3])
            img -= np.median(img) # smart or not???
            patch[:, j] = get_orig_patch(img, patch_size,
                                         data['rowc_' + filts[j]][i],
                                         data['colc_' + filts[j]][i])

        hdu = pf.PrimaryHDU(patch)
        hdu.writeto(patch_file)

def downsample(patch, factor, save_path=None):
    """
    Downsample the original patch.
    """
    new = downscale_local_mean(patch, (factor, factor))
    if save_path is not None:
        hdu = pf.PrimaryHDU(new)
        hdu.writeto(save_path)
    return new

def gen_feature_file(data_dir, save_path, factor=4, orig_size=25):
    """
    Save all the features into a single file for convenience.
    """
    data_files = glob.glob(data_dir + '*.fits')
    size = (orig_size - 1) / factor + 1
    features = np.zeros((len(data_files), size ** 2 * 5))
    for i in range(len(data_files)):
        f = pf.open(data_files[i])
        tmp = f[0].data
        f.close()
        dwn = np.array([])
        for j in range(5):
            dwn = np.append(dwn, downsample(tmp[:, j].reshape(orig_size, 
                                                              orig_size),
                                             factor))
        features[i, :] = dwn
    hdu = pf.PrimaryHDU(features)
    hdu.writeto(save_path)

if __name__ == '__main__':
    # Run the script
    img_dir = os.environ['IMGZDATA']
    photfile = img_dir + 'dr7photfirstdr724k_rfadely.fit'
    specfile = img_dir + 'dr7spec2nddr724k_rfadely.fit'
    patch_dir = img_dir + 'patches/dr7/orig/'
    img_dir += 'dr7/'

    if False:
        get_images(photfile, img_dir)

    if True:
        warnings.filterwarnings("ignore")
        targfile = 'dr724k_targets'
        make_orig_patches(photfile, specfile, img_dir, patch_dir, start=6000,
                          end=8000)

    if False:
        gen_feature_file(patch_dir, 'f')
