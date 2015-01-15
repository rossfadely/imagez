import os
import time
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
    nz = 6 - len(run)
    filled_run = '0' * nz + run
    nz = 4 - len(field)
    filled_field = '0' * nz + field
    frames = ['fpC-%s-%s-%s.fit.gz' % (filled_run, f + camcol,
                                       filled_field)
              for f in 'ugriz']
    cmd = 'wget http://das.sdss.org/imaging/'
    cmd += '%s/%s/corr/%s/' % (run, rerun, camcol)
    for i in range(5):
        file_path = save_dir + frames[i]
        if os.path.exists(file_path):
            pass
        else:
            os.system(cmd + frames[i])
            while True:
                time.sleep(5)
                if os.path.exists(file_path):
                    break
    return frames

def get_orig_patch(frame_path, patch_path, patch_size, row, col):
    """
    Extract a patch from the SDSS calibrated image.
    """
    if os.path.exists(patch_path):
        f = pf.open(patch_path)
        patch = f[0].data
        f.close()
    
    else:
        f = pf.open(frame_path)
        data = f[0].data
        f.close()

        assert patch_size % 2 == 1, 'Patch size should be odd'
        dlt = (patch_size - 1) / 2
        patch = data[row - dlt - 1:row + dlt, col - dlt - 1:col + dlt]

        hdu = pf.PrimaryHDU(patch)
        hdu.writeto(patch_path)
        
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

if __name__ == '__main__':
    run = '1239'
    rerun = '40'
    field = '176'
    camcol = '3'
    
    query_sdss_dr7('~/imagez/code/', run, rerun, camcol, field)
