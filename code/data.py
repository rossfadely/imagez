from skimage.transform import AffineTransform, rotate, warp

import os
import glob
import numpy as np
import pyfits as pf

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

def get_orig_patch(data, patch_size, row, col):
    """
    Extract a patch from the SDSS calibrated image.
    """
    row = np.round(row)
    col = np.round(col)
    assert patch_size % 2 == 1, 'Patch size should be odd'
    dlt = (patch_size - 1) / 2
    patch = data[row - dlt:row + dlt + 1, col - dlt:col + dlt + 1]
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

def make_centered_rotated_patches(photfile, patch_dir, img_dir, ind_file, name,
                                  PA_final=45.,
                                  patch_size=25, start=0, end=None,
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

    os.chdir(img_dir)
    for i in inds:
        print i
        patch_file = patch_dir + name + '_%s.fits' % str(data['specobjid'][i])
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
        od = np.zeros((patch_size ** 2., 5))
        for j in range(5):
            # check that image exists
            if not os.path.exists(frames[j]):
                print os.getcwd()
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
                pmx = patch.max()
                rng = pmx - floor
                patch = (patch - floor) / rng
                shift, rotation = None, None
                if do_shift:
                    # find subpixel shift and move to center of pixel
                    dltr = data['rowc_' + filts[j]][i] - flrr - 0.5
                    dltc = data['colc_' + filts[j]][i] - flrc - 0.5
                    shift = -1. * np.array([dltr, dltc])
                    tform = AffineTransform(translation=shift)
                    patch = warp(patch, tform)
                if do_rotation:
                    # rotate by the model angle
                    rotation = -45. - angles[i]
                    patch = rotate(patch, rotation)
                
                # restore the image brighness
                patch = patch * rng + floor
            out_patch_data[:, j] = patch.ravel()

        hdu = pf.PrimaryHDU(out_patch_data)
        hdu.writeto(patch_file)

def generate_zspec_file(photfile, zspec_file, Nsearch=10):
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
        if i % 200 == 0:
            print i
        inds[i] = i
        speczs[i] = photdata['specz'][i]
        speczerrs[i] = photdata['specz_err'][i]
        if i != (N - 1):
            if photdata['objid'][i] == photdata['objid'][i + 1]:
                ind = photdata['objid'][i:i + Nsearch] == photdata['objid'][i]
                matched_zs = photdata['specz'][i:i + Nsearch][ind]
                matched_zvars = photdata['specz_err'][i:i + Nsearch][ind] ** 2.
                ws = 1. / matched_zvars
                norm = ws.sum()
                speczs[i] = np.sum([ws[j] * matched_zs[j]
                                    for j in range(len(ws))])
                speczs[i] /= norm
                speczerrs[i] = 1. / np.sqrt(norm)
                i += len(ws) - 1
        i += 1

    ind = np.where(inds >= 0)[0]
    inds = inds[ind]
    speczs = speczs[ind]
    speczerrs = speczerrs[ind]
    f = open(zspec_file, 'w')
    f.write('# index specz speczerr')
    for i in range(inds.size):
        f.write('%d %0.8f %0.4e\n' % (inds[i], speczs[i], speczerrs[i]))
    f.close()

if __name__ == '__main__':
    # Run the script
    img_dir = os.environ['IMGZDATA']
    photfile = img_dir + 'dr12_main_50k_rfadely.fit'
    patch_dir = img_dir + 'patches/dr12/shifted/'
    data_dir = img_dir + 'patches/dr12/'
    img_dir += 'dr12/'
    zspec_file = img_dir + 'dr12_main_50k_z-ind.txt'

    if False:
        get_images(photfile, img_dir, start=32500, end=35000)

    if False:
        generate_zspec_file(photfile, zspec_file)

    if False:
        s, e = 0, None
        make_centered_rotated_patches(photfile, patch_dir, img_dir, zspec_file,
                                      's_r', start=s, end=e)
