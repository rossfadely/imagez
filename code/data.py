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
    return patch.ravel()

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

def gen_final_files(data_dir, save_base, photfile, specfile, factor=4,
                    orig_size=25, divide_out='avg', include=['psf']):
    """
    Take the original patches, apply any transformations, and save to 
    single file.  Also save redshifts.
    """
    target_path = save_base + '_z.txt'
    feature_path = save_base + '_features.fits'
    patch_paths = glob.glob(data_dir + '*fits')
    
    # Need photfile for extinction, etc.
    f = pf.open(photfile)
    photdata = f[1].data
    f.close()
    photdata = photdata[np.argsort(photdata['specobjid'])]

    # Need specinfo
    f = pf.open(specfile)
    specdata = f[1].data
    f.close()
    specdata = specdata[np.argsort(specdata['specobjid'])]

    # tranform one patch to get shape
    f = pf.open(patch_paths[0])
    d = f[0].data[:, 0].reshape(orig_size, orig_size)
    f.close()
    d = downsample(d, factor)
    patch_size = d.size

    Ndim = patch_size * 5
    if 'psf' in include:
        Ndim += 5
    Nsamples = len(patch_paths)
    patch_data = np.zeros((Nsamples, Ndim))

    targ_line = ''
    for i in range(Nsamples):
        if i % 100 == 0:
            print 'working on', i
        
        f = pf.open(patch_paths[i])
        d = f[0].data
        f.close()

        specobjid = np.int(patch_paths[i].split('_')[-1][:-5])
        ind = np.searchsorted(photdata['specobjid'], specobjid)

        includes = -1
        for j, f in enumerate('rugiz'):
            patch = d[:, j].reshape(orig_size, orig_size)
            patch *= photdata[ind]['extinction_' + f]
            patch = downsample(patch, factor).ravel()
            if j == 0: # always keep original r patch
                norm_patch = patch 
            elif divide_out == 'patch':
                patch /= norm_patch
            elif divide_out == 'avg':
                patch /= np.maximum(np.mean(norm_patch), 1.)
            patch_data[i, patch_size * j: patch_size * (j + 1)] = patch

            if 'psf' in include:
                name = 'mrrccpsf_' + f
                patch_data[i, includes] = np.sqrt(photdata[ind][name])
                includes -= 1

        ind = np.searchsorted(specdata['specobjid'], specobjid)
        targ_line += str(specdata['specobjid'][ind]) + ' '
        targ_line += str(specdata['z'][ind]) + ' '
        targ_line += str(specdata['zerr'][ind]) + '\n'

    f = open(target_path, 'w')
    f.write('# specobjid, z, zerr\n')
    f.write(targ_line)
    f.close()
    hdu = pf.PrimaryHDU(patch_data)
    hdu.writeto(feature_path, clobber=True)

def match_photozs_to_orig(patch_dir, photozfile, photfile, outpath):
    """
    Match photoz values to objects that have patches.
    """
    patch_paths = glob.glob(patch_dir + '*fits')

    f = pf.open(photfile)
    photdata = f[1].data
    f.close()
    photdata = photdata[np.argsort(photdata['specobjid'])]

    f = pf.open(photozfile)
    photozdata = f[1].data
    f.close()
    photozdata = photozdata[np.argsort(photozdata['objid'])]

    Nsamples = len(patch_paths)
    out_line = ''
    for i in range(Nsamples):
        if i % 100 == 0:
            print 'working on', i

        f = pf.open(patch_paths[i])
        d = f[0].data
        f.close()

        specobjid = np.int(patch_paths[i].split('_')[-1][:-5])
        ind = np.searchsorted(photdata['specobjid'], specobjid)
        ind = np.searchsorted(photozdata['objid'], photdata[ind]['objid'])
        assert (ind >= 0)
        d = photozdata[ind]
        out_line += ' '.join([str(specobjid), str(d['objid']), str(d['z']),
                              str(d['zerr'])]) + '\n'

    f = open(outpath, 'w')
    f.write('# specobjid, objid, z, zerr\n')
    f.write(out_line)
    f.close()

if __name__ == '__main__':
    # Run the script
    img_dir = os.environ['IMGZDATA']
    photfile = img_dir + 'dr7photfirstdr724k_rfadely.fit'
    specfile = img_dir + 'dr7spec2nddr724k_rfadely.fit'
    photozfile = img_dir + 'dr7photozdr724k_rfadely.fit'
    patch_dir = img_dir + 'patches/dr7/orig/'
    data_dir = img_dir + 'patches/dr7/'
    img_dir += 'dr7/'

    if False:
        get_images(photfile, img_dir)

    if False:
        make_img_list(photfile, img_dir, img_dir + 'dr724k_imgs.txt')

    if False:
        warnings.filterwarnings("ignore")
        s, e = 23000, 24000
        make_orig_patches(photfile, specfile, img_dir, patch_dir, start=s,
                          end=e)

    if False:
        gen_final_files(patch_dir, data_dir + 'dr7_fact4_avg_withpsf',
                        photfile, specfile)

    if True:
        outpath = data_dir + 'dr7_fact4_avg_withpsf_photoz.txt'
        match_photozs_to_orig(patch_dir, photozfile, photfile, outpath)
