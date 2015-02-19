import os
import numpy as np
import pyfits as pf

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.grid_search import GridSearchCV
from utils import process

def run_sklearn_model(regressor, parms, img_data, specz, process_kwargs,
                      cv=True):
    """
    Run a simple model on the data.
    """
    trn_x, tst_x, trn_ind, tst_ind = process(img_data, **process_kwargs)
    specz_trn = specz[trn_ind]
    specz_tst = specz[tst_ind]
    if cv:
        model = GridSearchCV(regressor(), parms, scoring='mean_squared_error')
    else:
        model = regressor(parms)
    model.fit(trn_x, specz_trn)
    predictions = model.predict(tst_x)
    return predictions, specz_tst, model, tst_ind

if __name__ == '__main__':
    from data import fetch_image_data, fetch_traditional_data
    from plots import plot_results

    data_file = os.environ['IMGZDATA'] + '/patches/dr12/shifted/'
    data_file += 's_r_5x5.fits'
    phot_file = os.environ['IMGZDATA'] + 'dr12_main_50k_rfadely.fit'
    spec_info_file = os.environ['IMGZDATA'] + '/dr12/dr12_main_50k_z-ind.txt'
    if False:
        data, spec_info, sdss = fetch_image_data(data_file, spec_info_file,
                                                 phot_file)
        process_kwargs = {'logscale':True, 'zm':True, 'zca':True}
    else:
        data, spec_info, sdss = fetch_traditional_data(phot_file,
                                                       spec_info_file)
        process_kwargs = {'logscale':False, 'zm':True, 'zca':True}

    seed = 8675309
    process_kwargs['seed'] = seed

    # linear model
    parms = {'alpha':[1.0e-8, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.]}
    parms = [1.e-8]
    pred, true, model, tst_ind = run_sklearn_model(Lasso, parms, data,
                                                   spec_info[:, 1],
                                                   process_kwargs, cv=False)
    print np.mean((pred - true) ** 2.)
    names = ['SDSS Photoz']
    results = [sdss[tst_ind]]
    names.append('Lasso')
    results.append(pred)

    # RF model
    parms = {'n_estimators':[32, 64, 128]}
    parms = 64
    pred, true, model, tst_ind = run_sklearn_model(RandomForestRegressor,
                                                   parms, data,
                                                   spec_info[:, 1],
                                                   process_kwargs,
                                                   cv=False)
    print np.mean((pred - true) ** 2.)
    names.append('Random Forest')
    results.append(pred)

    # knn model
    parms = {'n_neighbors':[2, 4, 8, 12, 24, 48]}
    parms = 4
    pred, true, model, tst_ind = run_sklearn_model(KNeighborsRegressor,
                                                   parms, data,
                                                   spec_info[:, 1],
                                                   process_kwargs,
                                                   cv=False)
    print np.mean((pred - true) ** 2.)
    names.append('kNN')
    results.append(pred)

    plot_results(spec_info[tst_ind, 1], results, names,
                 '../plots/simple_models_ugriz.png')
