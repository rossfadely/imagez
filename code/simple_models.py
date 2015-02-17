import os
import numpy as np
import pyfits as pf

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.grid_search import GridSearchCV
from utils import process

def run_sklearn_model(regressor, parms, img_data, specz):
    """
    Run a linear model on the data.
    """
    trn_x, tst_x, trn_ind, tst_ind = process(img_data)
    specz_trn = specz[trn_ind]
    specz_tst = specz[tst_ind]
    model = GridSearchCV(regressor(), parms, scoring='mean_squared_error')
    model.fit(trn_x, specz_trn)
    predictions = model.predict(tst_x)
    return predictions, specz_tst, model

if __name__ == '__main__':

    data_file = os.environ['IMGZDATA'] + '/patches/dr12/shifted/'
    data_file += 's_r_1x1.fits'
    f = pf.open(data_file)
    u = f[0].data
    size = u.shape[1]
    data = np.zeros((u.shape[0], 5 * size))
    for i in range(5):
        data[:, i * size:(i + 1) * size] = f[i].data
    f.close()

    spec_info = np.loadtxt(os.environ['IMGZDATA'] + 
                           '/dr12/dr12_main_50k_z-ind.txt')
    assert data.shape[0] == spec_info.shape[0]

    # linear model
    parms = {'alpha':[1.0e-8, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.]}
    pred, true, model = run_sklearn_model(Lasso, parms, data, spec_info[:, 1])
    print model.best_params_
    print np.mean((pred - true) ** 2.)

    # knn model
    parms = {'n_neighbors':[2, 4, 8, 12, 24, 48]}
    pred, true, model = run_sklearn_model(KNeighborsRegressor, parms, data,
                                          spec_info[:, 1])
    print model.best_params_
    print np.mean((pred - true) ** 2.)

    # RF model
    parms = {'n_estimators':[32, 64, 128, 256]}
    pred, true, model = run_sklearn_model(RandomForestRegressor, parms, data,
                                          spec_info[:, 1])
    print model.best_params_
    print np.mean((pred - true) ** 2.)
