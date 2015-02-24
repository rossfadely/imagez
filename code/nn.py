import os
import numpy as np

from utils import process
from lasagne import layers
from lasagne_wrapper import NeuralNet

def simple_net(data, specz, process_kwargs, learning_rate=0.01, momentum=0.9,
               Nlayers=2, Nhidden=64, Nepochs=50):
    """
    Fit a simple net to the data, using lasagne's NN packages.
    """
    y = np.array([specz]).T
    trn_x, tst_x, trn_ind, tst_ind = process(data, **process_kwargs)
    trn_y = y[trn_ind]
    tst_y = y[tst_ind]

    layer_defs=[('input', layers.InputLayer)]
    for i in range(Nlayers):
        layer_defs.append(('hidden', layers.DenseLayer))
    layer_defs.append(('output', layers.DenseLayer))

    model = NeuralNet(layers=layer_defs, input_shape=(None, trn_x.shape[1]),
                      hidden_num_units=Nhidden,
                      output_nonlinearity=None,
                      output_num_units=1,
                      update_learning_rate=learning_rate,
                      update_momentum=momentum,
                      regression=True, max_epochs=Nepochs, verbose=1)

    model.fit(trn_x.astype(np.float32), trn_y.astype(np.float32))
    predictions = model.predict(tst_x.astype(np.float32))
    return predictions, tst_y, model, tst_ind

def conv_net(data, specz, process_kwargs, learning_rate=0.01, momentum=0.9,
             Nepochs=50, architecture=[('conv', 32, 3), ('conv', 64, 3),
                                       ('conv', 128, 3), ('pool', 2),
                                       ('out')], gpu=False):
    """
    Fit a conv net to the data, using nolearn's wrappers on lasagne... oy vey
    """
    if gpu:
        convlayer = layers.cuda_convnet.Conv2DCCLayer
        poollayer = layers.cuda_convnet.MaxPool2DCCLayer
    else:
        convlayer = layers.Conv2DLayer
        poollayer = layers.MaxPool2DLayer


    y = np.array([specz]).T
    trn_x, tst_x, trn_ind, tst_ind = process(data, **process_kwargs)
    trn_y = y[trn_ind]
    tst_y = y[tst_ind]

    nn_kwargs = {'input_shape':(None, trn_x.shape[1], trn_x.shape[2],
                                trn_x.shape[3]),
                 'output_num_units':1, 'output_nonlinearity':None,
                 'update_learning_rate':learning_rate,
                 'update_momentum':momentum,
                 'regression':True, 'max_epochs':Nepochs, 'verbose':1}

    nconv, npool, nhidden = 1, 1, 1
    layer_defs=[('input', layers.InputLayer)]
    for i in range(len(architecture)):
        l = architecture[i]
        if l[0] == 'conv':
            layer_defs.append(('conv%d' % nconv, convlayer))
            nn_kwargs['conv%d_num_filters' % nconv] = l[1]
            nn_kwargs['conv%d_filter_size' % nconv] = (l[2], l[2])
            nconv += 1
        if l[0] == 'pool':
            layer_defs.append(('pool%d' % npool, poollayer))
            nn_kwargs['pool%d_ds' % npool] = (l[1], l[1])
            npool += 1
        if l[0] == 'dense':
            layer_defs.append(('hidden%d' % nhidden, layers.DenseLayer))
            nn_kwargs['hidden%d_num_units' % nhidden] = l[1]
            nhidden += 1
    layer_defs.append(('output', layers.DenseLayer))
    nn_kwargs['layers'] = layer_defs

    model = NeuralNet(**nn_kwargs)
    model.fit(trn_x.astype(np.float32), trn_y.astype(np.float32))
    model.fit(trn_x.astype(np.float32), trn_y.astype(np.float32))
    predictions = model.predict(tst_x.astype(np.float32))
    return predictions, tst_y, model, tst_ind

if __name__ == '__main__':
    from simple_models import run_sklearn_model
    from data import fetch_image_data, fetch_traditional_data
    from plots import plot_results

    from sklearn.linear_model import Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor

    data_file = os.environ['IMGZDATA'] + '/patches/dr12/shifted/'
    data_file += 's_r_5x5.fits'
    phot_file = os.environ['IMGZDATA'] + 'dr12_main_50k_rfadely.fit'
    spec_info_file = os.environ['IMGZDATA'] + '/dr12/dr12_main_50k_z-ind.txt'
    if True:
        data, spec_info, sdss = fetch_image_data(data_file, spec_info_file,
                                                 phot_file)
        process_kwargs = {'logscale':True, 'zm':True, 'zca':True}
    else:
        data, spec_info, sdss = fetch_traditional_data(phot_file,
                                                       spec_info_file)
        process_kwargs = {'logscale':False, 'zm':True, 'zca':True}

    seed = 0
    process_kwargs['seed'] = seed
    process_kwargs['test_frac'] = 0.2

    if False:
        process_kwargs['flatten'] = True
        pred, true, model, tst_ind = simple_net(data, spec_info[:, 1],
                                                process_kwargs, Nepochs=50,
                                                Nhidden=512,
                                                learning_rate=0.03)
        print np.mean((pred - true) ** 2.)
        print tst_ind[:10]

    if True:
        architecture=[('conv', 32, 3), ('conv', 64, 2), ('conv', 128, 2),
                      ('pool', 2)]
        pred, true, model, tst_ind = conv_net(data, spec_info[:, 1],
                                              process_kwargs, Nepochs=1,
                                              architecture=architecture)
        print np.mean((pred - true) ** 2.)

    if True:
        process_kwargs['flatten'] = True

        # linear model
        parms = [1.e-8]
        pred, true, model, tst_ind = run_sklearn_model(Lasso, parms, data,
                                                       spec_info[:, 1],
                                                       process_kwargs,
                                                       cv=False)

        print np.mean((pred - true) ** 2.)
        print tst_ind[:10]

        names = ['SDSS Photoz']
        results = [sdss[tst_ind]]
        names.append('Lasso')
        results.append(pred)

