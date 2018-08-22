from __future__ import division, print_function
import joblib
import numpy as np
from IPython.display import SVG
from scipy.spatial.distance import pdist, squareform, jaccard, cityblock
from scipy import stats

from multiprocessing import Pool
from copy import deepcopy

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model as LM

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, BayesianRidge, ElasticNet, Lasso

from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from pyearth import Earth
import os

import matplotlib.pyplot as plt
import re
import pdb

# Deep learning model with intermediate layer...
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=0)))

alphas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

def get_model_dict():
    # Model dictionaries 
    model_dict = {'ridge' : {'m' : Ridge, 'kw' : {'fit_intercept':True, 'alpha':0.1}, },
                  'rdgcv' : {'m' : RidgeCV, 'kw' : {'fit_intercept':True, 'alphas':alphas}, },
                  'rf'    : {'m' : RandomForestRegressor, 'kw' : {'n_estimators':100, 
                                                                  'n_jobs':4, 'max_depth':10}, },
                  'mars'  : {'m' : Earth, 'kw' : {}},
                  'svr'   : {'m' : SVR, 'kw' : {}, },
                  'svrl'   : {'m' : SVR, 'kw' : {'kernel' : 'linear'}, },              
                  'dl_l'   : {'m' : Pipeline, 
                              'kw' : {'steps': [('standardize', StandardScaler()),
                                                ('mlp', KerasRegressor(build_fn=larger_model, 
                                                                       epochs=100, batch_size=5, 
                                                                       verbose=0))
                                                ]},
                              },
                  }

    return model_dict


# Pull in data for a single target name
def get_data(tgt_name='COX-2'):
    data_dir = 'datasets/' + tgt_name + '/'
    preds = joblib.load(data_dir + tgt_name + '_predsu.npy')
    resps = joblib.load(data_dir + tgt_name + '_respu.npy')
    smiles = joblib.load(data_dir + tgt_name + '.smiu')
    dy = joblib.load(data_dir + tgt_name + '.dyu')
    return preds, resps, smiles, dy

def summarize(results):
    summary = {}
    for l in loss_dict.keys():
        vals = np.asarray([x[l] for x in results])
        if len(vals) == 1:
            summary[l] = {'loss' : vals[0]}
        else:
            summary[l] = {'loss_l' : np.percentile(vals, 5),
                          'loss' : np.mean(vals),                          
                          'loss_u' : np.percentile(vals, 95),
                         }
            pass
        pass
    return summary

# This function takes the vector of observed values and their mean (implies fewer computations of the mean)
def jackknife(vals, vals_bar):
    n = len(vals)
    var_JK = 0.0
    for v in vals:
        vals_bar_i = (n/(n-1)) * (vals_bar - v/n)
        #print(pow(vals_bar_i - vals_bar, 2))
        var_JK += pow(vals_bar_i - vals_bar, 2)
        pass

    sd_JK = pow( ((n-1)/n) * var_JK, .5)
   
    return sd_JK
    
def jackknife_summary(results, mdl_type=None):
    """

    :param results:
    :param mdl_type:
    :return:
    """
    summary = {}
    for l in loss_dict.keys():
        if mdl_type is None:
            vals = np.asarray([x[l] for x in results])
        else:
            vals = np.asarray([x[mdl_type][l] for x in results])
            pass

        if len(vals) == 1:
            summary[l] = {'loss' : vals[0]}
        else:
            vals_bar = np.mean(vals)
            sd = jackknife(vals, vals_bar)
            summary[l] = {'loss_l' : vals_bar - 2*sd,
                          'loss' : vals_bar,                          
                          'loss_u' : vals_bar + 2*sd,
                         }
            pass
        pass
    return summary

# Write the loss functions 

def avg_mse(predictions, responses, **kwargs):
    return mean_squared_error(responses, predictions) / mean_squared_error(responses, np.zeros_like(responses))


def Rank_loss(predictions, responses, **kwargs):

    assert 1/2 > 0
    tgt_val = kwargs.get('tgt_val')
    ranked = np.argsort(-predictions)
    found = responses[ranked] >= tgt_val
    # Number of actives
    N_gamma = np.sum(found)
    
    # Size of test sets
    N_test = predictions.shape[0]
    lt = kwargs.get('loss_type')
    #pdb.set_trace()
    if lt == 'min':
        # Equation (1) of the paper
        loss = 1/(N_test - N_gamma) * np.min(np.arange(N_test)[found])
    elif lt == 'avg':
        # Equation (2) of the paper
        loss = 1/N_gamma * 1/(N_test - N_gamma) * (np.sum(np.arange(N_test)[found]) - N_gamma * (N_gamma - 1)/2)
        pass
    
    assert loss >= 0
    assert loss <= 1
    return loss


def Active_rank_loss_avg(predictions, responses, **kwargs):
    kwargs.update({'loss_type' :'avg'})
    return Rank_loss(predictions, responses, **kwargs)

def Active_rank_loss_min(predictions, responses, **kwargs):
    kwargs.update({'loss_type' :'min'})
    return Rank_loss(predictions, responses, **kwargs)

# define the dictionary of losses used here
loss_dict = {'mse' :       {'func' : avg_mse,         'kw' : {}},
             'loss_min_90' :   {'func' : Active_rank_loss_min,  'kw' : {'frac_find' : 0.9}},
             'loss_avg_90' :   {'func' : Active_rank_loss_avg,  'kw' : {'frac_find' : 0.9}},
             'loss_min_95' :   {'func' : Active_rank_loss_min,  'kw' : {'frac_find' : 0.95}},
             'loss_avg_95' :   {'func' : Active_rank_loss_avg,  'kw' : {'frac_find' : 0.95}},
             'loss_min_99' :   {'func' : Active_rank_loss_min,  'kw' : {'frac_find' : 0.99}},
             'loss_avg_99' :   {'func' : Active_rank_loss_avg,  'kw' : {'frac_find' : 0.99}}}

def get_loss_dict():
    return loss_dict

def get_fn(od, frac_fit, kf=0, insample=False, fig=False, loss=None):
    fn = 'loss_' + str(frac_fit)
    if kf > 0:
        fn += '_kf_' + str(kf)
        pass
    if insample:
        fn += '_insample'
        pass

    ffn = os.path.join(od, fn)
    if fig:
        ffn = re.sub('[.]','', ffn)
        ffn = ffn + '_' + loss + '.pdf'
        pass
    return ffn
        

def runner(outdir, sorted_targets, model_dict, frac_fit=1.0, 
           tot_num_runs=40,
           kf=0, insample=False, use_pool=True, force_rerun=False):
    """

    :param outdir:
    :param sorted_targets:
    :param model_dict:
    :param frac_fit:
    :param tot_num_runs:
    :param kf:
    :param insample:
    :param use_pool:
    :param force_rerun:
    :return:
    """

    fnf = get_fn(outdir, frac_fit, kf=kf, insample=insample)
    if os.path.isfile(fnf) and not force_rerun:
        print('Already computed')
        return

    loss_hdr = {}
    if insample:
        num_runs = 1
    elif kf > 0:
        num_runs = int(tot_num_runs/kf)
    else:
        num_runs = tot_num_runs
    
    for tgt in sorted_targets:
        loss_hdr[tgt] = {}
        print ('Doing', tgt)
        preds, resps, _, dy = get_data(tgt)
        preds = preds + 0.
        for m in model_dict.keys():
            res = full_bootstrap(preds, resps, m, model_dict, frac_fit=frac_fit, num_kf=kf,
                                 insample=insample, num_runs=num_runs, use_pool=use_pool,
                                )
            joblib.dump(res, outdir + '/detail/' + tgt + '_' + m + '_' + str(frac_fit) + '_' + str(kf) + '.res')
            loss_hdr[tgt][m] = jackknife_summary(res)
        pass
    joblib.dump(loss_hdr, fnf)
    print('Completed')
    return

# Added put_title arg for plots for paper
def plotter(outdir, model_dict, sorted_targets, model_labels, frac_fit=1.0, kf=0, 
            insample=False, loss='mse', save=True, put_title=False, put_grid=False):
    
    fnf = get_fn(outdir, frac_fit, kf=kf, insample=insample)
    
    loss_hdr = joblib.load(fnf)
   
    for (i, method) in enumerate(sorted(model_dict.keys())):

        #losses = np.asarray([loss_hdr[x][method][loss]['loss'] for x in sorted_targets])
        losses = []
        for x in sorted_targets:
            try:
                losses.append(loss_hdr[x][method][loss]['loss'])
            except:
                print(x, method, loss)
                raise KeyError
        losses = np.asarray(losses)
        if insample:
            plt.plot(np.arange(25) + 0.05*i, losses,  label=method.upper())
        else:
            loss_l = np.asarray([loss_hdr[x][method][loss]['loss_l'] for x in sorted_targets])
            loss_u = np.asarray([loss_hdr[x][method][loss]['loss_u'] for x in sorted_targets])
            
            yerr = np.vstack((losses - loss_l, loss_u - losses))
            
            plt.errorbar(np.arange(25)+i*0.1, losses, capsize=10, yerr=yerr, label=method.upper())
            pass
        pass
        
    if insample:
        if put_title:
            title('Insample %s Loss' % loss.upper())
    else:
        ttl = 'OOS %s Loss' % loss.upper()
        if kf > 0:
            ttl += ' with %d fold CV' % kf
        if frac_fit < 1.0:
            ttl += ' Max activity in fit at %.1f' % frac_fit
        if put_title:
            title(ttl)
    pass

    plt.grid(put_grid)
    plt.tick_params(top=False, right=False)
    plt.xticks(np.arange(25), sorted_targets, rotation=-45)
    plt.legend(loc='best', fontsize = 'x-large', labels=model_labels)
    if loss=='mse':
        plt.ylabel('Expected mean squared error', fontsize='x-large')
    else:
        plt.ylabel('Expected loss', fontsize='x-large')
    
    if save:
        fnff = get_fn(outdir, frac_fit, kf=kf, insample=insample, fig=True, loss=loss)
        print(fnff)
        plt.savefig(fnff, bbox_inches='tight')

def probability_min(means, sigmas, K=1000, epsilon = 0.01):
    
    N = len(means)
    probs = np.zeros(N) # array of probability weights
    
    # check that none of the sigmas are zero to avoid NANs
    for ss in range(N):
        if(sigmas[ss]==0):
            sigmas[ss] = epsilon
            pass
        pass

    for i in range(N):
        xs = np.random.normal(loc=means[i],scale=sigmas[i],size=K)
        logterms = np.zeros(K)
        # Iterate over random draw from the i^th normal distribution
        for sim in range(K):
            x = xs[sim]
            logproductterm = 0
            for j in range(N):
                if(j != i):
                    logproductterm += stats.norm.logcdf(-(x-means[j])/sigmas[j]) 
                    pass
                pass
            logterms[sim] = logproductterm
            pass
        probs[i] = np.mean(np.exp(logterms))
        pass
    return probs

# Total score : this is change of ll_estimate
def model_score(frac_fit=1.0, losses=['loss_avg_90', 'mse', ], kf=0, K=1000, recompute_scores=False):
    if recompute_scores:
        fnf = get_fn(frac_fit, kf=0)
        loss_hdr = joblib.load(fnf)

        print(('%9s |' + '%9s |' * len(losses)) % tuple(['',] + losses))
        print('-' * (11 * (len(losses) + 1) - 1))

        M = len(model_dict.keys()) # number of models

        Loss_scores = []

        models = model_dict.keys()
        d = {'losses' : map(lambda x : x + '_' + str(frac_fit), losses)}

        for loss in losses:
            scores = np.zeros(M)
            for tgt in sorted_targets:
                tgt_means = []
                tgt_sigmas = []
                # For each model we extract the mean and SD
                for (i, method) in enumerate(models):
                    lmean = loss_hdr[tgt][method][loss]['loss']
                    llow = loss_hdr[tgt][method][loss]['loss_l']
                    lsigma = (lmean - llow)/2
                    tgt_means.append(lmean)
                    tgt_sigmas.append(lsigma)
                    pass
                # The arrays of means and SDs are used to compute probability of min expected loss
                weights = probability_min(tgt_means, tgt_sigmas, K=K)
                # A weight of 1: min loss with probability 1; weight of 0: min loss with probability 0

                scores += weights   
                pass
            Loss_scores.append(scores)



        for (i, method) in enumerate(models):
            d[method] = np.ravel(map(lambda x: x[i], Loss_scores))
            out = [method,]
            for (j, loss) in enumerate(losses):
                out.append(Loss_scores[j][i])
                pass
            print (('%9s |' + '%9.1f |' * len(losses)) % tuple(out))

        return d
    else:
        print('done this already, skipping...')






def nbs_run(kwargs):
    my_is = kwargs.get('is')
    my_oos = kwargs.get('oos')
    method = kwargs.get('method')
    preds = kwargs.get('preds') + 0.
    resps = kwargs.get('resps')
    model_dict = kwargs.get('model_dict')

    mdl = model_dict[method]['m'](**model_dict[method]['kw'])
    mdl.fit(preds[my_is], resps[my_is])
        
    predictions = mdl.predict(preds[my_oos])
    
    losses = {}
    for (l, v) in loss_dict.iteritems():
        
        # Worth noting, we're looking for the top frac_find _in_the_oos_data_
        # (not in the whole data - as otherwise we might be looking for something
        # that isn't there)
        if 'frac_find' in v['kw']:
            N = len(resps[my_oos])
            sorted_indices = np.argsort(resps[my_oos])
            n = int(N * v['kw']['frac_find'])
            tgt_val = resps[my_oos][sorted_indices[n]]
            v['kw'].update({'tgt_val' : tgt_val})
            
        losses[l] = v['func'](predictions, resps[my_oos], **v['kw'])
        pass
    
    return losses
    
    
def full_bootstrap(preds, resps, method, model_dict, num_runs=100, insample=False,
                  frac_fit=1.0, num_kf=0, use_pool=True):

    # losses is going to be a list of dicts, loss_type => value
    losses = []
    sorted_indices = np.argsort(resps)

    N = len(resps)
    M = int(N * frac_fit)
    
    # Choose insample values...
    idx_list = []
    for i in range(num_runs):
        if num_kf > 0:
            kf = KFold(n_splits=num_kf, shuffle=True)
            for (tr_i, tst_i) in kf.split(np.arange(M)):
                idcs = sorted_indices[tr_i]
                pass
            pass
        else:
            idcs_rand = np.random.choice(M, M)
            idcs = sorted_indices[idcs_rand]
            pass
        idx_list.append({'is' : idcs})
        pass
    
    # Add in all the other data...
    for d in idx_list:
        if insample:
            oos = d['is']
        else:
            oos = np.delete(np.arange(N), d['is'])
            pass
        d.update({'oos' : oos, 'preds' : preds, 'model_dict' : model_dict, 
                  'resps' : resps, 'method' : method})
        pass
            
    if use_pool:
        p = Pool(7)
        losses = p.map(nbs_run, idx_list)
        p.close()
        p.join()

    else:
        losses = [nbs_run(x) for x in idx_list]
        pass
    return losses


