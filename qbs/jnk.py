from __future__ import division, print_function
import numpy as np
from sklearn.model_selection import KFold
from utils import get_fn, get_data, jackknife_summary, get_loss_dict
import os
import joblib
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pdb

DEFAULT_INACTIVE_LEVEL = 3.5

def add_garbage(d, smiles, resps, all_smiles, fps, inactive_level=DEFAULT_INACTIVE_LEVEL):
    """

    :param d: dict of data for modelling
    :param smiles:  np.array of smiles of molecules with target activity
    :param resps: np.array of responses (activity levels) of target under consideration
    :param all_smiles: np.array of strings (smiles of all chembl molecules)
    :param fps:  np.array of 128-bit predictors - molecular fingerprints of all_smiles
    :param inactive_level: float - value that counts as 'inactive'
    :return: None
    """
    as_smiles = np.argsort(all_smiles)
    n = fps.shape[0]
    d['preds'] = fps
    d['resps'] = np.ones(n, dtype=float) * inactive_level
    smile_idces = all_smiles[as_smiles].searchsorted(smiles)
    d['resps'][as_smiles[smile_idces]] = resps

    oos = np.delete(np.arange(len(resps)), d['is'])
    
    garbage_idcs = np.delete(np.arange(n), as_smiles[smile_idces])
    
    # split garbage in 2
    kf = KFold(n_splits=2, shuffle=True)
    (tr_garb, tst_garb) = kf.split(garbage_idcs).next()
    tr_garb = garbage_idcs[tr_garb]
    tst_garb = garbage_idcs[tst_garb]
    
    tr_good = as_smiles[smile_idces[d['is']]]
    tst_good = as_smiles[smile_idces[oos]]

    d['is_good'] = tr_good
    d['is_garb'] = np.concatenate((tr_good, tr_garb))
    d['oos_good'] = tst_good
    d['oos_garb'] = np.concatenate((tst_good, tst_garb))


def runner(outdir, sorted_targets, model_dict,
           all_smiles, fps,  tot_num_runs=10, frac_fit=1.0,
           kf=0, insample=False, use_pool=True, force_rerun=False):
    """
    :param outdir:
    :param sorted_targets:
    :param model_dict:
    :param all_smiles:
    :param fps:
    :param tot_num_runs:
    :param frac_fit:
    :param kf:
    :param insample:
    :param use_pool:
    :param force_rerun:
    :return: None
    """
    fnf = get_fn(outdir, frac_fit, kf=kf, insample=insample)
    if not (not os.path.isfile(fnf) or force_rerun):
        print('Already computed')
        return

    model_types = ['mgood_pgood', 'mgood_pgarb', 'mgarb_pgood', 'mgarb_pgarb']

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
        preds, resps, smiles, dy = get_data(tgt)
        preds = preds + 0.
        for m in model_dict.keys():
            res = full_bootstrap(preds=preds, resps=resps, smiles=smiles, method=m, model_dict=model_dict,
                                 all_smiles=all_smiles, fps=fps, num_kf=kf,
                                 insample=insample, num_runs=num_runs, use_pool=use_pool)
            loss_hdr[tgt][m] = dict((mdl_type, jackknife_summary(res, mdl_type)) for mdl_type in model_types)
        pass
    joblib.dump(loss_hdr, fnf)
    print('Completed')
    return


# Added put_title arg for plots for paper
def plotter(outdir, model_dict, sorted_targets, model_types, model_labels, frac_fit=1.0, kf=0,
            insample=False, loss='mse', save=True, put_title=False, put_grid=False):
    """

    :param outdir:
    :param model_dict:
    :param sorted_targets:
    :param model_labels:
    :param frac_fit:
    :param kf:
    :param insample:
    :param loss:
    :param save:
    :param put_title:
    :param put_grid:
    :return:
    """

    #model_types = ['mgood_pgood', 'mgood_pgarb', 'mgarb_pgood', 'mgarb_pgarb']
    fnf = get_fn(outdir, frac_fit, kf=kf, insample=insample)

    loss_hdr = joblib.load(fnf)

    for mdl_type in model_types:
        for (i, method) in enumerate(sorted(model_dict.keys())):

            # losses = np.asarray([loss_hdr[x][method][loss]['loss'] for x in sorted_targets])
            losses = []
            for x in sorted_targets:
                try:
                    losses.append(loss_hdr[x][method][mdl_type][loss]['loss'])
                except:
                    print(x, method, loss)
                    raise KeyError
            losses = np.asarray(losses)
            if insample:
                plt.plot(np.arange(25) + 0.05 * i, losses, label=method.upper())
            else:
                loss_l = np.asarray([loss_hdr[x][method][mdl_type][loss]['loss_l'] for x in sorted_targets])
                loss_u = np.asarray([loss_hdr[x][method][mdl_type][loss]['loss_u'] for x in sorted_targets])

                yerr = np.vstack((losses - loss_l, loss_u - losses))

                plt.errorbar(np.arange(25) + i * 0.1, losses, capsize=10, yerr=yerr, label=mdl_type + ' ' + method.upper())
                pass
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
    plt.legend(loc='best', fontsize='x-large', labels=model_labels)
    if loss == 'mse':
        plt.ylabel('Expected mean squared error', fontsize='x-large')
    else:
        plt.ylabel('Expected loss', fontsize='x-large')

    if save:
        fnff = get_fn(outdir, frac_fit, kf=kf, insample=insample, fig=True, loss=loss)
        print(fnff)
        plt.savefig(fnff, bbox_inches='tight')


def nbs_run(kwargs):
    """
    :param kwargs:
    :return:
    """
    method = kwargs.get('method')
    preds = kwargs.get('preds') + 0.
    resps = kwargs.get('resps')
    model_dict = kwargs.get('model_dict')
    assert model_dict is not None

    model_type_dict = {'mgood_pgood' : {'train': 'is_good', 'test' : 'oos_good'},
                       'mgood_pgarb' : {'train': 'is_good', 'test' : 'oos_garb'},
                       'mgarb_pgood' : {'train': 'is_garb', 'test' : 'oos_good'},
                       'mgarb_pgarb' : {'train': 'is_garb', 'test' : 'oos_garb'},
                       }

    inactive_level = kwargs.get('inactive_level', DEFAULT_INACTIVE_LEVEL)
    losses = {}

    for mdl_type in model_type_dict.keys():
        losses[mdl_type] = {}
        is_type = model_type_dict[mdl_type]['train']
        oos_type = model_type_dict[mdl_type]['test']

        loss_dict = get_loss_dict()

        mdl = model_dict[method]['m'](**model_dict[method]['kw'])


        my_is = kwargs.get(is_type)
        my_oos = kwargs.get(oos_type)

        mdl.fit(preds[my_is], resps[my_is])

        predictions = mdl.predict(preds[my_oos])

        for (l, v) in loss_dict.iteritems():
        
            # Worth noting, we're looking for the top frac_find _in_the_oos_data_
            # (not in the whole data - as otherwise we might be looking for something
            # that isn't there)
            if 'frac_find' in v['kw']:
                big_n = sum(resps[my_oos]> inactive_level)
                sorted_indices = np.argsort(-resps[my_oos])
                n = int(big_n * (1 - v['kw']['frac_find']))
                tgt_val = resps[my_oos][sorted_indices[n]]
                v['kw'].update({'tgt_val' : tgt_val})
            losses[mdl_type][l] = v['func'](predictions, resps[my_oos], **v['kw'])
            pass
        pass
    return losses



def full_bootstrap(preds, resps, smiles, method, model_dict, all_smiles, fps,
                   num_runs=100, insample=False,
                  frac_fit=1.0, num_kf=0, use_pool=True):
    """
    :param preds:
    :param resps:
    :param smiles:
    :param method:
    :param model_dict:
    :param all_smiles:
    :param fps:
    :param num_runs:
    :param insample:
    :param frac_fit:
    :param num_kf:
    :param use_pool:
    :return:
    """

    # losses is going to be a list of dicts, loss_type => value                                                                                                                                        
    losses = []
    sorted_indices = np.argsort(resps)

    N = len(resps)
    M = int(N * frac_fit)

    # Choose insample values...                                                                                                                                                                        
    idx_list = []
    for i in range(num_runs):
        idcs_rand = np.random.choice(M, M)
        idcs = sorted_indices[idcs_rand]
        idx_list.append({'is' : idcs})
        pass

    # Add in all the other data...                                                                                                                                                                     
    for d in idx_list:
        
        add_garbage(d, smiles, resps, all_smiles, fps)
        d.update({'method' : method})
        d.update({'model_dict' : model_dict})
        if insample:
            d.update({'oos' : d['is']})
            pass        
        pass

    if use_pool:
        p = Pool(4)
        losses = p.map(nbs_run, idx_list)
        p.close()
        p.join()

    else:
        losses = [nbs_run(x) for x in idx_list]
        pass
    return losses

