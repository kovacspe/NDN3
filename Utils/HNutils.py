"""Utility functions to assist with creating, training and analyzing HN datasets"""

#from __future__ import division
#from __future__ import print_function
import numpy as np
import scipy.io as sio
from copy import deepcopy
import NDN3 

## HNspecific loading
def data_loader( filename ):
    matdat = sio.loadmat(filename)
    #matdat = sio.loadmat('Data/'+exptname+'py.mat')
    #print('Loaded '+exptname+'py.mat')
    TRcued = matdat['cued']
    TRchoice = matdat['choice']
    TRsig = matdat['signal']  # Ntr x 2 (sorted by RF)
    TRstr = matdat['strength']  # Ntr x 2 (sorted by RF)
    TRstim = matdat['cued_stim']  # Ntr x 4 (sorted by cued, then uncued)
    Robs = matdat['Robs']
    used_inds = matdat['used_inds'][:,0] - 1
    #modvars = matdat['moduvar']
    stim = np.expand_dims(np.multiply(TRsig[:,0], TRstr[:,0]), axis=1)  # stim strength and direction combined
    stimL = matdat['stimL']
    stimR = matdat['stimR']
    blks = matdat['blks']
    dislist = matdat['disp_list'][:,0]
    #strlist = matdat['Dc_list'][:,0]
    Ntr = blks.shape[0]
    Nframes = np.min(np.diff(blks))
    NT, NC = Robs.shape
    ND = len(dislist)
    stimlist = np.unique(stim)
    Nstim = len(stimlist)
    
    CHname = [None]*NC
    # Detect disparities used for decision (indexed by stimulus number)
    decision_stims = np.where(matdat['disp_list'] == np.unique(matdat['cued_stim'][:,0]))[0]
    for cc in range(NC):
        CHname[cc] = matdat['CHnames'][0][cc][0]
    expt_info = {'exptname':filename, 'CHnames': CHname, 'blks':blks, 'dec_stims': decision_stims, 
                 'DispList':dislist, 'StimList':stimlist, 
                 'stimL': stimL, 'stimR':stimR, 'Robs':Robs, 'used_inds': used_inds}
    
    twin = range(25,Nframes)
    Rtr = np.zeros([Ntr, NC], dtype='float32')
    for nn in range(Ntr):
        Rtr[nn,:] = np.sum(Robs[np.add(twin, blks[nn,0]), :], axis=0)

    print("%d frames, %d units, %d trials with %d frames each"%(NT, NC, Ntr, Nframes))
    
    use_random = False
    # Cued and uncued trials
    trC = np.where(TRcued[:,0] > 0)[0]
    trU = np.where(TRcued[:,0] < 0)[0]
    # zero-strength trials
    tr0 = np.where(TRstr[:,0] == 0)[0]
    # sort by cued/uncued
    tr0C = np.where((TRstr[:,0] == 0) & (TRcued[:,0] > 0))[0]
    tr0U = np.where((TRstr[:,0] == 0) & (TRcued[:,0] < 0))[0]
    # for purposes of cross-validation, do the same for non-zero-strength trials
    trXC = np.where((TRstr[:,0] != 0) & (TRcued[:,0] > 0))[0]
    trXU = np.where((TRstr[:,0] != 0) & (TRcued[:,0] < 0))[0]

    # Assign train and test indices sampled evenly from each subgroup (note using default 4-fold)
    Ut0C, Xt0C = train_test_assign( tr0C, use_random=use_random )
    Ut0U, Xt0U = train_test_assign( tr0U, use_random=use_random )
    UtXC, XtXC = train_test_assign( trXC, use_random=use_random )
    UtXU, XtXU = train_test_assign( trXU, use_random=use_random )

    # Putting together for larger groups
    Ut0 = np.sort( np.concatenate( (Ut0C, Ut0U), axis=0 ) )
    Xt0 = np.sort( np.concatenate( (Xt0C, Xt0U), axis=0 ) )
    UtC = np.sort( np.concatenate( (Ut0C, UtXC), axis=0 ) )
    XtC = np.sort( np.concatenate( (Xt0C, XtXC), axis=0 ) )
    UtU = np.sort( np.concatenate( (Ut0U, UtXU), axis=0 ) )
    XtU = np.sort( np.concatenate( (Xt0U, XtXU), axis=0 ) )

    Ut = np.sort( np.concatenate( (Ut0, UtXC, UtXU), axis=0 ) )
    Xt = np.sort( np.concatenate( (Xt0, XtXC, XtXU), axis=0 ) )
    
    trs = {'c':trC, 'u':trU, '0':tr0, '0c': tr0C, '0u': tr0U}
    Utr = {'all':Ut, '0':Ut0, 'c':UtC, 'u':UtU, '0c':Ut0C, '0u':Ut0U}
    Xtr = {'all':Xt, '0':Xt0, 'c':XtC, 'u':XtU, '0c':Xt0C, '0u':Xt0U}

    # Additional processing check
    # Cued and uncued stim
    Cstim = np.multiply(TRstim[:,1], np.sign(TRstim[:,0])) # Cued stim
    Ustim = np.multiply(TRstim[:,3], np.sign(TRstim[:,2]))  # Uncued stim
    f_far = np.zeros([Nstim,2])
    for nn in range(Nstim):
        tr1 = np.where(Cstim == stimlist[nn])[0]
        tr2 = np.where(Ustim == stimlist[nn])[0]
        f_far[nn,0] = np.sum(TRchoice[tr1] > 0)/len(tr1)
        f_far[nn,1] = np.sum(TRchoice[tr2] > 0)/len(tr2)

    return expt_info, Rtr, TRchoice, stim, TRsig, TRcued, TRstim, Ntr, Nstim, NC, trs, Utr, Xtr, f_far


def channel_list_scrub( fnames, subset=None, display_names=True ):
    chnames = []
    if subset is None:
        subset = list(range(len(fnames)))
    for nn in subset:
        fn = fnames[nn]
        a = fn.find('c')   # finds the 'ch'
        b = fn.find('s')-1  # finds the 'sort'
        chn = deepcopy(fn[a:b])
        chnames.append(chn)
    if display_names:
        print(chnames)
    return chnames


## GENERAL UTILITY FUNCTIONS
def train_test_assign( trial_ns, fold=4, use_random=True ):
    num_tr = len(trial_ns)
    if use_random:
        permu = np.random.permutation(num_tr)
        xtr = np.sort( trial_ns[ permu[range(np.floor(num_tr/fold).astype(int))] ]) 
        utr = np.sort( trial_ns[ permu[range(np.floor(num_tr/fold).astype(int), num_tr)] ])
    else:
        xtr = trial_ns[np.array(range(fold//2, num_tr, fold), dtype='int32')]
        utr = trial_ns[np.array(list(set(list(range(num_tr)))-set(xtr)), dtype='int32')]
    return utr, xtr


def drift_design_matrix(num_trials, drift_spacing=100, to_plot=False):
    xminus = 1-np.array(range(drift_spacing))/drift_spacing
    xplus = np.array(range(drift_spacing))/drift_spacing

    anchor_pnts = list(range(0, num_trials, drift_spacing))
    NW = len(anchor_pnts)-1

    Xoffset = np.zeros([num_trials, NW], dtype='float32')
    for nn in range(NW):
        #if nn < (NW-1):
        Xoffset[range(drift_spacing*nn, drift_spacing*(nn+1)),nn] = xminus
        if nn > 0:
            Xoffset[range(drift_spacing*(nn-1), drift_spacing*(nn)),nn] = xplus
    #Xoffset[win_spacing*(NW-1):,nn] = 1.0
    if to_plot:
        plt.imshow(Xoffset, aspect='auto')
        plt.show()
        
    return Xoffset, NW


def vector_regression(xs, ys):
    svec = np.divide(
        np.mean(np.multiply(ys, np.expand_dims(xs, axis=1)), axis=0) - np.mean(xs)*np.mean(ys, axis=0),
        np.var(xs))
    bvec = np.mean(ys, axis=0)-svec*np.mean(xs)
    return svec, bvec


def vec_norm(v):
    return np.sqrt(np.sum(np.square(v)))


def R2fromMSE(mses, data):
    return 1-np.divide(mses, np.var(data, axis=0))


# Parse model terms to separate and eliminate variance due to drift and generate other variances
def adjR2( damod, s, r, i_c, i_u ):
    """Calculate R2s accounting for predicted drift. In poorly predicted cells, most of the variance is due to
    drift, so this would make it look like the models are good at predicting stim and choice, when in fact they
    are just predicting drifts."""
    num_cells = r.shape[1]
    # Robs adjusted for drift (to extent possible with model)
    drift = damod.generate_prediction(input_data=s, ffnet_target=0)
    driftC = drift[i_c,:]
    driftU = drift[i_u,:]
    drift_var_frac = np.divide( np.var(drift, axis=0), np.var(r, axis=0))
    
    RadjC = deepcopy(r[i_c,:])-driftC
    RadjU = deepcopy(r[i_u,:])-driftU
    predCadj = damod.generate_prediction(input_data=s, data_indxs=i_c)-driftC
    predUadj = damod.generate_prediction(input_data=s, data_indxs=i_u)-driftU

    # Adjusted R2s for drift
    R2s = np.zeros([num_cells,2])
    R2s[:,0] = 1 - np.divide( np.mean(np.square(RadjC-predCadj), axis=0), np.var(RadjC, axis=0) )
    R2s[:,1] = 1 - np.divide( np.mean(np.square(RadjU-predUadj), axis=0), np.var(RadjU, axis=0) )
    return R2s, drift_var_frac


def multibar_plot( datas, subplot_info=None, fig_width=None ):

    tot_w = 0.7
    N = len(datas[0])
    clrs = 'brgcybrgcy'
    num_bars = len(datas)
    #assert N > 1, 'not using this write: need list of datasets'
    for nn in range(num_bars):
        assert len(datas[nn]) == N, 'data length much match'

    x = np.arange(N)
    width = tot_w/num_bars # individual bar width
    bar_locs = np.arange(-tot_w/2+width/2, tot_w/2+width/2, width)
    if subplot_info is None:
        fig, ax = plt.subplots()
        if fig_width is not None:
            assert fig_width <= 1, 'fig_width max value is 1'
            fig.set_size_inches(16*fig_width, 2)
    else:
        assert len(subplot_info) == 3, 'must have subplot arguments in subplot_info'
        ax = plt.subplot(subplot_info[0], subplot_info[1], subplot_info[2] )

    for nn in range(num_bars):
        ax.bar( x+bar_locs[nn], datas[nn], width, color=clrs[nn], edgecolor='k' )
    if subplot_info is None:
        plt.show()
        
        
from IPython.display import display, HTML
def title_write( title_text, lvl = 2 ):
    if lvl == 1:
        display(HTML('<h1>{}</h1><br/>'.format(title_text)))
    elif lvl == 2:
        display(HTML('<h2>{}</h2><br/>'.format(title_text)))
    elif lvl == 3:
        display(HTML('<h3>{}</h3><br/>'.format(title_text)))
    else:
        display(HTML('<h4>{}</h4><br/>'.format(title_text)))
        
def mycorr( a, b ):
    anorm = (deepcopy(a)-np.mean(a)) / np.maximum(np.std(a), 1e-6)
    bnorm = (deepcopy(b)-np.mean(b)) / np.maximum(np.std(b), 1e-6)
    cor = np.mean(np.multiply(anorm, bnorm))
    return cor


def calc_pop_angle( v1, v2 ):
    nv1 = deepcopy(v1) / np.maximum(vec_norm(v1), 1e-6)
    nv2 = deepcopy(v2) / np.maximum(vec_norm(v2), 1e-6)
    return np.arccos(np.matmul(nv1.T, nv2))*180/np.pi
    

def LNdecoder(inputs, output, indx_train, indx_test, silent=False, l1reg = 1.0, fit_params=None, nreps=1):

    if len(inputs.shape) == 1:
        nlvs = 1
    else:
        nlvs = inputs.shape[1]
    if fit_params is None:
        fit_params = adam_dec

    decoder_par = NDNutils.ffnetwork_params( 
        input_dims=[nlvs,1,1], layer_sizes=[1], layer_types=['normal'], act_funcs=['tanh'], verbose=False,
        reg_list={'l2':[l1reg]} )
    
    Dmods = []
    mses = np.zeros(nreps)
    for nn in range(nreps):
        decLN = NDN.NDN( decoder_par, tf_seed = 5+nn, noise_dist='gaussian')
        #_= decLN.train(input_data=inputs, output_data=output, train_indxs=indx_train, test_indxs=indx_test, 
        #               learning_alg='lbfgs', opt_params=lbfgs_param) 
        _= decLN.train(input_data=inputs, output_data=output, train_indxs=indx_train, test_indxs=indx_test,
                        learning_alg='adam', opt_params=fit_params, silent=True, output_dir=output_dir)
        Dmods.append(decLN.copy_model())
        mses[nn] = decLN.eval_models(input_data=inputs, output_data=output, data_indxs=indx_test)[0]
        if not silent:
            print("  %2d of %d:  %0.4f"%(nn, nreps, mses[nn]))
    b = np.argmin(mses)
    mse = mses[b]
    decLN = Dmods[b].copy_model()
    R2 = 1-mse/np.var(output[indx_test])
    Dpred = decLN.generate_prediction(input_data=inputs) 
    if not silent:
        print("test-R2 = %0.3f"%R2)
    ws = deepcopy(decLN.networks[0].layers[0].weights[:,0])
    bs = deepcopy(decLN.networks[0].layers[0].biases[0,:])
    return R2, Dpred, [ws, bs]


def NLdecoder(inputs, output, indx_train, indx_test, silent=False, l1reg = 1.0, hidden_unit_frac=2):
        
    if len(inputs.shape) == 1:
        nlvs = 1
    else:
        nlvs = inputs.shape[1]

    decoder_par = NDNutils.ffnetwork_params( 
        input_dims=[nlvs,1,1], layer_sizes=[nlvs//hidden_unit_frac, 1], layer_types=['normal','normal'], 
        act_funcs=['relu', 'tanh'], verbose=False, reg_list={'l2':[l1reg, l1reg]} )
    
    decNL = NDN.NDN( decoder_par, tf_seed = 5, noise_dist='gaussian')
    #_= decNL.train(input_data=inputs, output_data=output, train_indxs=indx_train, test_indxs=indx_test, 
    #               learning_alg='lbfgs', opt_params=lbfgs_param) 
    _= decNL.train(input_data=inputs, output_data=output, train_indxs=indx_train, test_indxs=indx_test,
                    learning_alg='adam', opt_params=adam_dec, silent=silent, output_dir=output_dir) 
    mse = decNL.eval_models(input_data=inputs, output_data=output, data_indxs=indx_test)[0]
    R2 = 1-mse/np.var(output[indx_test])
    Dpred = decNL.generate_prediction(input_data=inputs) 
    if not silent:
        print("test-R2 = %0.3f"%R2)
        
    ws1 = deepcopy(decNL.networks[0].layers[0].weights)
    bs1 = deepcopy(decNL.networks[0].layers[0].biases[0,:])
    ws2 = deepcopy(decNL.networks[0].layers[1].weights[:,0])
    bs2 = deepcopy(decNL.networks[0].layers[1].biases[0,:])

    return R2, Dpred, [ws1, bs1, ws2, bs2]


def parse_channel_name( cn, area=None ):
    indices = [i for i, a in enumerate(cn) if a == '_']
    exptname = cn[:indices[1]]
    if len(indices) == 5:
        # then only one electrode in experiment
        chname = cn[indices[1]+1:indices[3]]
        head = None
    else:
        if cn[indices[1]+2] == 'A':
            head = 'a'
        elif cn[indices[1]+2] == 'B':
            head = 'b'
        elif cn[indices[1]+2] == 'C':
            head = 'c'
        elif cn[indices[1]+2] == 'D':
            head = 'd'
        else:
            print( 'problem parsing headstage info:', cn )
        chname = cn[indices[2]+1:indices[4]]

    if area is None:
        return exptname, head, chname
    else:
        if head is None:
            fname = exptname+'_'+area+'_py.mat'
        else:
            fname = exptname+'_'+area+head+'_py.mat'
        return fname, chname
    
def HNchannels( HNlist, master_list ):
    NCtot = len(master_list)
    CHabbrev = []
    for nn in range(NCtot):
        fn, cn = parse_channel_name( master_list[nn], area='V2')
        CHabbrev.append(deepcopy(cn))
    CHlist = sorted(deepcopy(HNlist), key = lambda x: (len (x), x))
    locs = np.zeros(len(CHlist), dtype='int32')
    for nn in range(len(CHlist)):
        if CHlist[nn] in CHabbrev:
            locs[nn] = CHabbrev.index(CHlist[nn])
        else:
            locs[nn] = -1
            print( "PROBLEM: %s not found in master_list"%CHlist[nn])
    locs = locs[np.where(locs >= 0)[0]]
    return locs, CHlist