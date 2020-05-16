"""Utility functions to assist with creating, training and analyzing NDN 
models.
"""

from __future__ import division
from __future__ import print_function
import numpy as np
from copy import deepcopy
from scipy.linalg import toeplitz


def ffnetwork_params(
        input_dims=None,
        layer_sizes=None,
        layer_types=None,
        act_funcs='relu',
        time_expand=None,
        ei_layers=None,
        reg_list=None,
        xstim_n=0,
        ffnet_n=None,
        normalization=None,
        network_type='normal',
        verbose=True,
        log_activations=False,
        conv_filter_widths=None,  # the below are for convolutional network
        shift_spacing=1,
        dilation=1):
    """generates information for the network_params dict that is passed to the
    NDN constructor.
    
    Args:
        input_dims (list of ints, optional): list of the form
            [num_lags, num_x_pix, num_y_pix] that describes the input size for
            the network. If only a single dimension is needed, use
            [1, input_size, 1].
        layer_sizes (list of ints): number of subunits in each layer of
            the network. Last layer should match number of neurons (Robs). Each 
            entry can be a 3-d list, if there is a spatio-filter/temporal 
            arrangement.
        layer_types (list of strs): a string for each layer, specifying what type
            of layer is should be. Layer types include the following:
            ['normal' | 'sep' | | 'conv' | 'convsep' | 'biconv' | 'add' | 'spike_history']
        act_funcs: (str or list of strs, optional): activation function for
            network layers; replicated if a single element.
            ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 
            'quad' | 'lin'
        time_expand: (list of ints, optional): How much the input to the given layer
            should be "time_embedded" so the layer processes that many lags.
            DEFAULT is all zeros (which None also converts from)
        ei_layers (`None` or list of ints, optional): if not `None`, it should
            be a list of the number of inhibitory units for each layer other
            than the output layer, so list should be of length one less than
            layer_sizes. All the non-inhibitory units are of course excitatory,
            and having `None` for a layer means it will be unrestricted.
        reg_list (dict, optional): each key corresponds to a type of regularization
            (refer to regularization documentation for a complete list). An example
            using l2 regularization looks like:
            {'l2': [l2_layer_0_val, l2_layer_1_val, ..., l2_layer_-1_val}.
            If a single value is given like {'l2': l2_val} then that value is
            applied to all layers in the network. If the list is shorter than
            the number of layers, regularization will default to 'None'.
        normalization (list of ints, optional): specifies normalization for each
            layer, with '0' corresponding to layer default (usually no reg), and
            '1' usually tuning on. See layer-specific docs for layers with more
            complex regularization
        xstim_n (int or `None`): index into external list of input matrices
            that specifies which input to process. It should be `None` if the 
            network will be directed internally (see ffnet_n)
        ffnet_n (int or `None`): internal network that this network receives 
            input from (has to be `None` if xstim_n is not `None`)
        network_type (str, optional): specify type of network
            ['normal'] | 'sep' |
        verbose (bool, optional): `True` to print network specifications
        log_activations (bool, optional): `True` to log layer activations for
            viewing in tensorboard

        FOR convolution-specific parameters:
        conv_filter_widths (list of ints, optional): spatial dimension of
            filter (if different than stim_dims)
        shift_spacing (int, optional): stride used by convolution operation (default 1)
        dilation (int, optional): dilation used by convolution operation (default 1)
        
    Returns:
        dict: params to initialize an `FFNetwork` object
        
    Raises:
        TypeError: If `layer_sizes` is not specified
        TypeError: If both `xstim_n` and `ffnet_n` are `None`
        ValueError: If `reg_list` is a list and its length does not equal 
            the number of layers
        
    """

    if layer_sizes is None:
        raise TypeError('Must specify layer_sizes.')

    if layer_types is None:
        raise TypeError('Must explicitly specify layer_types.')

    if xstim_n is not None:
        if not isinstance(xstim_n, list):
            xstim_n = [xstim_n]
    elif ffnet_n is None:
        TypeError('Must assign some input source.')

    if network_type is 'side':
        xstim_n = None

    if ffnet_n is not None:
        if not isinstance(ffnet_n, list):
            ffnet_n = [ffnet_n]

    num_layers = len(layer_sizes)
    if len(layer_types) != num_layers:
        TypeError('Must assign the correct number of layer types.')

    # Process input_dims, if applicable
    if input_dims is not None:
        input_dims = expand_input_dims_to_3d(input_dims)

    if time_expand is None:
        time_expand = [0]*num_layers
    else:
        if not isinstance(time_expand, list):
            time_expand = [time_expand]
        if len(time_expand) < num_layers:
            time_expand += [0]*(num_layers-len(time_expand))

    # Establish positivity constraints
    pos_constraints = [None] * num_layers
    num_inh_layers = [0] * num_layers

    # Establish normalization
    norm_vals = [0] * num_layers
    if normalization is not None:
        for nn in range(len(normalization)):
            norm_vals[nn] = normalization[nn]

    if ei_layers is not None:
        for nn in range(len(ei_layers)):
            if ei_layers[nn] is not None:
                if ei_layers[nn] >= 0:
                    num_inh_layers[nn] = ei_layers[nn]
                    if nn < (num_layers-1):
                        pos_constraints[nn+1] = True
            else:
                num_inh_layers[nn] = 0
    if not isinstance(act_funcs, list):
        act_funcs = [act_funcs] * num_layers

    # Reformat regularization information into regularization for each layer
    reg_initializers = []
    for nn in range(num_layers):
        reg_initializers.append({})
        if reg_list is not None:
            #for reg_type, reg_val_list in reg_list.iteritems():  # python3 replace
            for reg_type in reg_list:
                reg_val_list = reg_list[reg_type]
                if not isinstance(reg_val_list, list):
                    if reg_val_list is not None:
                        reg_initializers[0][reg_type] = reg_val_list  # only set first value
                else:
                    if nn < len(reg_val_list):
                        if reg_val_list[nn] is not None:
                            reg_initializers[nn][reg_type] = reg_val_list[nn]

    # Make default weights initializers
    weight_inits = ['normal']*num_layers

    network_params = {
        'network_type': network_type,
        'xstim_n': xstim_n,
        'ffnet_n': ffnet_n,
        'input_dims': input_dims,
        'layer_sizes': layer_sizes,
        'layer_types': layer_types,
        'activation_funcs': act_funcs,
        'normalize_weights': norm_vals,
        'time_expand': time_expand,
        'reg_initializers': reg_initializers,
        'weights_initializers': weight_inits,
        'num_inh': num_inh_layers,
        'pos_constraints': pos_constraints,
        'log_activations': log_activations}

    # if convolutional, add the following convolution-specific fields
    if conv_filter_widths is not None:
        if not isinstance(conv_filter_widths, list):
            conv_filter_widths = [conv_filter_widths]
        while len(conv_filter_widths) < num_layers:
            conv_filter_widths.append(None)
    else:
        conv_filter_widths = [None]*num_layers

    network_params['conv_filter_widths'] = conv_filter_widths

    if shift_spacing is not None:
        if not isinstance(shift_spacing, list):
            shift_spacing = [shift_spacing]*num_layers
        while len(shift_spacing) < num_layers:
            shift_spacing.append(None)
    else:
        shift_spacing = [1]*num_layers
    network_params['shift_spacing'] = shift_spacing

    if dilation is not None:
        if not isinstance(dilation, list):
            dilation = [dilation]*num_layers
        while len(dilation) < num_layers:
            dilation.append(None)
    else:
        dilation = [1]*num_layers
    network_params['dilation'] = dilation

    if verbose:
        if input_dims is not None:
            print('Input dimensions: ' + str(input_dims))
        for nn in range(num_layers):
            s = str(nn) + ': ' + layer_types[nn] + ' (' + act_funcs[nn] + '):  \t[E' + \
                str(layer_sizes[nn]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + '] '
            if norm_vals[nn] != 0:
                s += 'N'  # + str(norm_vals[nn])
            if pos_constraints[nn]:
                s += '+'
            if conv_filter_widths[nn] is not None:
                s += '  \tfilter width = ' + str(conv_filter_widths[nn])
            print(s)
    return network_params
# END FFNetwork_params


def expand_input_dims_to_3d(input_size):
    """Utility function to turn inputs into 3-d vectors"""

    if not isinstance(input_size, list):
        input3d = [input_size, 1, 1]
    else:
        input3d = input_size[:]
    while len(input3d) < 3:
        input3d.append(1)

    return input3d


def concatenate_input_dims(parent_input_size, added_input_size):
    """Utility function to concatenate two sets of input_dims vectors
    -- parent_input_size can be none, if added_input_size is first
    -- otherwise its assumed parent_input_size is already 3-d, but
    added input size might have to be formatted.
    
    Args:
        parent_input_size (type): description
        added_input_size (type): description
        
    Returns:
        type: description
        
    Raises:
    
    """

    cat_dims = expand_input_dims_to_3d(added_input_size)

    if parent_input_size is not None:
        # Sum full vector along the first dimension ("filter" dimension)
        assert parent_input_size[1] == cat_dims[1], \
            'First dimension of inputs do not agree.'
        assert parent_input_size[2] == cat_dims[2], \
            'Last dimension of inputs do not agree.'
        cat_dims[0] += parent_input_size[0]

    return cat_dims


def shift_mat(m, sh, dim, zpad=True):
    """Modern version of shift_mat_zpad, good up to 4-dimensions, with z-pad as option"""
    shm = np.roll(m.copy(), sh, dim)
    if zpad:
        assert m.ndim < 5, 'Cannot do more than 4 dimensions.'
        L = m.shape[dim]
        if abs(sh) >= L:
            return np.zeros(m.shape)
        if sh < 0:
            ztar = range(L-abs(sh),L)
        else:
            ztar = range(abs(sh))
        if m.ndim == 1:
            if dim == 0:
                shm[ztar] = 0
        elif m.ndim == 2:
            if dim == 0:
                shm[ztar, :] = 0
            else:
                shm[:, ztar] = 0
        elif m.ndim == 3:
            if dim == 0:
                shm[ztar, :, :] = 0
            elif dim == 1:
                shm[:, ztar, :] = 0
            elif dim == 2:
                shm[:, :, ztar] = 0
        elif m.ndim == 3:
            if dim == 0:
                shm[ztar ,: ,:] = 0
            elif dim == 1:
                shm[:, ztar, :] = 0
            elif dim == 2:
                shm[: ,: ,ztar] = 0
        elif m.ndim == 4:
            if dim == 0:
                shm[ztar ,: ,: ,:] = 0
            elif dim == 1:
                shm[:, ztar, :, :] = 0
            elif dim == 2:
                shm[:, :, ztar, :] = 0
            elif dim == 3:
                shm[:, :, :, ztar] = 0
    return shm


def shift_mat_zpad(x, shift, dim=0):
    """Takes a vector or matrix and shifts it along dimension dim by amount 
    shift using zero-padding. Positive shifts move the matrix right or down.
    
    Args:
        x (type): description
        shift (type): description
        dim (type): description
        
    Returns:
        type: description
            
    Raises:
            
    """

    assert x.ndim < 3, 'only works in 2 dims or less at the moment.'
    if x.ndim == 1:
        oneDarray = True
        xcopy = np.zeros([len(x), 1])
        xcopy[:, 0] = x
    else:
        xcopy = x.copy()
        oneDarray = False
    sz = list(np.shape(xcopy))

    if sz[0] == 1:
        dim = 1

    if dim == 0:
        if shift >= 0:
            a = np.zeros((shift, sz[1]))
            b = xcopy[0:sz[0]-shift, :]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((-shift, sz[1]))
            b = xcopy[-shift:, :]
            xshifted = np.concatenate((b, a), axis=dim)
    elif dim == 1:
        if shift >= 0:
            a = np.zeros((sz[0], shift))
            b = xcopy[:, 0:sz[1]-shift]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((sz[0], -shift))
            b = xcopy[:, -shift:]
            xshifted = np.concatenate((b, a), axis=dim)

    # If the shift in one direction is bigger than the size of the stimulus in
    # that direction return a zero matrix
    if (dim == 0 and abs(shift) > sz[0]) or (dim == 1 and abs(shift) > sz[1]):
        xshifted = np.zeros(sz)

    # Make into single-dimension if it started that way
    if oneDarray:
        xshifted = xshifted[:,0]

    return xshifted
# END shift_mat_zpad


def create_time_embedding(stim, pdims, up_fac=1, tent_spacing=1):
    """All the arguments starting with a p are part of params structure which I 
    will fix later.
    
    Takes a Txd stimulus matrix and creates a time-embedded matrix of size 
    Tx(d*L), where L is the desired number of time lags. If stim is a 3d array, 
    the spatial dimensions are folded into the 2nd dimension. 
    
    Assumes zero-padding.
     
    Optional up-sampling of stimulus and tent-basis representation for filter 
    estimation.
    
    Note that xmatrix is formatted so that adjacent time lags are adjacent 
    within a time-slice of the xmatrix, thus x(t, 1:nLags) gives all time lags 
    of the first spatial pixel at time t.
    
    Args:
        stim (type): simulus matrix (time must be in the first dim).
        pdims (list/array): length(3) list of stimulus dimensions
        up_fac (type): description
        tent_spacing (type): description
        
    Returns:
        numpy array: time-embedded stim matrix
        
    """

    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))

    # No support for more than two spatial dimensions
    if len(sz) > 3:
        print('More than two spatial dimensions not supported, but creating' +
              'xmatrix anyways...')

    # Check that the size of stim matches with the specified stim_params
    # structure
    if np.prod(pdims[1:]) != sz[1]:
        print('Stimulus dimension mismatch')
        raise ValueError

    modstim = stim.copy()
    # Up-sample stimulus if required
    if up_fac > 1:
        # Repeats the stimulus along the time dimension
        modstim = np.repeat(modstim, up_fac, 0)
        # Since we have a new value for time dimension
        sz = list(np.shape(modstim))

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append(
            np.arange(1, tent_spacing) / tent_spacing,
            1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + \
                            shift_mat_zpad(modstim,
                                           ii-tent_spacing+1,
                                           0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing

    # If tent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using
    # tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).
    # Since isempty(tent_spa...) is equivalent to its value being 1 I added
    # this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])),
                        np.concatenate((modstim[0], np.zeros(pdims[0] - 1)),
                                       axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros((sz[0], np.prod(pdims)))
        for lag in range(pdims[0]):
            for xx in range(0, sz[1]):
                xmat[:, xx*pdims[0]+lag] = shift_mat_zpad(
                    modstim[:, xx], lag_spacing * lag, 0)

    return xmat
# END create_time_embedding


def create_NL_embedding(stim, bounds):
    """Format for applying input nonlinearity to stimulus"""
    NT, NF = stim.shape
    NNL = len(bounds)
    NLstim = np.zeros([NT, NF, NNL])
    for nn in range(NNL):
        tmp = np.zeros([NT, NF])
        a = np.where(stim < bounds[nn])
        b = np.where(stim >= bounds[nn])
        if nn == 0:
            tmp[a] = 1
        else:
            tmp[a] = np.add(np.add(stim[a], -bounds[nn])/(bounds[nn]-bounds[nn-1]), 1)
        if nn < NNL-1:
            tmp[b] = np.add(np.add(-stim[b], bounds[nn])/(bounds[nn+1]-bounds[nn]), 1)
        else:
            tmp[b] = 1
        tmp = np.maximum(tmp, 0)
        NLstim[:, :, nn] = tmp.copy()
    print("Generated NLstim: %d x %d x %d" % (NT, NF, NNL))
    return np.reshape(NLstim, [NT, NF*NNL])


def reg_path(
        ndn_mod=None,
        input_data=None,
        output_data=None,
        train_indxs=None,
        test_indxs=None,
        blocks=None,
        reg_type='l1',
        reg_vals=[1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1],
        ffnet_target=0,
        layer_target=0,
        data_filters=None,
        opt_params=None,
        fit_variables=None,
        learning_alg = 'adam',
        output_dir=None,
        cumulative=False,
        silent=True):
    """perform regularization over reg_vals to determine optimal cross-validated loss

        Args:

        Returns:
            dict: params to initialize an `FFNetwork` object

        Raises:
            TypeError: If `layer_sizes` is not specified
    """

    if ndn_mod is None:
        raise TypeError('Must specify NDN to regularize.')
    if input_data is None:
        raise TypeError('Must specify input_data.')
    if output_data is None:
        raise TypeError('Must specify output_data.')
    if train_indxs is None:
        raise TypeError('Must specify training indices.')
    if test_indxs is None:
        raise TypeError('Must specify testing indices.')

    num_regs = len(reg_vals)

    LLxs = np.zeros([num_regs], dtype='float32')
    test_mods = []

    for nn in range(num_regs):
        if not silent:
            if cumulative:
                print('\nCumulative regularization test: %s = %s:\n' % (reg_type, str(reg_vals[nn])))
            else:
                print('\nRegularization test: %s = %s:\n' % (reg_type, str(reg_vals[nn])))

        if not cumulative or (nn == 0):
            test_mod = ndn_mod.copy_model()  # start from same base_model
        # otherwise will continue with same test model

        if isinstance(layer_target, list):
            for mm in range(len(layer_target)):
                test_mod.set_regularization(reg_type, reg_vals[nn], ffnet_target, layer_target[mm])
        else:
            test_mod.set_regularization(reg_type, reg_vals[nn], ffnet_target, layer_target)
        test_mod.train(input_data=input_data, output_data=output_data, silent=silent,
                       train_indxs=train_indxs, test_indxs=test_indxs, blocks=blocks,
                       data_filters=data_filters, fit_variables=fit_variables,
                       learning_alg=learning_alg, opt_params=opt_params, output_dir=output_dir)
        LLxs[nn] = np.mean(
            test_mod.eval_models(input_data=input_data, output_data=output_data, blocks=blocks,
                                 data_indxs=test_indxs, data_filters=data_filters))
        test_mods.append(test_mod.copy_model())
        print('%s (%s = %s): %s' % (nn, reg_type, reg_vals[nn], LLxs[nn]))

    return LLxs, test_mods
# END reg_path


def generate_spike_history(robs, nlags, neg_constraint=True, reg_par=0,
                           xstim_n=1):
    """Will generate X-matrix that contains Robs information for each cell. It will
    use the default resolution of robs, and simply go back a certain number of lags.
    To have it applied to the corresponding neuron, will need to use add_layers"""

    NC = robs.shape[1]

    Xspk = create_time_embedding(shift_mat_zpad(robs, 1, dim=0), pdims=[nlags, NC, 1])
    ffnetpar = ffnetwork_params(layer_sizes=[NC], input_dims=[nlags, NC, 1],
                                act_funcs='lin', reg_list={'d2t': reg_par},
                                xstim_n=xstim_n, verbose=False,
                                network_type='spike_history')
    # Constrain spike history terms to be negative
    ffnetpar['pos_constraints'] = [neg_constraint]
    ffnetpar['num_inh'] = [NC]
    ffnetpar['layer_types'] = ['spike_history']

    return Xspk, ffnetpar
# END generate_spike_history


def process_blocks(block_inds, data_filters, batch_size=2000, skip=20):
    """processes blocked-stimuli for train. Note that it assumes matlab indexing (starting with 1)"""

    if skip is None:
        print("WARNING: no time-spread entered for using blocks. Setting to 12.")
        skip = 12

    mod_df = deepcopy(data_filters)
    if len(mod_df) > 1:  # then multiple outputs and data-filters (to deal with layer
        print('WARNING: Multiple data_filters not implemented in block-processing.')

    val_inds = np.zeros(mod_df[0].shape[0])
    num_blocks = block_inds.shape[0]
    av_size = 0
    block_lists = []
    for nn in range(num_blocks):
        val_inds[range(block_inds[nn, 0]+skip-1, block_inds[nn, 1])] = 1.0
        av_size += np.max([block_inds[nn, 1] - block_inds[nn, 0] - skip+1, 0])
        block_lists.append(np.array(range(block_inds[nn, 0]-1, block_inds[nn, 1]), dtype='int'))

    comb_number = np.round(batch_size/av_size*num_blocks).astype(int)
    for nn in range(len(data_filters)):
        mod_df[nn] = data_filters[nn]*np.expand_dims(val_inds,1)

    return block_lists, mod_df, comb_number


def make_block_indices( block_lims, lag_skip=0):
    """return all the indices within the blocks passed. Similar to function above: perhaps
    could be combined in the future if above could process things less complicated."""

    block_inds = np.array([])
    for nn in range(block_lims.shape[0]):
        if block_lims[nn,0]-1+lag_skip < block_lims[nn,1]:
            block_inds = np.concatenate((block_inds, np.array(
                range(block_lims[nn,0]-1+lag_skip, block_lims[nn,1]))), axis=0)
    return block_inds.astype(int)



def generate_xv_folds(nt, num_folds=5, num_blocks=3, which_fold=None):
    """Will generate unique and cross-validation indices, but subsample in each block
        NT = number of time steps
        num_folds = fraction of data (1/fold) to set aside for cross-validation
        which_fold = which fraction of data to set aside for cross-validation (default: middle of each block)
        num_blocks = how many blocks to sample fold validation from"""

    test_inds = []
    NTblock = np.floor(nt/num_blocks).astype(int)
    block_sizes = np.zeros(num_blocks, dtype='int32')
    block_sizes[range(num_blocks-1)] = NTblock
    block_sizes[num_blocks-1] = nt-(num_blocks-1)*NTblock

    if which_fold is None:
        which_fold = num_folds//2
    else:
        assert which_fold < num_folds, 'Must choose XV fold within num_folds =' + str(num_folds)

    # Pick XV indices for each block
    cnt = 0
    for bb in range(num_blocks):
        tstart = np.floor(block_sizes[bb] * (which_fold / num_folds))
        if which_fold < num_folds-1:
            tstop = np.floor(block_sizes[bb] * ((which_fold+1) / num_folds))
        else: 
            tstop = block_sizes[bb]

        test_inds = test_inds + list(range(int(cnt+tstart), int(cnt+tstop)))
        cnt = cnt + block_sizes[bb]

    test_inds = np.array(test_inds, dtype='int')
    train_inds = np.setdiff1d(np.arange(0, nt, 1), test_inds)

    return train_inds, test_inds


def spikes_to_robs(spks, num_time_pts, dt):
    """
    Description
    
    Args:
        spks (type): description
        num_time_pts (type): description
        dt (type): description
        
    Returns:
        type: description

    """

    bins_to_use = range(num_time_pts + 1) * dt
    robs, bin_edges = np.histogram(spks.flatten(), bins=bins_to_use.flatten())
    robs = np.expand_dims(robs, axis=1)

    return robs


def tent_basis_generate( xs=None, num_params=None, doubling_time=None, init_spacing=1, first_lag=0 ):
    """Computes tent-bases over the range of 'xs', with center points at each value of 'xs'
    Alternatively (if xs=None), will generate a list with init_space and doubling_time up to
    the total number of parameters. Must specify xs OR num_params. 
    
    Defaults:
        doubling_time = num_params
        init_space = 1"""

    # Determine anchor-points
    if xs is not None:
        tbx = np.array(xs,dtype='int32')
        if num_params is not None: 
            print( 'Warning: will only use xs input -- num_params is ignored.' )
    else:
        assert num_params is not None, 'Need to specify either xs or num_params'
        if doubling_time is None:
            doubling_time = num_params+1  # never doubles
        tbx = np.zeros( num_params, dtype='int32' )
        cur_loc, cur_spacing, sp_count = first_lag, init_spacing, 0
        for nn in range(num_params):
            tbx[nn] = cur_loc
            cur_loc += cur_spacing
            sp_count += 1
            if sp_count == doubling_time:
                sp_count = 0
                cur_spacing *= 2

    # Generate tent-basis given anchor points
    NB = len(tbx)
    NX = (np.max(tbx)+1).astype(int)
    tent_basis = np.zeros([NX,NB], dtype='float32')
    for nn in range(NB):
        if nn > 0:
            dx = tbx[nn]-tbx[nn-1]
            tent_basis[range(tbx[nn-1], tbx[nn]+1), nn] = np.array(list(range(dx+1)))/dx
        elif tbx[0] > 0:  # option to have function go to zero at beginning
            dx = tbx[0]
            tent_basis[range(tbx[nn]+1), nn] = np.array(list(range(dx+1)))/dx
        if nn < NB-1:
            dx = tbx[nn+1]-tbx[nn]
            tent_basis[range(tbx[nn], tbx[nn+1]+1), nn] = 1-np.array(list(range(dx+1)))/dx

    return tent_basis


######## GPU picking ########
import subprocess, re, os, sys

def run_command(cmd):
    """Run command, return output as string."""

    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""

    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def setup_one_gpu(gpu_choice=None):
    assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    if gpu_choice is None:
        print('\n---> setting up GPU with largest available memory:')
        gpu_id = pick_gpu_lowest_memory()
    else:
        gpu_id = gpu_choice
    print("   ...picking GPU # " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def setup_no_gpu():
    if 'tensorflow' in sys.modules:
        print("Warning, GPU setup must happen before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = ''


def assign_gpu(gpu_choice=None):
    print('*******************************************************************************************')

    print('---> getting list of available GPUs:')
    print(list_available_gpus())
    print('\n---> getting GPU memory map:')
    print(gpu_memory_map())
        
    setup_one_gpu(gpu_choice=gpu_choice)

    print('*******************************************************************************************')

    print('\nDone!')
    return pick_gpu_lowest_memory()
