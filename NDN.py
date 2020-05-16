"""Neural deep network"""

from __future__ import print_function
from __future__ import division
from copy import deepcopy

import os
import warnings
import shutil

import numpy as np
import tensorflow as tf

from .ffnetwork import FFNetwork
from .ffnetwork import SideNetwork
from .NDNutils import concatenate_input_dims
from .NDNutils import process_blocks
from .NDNutils import make_block_indices

class NDN(object):
    """Tensorflow (tf) implementation of Neural Deep Network class

    Attributes:
        num_networks (int): number of FFNetworks in model
        network_list (list of `FFNetwork` objects): list containing all
            FFNetworks in model
        num_input_streams (int): number of distinct input matrices processed by
            the various FFNetworks in model
        ffnet_out (list of ints): indices into network_list of all FFNetworks
            whose outputs are used in calculating the cost function
        input_sizes (list of lists): list of the form
            [num_lags, num_x_pix, num_y_pix] that describes the input size for
            each input stream
        output_sizes (list of ints): number of output units for each output
            stream
        noise_dist (str): specifies the probability distribution used to define
            the cost function
            ['poisson'] | 'gaussian' | 'bernoulli'
        tf_seed (int): rng seed for both tensorflow and numpy, which allows
            for reproducibly random initializations of parameters
        cost
        unit_cost
        cost_reg
        cost_penalized
        graph (tf.Graph object): tensorflow graph of the model
        saver (tf.train.Saver object): for saving and restoring models
        merge_summaries (tf.summary.merge_all op): for collecting all summaries
        init (tf.global_variables_initializer op): for initializing variables
        sess_config (tf.ConfigProto object): specifies configurations for
            tensorflow session, such as GPU utilization

    Notes:
        One assumption is that the output of all FFnetworks -- whether or not
        they are multi-dimensional -- project down into one dimension when
        input into a second network. As a result, concatenation of multiple
        streams will always be in one dimension.

    """

    _allowed_learning_algs = ['adam', 'lbfgs']
    _allowed_data_pipeline_types = ['data_as_var', 'feed_dict', 'iterator']
    _log_min = 1e-6  # constant to add to all arguments to logarithms

    _allowed_noise_dists = ['gaussian', 'poisson', 'bernoulli']
    _allowed_network_types = ['normal', 'side']
    _allowed_layer_types = ['normal', 'conv', 'sep', 'convsep', 'add', 'biconv', 'spike_history']

    def __init__(
            self,
            network_list=None,
            noise_dist='poisson',
            ffnet_out=-1,
            input_dim_list=None,
            batch_size=None,
            tf_seed=0):
        """Constructor for network-NIM class

        Args:
            network_list (list of dicts): created using
                `NDNutils.FFNetwork_params`
            noise_dist (str, optional): specifies the probability distribution
                used to define the cost function
                ['poisson'] | 'gaussian' | 'bernoulli'
            ffnet_out (int or list of ints, optional): indices into
                network_list that specifes which network outputs are used for
                the cost function; defaults to last network in `network_list`
            input_dim_list (list of lists): list of the form
                [num_lags, num_x_pix, num_y_pix] that describes the input size
                for each input stream
            tf_seed (int)

        Raises:
            TypeError: If `network_list` is not specified
            ValueError: If `input_dim_list` does not match inputs specified for
                each FFNetwork in `network_list`
            ValueError: If any element of `ffnet_out` is larger than the length
                of `network_list`

        """

        if network_list is None:
            raise TypeError('Must specify network list.')

        # from old network.py
        self.num_examples = 0
        self.filter_data = False
        # for tf.data API / 'iterator' pipeline
        self.data_pipe_type = 'data_as_var'
        self.batch_size = None
        self.dataset_types = None
        self.dataset_shapes = None
        self.poisson_unit_norm = None
        # end from old network.py

        self.batch_size = batch_size
        self.time_spread = None

        # Set network_list
        if not isinstance(network_list, list):
            network_list = [network_list]
        self.num_networks = len(network_list)
        self.network_list = deepcopy(network_list)

        # Determine number of inputs
        measured_input_list = set()
        for nn in range(self.num_networks):
            if network_list[nn]['xstim_n'] is not None:
                measured_input_list |= set(network_list[nn]['xstim_n'])
        measured_input_list = list(measured_input_list)

        if input_dim_list is None:
            input_dim_list = [None] * len(measured_input_list)
        elif len(input_dim_list) < len(measured_input_list):
                ValueError('Something_wrong with inputs')

        self.num_input_streams = len(input_dim_list)

        if not isinstance(ffnet_out, list):
            ffnet_out = [ffnet_out]
        for nn in range(len(ffnet_out)):
            if ffnet_out[nn] > self.num_networks:
                ValueError('ffnet_out has values that are too big')

        self.ffnet_out = ffnet_out[:]
        self.input_sizes = input_dim_list[:]
        # list of output sizes (for Robs placeholders)
        self.output_sizes = [0] * len(ffnet_out)
        self.noise_dist = noise_dist
        self.tf_seed = tf_seed

        self._define_network()

        # set parameters for graph (constructed for each train)
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None
    # END NDN.__init__

    def _define_network(self):
        # Create the FFnetworks

        self.networks = []

        for nn in range(self.num_networks):
            # Tabulate network inputs. Note that multiple inputs assumed to be
            # combined along 2nd dim, and likewise 1-D outputs assumed to be
            # over 'space'
            input_dims_measured = None

            if self.network_list[nn]['ffnet_n'] is not None:
                ffnet_n = self.network_list[nn]['ffnet_n']
                for mm in ffnet_n:
                    assert mm <= self.num_networks, 'Too many ffnetworks referenced.'
                    # print('network %i:' % nn, mm, input_dims_measured, self.networks[mm].layers[-1].output_dims)
                    input_dims_measured = concatenate_input_dims(
                        input_dims_measured,
                        self.networks[mm].layers[-1].output_dims)

            # Determine external inputs
            if self.network_list[nn]['xstim_n'] is not None:
                xstim_n = self.network_list[nn]['xstim_n']
                for mm in xstim_n:
                    # First see if input is not specified at NDN level
                    if self.input_sizes[mm] is None:
                        # then try to scrape from network_params
                        assert self.network_list[nn]['input_dims'] is not None, \
                            'External input size not defined.'
                        self.input_sizes[mm] = self.network_list[nn]['input_dims']
                    input_dims_measured = concatenate_input_dims(
                        input_dims_measured, self.input_sizes[mm])

            # Now specific/check input to this network
            if self.network_list[nn]['input_dims'] is None:
                if self.network_list[nn]['network_type'] != 'side':
                    self.network_list[nn]['input_dims'] = input_dims_measured
                # print('network %i:' % nn, input_dims_measured)
            else:
                # print('network %i:' % nn, network_list[nn]['input_dims'], input_dims_measured )
                assert self.network_list[nn]['input_dims'] == \
                       list(input_dims_measured), 'Input_dims dont match. network '+str(nn)

            # Build networks
            if self.network_list[nn]['network_type'] == 'side':
                assert len(self.network_list[nn]['ffnet_n']) == 1, \
                    'only one input to a side network'
                network_input_params = \
                    self.network_list[self.network_list[nn]['ffnet_n'][0]]
                self.networks.append(
                    SideNetwork(
                        scope='side_network_%i' % nn,
                        input_network_params=network_input_params,
                        params_dict=self.network_list[nn]))
            else:
                self.networks.append(
                    FFNetwork(
                        scope='network_%i' % nn,
                        params_dict=self.network_list[nn]))

            # Enforce temporal spread equal at least to input lags
            #self.networks[nn].time_spread = np.maximum(self.networks[nn].time_spread, 
            #                                    self.networks[nn].layers[0].num_lags)
            
            # Track temporal spread
            if self.networks[nn].time_spread > 0:
                if self.time_spread is None:
                    self.time_spread = self.networks[nn].time_spread
                else:
                    # For now assume serial procuessing (fix latter for parallel?)
                    self.time_spread += self.networks[nn].time_spread

        # Assemble outputs
        for nn in range(len(self.ffnet_out)):
            ffnet_n = self.ffnet_out[nn]
            self.output_sizes[nn] = np.prod(self.networks[ffnet_n].layers[-1].output_dims)
    # END NDN._define_network

    def _build_graph(
            self,
            learning_alg='adam',
            opt_params=None,
            fit_variables=None,
            batch_size=None,
            use_dropout=False):
        """NDN._build_graph"""

        # Take care of optimize parameters if necessary
        if opt_params is None:
            opt_params = self.optimizer_defaults({}, learning_alg)
        # Update batch-size as needed
        if opt_params['batch_size'] is not None:
            self.batch_size = opt_params['batch_size']

        # overwrite unit_cost_norm with opt_params value (if specified)
        if opt_params['poisson_unit_norm'] is not None:
            self.poisson_unit_norm = opt_params['poisson_unit_norm']

        self.graph = tf.Graph()  # must be initialized before graph creation

        # for specifying device
        if opt_params['use_gpu']:
            self.sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build model graph
        with self.graph.as_default():

            np.random.seed(self.tf_seed)
            tf.set_random_seed(self.tf_seed)

            # define pipeline for feeding data into model
            with tf.variable_scope('data'):
                self._initialize_data_pipeline()

            # Build network graph
            for nn in range(self.num_networks):

                if self.network_list[nn]['network_type'] == 'side':

                    # Specialized inputs to side-network
                    assert self.network_list[nn]['xstim_n'] is None, \
                        'Cannot have any external inputs into side network.'
                    assert len(self.network_list[nn]['ffnet_n']) == 1, \
                        'Can only have one network input into a side network.'
                    # Pass the entire network into the input of side network
                    input_network_n = self.network_list[nn]['ffnet_n'][0]
                    assert input_network_n < nn, \
                        'Must create network for side network first.'
                    input_cat = self.networks[input_network_n]

                else:   # assume normal network
                    # Assemble input streams -- implicitly along input axis 1
                    # (0 is T)
                    input_cat = None
                    if self.network_list[nn]['xstim_n'] is not None:
                        for ii in self.network_list[nn]['xstim_n']:
                            if input_cat is None:
                                input_cat = self.data_in_batch[ii]
                            else:
                                input_cat = tf.concat(
                                    (input_cat, self.data_in_batch[ii]),
                                    axis=1)
                    if self.network_list[nn]['ffnet_n'] is not None:
                        for ii in self.network_list[nn]['ffnet_n']:
                            if input_cat is None:
                                input_cat = \
                                    self.networks[ii].layers[-1].outputs
                            else:
                                input_cat = \
                                    tf.concat(
                                        (input_cat,
                                         self.networks[ii].layers[-1].outputs),
                                        axis=1)
                # first argument for T/FFnetworks is inputs but it is input_network for scaffold
                # so no keyword argument used below
                self.networks[nn].build_graph(input_cat, params_dict=self.network_list[nn],
                                              batch_size=batch_size, use_dropout=use_dropout)

            # Define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # Define optimization routine
            var_list = self._build_fit_variable_list(fit_variables)

            with tf.variable_scope('optimizer'):
                self._define_optimizer(
                    learning_alg=learning_alg,
                    opt_params=opt_params,
                    var_list=var_list)

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            # To use once backward compat is possible: self.saver = tf.compat.v1.train.Saver()
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            # Not backward-compatible: self.merge_summaries = tf.compat.v1.summary.merge_all()
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
    # END NDN._build_graph

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        cost = []
        self.cost_iter = [] ###
        unit_cost = []
        for nn in range(len(self.ffnet_out)):
            if self.time_spread is None:
                time_spread = 0
                #data_out = self.data_out_batch[nn]
            else:
                time_spread = self.time_spread
                #data_out = tf.slice(self.data_out_batch[nn], [self.time_spread, 0], [-1, -1])

            if self.filter_data:
                # this will zero out predictions where there is no data, matching Robs here
                pred_tmp = tf.multiply(
                    self.networks[self.ffnet_out[nn]].layers[-1].outputs,
                    self.data_filter_batch[nn])
            else:
                pred_tmp = self.networks[self.ffnet_out[nn]].layers[-1].outputs

            #pred = pred_tmp[time_spread:, :]

            if time_spread == 0:
                #nt = tf.constant(self.batch_size, dtype=tf.float32)
                data_out = self.data_out_batch[nn]
                pred = pred_tmp
                #nt = tf.cast(tf.shape(pred)[0], tf.float32)
            else:
                data_out = tf.slice(self.data_out_batch[nn], [time_spread, 0], [-1, -1])
                pred = tf.slice(pred_tmp, [time_spread, 0], [-1, -1])  # [self.batch_size-self.time_spread, -1])
                # effective_batch_size is self.batch_size - self.time_spread
            nt = tf.cast(tf.shape(pred)[0], tf.float32)  # already sliced 

            # define cost function
            if self.noise_dist == 'gaussian':
                with tf.name_scope('gaussian_loss'):
                    cost.append(tf.nn.l2_loss(data_out - pred) / nt * 2)  # x2: l2_loss gives half the norm (!)
                    #unit_cost.append(tf.reduce_mean(tf.square(data_out-pred), axis=0))
                    unit_cost.append(tf.reduce_sum(tf.square(data_out-pred), axis=0)) # unnormalized
                    # time_spread in indexing below -- slightly quicker it seems, but not by much...
                    # cost.append(tf.nn.l2_loss(data_out[time_spread:, :] - pred) / nt * 2)
                    # unit_cost.append(tf.reduce_mean(tf.square(data_out[time_spread:, :]-pred), axis=0))

            elif self.noise_dist == 'poisson':
                with tf.name_scope('poisson_loss'):

                    if self.poisson_unit_norm is not None:
                        # normalize based on rate * time (number of spikes)
                        cost_norm = tf.maximum(tf.multiply(self.poisson_unit_norm[nn], nt), 1e-8)
                    else:
                        cost_norm = nt

                    cost.append(-tf.reduce_sum(tf.divide(
                        tf.multiply(data_out, tf.log(self._log_min + pred)) - pred,
                        #tf.multiply(data_out[time_spread:, :], tf.log(self._log_min + pred)) - pred,
                        cost_norm)))

                    # unit_cost does not directly take cost_norm into account -- needs to be normed
                    unit_cost.append(-tf.reduce_sum(
                        tf.multiply(data_out, tf.log(self._log_min + pred)) - pred,
                        axis=0))

            elif self.noise_dist == 'bernoulli':
                with tf.name_scope('bernoulli_loss'):
                    # Check per-cell normalization with cross-entropy
                    # cost_norm = tf.maximum(
                    #   tf.reduce_sum(data_out, axis=0), 1)
                    cost.append(tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=data_out, logits=pred)))
                    #unit_cost.append(tf.reduce_mean(
                    unit_cost.append(tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=data_out, logits=pred), axis=0))
            else:
                TypeError('Cost function not supported.')

        self.cost = tf.add_n(cost)
        self.unit_cost = unit_cost # this is not yet normalized

        # Add regularization penalties
        reg_costs = []
        with tf.name_scope('regularization'):
            for nn in range(self.num_networks):
                reg_costs.append(self.networks[nn].define_regularization_loss())
        self.cost_reg = tf.add_n(reg_costs)

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        # with tf.variable_scope('summaries'):
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('cost_penalized', self.cost_penalized)
        tf.summary.scalar('reg_pen', self.cost_reg)
        # For future: these are not backwards-compatible, but will replace the above
        # tf.compat.v1.summary.scalar('cost', self.cost)
        # tf.compat.v1.summary.scalar('cost_penalized', self.cost_penalized)
        # tf.compat.v1.summary.scalar('reg_pen', self.cost_reg)
    # END NDN._define_loss

    def _assign_model_params(self, sess):
        """Functions assigns parameter values to randomly initialized model"""
        with self.graph.as_default():
            for nn in range(self.num_networks):
                self.networks[nn].assign_model_params(sess)

    def _write_model_params(self, sess):
        """Pass write_model_params down to the multiple networks"""
        for nn in range(self.num_networks):
            self.networks[nn].write_model_params(sess)

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates
        parameter values"""
        with self.graph.as_default():
            for nn in range(self.num_networks):
                self.networks[nn].assign_reg_vals(sess)

    def _build_fit_variable_list(self, fit_parameter_list):
        """Generates variable list to fit if argument is not none.
        'fit_parameter_list' is generated by a """
        var_list = None
        if fit_parameter_list is not None:
            var_list = []
            for nn in range(self.num_networks):
                var_list += self.networks[nn].build_fit_variable_list(
                    fit_parameter_list[nn])
        return var_list
    # END _generate_variable_list

    def fit_variables(self, layers_to_skip=None, fit_biases=False):
        """Generates a list-of-lists-of-lists of correct format to specify all
        the variables to fit, as an argument for network.train

        Args:
            layers_to_skip (list of lists, optional): list of layers to skip
                for each network. If just single list, defaults to skipping
                those layers in the first network
            fit_biases (bool or list of bools): specify whether or not to fit
                biases for each network
                DEFAULT: False

        """

        if layers_to_skip is None:
            layers_to_skip = [[]] * self.num_networks
        else:
            # Assume a single list is referring to the first network by default
            if not isinstance(layers_to_skip, list):
                layers_to_skip = [[layers_to_skip]]
            else:
                if not isinstance( layers_to_skip[0], list):
                    # then assume just meant for first network
                    layers_to_skip = [layers_to_skip]

        if not isinstance(fit_biases, list):
            fit_biases = [fit_biases]*self.num_networks
        # here assume a single list is referring to each network
        assert len(fit_biases) == self.num_networks, \
            'fit_biases must be a list over networks'
        for nn in range(self.num_networks):
            if not isinstance(fit_biases[nn], list):
                fit_biases[nn] = [fit_biases[nn]]*len(self.networks[nn].layers)
        fit_list = [[]]*self.num_networks
        for nn in range(self.num_networks):
            fit_list[nn] = [{}]*self.networks[nn].num_layers
            for layer in range(self.networks[nn].num_layers):
                fit_list[nn][layer] = {'weights': True, 'biases': False}
                fit_list[nn][layer]['biases'] = fit_biases[nn][layer]
                if nn < len(layers_to_skip):
                    if layer in layers_to_skip[nn]:
                        fit_list[nn][layer]['weights'] = False
                        fit_list[nn][layer]['biases'] = False
        return fit_list
        # END NDN.set_fit_variables

    def set_partial_fit(self, ffnet_target=0, layer_target=None, value=None):
        """ Assign partial_fit values

        Args:
            ffnet_target (int): which network
            layer_target (int): which layer
            value (0, 1, or None): partial_fit value to be assigned
                - 0 ---> fit only temporal part
                - 1 ---> fit only spatial part
                - anything else ---> fit everything
        """

        if value == 0:
            translation = 'only --temporal--'
        elif value == 1:
            translation = 'only --spatial--'
        else:
            translation = '--everything--'

        if type(self.networks[ffnet_target].layers[layer_target]) is SepLayer or ConvSepLayer:
            self.networks[ffnet_target].layers[layer_target].partial_fit = value
            self.networks[ffnet_target].layers[layer_target].reg.partial_fit = value

            print('....partial_fit value for --net%sL%s-- set to %s, %s will be fit'
                  % (ffnet_target, layer_target, value, translation))
        else:
            raise ValueError('partial fit should be used only with Seplayer families.')
    # END NDN.set_partial_fit

    def set_regularization(self, reg_type, reg_val, ffnet_target=0, layer_target=None):
        """Add or reassign regularization values

        Args:
            reg_type (str): see allowed_reg_types in regularization.py
            reg_val (int): corresponding regularization value
            ffnet_target (int): which network to assign regularization to
                DEFAULT: 0
            layer_target (int or list of ints): specifies which layers the
                current reg_type/reg_val pair is applied to
                DEFAULT: all layers in ffnet_target

        """

        if layer_target is None:
            # set all layers
            for nn in range(self.num_networks):
                layer_target = range(self.networks[nn].num_layers)
        elif not isinstance(layer_target, list):
                layer_target = [layer_target]

        # set regularization at the layer level
        for layer in layer_target:
            self.networks[ffnet_target].layers[layer].set_regularization(
                reg_type, reg_val)
    # END set_regularization

    def get_LL(self, input_data, output_data, data_indxs=None,
               data_filters=None, use_gpu=False):
        """Get cost from loss function and regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            data_filters (numpy array, optional):
            use_gpu (boolean, optional): whether or not to use GPU for
                computation (default=False)
        Returns:
            float: cost function evaluated on `input_data`

        """

        # check input
        input_data, output_data, data_filters = self._data_format(input_data, output_data, data_filters)
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # build datasets if using 'iterator' pipeline
        if self.data_pipe_type == 'iterator':
            dataset = self._build_dataset(
                input_data=input_data,
                output_data=output_data,
                data_filters=data_filters,
                indxs=data_indxs,
                training_dataset=False,
                batch_size=self.num_examples)
            # store info on dataset for buiding data pipeline
            self.dataset_types = dataset.output_types
            self.dataset_shapes = dataset.output_shapes
            # build iterator object to access elements from dataset
            # iterator_tr = dataset.make_one_shot_iterator()

        # Potentially place graph operations on CPU
        if not use_gpu:
            #temp_config = tf.ConfigProto(device_count={'GPU': 0})
            with tf.device('/cpu:0'):
                self._build_graph()
        else:
            #temp_config = tf.ConfigProto(device_count={'GPU': 1})
            self._build_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data, data_filters=data_filters)

            cost_tr = self._get_test_cost(
                sess=sess,
                input_data=input_data,
                output_data=output_data,
                data_filters=data_filters,
                test_indxs=data_indxs,
                test_batch_size=self.batch_size)

        return cost_tr
    # END get_LL

    def eval_models(self, input_data=None, output_data=None, data_indxs=None, blocks=None,
                    data_filters=None, nulladjusted=False, use_gpu=False, use_dropout=False):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            data_filters (time x output_dim numpy array, optional): array of zeros and ones showing
                where output_data is valid
            blocks (num_blocks x 2): if data is broken into blocks, then corresponds to start- and
                end-indices of each block in data. If set, then interpret data_indxs as numbering
                the blocks to use (rather than indices themselves)
            nulladjusted (bool): subtracts the log-likelihood of a "null" model that has constant
                [mean] firing rate
            use_gpu (boolean, default False): forces eval_models on CPU (to avoid potential memory
                issues on GPU that lead to kernel crashing)
            use_dropout (boolean, default False): whether to use dropout settings

        Returns:
            numpy array: value of log-likelihood for each unit in model. For
                Poisson noise distributions, if not null-adjusted, this will
                return the *negative* log-likelihood. Null-adjusted will be
                positive if better than the null-model.
        """

        # check input
        input_data, output_data, mod_df = self._data_format(input_data, output_data, data_filters)
        var_batch_size = self._variable_batch_size_permitted()
        if not var_batch_size:
            assert blocks is None, 'Cannot use blocks: variable batch size not permitted.'
            
        # arrange parameters that affect data processing
        if self.time_spread is None:
            time_spread = 0
        else:
            time_spread = self.time_spread

        if blocks is None:
            if data_indxs is None:
                data_indxs = np.arange(self.num_examples)

            # Handle constant batch-size
            if self.batch_size is None:
                batch_size = data_indxs.shape[0]
                # note this could crash if batch_size too large. but crash cause should be clear...
            else:
                batch_size = np.minimum(self.batch_size, data_indxs.shape[0])
            num_batches_test = data_indxs.shape[0] // batch_size

        else:  # process block information (variable batch-size)
            self.filter_data = True
            batch_size = None
            if data_indxs is None:
                data_indxs = np.arange(blocks.shape[0])
            if self.time_spread > 0:
                block_lists, mod_df, _ = process_blocks(blocks, mod_df, skip=self.time_spread)
            else:  # enter default time spread in between blocks
                print("WARNING: no time-spread entered for using blocks. Setting to 20.")
                self.time_spread = 12
            # data_indxs is number of blocks    
            num_batches_test = len(data_indxs)
   
        # build datasets if using 'iterator' pipeline -- curently not operating...
        #if self.data_pipe_type == 'iterator':
        #    dataset = self._build_dataset(
        #        input_data=input_data,
        #        output_data=output_data,
        #        data_filters=data_filters,
        #        indxs=data_indxs,
        #        training_dataset=False,
        #        batch_size=self.num_examples)
        #    # store info on dataset for buiding data pipeline
        #    self.dataset_types = dataset.output_types
        #    self.dataset_shapes = dataset.output_shapes
        #    # build iterator object to access elements from dataset
        #    iterator = dataset.make_one_shot_iterator()

        # Place graph operations on CPU
        if not use_gpu:
            with tf.device('/cpu:0'):
                self._build_graph(batch_size=batch_size, use_dropout=use_dropout)
        else:
            self._build_graph(batch_size=batch_size, use_dropout=use_dropout)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self._restore_params(sess, input_data, output_data, data_filters=mod_df)

            for batch_test in range(num_batches_test):
                if blocks is None:
                    batch_indxs_test = data_indxs[batch_test*batch_size:(batch_test+1)*batch_size]
                    # Incorporate time-spread for constant batch size by zeroing-out ignored data 
                    # for normalization purposes (below).
                    if time_spread > 0:
                        for nn in range(len(self.ffnet_out)):
                            mod_df[nn][
                                data_indxs[batch_test*batch_size:
                                            (batch_test*batch_size+time_spread)], :] = 0  
                else:
                    # note for blocks, data_indxs is actually numbering the blocks to use
                    # Also note: mod_df already takes time_spread into account above
                    batch_indxs_test = block_lists[data_indxs[batch_test]]

                if self.data_pipe_type == 'data_as_var':
                    feed_dict = {self.indices: batch_indxs_test}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=mod_df,
                        batch_indxs=batch_indxs_test)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: data_indxs}
 
                if batch_test == 0:
                    unit_cost = sess.run(self.unit_cost, feed_dict=feed_dict)
                else:
                    unit_cost = np.add(unit_cost, sess.run(self.unit_cost, feed_dict=feed_dict))
 
            # Add fractional batch at end
            if blocks is None:
                if var_batch_size and (data_indxs.shape[0]-num_batches_test*batch_size) > time_spread:
                    batch_indxs_test = data_indxs[range(num_batches_test*batch_size, data_indxs.shape[0])]
                    feed_dict = {self.indices: batch_indxs_test}
                    unit_cost = np.add(unit_cost, sess.run(self.unit_cost, feed_dict=feed_dict))
                else:
                    # Zero-out fractional-block data at end that is not used in unit_cost calc
                    for nn in range(len(self.ffnet_out)):
                        mod_df[nn][data_indxs[range(num_batches_test*batch_size, data_indxs.shape[0])], :] = 0
            # thus far, this is the summed (not normalized) LL for each neuron

        # Normalize unit_cost dependent on noise_distributions 
        ll_neuron = []
        for nn in range(len(self.ffnet_out)):
            if self.noise_dist == 'gaussian':  # normalize by length of non-zero Robs
                unit_norms = np.sum(mod_df[nn][data_indxs, :], axis=0)
            else:  # normalize by number of spikes for Poisson or Bernoulli
                if blocks is None:
                    unit_norms = np.sum(np.multiply(
                            output_data[nn][data_indxs, :], 
                            mod_df[nn][data_indxs, :].astype('float32')), 
                        axis=0)
                else:
                    indxs_test = make_block_indices(blocks[data_indxs, :])
                    unit_norms = np.sum( np.multiply(output_data[nn][indxs_test, :], 
                            mod_df[nn][indxs_test, :].astype('float32')), axis=0) 
                    #unit_norms = np.multiply(deepcopy(self.poisson_unit_norm), len(data_indxs))
                    #np.sum(mod_df[nn][data_indxs, :], axis=0)) # default only normalized by total time vs. amt of data
            ll_neuron.append(np.divide(np.squeeze(unit_cost[nn]), np.maximum(unit_norms, 1)))   
        
        if nulladjusted:
            # note that ll_neuron is negative of the true log-likelihood,
            # but get_null_ll is not (so + is actually subtraction)
            for nn, temp_data in enumerate(output_data):
                if blocks is None:
                    null_lls = self.get_null_ll(temp_data[data_indxs, :], mod_df[nn][data_indxs, :])
                else: 
                    indxs_test = make_block_indices(blocks[data_indxs, :], lag_skip=self.time_spread)
                    null_lls = self.get_null_ll(temp_data[indxs_test, :], mod_df[nn][indxs_test, :])

                ll_neuron[nn] = -ll_neuron[nn] - null_lls

        if len(output_data) == 1:
            return ll_neuron[0]
        else:
            return ll_neuron
# END NDN.eval_models

    def generate_prediction(self, input_data, data_indxs=None, use_gpu=False,
                            ffnet_target=-1, layer_target=-1, use_dropout=False, pre_activation=False):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used
            use_gpu (True or False): Obvious
            ffnet_target (int, optional): index into `network_list` that specifies
                which FFNetwork to generate the prediction from
            layer_target (int, optional): index into layers of network_list[ffnet_target]
                that specifies which layer to generate prediction from
            use_dropout (Boolean): whether to use dropout, default is False
            pre_activation (Boolean): whether to use the output before the activation function

        Returns:
            numpy array: pred values from network_list[ffnet_target].layers[layer]

        Raises:
            ValueError: If `layer` index is larger than number of layers in
                network_list[ffnet_target]

        """

        # Set up model to handle prediction
        self.filter_data = False

        # validate function inputs (includes generating dummy output_data)
        input_data, output_data, _ = self._data_format(input_data, None, None)

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)
        if layer_target >= len(self.networks[ffnet_target].layers):
                ValueError('This layer does not exist.')
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # change data_pipe_type to feed_dict
        # original_pipe_type = deepcopy(self.data_pipe_type)
        # self.data_pipe_type = 'data_as_var'

        if (self.batch_size is None) or (self.batch_size > data_indxs.shape[0]):
            batch_size_save = self.batch_size
            self.batch_size = data_indxs.shape[0]
        else:
            batch_size_save = None

        # Get prediction for complete range
        #num_batches_test = data_indxs.shape[0] // self.batch_size
        num_batches_test = np.ceil(data_indxs.shape[0]/self.batch_size).astype(int)

        # Place graph operations on CPU
        if not use_gpu:
            temp_config = tf.ConfigProto(device_count={'GPU': 0})
            with tf.device('/cpu:0'):
                self._build_graph(batch_size=self.batch_size, use_dropout=use_dropout)
        else:
            temp_config = tf.ConfigProto(device_count={'GPU': 1})
            self._build_graph(batch_size=self.batch_size, use_dropout=use_dropout)

        with tf.Session(graph=self.graph, config=temp_config) as sess:

            self._restore_params(sess, input_data, output_data)
            t0 = 0  # default unless there is self.time_spread

            for batch_test in range(num_batches_test):

                if self.time_spread is None:
                    indx_beg = batch_test * self.batch_size
                else:
                    if self.time_spread < batch_test * self.batch_size:
                        indx_beg = batch_test * self.batch_size - self.time_spread
                        t0 = self.time_spread
                    else:
                        indx_beg = 0
                        t0 = batch_test * self.batch_size

                indx_end = (batch_test + 1) * self.batch_size
                if indx_end > data_indxs.shape[0]:
                    indx_end = data_indxs.shape[0]

                batch_indxs_test = data_indxs[indx_beg:indx_end]
                if self.data_pipe_type == 'data_as_var':
                    feed_dict = {self.indices: batch_indxs_test}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=None,
                        batch_indxs=batch_indxs_test)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: data_indxs}

                outputs = self.networks[ffnet_target].layers[layer_target].outputs

                if pre_activation:
                    nodes = [n.name for n in self.graph.as_graph_def().node]
                    name = outputs.name[:outputs.name.find(':')] # removing : gets the operation
                    id = [id for id in range(len(nodes)) if nodes[id]==name]
                    opname = nodes[id[0]-1] # the operation before the activation function
                    # get that operation
                    op = self.graph.get_operation_by_name(opname)
                    # get the associated tensor
                    outputs = op.values()
                    pred_tmp = sess.run(outputs, feed_dict=feed_dict)
                    pred_tmp = np.squeeze(pred_tmp)
                else:
                    pred_tmp = sess.run(outputs, feed_dict=feed_dict)

                if batch_test == 0:
                    pred = pred_tmp[t0:]
                else:
                    pred = np.concatenate((pred, pred_tmp[t0:]), axis=0)

        # change the data_pipe_type to original
        # self.data_pipe_type = original_pipe_type
        if batch_size_save is not None:
            self.batch_size = batch_size_save

        return pred
    # END generate_prediction

    def get_reg_pen(self):
        """Return reg penalties in a dictionary"""

        reg_dict = {}
        if self.graph is None:
            self._build_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # initialize all parameters randomly
            sess.run(self.init)

            # overwrite randomly initialized values of model with stored values
            self._assign_model_params(sess)

            # update regularization parameter values
            self._assign_reg_vals(sess)

            with tf.name_scope('get_reg_pen'):  # to keep the graph clean-ish
                for nn in range(self.num_networks):
                    for layer in range(self.networks[nn].num_layers):
                        reg_dict['net%iL%i' % (nn, layer)] = \
                            self.networks[nn].layers[layer].get_reg_pen(sess)

        return reg_dict

    def copy_model(self, tf_seed=0):
        """Makes an exact copy of model without further elaboration."""

        # Assemble network_list
        target = NDN(self.network_list, ffnet_out=self.ffnet_out,
                     noise_dist=self.noise_dist, tf_seed=tf_seed)

        target.poisson_unit_norm = self.poisson_unit_norm
        target.data_pipe_type = self.data_pipe_type
        target.batch_size = self.batch_size
        target.time_spread = self.time_spread

        # Copy all the parameters
        for nn in range(self.num_networks):
            target.networks[nn].copy_ffnetwork_params( self.networks[nn] )
        return target

    def matlab_export(self, filename):
        """Exports all weights and biases to matlab-readable file with given filename"""

        import scipy.io as sio
        matdata = {}
        for nn in range(self.num_networks):
            for ll in range(len(self.networks[nn].layers)):
                wstring = 'ws' + str(nn) + str(ll)
                matdata[wstring] = self.networks[nn].layers[ll].weights
                bstring = 'bs' + str(nn) + str(ll)
                matdata[bstring] = self.networks[nn].layers[ll].biases

        sio.savemat(filename, matdata)

    def initialize_output_layer_bias(self, robs):
        """Sets biases in output layer(w) to explain mean firing rate, using Robs"""

        if robs is not list:
            robs = [robs]
        for nn in range(len(self.ffnet_out)):
            frs = np.mean(robs[nn], axis=0, dtype='float32')
            assert len(frs) == len(
                self.networks[self.ffnet_out[nn]].layers[-1].biases[0]), \
                'Robs length wrong'
            self.networks[self.ffnet_out[nn]].layers[-1].biases = frs
    # END NDN.initialize_output_layer_bias

    def get_null_ll(self, robs, data_filters=None):
        """Calculates null-model (constant firing rate) likelihood, given Robs
        (which determines what firing rate for each cell)"""

        if self.noise_dist == 'gaussian':
            # In this case, LLnull is just var of data that exists
            if data_filters is None:
                null_lls = np.var(robs, axis=0)
            else:
                rbars = np.divide(np.multiply(data_filters, np.sum(robs, axis=0), np.sum(data_filters, axis=0)))
                rbars = np.divide(np.sum(np.multiply(data_filters, robs), axis=0), 
                                    np.maximum((np.sum(data_filters, axis=0), 1)))
                rbars = np.divide(np.multiply(data_filters, np.sum(robs, axis=0)), np.sum(data_filters, axis=0)) # Correct IMHO

                null_lls = np.divide(np.sum(np.square(np.add(robs, -rbars))), np.sum(data_filters, axis=0))

        elif self.noise_dist == 'poisson':
            # while the 'correct' null_ll would be `np.multiply(np.log(rbars) - 1.0, rbars)` the ll
            # ..for poisson is normalized (f:_define_loss:cost_norm) with `self.poisson_unit_norm`
            # that defaults (f:set_poisson_norm) to mean of the output data. Ergo null_ll needs to
            # be normalized as well -> `rbars*(np.log(rbars) - 1)/rbars` -> `np.log(rbars) - 1`
            # See: #7
            # Now can explicitly take into account data_filters
            if data_filters is None:
                rbars = np.mean(robs, axis=0)
            else:
                rbars = np.divide(np.sum(np.multiply(robs, data_filters), axis=0), 
                                    np.maximum(np.sum(data_filters, axis=0),1))
            null_lls = np.log(rbars+self._log_min) - 1.0
        # elif self.noise_dist == 'bernoulli':
        else:
            null_lls = [0] * robs.shape[1]
            print('Not worked out yet')

        return null_lls
    # END NDN.get_null_ll

    def set_poisson_norm(self, data_out, data_indxs=None, data_filters=None, blocks=None, mult=1):
        """Calculates the average probability of spike per bin to normalize the Poisson likelihood. 
        It Will include multiplying factor if this is part of the argument."""

        if type(data_out) is not list:
            data_out = [data_out]
        if data_filters is not None:
            if type(data_filters) is not list:
                data_filters = [data_filters]

        if blocks is not None:
            if data_indxs is None:
                data_indxs = range(blocks.shape[0])
            indxs = []
            for nn in range(len(data_indxs)):
                indxs = np.concatenate((indxs, np.array(range(blocks[data_indxs[nn], 0]-1,
                                                              blocks[data_indxs[nn], 1]))), axis=0)
            indxs = indxs.astype(int)
        else:
            if data_indxs is None:
                indxs = range(data_out[0].shape[0])
            else:
                indxs = data_indxs

        self.poisson_unit_norm = []
        for ii, temp_data in enumerate(data_out):
            nc = self.network_list[self.ffnet_out[ii]]['layer_sizes'][-1]
            assert nc == temp_data.shape[1], 'Output of network must match robs'
            #if blocks is None:
            #    indxs = range(temp_data.shape[0])
            if data_filters is not None:
                assert nc == data_filters[ii].shape[1], 'Output of network must match data_filters'
                self.poisson_unit_norm.append(np.maximum(
                    np.divide(np.sum(np.multiply(temp_data[indxs, :],
                                                 data_filters[ii][indxs, :]).astype('float32'), axis=0),
                              np.sum(data_filters[ii].astype('float32') / mult, axis=0)),
                    1e-8))
            else:
                self.poisson_unit_norm.append(np.maximum(np.mean(temp_data[indxs, :].astype('float32'), axis=0), 1e-8))
    # END NDN.set_poisson_norm ####################################################

    def _initialize_data_pipeline(self):
        """Define pipeline for feeding data into model"""

        if self.data_pipe_type == 'data_as_var':
            # define indices placeholder to specify subset of data
            self.indices = tf.placeholder(
                dtype=tf.int32,
                shape=None,
                name='indices_ph')

            # INPUT DATA
            self.data_in_ph = [None] * len(self.input_sizes)
            self.data_in_var = [None] * len(self.input_sizes)
            self.data_in_batch = [None] * len(self.input_sizes)
            for i, input_size in enumerate(self.input_sizes):
                # reduce input_sizes to single number if 3-D
                num_inputs = np.prod(input_size)
                # placeholders for data
                self.data_in_ph[i] = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.num_examples, num_inputs],
                    name='input_ph_%02d' % i)
                # turn placeholders into variables so they get put on GPU
                self.data_in_var[i] = tf.Variable(
                    self.data_in_ph[i],  # initializer for Variable
                    trainable=False,     # no GraphKeys.TRAINABLE_VARS
                    collections=[],      # no GraphKeys.GLOBAL_VARS
                    name='input_var_%02d' % i)
                # use selected subset of data
                self.data_in_batch[i] = tf.gather(
                    self.data_in_var[i],
                    self.indices,
                    name='input_batch_%02d' % i)

            # OUTPUT DATA
            self.data_out_ph = [None] * len(self.output_sizes)
            self.data_out_var = [None] * len(self.output_sizes)
            self.data_out_batch = [None] * len(self.output_sizes)
            for i, output_size in enumerate(self.output_sizes):
                # placeholders for data
                self.data_out_ph[i] = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.num_examples, output_size],
                    name='output_ph_%02d' % i)
                # turn placeholders into variables so they get put on GPU
                self.data_out_var[i] = tf.Variable(
                    self.data_out_ph[i],  # initializer for Variable
                    trainable=False,      # no GraphKeys.TRAINABLE_VARS
                    collections=[],       # no GraphKeys.GLOBAL_VARS
                    name='output_var_%02d' % i)
                # use selected subset of data
                self.data_out_batch[i] = tf.gather(
                    self.data_out_var[i],
                    self.indices,
                    name='output_batch_%02d' % i)

            # DATA FILTERS
            if self.filter_data:
                self.data_filter_ph = [None] * len(self.output_sizes)
                self.data_filter_var = [None] * len(self.output_sizes)
                self.data_filter_batch = [None] * len(self.output_sizes)
                for ii, output_size in enumerate(self.output_sizes):
                    # placeholders for data
                    self.data_filter_ph[i] = tf.placeholder(
                        dtype=tf.float32,
                        shape=[self.num_examples, output_size],
                        name='data_filter_ph_%02d' % i)
                    # turn placeholders into variables so they get put on GPU
                    self.data_filter_var[i] = tf.Variable(
                        self.data_filter_ph[i],  # initializer for Variable
                        trainable=False,  # no GraphKeys.TRAINABLE_VARS
                        collections=[],  # no GraphKeys.GLOBAL_VARS
                        name='output_filter_%02d' % i)
                    # use selected subset of data
                    self.data_filter_batch[i] = tf.gather(
                        self.data_filter_var[i],
                        self.indices,
                        name='output_filter_%02d' % i)

        elif self.data_pipe_type == 'feed_dict':
            # INPUT DATA
            self.data_in_batch = [None] * len(self.input_sizes)
            for i, input_size in enumerate(self.input_sizes):
                # reduce input_sizes to single number if 3-D
                num_inputs = np.prod(input_size)
                # placeholders for data
                self.data_in_batch[i] = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, num_inputs],
                    name='input_batch_%02d' % i)

            # OUTPUT DATA
            self.data_out_batch = [None] * len(self.output_sizes)
            for i, output_size in enumerate(self.output_sizes):
                # placeholders for data
                self.data_out_batch[i] = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, output_size],
                    name='output_batch_%02d' % i)

            # DATA FILTERS
            if self.filter_data:
                self.data_filter_batch = [None] * len(self.output_sizes)
                for i, output_size in enumerate(self.output_sizes):
                    # placeholders for data
                    self.data_filter_batch[i] = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None, output_size],
                        name='data_filter_%02d' % i)

        elif self.data_pipe_type == 'iterator':
            # build iterator object to access elements from dataset; make
            # 'initializable' so that we can easily switch between training and
            # xv datasets
            self.iterator_handle = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.iterator_handle,
                self.dataset_types,
                self.dataset_shapes)
            next_element = self.iterator.get_next()

            # pull input/output/filter data out of 'next_element'
            self.data_in_batch = [None] * len(self.input_sizes)
            for i, _ in enumerate(self.input_sizes):
                name = 'input_%02d' % i
                self.data_in_batch[i] = next_element[name]

            self.data_out_batch = [None] * len(self.output_sizes)
            for i, _ in enumerate(self.output_sizes):
                name = 'output_%02d' % i
                self.data_out_batch[i] = next_element[name]

            if self.filter_data:
                self.data_filter_batch = [None] * len(self.output_sizes)
                for i, _ in enumerate(self.output_sizes):
                    name = 'filter_%02d' % i
                    self.data_filter_batch[i] = next_element[name]

    # END Network._initialize_data_pipeline

    def _define_optimizer(self, learning_alg='adam', opt_params=None,
                          var_list=None):
        """Define one step of the optimization routine
        L-BGFS algorithm described
        https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.minimize-lbfgsb.html
        """

        if learning_alg == 'adam':
            # self.train_step = tf.compat.v1.train.AdamOptimizer(  # not backwards compatible
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=opt_params['learning_rate'],
                beta1=opt_params['beta1'],
                beta2=opt_params['beta2'],
                epsilon=opt_params['epsilon']). \
                minimize(self.cost_penalized, var_list=var_list)
        elif learning_alg == 'lbfgs':
            self.train_step = tf.contrib.opt.ScipyOptimizerInterface(
                self.cost_penalized,
                var_list=var_list,
                method='L-BFGS-B',
                options={
                    'maxiter': opt_params['maxiter'],
                    'gtol': opt_params['grad_tol'],
                    'ftol': opt_params['func_tol'],
                    'eps': opt_params['eps'],
                    'disp': opt_params['display']})
    # END _define_optimizer

    def train(
            self,
            input_data=None,
            output_data=None,
            train_indxs=None,
            test_indxs=None,
            blocks=None,  # Added
            fit_variables=None,
            use_dropout=True,
            data_filters=None,
            learning_alg='adam',
            opt_params=None,
            output_dir=None,
            silent=False):
        """Network training function. Note that the "early_stop_mode" is a critical part of the training, and can
        be set to the following:
            0: will stop after a certain number of iterations (specified by 'epochs_training')
            1: will use early stopping on test data -- note in both cases regulariation
            2: will use early stopping on training data
            3: will use delta criteria for stoppying (more detail?)

        Args:
            input_data (list): input to network; each element should be a
                time x input_dim numpy array
            output_data (list of matrices): desired output of network; each
                element should be a time x output_dim numpy array
            train_indxs (numpy array, optional): subset of data to use for
                training
            test_indxs (numpy array, optional): subset of data to use for
                testing; if available these are used when displaying updates,
                and are also the indices used for early stopping if enabled
            blocks (list-of-lists): pass in blocks explicitly for temporal convolutions to work. If none
                then will divide into blocks according to opt_params, but otherwise will use these blocks
                and train and test indx will correspond to block numbers
            fit_variables (list-of-lists, optional): default none
                Generated by 'fit_variables' (if not none) to reference which
                variables in the model to fit.
            use_dropout (binary): obvious.
            data_filters (list of matrices): matrices as same size as
                output_data that zeros out predictions where data is absent
            learning_alg (str, optional): algorithm used for learning
                parameters.
                'lbfgs' | ['adam']
            opt_params: dictionary with optimizer-specific parameters; see
                network.optimizer_defaults method for valid key-value pairs and
                corresponding default values.
            output_dir (str, optional): absolute path for saving checkpoint
                files and summary files; must be present if either
                `epochs_ckpt` or `epochs_summary` values in `opt_params` is not
                `None`. If `output_dir` is not `None`, regardless of checkpoint
                or summary settings, the graph will automatically be saved.
                Must be present if early_stopping is desired to restore the
                best fit, otherwise it will restore the model at break point.
            silent (Boolean, optional), slients standard fitting output (i.e. non-verbose)
                default is False

        Returns:
            int: number of total training epochs

        Raises:
            ValueError: If `input_data` and `output_data` don't share time dim
            ValueError: If data time dim doesn't match that specified in model
            ValueError: If `epochs_ckpt` value in `opt_params` is not `None`
                and `output_dir` is `None`
            ValueError: If `epochs_summary` in `opt_params` is not `None` and
                `output_dir` is `None`
            ValueError: If `early_stop` > 0 and `test_indxs` is 'None'

        """

        self.num_examples = 0
        self.filter_data = False

        # check inputs and outputs
        input_data, output_data, data_filters = self._data_format(input_data, output_data, data_filters)

        # Check format of opt_params (and add some defaults)
        if opt_params is None:
            opt_params = {}
        opt_params = self.optimizer_defaults(opt_params, learning_alg)

        # update data pipeline type before building tensorflow graph
        self.data_pipe_type = opt_params['data_pipe_type']

        if train_indxs is None:
            train_indxs = np.arange(self.num_examples)

        # Check values entered
        if learning_alg is 'adam':
            if opt_params['epochs_ckpt'] is not None and output_dir is None:
                raise ValueError('output_dir must be specified to save model')
            if opt_params['epochs_summary'] is not None and output_dir is None:
                raise ValueError('output_dir must be specified to save summaries')
            if opt_params['early_stop'] > 0 and test_indxs is None:
                raise ValueError('test_indxs must be specified for early stopping')

        # Handle blocks, if necessary
        if blocks is not None:
            if data_filters is None:  # produce data-filters for incorporating gaps
                data_filters = [np.ones(output_data[0].shape, dtype='float32')]
            self.filter_data = True

            block_lists, data_filters, batch_comb = process_blocks(
                    blocks, data_filters, opt_params['batch_size'], skip=self.time_spread)

        # build datasets if using 'iterator' pipeline
        if self.data_pipe_type is 'iterator':
            dataset_tr = self._build_dataset(
                input_data=input_data,
                output_data=output_data,
                data_filters=data_filters,
                indxs=train_indxs,
                training_dataset=True,
                batch_size=opt_params['batch_size'])
            # store info on dataset for buiding data pipeline
            self.dataset_types = dataset_tr.output_types
            self.dataset_shapes = dataset_tr.output_shapes
            if test_indxs is not None:
                dataset_test = self._build_dataset(
                    input_data=input_data,
                    output_data=output_data,
                    data_filters=data_filters,
                    indxs=test_indxs,
                    training_dataset=False,
                    batch_size=opt_params['batch_size'])
            else:
                dataset_test = None

        # Set Poisson_unit_norm if specified
        # overwrite unit_cost_norm with opt_params value
        if opt_params['poisson_unit_norm'] is not None:
            self.poisson_unit_norm = opt_params['poisson_unit_norm']
        elif (self.noise_dist == 'poisson') and (self.poisson_unit_norm is None):
            self.set_poisson_norm(output_data, data_filters=None, blocks=blocks)
            # note the default should not include data_filters, since it would not accurately
            # reflect the sampling of batches: meaning with data_filters, the normalizing
            # firing rate would be especially high.
            # self.set_poisson_norm(output_data, data_filters=data_filters, blocks=blocks)

        # Build graph: self.build_graph must be defined in child of network
        self._build_graph(
            learning_alg=learning_alg,
            opt_params=opt_params,
            fit_variables=fit_variables,
            batch_size=opt_params['batch_size'],
            use_dropout=use_dropout)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            # handle output directories
            train_writer = None
            test_writer = None
            if output_dir is not None:

                # remake checkpoint directory
                if opt_params['epochs_ckpt'] is not None:
                    ckpts_dir = os.path.join(output_dir, 'ckpts')
                    if os.path.isdir(ckpts_dir):
                        tf.gfile.DeleteRecursively(ckpts_dir)
                    os.makedirs(ckpts_dir)

                # remake training summary directories
                summary_dir_train = os.path.join(
                    output_dir, 'summaries', 'train')
                if os.path.isdir(summary_dir_train):
                    tf.gfile.DeleteRecursively(summary_dir_train)
                    #tf.io.gfile.rmtree(summary_dir_train)  # not backwards compatible
                os.makedirs(summary_dir_train)
                #train_writer = tf.compat.v1.summary.FileWriter(  # backward-compatible
                train_writer = tf.summary.FileWriter(
                    summary_dir_train, graph=sess.graph)

                # remake testing summary directories
                summary_dir_test = os.path.join(
                    output_dir, 'summaries', 'test')
                if test_indxs is not None:
                    if os.path.isdir(summary_dir_test):
                        tf.gfile.DeleteRecursively(summary_dir_test)
                    os.makedirs(summary_dir_test)
                    #test_writer = tf.compat.v1.summary.FileWriter(   # not backwards compatible
                    test_writer = tf.summary.FileWriter(
                        summary_dir_test, graph=sess.graph)

            # overwrite initialized values of network with stored values
            self._restore_params(sess, input_data, output_data, data_filters)

            if self.data_pipe_type is 'data_as_var':
                # select learning algorithm
                if learning_alg is 'adam':
                    if blocks is None:
                        epoch = self._train_adam(
                            sess=sess,
                            train_writer=train_writer,
                            test_writer=test_writer,
                            train_indxs=train_indxs,
                            test_indxs=test_indxs,
                            data_filters=data_filters,
                            opt_params=opt_params,
                            output_dir=output_dir,
                            silent=silent)
                    else:
                        epoch = self._train_adam_block(
                            sess=sess,
                            train_writer=train_writer,
                            test_writer=test_writer,
                            block_lists=block_lists,
                            batch_comb=batch_comb,
                            train_blocks=train_indxs,
                            test_blocks=test_indxs,
                            data_filters=data_filters,
                            opt_params=opt_params,
                            output_dir=output_dir,
                            silent=silent)

                elif learning_alg is 'lbfgs':
                    self.train_step.minimize(
                        sess, feed_dict={self.indices: train_indxs})
                    epoch = float('NaN')
                else:
                    raise ValueError('Invalid learning algorithm')

            elif self.data_pipe_type is 'feed_dict':
                # select learning algorithm
                if learning_alg is 'adam':
                    if blocks is None:
                        epoch = self._train_adam(
                            sess=sess,
                            train_writer=train_writer,
                            test_writer=test_writer,
                            train_indxs=train_indxs,
                            test_indxs=test_indxs,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            opt_params=opt_params,
                            output_dir=output_dir,
                            silent=silent)
                    else:
                        epoch = self._train_adam_block(
                            sess=sess,
                            train_writer=train_writer,
                            test_writer=test_writer,
                            block_lists=block_lists,
                            batch_comb=batch_comb,
                            train_blocks=train_indxs,
                            test_blocks=test_indxs,
                            data_filters=data_filters,
                            opt_params=opt_params,
                            output_dir=output_dir,
                            silent=silent)

                elif learning_alg is 'lbfgs':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,  # this line needed?
                        batch_indxs=train_indxs)

                    self.train_step.minimize(sess, feed_dict=feed_dict)
                    epoch = float('NaN')
                else:
                    raise ValueError('Invalid learning algorithm')

            elif self.data_pipe_type is 'iterator':
                # select learning algorithm
                if learning_alg is 'adam':
                    epoch = self._train_adam(
                        sess=sess,
                        train_writer=train_writer,
                        test_writer=test_writer,
                        train_indxs=train_indxs,
                        test_indxs=test_indxs,
                        data_filters=data_filters,
                        dataset_tr=dataset_tr,
                        dataset_test=dataset_test,
                        opt_params=opt_params,
                        output_dir=output_dir,
                        silent=silent)

                elif learning_alg is 'lbfgs':
                    raise ValueError(
                        'Use of iterator pipeline with lbfgs not supported')
                else:
                    raise ValueError('Invalid learning algorithm')

            # write out weights/biases to numpy arrays before session closes
            self._write_model_params(sess)

        return epoch
    # END train

    def _train_adam(
            self,
            sess=None,
            train_writer=None,
            test_writer=None,
            train_indxs=None,
            test_indxs=None,
            input_data=None,
            output_data=None,
            data_filters=None,
            dataset_tr=None,
            dataset_test=None,
            opt_params=None,
            output_dir=None,
            silent=False):
        """Training function for adam optimizer to clean up code in `train`"""

        epochs_training = opt_params['epochs_training']
        epochs_ckpt = opt_params['epochs_ckpt']
        epochs_summary = opt_params['epochs_summary']

        # Inherit batch size for future uses, and adapt to current training
        self.batch_size = opt_params['batch_size']
        if self.batch_size > train_indxs.shape[0]:
            self.batch_size = train_indxs.shape[0]
        num_batches_tr = train_indxs.shape[0] // opt_params['batch_size']
        if self.data_pipe_type != 'data_as_var':
            assert self.batch_size is not None, 'Need to assign batch_size to train.'

        early_stop_mode = opt_params['early_stop_mode']
        MAPest = opt_params['MAPest']

        assert early_stop_mode < 10, 'early_stop_mode now must be set 0-3.'

        if early_stop_mode > 0:
            prev_costs = np.multiply(np.ones(opt_params['early_stop']), float('NaN'))

        if opt_params['run_diagnostics']:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # build iterator handles if using that input pipeline type
        if self.data_pipe_type == 'iterator':
            # build iterator object to access elements from dataset
            iterator_tr = dataset_tr.make_one_shot_iterator()
            # get string handle of iterator
            iter_handle_tr = sess.run(iterator_tr.string_handle())

            if test_indxs is not None:
                # build iterator object to access elements from dataset
                iterator_test = dataset_test.make_one_shot_iterator()
                # get string handle of iterator
                iter_handle_test = sess.run(iterator_test.string_handle())

        # used in early_stopping
        best_epoch = 0
        best_cost = float('Inf')
        chkpted = False

        if self.time_spread is not None:
            # get number of batches and their order for train indxs
            batch_order = np.arange(num_batches_tr)

        # start training loop
        for epoch in range(epochs_training):

            # shuffle data before each pass
            if self.time_spread is None:
                train_indxs_perm = np.random.permutation(train_indxs)
            else:
                batch_order_perm = np.random.permutation(batch_order)

            # pass through dataset once
            for batch in range(num_batches_tr):

                if (self.data_pipe_type == 'data_as_var') or (self.data_pipe_type == 'feed_dict'):
                    # get training indices for this batch
                    if self.time_spread is None:
                        batch_indxs = train_indxs_perm[batch * opt_params['batch_size']:
                                                       (batch + 1) * opt_params['batch_size']]
                    else:
                        batch_indxs = train_indxs[batch_order_perm[batch] * self.batch_size:
                                                  (batch_order_perm[batch]+1) * self.batch_size]

                # one step of optimization routine
                if self.data_pipe_type == 'data_as_var':
                    # get the feed_dict for batch_indxs
                    feed_dict = {self.indices: batch_indxs}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        batch_indxs=batch_indxs)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: iter_handle_tr}

                sess.run(self.train_step, feed_dict=feed_dict)

            # print training updates -- recalcs all training data (no need to permute)
            if opt_params['display'] is not None and not silent and \
                    ((epoch % opt_params['display'] == opt_params['display']-1) or (epoch == 0)):

                cost_tr, cost_test = 0, 0
                for batch_tr in range(num_batches_tr):
                    # Will be contiguous data: no need to change time-spread
                    batch_indxs_tr = train_indxs[batch_tr * self.batch_size:
                                                 (batch_tr+1) * self.batch_size]

                    if self.data_pipe_type == 'data_as_var':
                        feed_dict = {self.indices: batch_indxs_tr}
                    elif self.data_pipe_type == 'feed_dict':
                        feed_dict = self._get_feed_dict(
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            batch_indxs=batch_indxs_tr)
                    elif self.data_pipe_type == 'iterator':
                        feed_dict = {self.iterator_handle: iter_handle_tr}

                    cost_tr += sess.run(self.cost, feed_dict=feed_dict)

                cost_tr /= num_batches_tr
                reg_pen = sess.run(self.cost_reg)

                if test_indxs is not None:
                    if self.data_pipe_type == 'data_as_var' or \
                            self.data_pipe_type == 'feed_dict':
                        cost_test = self._get_test_cost(
                            sess=sess,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            test_indxs=test_indxs,
                            test_batch_size=self.batch_size)
                    elif self.data_pipe_type == 'iterator':
                        cost_test = self._get_test_cost(
                            sess=sess,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            test_indxs=iter_handle_test,
                            test_batch_size=self.batch_size)

                    #if MAPest:  # then add reg_penalty to test cost (this is just for display)
                    #    cost_tr += reg_pen
                    #    cost_test += reg_pen

                # print additional testing info
                print('Epoch %04d:  avg train cost = %10.4f,  '
                      'avg test cost = %10.4f,  '
                      'reg penalty = %10.4f'
                      % (epoch, cost_tr / np.sum(self.output_sizes),
                         cost_test / np.sum(self.output_sizes),
                         reg_pen / np.sum(self.output_sizes)))

            # save model checkpoints
            if epochs_ckpt is not None and (
                    epoch % epochs_ckpt == epochs_ckpt - 1 or epoch == 0):
                save_file = os.path.join(output_dir, 'ckpts', str('epoch_%05g.ckpt' % epoch))
                self.checkpoint_model(sess, save_file)

            # save model summaries
            if epochs_summary is not None and ((epoch % epochs_summary == epochs_summary-1) or (epoch == 0)):

                # TODO: what to use with feed_dict?
                if opt_params['run_diagnostics']:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'epoch_%d' % epoch)
                else:
                    summary = sess.run(self.merge_summaries, feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch)
                train_writer.flush()

                if test_indxs is not None:
                    # FIXME: Assuming 'data_as_var' and `test_indxs` being small enough to fit in one batch
                    if self.data_pipe_type == 'data_as_var':
                        feed_dict = {self.indices: test_indxs}
                    else:
                        raise NotImplementedError("Other data pipelines are not yet implemented for test_indxs")
                    if opt_params['run_diagnostics']:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                        test_writer.add_run_metadata(run_metadata, 'epoch_%d' % epoch)
                    else:
                        summary = sess.run(self.merge_summaries, feed_dict=feed_dict)
                    test_writer.add_summary(summary, epoch)
                    test_writer.flush()

            if opt_params['early_stop_mode'] > 0:

                # if you want to suppress that useless warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_before = np.nanmean(prev_costs)

                if (self.data_pipe_type == 'data_as_var') or (self.data_pipe_type == 'feed_dict'):

                    if early_stop_mode == 2:
                        data_indxs = train_indxs
                    else:
                        data_indxs = test_indxs

                    cost_test = self._get_test_cost(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_indxs=data_indxs,
                        test_batch_size=opt_params['batch_size'])

                elif self.data_pipe_type == 'iterator':
                    assert not early_stop_mode == 2, 'curently doesnt work for esm 2'

                    cost_test = self._get_test_cost(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_indxs=iter_handle_test,
                        test_batch_size=opt_params['batch_size'])

                if MAPest:
                    cost_test += sess.run(self.cost_reg)

                prev_costs = np.roll(prev_costs, 1)
                prev_costs[0] = cost_test

                mean_now = np.nanmean(prev_costs)

                # Calculate change in relevant condition
                delta = (mean_before - mean_now) - opt_params['func_tol']
                #delta = (mean_before - mean_now) / mean_before
                # to check and refine the condition on chkpting best model
                # print(epoch, delta, 'delta condition:', delta < 1e-4)

                if cost_test < best_cost:
                    # update best cost and the epoch that it happened at
                    best_cost = cost_test
                    best_epoch = epoch
                    # chkpt model if desired
                    if output_dir is not None:
                        if (early_stop_mode == 1) or (early_stop_mode == 2):
                            save_file = os.path.join(output_dir, 'bstmods', 'best_model')
                            self.checkpoint_model(sess, save_file)
                            chkpted = True
                        elif (early_stop_mode == 2) and (delta < 5e-5):
                            save_file = os.path.join(output_dir, 'bstmods', 'best_model')
                            self.checkpoint_model(sess, save_file)
                            chkpted = True

                if (early_stop_mode > 0):
                    if epoch > opt_params['early_stop'] and (delta < 0):
                        if not silent:
                            print('\n*** early stop criteria met...stopping train now...')
                            print('     ---> number of epochs used: %d,  '
                                  'end cost: %04f' % (epoch, cost_test))
                            print('     ---> best epoch: %d,  '
                                  'best cost: %04f\n' % (best_epoch, best_cost))
                        # restore saved variables into tf Variables
                        if output_dir is not None and chkpted and early_stop_mode > 0:
                            # save_file exists only if chkpted is True
                            self.saver.restore(sess, save_file)
                            # delete files before break to clean up space
                            shutil.rmtree(os.path.join(output_dir, 'bstmods'), ignore_errors=True)
                        break
        return epoch
        #    return epoch
        # END _train_adam

    def _get_test_cost(self, sess, input_data, output_data, data_filters,
                       test_indxs, test_batch_size=None):
        """Utility function to clean up code in `_train_adam` method"""

        if test_batch_size is not None:
            if test_batch_size > test_indxs.shape[0]:
                test_batch_size = test_indxs.shape[0]
            num_batches_test = test_indxs.shape[0] // test_batch_size
            #print(test_indxs.shape[0], test_batch_size, num_batches_test)
            cost_test = 0
            for batch_test in range(num_batches_test):
                batch_indxs_test = test_indxs[batch_test * test_batch_size:
                                              (batch_test+1) * test_batch_size]
                if self.data_pipe_type == 'data_as_var':
                    feed_dict = {self.indices: batch_indxs_test}
                elif self.data_pipe_type == 'feed_dict':
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        batch_indxs=batch_indxs_test)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: test_indxs}
                cost_test += sess.run(self.cost, feed_dict=feed_dict)
            cost_test /= num_batches_test
        else:
            if self.data_pipe_type == 'data_as_var':
                feed_dict = {self.indices: test_indxs}
            elif self.data_pipe_type == 'feed_dict':
                feed_dict = self._get_feed_dict(
                    input_data=input_data,
                    output_data=output_data,
                    data_filters=data_filters,
                    batch_indxs=test_indxs)
            elif self.data_pipe_type == 'iterator':
                feed_dict = {self.iterator_handle: test_indxs}
            cost_test = sess.run(self.cost, feed_dict=feed_dict)
        return cost_test

    def _get_feed_dict(
            self,
            input_data=None,
            output_data=None,
            batch_indxs=None,
            data_filters=None):
        """Generates feed dict to be used with the `feed_dict` data pipeline"""

        #if self.time_spread is not None:
        #    print('Is _get_feed_dict doing what we want with time-spread data?')

        if batch_indxs is None:
            batch_indxs = np.arange(input_data[0].shape[0])

        feed_dict = {}
        if input_data is not None:
            for i, temp_data in enumerate(input_data):
                feed_dict[self.data_in_batch[i]] = \
                    temp_data[batch_indxs, :]
        if output_data is not None:
            for i, temp_data in enumerate(output_data):
                feed_dict[self.data_out_batch[i]] = temp_data[batch_indxs, :]
        if data_filters is not None:
            for i, temp_data in enumerate(data_filters):
                feed_dict[self.data_filter_batch[i]] = temp_data[batch_indxs, :]
        return feed_dict
    # END _get_feed_dict

    # REDUNDANT TO WORK DIFFERENTLY
    def _train_adam_block(
            self,
            sess=None,
            train_writer=None,
            test_writer=None,
            train_blocks=None,  # here
            test_blocks=None,  # here
            block_lists=None,  # here
            batch_comb=1,  # here
            input_data=None,
            output_data=None,
            data_filters=None,
            #dataset_tr=None,
            #dataset_test=None,
            opt_params=None,
            output_dir=None,
            silent=False):
        """Training function for adam optimizer to clean up code in `train`"""

        epochs_training = opt_params['epochs_training']
        epochs_ckpt = opt_params['epochs_ckpt']
        epochs_summary = opt_params['epochs_summary']

        # Inherit batch size if relevant
        self.batch_size = opt_params['batch_size']
        if self.data_pipe_type != 'data_as_var':
            assert self.batch_size is not None, 'Need to assign batch_size to train.'
        early_stop_mode = opt_params['early_stop_mode']

        MAPest = opt_params['MAPest']

        assert early_stop_mode < 10, 'early_stop_mode now must be set 0-3.'
        if early_stop_mode > 0:
            prev_costs = np.multiply(np.ones(opt_params['early_stop']), float('NaN'))

        #num_batches_tr = train_indxs.shape[0] // opt_params
        if data_filters is None:  # then make basic data_filters
            df = []
            for nn in range(len(output_data)):
                df.append(np.ones(output_data[nn].shape, dtype='int'))
        else:
            df = deepcopy(data_filters)

        num_batches_tr = len(train_blocks)
        num_batches_te = len(test_blocks)
        num_comb_batch_train_step = np.floor(num_batches_tr / batch_comb).astype(int)

        assert num_batches_tr+num_batches_te == len(block_lists), 'Incorrect number of train/test blocks.'

        if opt_params['run_diagnostics']:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # build iterator handles if using that input pipeline type
        if self.data_pipe_type == 'iterator':
            # build iterator object to access elements from dataset
            iterator_tr = dataset_tr.make_one_shot_iterator()
            # get string handle of iterator
            iter_handle_tr = sess.run(iterator_tr.string_handle())

            if test_indxs is not None:
                # build iterator object to access elements from dataset
                iterator_test = dataset_test.make_one_shot_iterator()
                # get string handle of iterator
                iter_handle_test = sess.run(iterator_test.string_handle())

        # used in early_stopping
        best_epoch = 0
        best_cost = float('Inf')
        chkpted = False

        # if self.time_spread is not None:
        # get number of batches and their order for train indxs
        #   num_batches_tr = train_indxs.shape[0] // self.batch_size
        batch_order = np.arange(num_batches_tr)

        # start training loop
        for epoch in range(epochs_training):

            # shuffle data before each pass
            #if self.time_spread is None:
            #    train_indxs_perm = np.random.permutation(train_indxs)
            #else:
            batch_order_perm = np.random.permutation(batch_order)

            # pass through dataset once
            for batch in range(num_comb_batch_train_step):

                #if (self.data_pipe_type == 'data_as_var') or (self.data_pipe_type == 'feed_dict'):
                    # get training indices for this batch
                    # if self.time_spread is None:
                    #    batch_indxs = train_indxs_perm[batch * opt_params['batch_size']:
                    #                                   (batch + 1) * opt_params['batch_size']]
                    #else:
                    #    batch_indxs = train_indxs[batch_order_perm[batch] * self.batch_size:
                    #                              (batch_order_perm[batch]+1) * self.batch_size]

                for nn in range(batch_comb):
                    block_num = train_blocks[batch_order_perm[batch*batch_comb+nn]]
                    if nn == 0:
                        batch_indxs = block_lists[block_num]
                    else:
                        batch_indxs = np.concatenate((batch_indxs, block_lists[block_num]))

                # one step of optimization routine
                if self.data_pipe_type == 'data_as_var':
                    # get the feed_dict for batch_indxs
                    #feed_dict = {self.indices: batch_indxs}
                    #feed_dict = {self.indices: block_inds[batch_order_perm[batch]]}
                    feed_dict = {self.indices: batch_indxs}
                elif self.data_pipe_type == 'feed_dict':
                    print('*** likely problem with variable batch sizes for feed_dict mode')
                    feed_dict = self._get_feed_dict(
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        batch_indxs=batch_indxs)
                        #batch_indxs=block_inds[batch_order_perm[batch]])
                        #batch_indxs=batch_indxs)
                elif self.data_pipe_type == 'iterator':
                    feed_dict = {self.iterator_handle: iter_handle_tr}

                sess.run(self.train_step, feed_dict=feed_dict)

            # print training updates -- recalcs all training data (no need to permute)
            if opt_params['display'] is not None and not silent and \
                    ((epoch % opt_params['display'] == opt_params['display']-1) or (epoch == 0)):

                #cost_tr, cost_test = 0, 0
                #for batch_tr in range(num_batches_tr):
                    # Will be contiguous data: no need to change time-spread
                    #batch_indxs_tr = train_indxs[batch_tr * opt_params['batch_size']:
                    #                             (batch_tr+1) * opt_params['batch_size']]
                #    if self.data_pipe_type == 'data_as_var':
                        #feed_dict = {self.indices: batch_indxs_tr}
                #        feed_dict = {self.indices: block_lists[train_blocks[batch_tr]]}
                #    elif self.data_pipe_type == 'feed_dict':
                #        feed_dict = self._get_feed_dict(
                #            input_data=input_data,
                #            output_data=output_data,
                #            data_filters=mod_df,
                #            batch_indxs=block_lists[train_blocks[batch_tr]])
                #            #batch_indxs=batch_indxs_tr)
                #    elif self.data_pipe_type == 'iterator':
                #        feed_dict = {self.iterator_handle: iter_handle_tr}

                #    cost_tr += sess.run(self.cost, feed_dict=feed_dict)

                #cost_tr /= num_batches_tr
                if self.data_pipe_type == 'data_as_var' or self.data_pipe_type == 'feed_dict':
                    cost_tr = self._get_test_cost_block(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_blocks=train_blocks,
                        block_inds=block_lists)
                elif self.data_pipe_type == 'iterator':
                    cost_tr = self._get_test_cost(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_indxs=iter_handle_train,
                        test_batch_size=opt_params['batch_size'])
                reg_pen = sess.run(self.cost_reg)

                if test_blocks is not None:
                    if self.data_pipe_type == 'data_as_var' or self.data_pipe_type == 'feed_dict':
                        cost_test = self._get_test_cost_block(
                            sess=sess,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            test_blocks=test_blocks,
                            block_inds=block_lists)
                            #test_indxs=test_indxs,
                            #test_batch_size=opt_params['batch_size'])
                    elif self.data_pipe_type == 'iterator':
                        cost_test = self._get_test_cost(
                            sess=sess,
                            input_data=input_data,
                            output_data=output_data,
                            data_filters=data_filters,
                            test_indxs=iter_handle_test,
                            test_batch_size=opt_params['batch_size'])

                # print additional testing info
                print('Epoch %04d:  avg train cost = %10.4f,  '
                      'avg test cost = %10.4f,  reg penalty = %10.4f'
                      % (epoch, cost_tr / np.sum(self.output_sizes),
                         cost_test / np.sum(self.output_sizes), reg_pen / np.sum(self.output_sizes)))

            # save model checkpoints
            if epochs_ckpt is not None and (
                    epoch % epochs_ckpt == epochs_ckpt - 1 or epoch == 0):
                save_file = os.path.join(output_dir, 'ckpts', str('epoch_%05g.ckpt' % epoch))
                self.checkpoint_model(sess, save_file)

            # save model summaries
            if epochs_summary is not None and ((epoch % epochs_summary == epochs_summary-1) or (epoch == 0)):

                # TODO: what to use with feed_dict?
                if opt_params['run_diagnostics']:
                    summary = sess.run(
                        self.merge_summaries,
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'epoch_%d' % epoch)
                else:
                    summary = sess.run(self.merge_summaries, feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch)
                train_writer.flush()

                if test_blocks is not None:
                    # FIXME: test_blocks / test_indxs are actually not used -> broken
                    if opt_params['run_diagnostics']:
                        summary = sess.run(
                            self.merge_summaries,
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                        test_writer.add_run_metadata(run_metadata, 'epoch_%d' % epoch)
                    else:
                        summary = sess.run(self.merge_summaries, feed_dict=feed_dict)
                    test_writer.add_summary(summary, epoch)
                    test_writer.flush()

            if opt_params['early_stop'] > 0:

                # if you want to suppress that useless warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_before = np.nanmean(prev_costs)

                if (self.data_pipe_type == 'data_as_var') or (self.data_pipe_type == 'feed_dict'):

                    if early_stop_mode == 2:
                        data_blocks = train_blocks
                    else:
                        data_blocks = test_blocks

                    cost_test = self._get_test_cost_block(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_blocks=data_blocks,
                        block_inds=block_lists)
                elif self.data_pipe_type == 'iterator':
                    assert not early_stop_mode == 2, 'curently doesnt work for esm 2'

                    cost_test = self._get_test_cost(
                        sess=sess,
                        input_data=input_data,
                        output_data=output_data,
                        data_filters=data_filters,
                        test_indxs=iter_handle_test,
                        test_batch_size=opt_params['batch_size'])

                if MAPest:
                    cost_test += sess.run(self.cost_reg)

                prev_costs = np.roll(prev_costs, 1)
                prev_costs[0] = cost_test

                mean_now = np.nanmean(prev_costs)

                delta = (mean_before - mean_now) - opt_params['func_tol']
                #delta = (mean_before - mean_now) / mean_before
                # to check and refine the condition on chkpting best model
                # print(epoch, delta, 'delta condition:', delta < 1e-4)

                # to check and refine the condition on chkpting best model
                # print(epoch, delta, 'delta condition:', delta < 1e-4)

                if cost_test < best_cost:
                    # update best cost and the epoch that it happened at
                    best_cost = cost_test
                    best_epoch = epoch
                    # chkpt model if desired
                    if output_dir is not None:
                        if (early_stop_mode == 1) or (early_stop_mode == 2):
                            save_file = os.path.join(output_dir, 'bstmods', 'best_model')
                            self.checkpoint_model(sess, save_file)
                            chkpted = True
                        elif (early_stop_mode == 2) and (delta < 5e-5):
                            save_file = os.path.join(output_dir, 'bstmods', 'best_model')
                            self.checkpoint_model(sess, save_file)
                            chkpted = True

                if (early_stop_mode > 0):
                    if epoch > opt_params['early_stop'] and (delta < 0):
                        if not silent:
                            print('\n*** early stop criteria met...stopping train now...')
                            print('     ---> number of epochs used: %d,  '
                                  'end cost: %04f' % (epoch, cost_test))
                            print('     ---> best epoch: %d,  '
                                  'best cost: %04f\n' % (best_epoch, best_cost))
                        # restore saved variables into tf Variables
                        if output_dir is not None and chkpted and early_stop_mode > 0:
                            # save_file exists only if chkpted is True
                            self.saver.restore(sess, save_file)
                            # delete files before break to clean up space
                            shutil.rmtree(os.path.join(output_dir, 'bstmods'), ignore_errors=True)
                        break

        return epoch
        #    return epoch
        # END _train_adam_block

    def _get_test_cost_block(self, sess, input_data, output_data, data_filters, test_blocks, block_inds):
        """Utility function to clean up code in `_train_adam` method"""

        num_batches_test = len(test_blocks)

        cost_test = 0
        for batch_test in range(num_batches_test):
            # batch_indxs_test = test_indxs[batch_test * test_batch_size:(batch_test+1) * test_batch_size]
            if self.data_pipe_type == 'data_as_var':
                # feed_dict = {self.indices: batch_indxs_test}
                feed_dict = {self.indices: block_inds[test_blocks[batch_test]]}
            elif self.data_pipe_type == 'feed_dict':
                feed_dict = self._get_feed_dict(
                    input_data=input_data,
                    output_data=output_data,
                    data_filters=data_filters,
                    batch_indxs=block_inds[test_blocks[batch_test]])
                    #batch_indxs=batch_indxs_test)
            elif self.data_pipe_type == 'iterator':
                feed_dict = {self.iterator_handle: test_indxs}
            cost_test += sess.run(self.cost, feed_dict=feed_dict)

        cost_test /= num_batches_test

        return cost_test
    # END _get_test_cost_block

    def _build_dataset(self, input_data, output_data, data_filters=None,
                       indxs=None, batch_size=32, training_dataset=True):
        """Generates tf.data.Dataset object to be used with the `iterator` data
        pipeline"""

        # keep track of input tensors
        tensors = {}

        # INPUT DATA
        for i, input_size in enumerate(self.input_sizes):
            name = 'input_%02d' % i
            # add data to dict of input tensors
            tensors[name] = input_data[i][indxs, :]

        # OUTPUT DATA
        for i, output_size in enumerate(self.output_sizes):
            name = 'output_%02d' % i
            # add data to dict of input tensors
            tensors[name] = output_data[i][indxs, :]

        # DATA FILTERS
        if self.filter_data:
            for i, output_size in enumerate(self.output_sizes):
                name = 'filter_%02d' % i
                tensors[name] = data_filters[i][indxs, :]

        # construct dataset object from placeholder dict
        dataset = tf.data.Dataset.from_tensor_slices(tensors)

        if training_dataset:
            # auto shuffle data
            dataset = dataset.shuffle(buffer_size=10000)

        if batch_size > 0:
            # auto batch data
            dataset = dataset.batch(batch_size)

        # repeat (important that this comes after shuffling and batching)
        dataset = dataset.repeat()
        # prepare each batch on cpu while running previous through model on
        # GPU
        dataset = dataset.prefetch(buffer_size=1)

        return dataset

    def _restore_params(self, sess, input_data, output_data,
                        data_filters=None):
        """Restore model parameters from numpy matrices and update
        regularization values from list. This function is called by any other
        function that needs to initialize a new session to run parts of the
        graph."""

        # initialize all parameters randomly
        sess.run(self.init)

        if self.data_pipe_type == 'data_as_var':
            # check input
            if type(input_data) is not list:
                input_data = [input_data]
            if type(output_data) is not list:
                output_data = [output_data]

            # initialize input/output data
            for i, temp_data in enumerate(input_data):
                sess.run(self.data_in_var[i].initializer,
                         feed_dict={self.data_in_ph[i]: temp_data})
            for i, temp_data in enumerate(output_data):
                sess.run(self.data_out_var[i].initializer,
                         feed_dict={self.data_out_ph[i]: temp_data})
                if self.filter_data:
                    sess.run(
                        self.data_filter_var[i].initializer,
                        feed_dict={self.data_filter_ph[i]: data_filters[i]})

        # overwrite randomly initialized values of model with stored values
        self._assign_model_params(sess)

        # update regularization parameter values
        self._assign_reg_vals(sess)

#    def _assign_model_params(self, sess):
#        """Assigns parameter values previously stored in numpy arrays to
#        tf Variables in model; function needs to be implemented by specific
#        model"""
#        raise NotImplementedError()

#    def _assign_reg_vals(self, sess):
#        """Loops through all current regularization penalties and updates
#        parameter values in the tf Graph; needs to be implemented by specific
#        model"""
#        raise NotImplementedError()

    def checkpoint_model(self, sess, save_file):
        """Checkpoint model parameters in tf Variables

        Args:
            sess (tf.Session object): current session object to run graph
            save_file (str): full path to output file

        """

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        self.saver.save(sess, save_file)
        # print('Model checkpointed to %s' % save_file)

    def restore_model(self, save_file, input_data=None, output_data=None):
        """Restore previously checkpointed model parameters in tf Variables

        Args:
            save_file (str): full path to saved model
            input_data (time x input_dim numpy array, optional): input to
                network; required if self.data_pipe_type is `data_as_var`
            output_data (time x output_dim numpy array, optional): desired
                output of network; required if self.data_pipe_type is
                `data_as_var`

        Raises:
            ValueError: If `save_file` is not a valid filename
        """

        if not os.path.isfile(save_file + '.meta'):
            raise ValueError(str('%s is not a valid filename' % save_file))

        # check input
        if self.data_pipe_type == 'data_as_var':
            if type(input_data) is not list:
                input_data = [input_data]
            if type(output_data) is not list:
                output_data = [output_data]
            self.num_examples = input_data[0].shape[0]
            for temp_data in input_data:
                if temp_data.shape[0] != self.num_examples:
                    raise ValueError(
                        'Input data dims must match across input_data.')
            for nn, temp_data in enumerate(output_data):
                if temp_data.shape[0] != self.num_examples:
                    raise ValueError('Output dim0 must match model values')

        # Build graph: self._build_graph must be defined in child of network
        self._build_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # initialize tf params in new session
            self._restore_params(sess, input_data, output_data)
            # restore saved variables into tf Variables
            self.saver.restore(sess, save_file)
            # write out weights/biases to numpy arrays before session closes
            self._write_model_params(sess)

    def save_model(self, save_file):
        """Save full network object using dill (extension of pickle)

        Args:
            save_file (str): full path to output file
        """

        import sys
        import dill

        tmp_ndn = self.copy_model()
        sys.setrecursionlimit(10000)  # for dill calls to pickle

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        with open(save_file, 'wb') as f:
            dill.dump(tmp_ndn, f)
        print('Model pickled to %s' % save_file)
    # END NDN3.save_model

    def _data_format(self, input_data, output_data=None, data_filters=None):

        if type(input_data) is not list:
            input_data = [input_data]
        self.num_examples = input_data[0].shape[0]
        num_outputs = len(self.ffnet_out)

        if output_data is not None:
            if type(output_data) is not list:
                output_data = [output_data]
            # add dimensions to output data as needed
            for nn in range(len(output_data)):
                if len(output_data[nn].shape) == 1:
                    output_data[nn] = np.expand_dims(output_data[nn], 1)
        else:  # generate dummy data
            output_data = [None] * num_outputs
            for nn in range(num_outputs):
                output_data[nn] = np.zeros(
                    [self.num_examples, np.prod(self.networks[self.ffnet_out[nn]].layers[-1].output_dims)],
                    dtype='float32')

        if data_filters is not None:
            self.filter_data = True
            if type(data_filters) is not list:
                data_filters = [data_filters]
            assert len(data_filters) == len(output_data), 'Number of data filters must match output data.'
            # Also make sure Robs is zeroed out when data_filters is zero (for Poisson calc)
            output_data[nn] = np.multiply(output_data[nn], data_filters[nn])
        else:
            if self.filter_data:
                self.filter_data = False
                print('WARNING: not using data-filter despite previously using.')
            # Make dummy data-filters to be consistent with future operations
            data_filters = [None]*num_outputs
            for nn in range(num_outputs):
                data_filters[nn] = np.ones(output_data[nn].shape, dtype='float32')

        for temp_data in input_data:
            if temp_data.shape[0] != self.num_examples:
                raise ValueError('Input data dims must match across input_data.')
        for nn, temp_data in enumerate(output_data):
            if temp_data.shape[0] != self.num_examples:
                print('in', nn, temp_data.shape, self.num_examples)
                raise ValueError('Output dim0 must match model values')
            if len(temp_data.shape) < 2:
                output_data[nn] = np.expand_dims(temp_data, axis=1)
            # Protect single-dimension outputs
            if len(temp_data.shape) == 1:
                output_data[nn] = np.expand_dims(temp_data, axis=1)
            if self.filter_data:
                # Protect single-dimension outputs
                if len(data_filters[nn].shape) == 1:
                    data_filters[nn] = np.expand_dims(data_filters[nn], axis=1)
                assert data_filters[nn].shape == output_data[nn].shape, 'data_filter sizes must match output_data'
        return input_data, output_data, data_filters

    def _variable_batch_size_permitted( self ):
        """determines whether model can use variable batch_sizes"""
        
        if self.data_pipe_type == 'data_as_var':
            can_we = True
        else:
            can_we = False
        
        # Look for time_expansions not paired with temporal layers
        for nn in range(self.num_networks):
            if np.sum(self.network_list[nn]['time_expand']) > 0:
                for ll in range(len(self.networks[nn].layers)):
                    if (self.network_list[nn]['time_expand'][ll] > 0) and \
                            ((self.network_list[nn]['layer_types'][ll] != 'temporal') or \
                                (self.network_list[nn]['layer_types'][ll] != 'sp_temporal')):
                        can_we = False
        return can_we


    # noinspection PyInterpreter
    @classmethod
    def load_model(cls, save_file):
        """Restore previously saved network object

        Args:
            save_file (str): full path to saved model

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        import dill

        if not os.path.isfile(save_file):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with open(save_file, 'rb') as f:
            return dill.load(f)

    @classmethod
    def optimizer_defaults(cls, opt_params, learning_alg):
        """Sets defaults for different optimizers

        In the `opt_params` dictionary, the `display` and `use_gpu` keys are
        available for all optimizers. The following keys are used exclusively
        for lbfgs: `max_iter`, `func_tol`, `grad_tol` and `eps`. The remaining
        keys are all specific to the adam optimizer.

        Args:
            opt_params: dictionary with optimizer-specific parameters
            opt_params['display'] (int, optional): For adam, this defines the
                number of epochs between updates to the console. Becomes
                boolean for lbfgs, prints detailed optimizer info for each
                iteration.
                DEFAULT: 0/False
            opt_params['use_gpu'] (bool, optional): `True` to fit model on gpu.
                DEFAULT: False
            opt_params['data_pipe_type'] (int, optional): specify how data
                should be fed to the model.
                0: pin input/output data to tf.Variable; when fitting models
                    with a GPU, this puts all data on the GPU and avoids the
                    overhead associated with moving data from CPU to GPU. Works
                    well when data+model can fit on GPU
                1: standard use of feed_dict
                DEFAULT: 0
            opt_params['max_iter'] (int, optional): maximum iterations for
                lbfgs algorithm.
                DEFAULT: 500
            opt_params['func_tol'] (float, optional): see lbfgs method in SciPy
                optimizer. Also can be used for early stopping in adam
                DEFAULT: 2.22e-09
            opt_params['grad_tol'] (float, optional): see lbfgs method in SciPy
                optimizer.
                DEFAULT: 1e-05
            opt_params['eps'] (float, optional): see lbfgs method in SciPy
                optimizer.
                DEFAULT: 1e-08
            opt_params['learning_rate'] (float, optional): learning rate used
                by adam.
                DEFAULT: 1e-3.
            opt_params['batch_size'] (int, optional): number of data points to
                use for each iteration of training.
                DEFAULT: 128
            opt_params['epochs_training'] (int, optional): max number of
                epochs.
                DEFAULT: 100
            opt_params['epochs_ckpt'] (int, optional): number of epochs between
                saving checkpoint files.
                DEFAULT: `None`
            opt_params['early_stop_mode'] (int, optional): different options include
                0: don't chkpt, return the last model after loop break
                1: chkpt all models and choose the best one from the pool
                2: chkpt in a smart way, when training session is about to converge
                DEFAULT: `0`
            opt_params['MAPest'] (boolean, optional)
                True: include regularization penalty in judging early_stopping criteria
                False: only use test_set LL for early stopping
                DEFAULT: True
            opt_params['early_stop'] (int, optional): if greater than zero,
                training ends when the cost function evaluated on test_indxs is
                not lower than the maximum over that many previous checks.
                (Note that when early_stop > 0 and early_stop_mode = 1, early
                stopping will come in effect after epoch > early_stop pool size)
                DEFAULT: 0
            opt_params['beta1'] (float, optional): beta1 (1st momentum term)
                for Adam
                DEFAULT: 0.9
            opt_params['beta2'] (float, optional): beta2 (2nd momentum term)
                for Adam
                DEFAULT: 0.999
            opt_params['epsilon'] (float, optional): epsilon parameter in
                Adam optimizer
                DEFAULT: 1e-4 (note normal Adam default is 1e-8)
            opt_params['epochs_summary'] (int, optional): number of epochs
                between saving network summary information.
                DEFAULT: `None`
            opt_params['run_diagnostics'] (bool, optional): `True` to record
                compute time and memory usage of tensorflow ops during training
                and testing. `epochs_summary` must not be `None`.
                DEFAULT: `False`
            opt_params['poisson_unit_norm'] (None, or list of numbers, optional):
                'None' will not normalize, but list of length NC will. This can
                be set using NDN function set_poisson_norm
            learning_alg (str): 'adam' and 'lbfgs' currently supported
        """

        # Non-optimizer specific defaults
        if 'display' not in opt_params:
            opt_params['display'] = None
        if 'use_gpu' not in opt_params:
            opt_params['use_gpu'] = False
        if 'data_pipe_type' not in opt_params:
            opt_params['data_pipe_type'] = 'data_as_var'
        if 'poisson_unit_norm' not in opt_params:
            opt_params['poisson_unit_norm'] = None
        if 'epochs_ckpt' not in opt_params:
            opt_params['epochs_ckpt'] = None

        if learning_alg is 'adam':
            if 'learning_rate' not in opt_params:
                opt_params['learning_rate'] = 1e-4
            if 'batch_size' not in opt_params:
                opt_params['batch_size'] = None
            if 'epochs_training' not in opt_params:
                opt_params['epochs_training'] = 100
            if 'early_stop_mode' not in opt_params:
                opt_params['early_stop_mode'] = 0
            if 'MAPest' not in opt_params:
                opt_params['MAPest'] = True
            if 'func_tol' not in opt_params:
                opt_params['func_tol'] = 0
            if 'epochs_summary' not in opt_params:
                opt_params['epochs_summary'] = None
            if 'early_stop' not in opt_params:
                opt_params['early_stop'] = 1
            if 'beta1' not in opt_params:
                opt_params['beta1'] = 0.9
            if 'beta2' not in opt_params:
                opt_params['beta2'] = 0.999
            if 'epsilon' not in opt_params:
                opt_params['epsilon'] = 1e-8
            if 'run_diagnostics' not in opt_params:
                opt_params['run_diagnostics'] = False

        else:  # lbfgs
            if 'maxiter' not in opt_params:
                opt_params['maxiter'] = 500
            # The iteration will stop when
            # max{ | proj g_i | i = 1, ..., n} <= pgtol
            # where pg_i is the i-th component of the projected gradient.
            if 'func_tol' not in opt_params:
                opt_params['func_tol'] = 2.220446049250313e-09  # ?
            if 'grad_tol' not in opt_params:
                opt_params['grad_tol'] = 1e-05
            if 'eps' not in opt_params:
                opt_params['eps'] = 1e-08
            # Convert display variable to boolean
            if opt_params['display'] is None:
                opt_params['display'] = False
            if 'batch_size' not in opt_params:
                opt_params['batch_size'] = None

        return opt_params
    # END network.optimizer_defaults

    # FROM TNDN
    def _set_batch_size(self, new_batch_size):
        """
        :param new_size:
        :return:
        """

        # TNDN
        if hasattr(self, 'batch_size'):
            self.batch_size = new_batch_size

        # TFFNetworks, layers
        for nn in range(self.num_networks):
            if hasattr(self.networks[nn], 'batch_size'):
                self.networks[nn].batch_size = new_batch_size
            for mm in range(len(self.networks[nn].layers)):
                if hasattr(self.networks[nn].layers[mm], 'batch_size'):
                    self.networks[nn].layers[mm].batch_size = new_batch_size
    # END TNDN._set_batch_size

    def _set_time_spread(self, new_time_spread):
        """
        :param new_size:
        :return:
        """

        # TNDN
        if hasattr(self, 'time_spread'):
            self.time_spread = new_time_spread

        # TFFNetworks, layers
        for nn in range(self.num_networks):
            if hasattr(self.networks[nn], 'time_spread'):
                self.networks[nn].time_spread = new_time_spread
            for mm in range(len(self.networks[nn].layers)):
                if hasattr(self.networks[nn].layers[mm], 'time_spread'):
                    self.networks[nn].layers[mm].time_spread = new_time_spread
    # END TNDN._set_time_spread
