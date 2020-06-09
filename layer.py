"""Basic layer definitions"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from .regularization import Regularization
from .regularization import SepRegularization
from .regularization import UnitRegularization

from copy import deepcopy
from sklearn.preprocessing import normalize as sk_normalize


class Layer(object):
    """Implementation of fully connected neural network layer

    Attributes:
        scope (str): name scope for variables and operations in layer
        input_dims (list of ints): 3-dimensional internal representation of dimensions of input, with input_dims[0]
            combining num_filts and num_lags, and other two are spatial dimensions.
        num_lags (int): number of lags, to disambiguate first dimension of input_dims, and specifically used for
            temporal layers
        output_dims (list): outputs of layer
        outputs (tf Tensor): output of layer
        weights_ph (tf placeholder): placeholder for weights in layer
        biases_ph (tf placeholder): placeholder for biases in layer
        weights_var (tf Tensor): weights in layer
        biases_var (tf Tensor): biases in layer
        weights (numpy array): shadow variable of `weights_var` that allows for 
            easier manipulation outside of tf sessions
        biases (numpy array): shadow variable of `biases_var` that allows for 
            easier manipulation outside of tf sessions
        activation_func (tf activation function): activation function in layer
        reg (Regularization object): holds regularizations values and matrices
            (as tf constants) for layer
        ei_mask_var (tf constant): mask of +/-1s to multiply output of layer
        ei_mask (list): mask of +/-1s to multiply output of layer; shadows 
            `ei_mask_tf` for easier manipulation outside of tf sessions
        pos_constraint (None, valued): positivity constraint on weights in layer
        num_filters (int): equal to output_dims
        filter_dims (list of ints): equal to input_dims
        normalize_weights (int): defines normalization type for weights in 
            layer
        log (bool): use tf summary writers on layer output

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            filter_dims=None,
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for Layer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensionality, up to 4 dimensions. Should be in the
                form of: [# filts, # space dim1, # space dim2, num_lags]
                This will be converted to 3-dim internal representation where num_lags and # filt are combined,
                and t-regularization will operate on first dim of internal rep. [also max_filt]
                Default is none, which means will be determined at the build time
            output_dims (int or list of ints): size of outputs of layer, arranged as multi-dimensional list as well
            filter_dims (int or list of ints): dimensions of filters: if left blank will be determined by input_dims,
                but might be different for convolutional reasons (not implemented in layer, but its children)
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' or 'lin' | 'softplus' | 'leaky_relu' |
                'elu' | 'quad' | 'requ' (rect quadratic)
            normalize_weights (int): 1 to normalize weights, -1 to apply maxnorm, and 0 to leave unnormalized
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional)bias_init: initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer 
                activations

        Raises:
            TypeError: If `variable_scope` is not specified
            TypeError: If `input_dims` is not specified
            TypeError: If `output_dims` is not specified
            ValueError: If `activation_func` is not a valid string
            ValueError: If `num_inh` is greater than total number of units
            ValueError: If `weights_initializer` is not a valid string
            ValueError: If `biases_initializer` is not a valid string

        """
        
        _allowed_act_funcs = ['lin', 'relu', 'leaky_relu', 'softplus', 'exp',
                              'sigmoid', 'tanh', 'quad', 'elu', 'requ']

        if activation_func in _allowed_act_funcs:
            self.act_func = activation_func
            self.nl_param = 0.1    # right now only used for leaky_relu
        else:
            raise ValueError('Invalid activation function ''%s''' % activation_func)

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify layer scope')
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify both input and output dimensions')

        self.scope = scope

        # Parse input and filter dimensions: note that internal variable input_dims will be 3-D, defined by
        #    [# of non-spatial elements, # space dim1, #space dim2]. Any additional non-spatial dims will be
        #    broken up in separarte internal variable 'internal dims', which can be list of one or greater

        # Make input, output, and filter sizes explicit
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # input_dims = [1, input_dims, 1]
            input_dims = [input_dims, 1, 1]

        # Internal representation will be 3-dimensional, combining num_lags with input_dims[0]
        if len(input_dims) > 3:
            self.num_lags = input_dims[3]
            input_dims[0] *= input_dims[3]
        else:
            self.num_lags = 1
        self.input_dims = deepcopy(input_dims[:3])

        if isinstance(output_dims, list):
            while len(output_dims) < 3:
                output_dims.append(1)
            num_outputs = np.prod(output_dims)
        else:
            num_outputs = output_dims
            output_dims = [output_dims, 1, 1]

        self.output_dims = output_dims[:]
        # default to have N filts for N outputs in base layer class

        # print('layer', scope, self.input_dims, self.internal_dims)

        if filter_dims is None:
            if self.num_lags == 1:
                filter_dims = self.input_dims
            else:
                #filter_dims = self.input_dims + [self.num_lags]
                filter_dims = self.input_dims

        self.filter_dims = filter_dims[:]
        num_inputs = np.prod(self.filter_dims)
        self.num_filters = num_outputs

        # create excitatory/inhibitory mask
        if num_inh > num_outputs:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_outputs - num_inh) + [-1] * num_inh

        # save positivity constraint on weights
        self.pos_constraint = pos_constraint
        self.normalize_weights = normalize_weights

        # use tf's summary writer to save layer activation histograms
        if log_activations:
            self.log = True
        else:
            self.log = False
        
        # Set up layer regularization
        self.reg = Regularization(
            input_dims=filter_dims,
            num_outputs=num_outputs,
            vals=reg_initializer)

        # Initialize weight values
        weight_dims = (num_inputs, num_outputs)

        if weights_initializer == 'trunc_normal':
            init_weights = np.abs(np.random.normal(size=weight_dims, scale=0.1))
        elif weights_initializer == 'normal':
            init_weights = np.random.normal(size=weight_dims, scale=0.1)
        elif weights_initializer == 'zeros':
            init_weights = np.zeros(shape=weight_dims, dtype='float32')
        elif weights_initializer == 'ones':
            init_weights = np.ones(shape=weight_dims, dtype='float32')
        else:
            raise ValueError('Invalid weights_initializer ''%s''' %
                             weights_initializer)
        if pos_constraint is not None:
            init_weights = np.maximum(init_weights, 0)
        if normalize_weights > 0:
            init_weights = sk_normalize(init_weights, axis=0)
        elif normalize_weights < 0:
            init_weights = np.divide(init_weights, np.maximum(np.sqrt(np.sum(np.square(init_weights), axis=0)), 1))

        # Initialize numpy array that will feed placeholder
        self.weights = init_weights.astype('float32')

        # Initialize bias values
        bias_dims = (1, num_outputs)
        if biases_initializer == 'trunc_normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'zeros':
            init_biases = np.zeros(shape=bias_dims, dtype='float32')
        else:
            raise ValueError('Invalid biases_initializer ''%s''' %
                             biases_initializer)
        # Initialize numpy array that will feed placeholder
        self.biases = init_biases.astype('float32')

        # Define tensorflow variables as placeholders
        self.weights_ph = None
        self.weights_var = None
        self.biases_ph = None
        self.biases_var = None
        self.outputs = None
    # END Layer.__init__

    def _define_layer_variables(self):
        # Define tensor-flow versions of variables (placeholder and variables)

        with tf.name_scope('weights_init'):
            self.weights_ph = tf.placeholder_with_default(
                self.weights,
                shape=self.weights.shape,
                name='weights_ph')
            self.weights_var = tf.Variable(
                self.weights_ph,
                dtype=tf.float32,
                name='weights_var')

        # Initialize biases placeholder/variable
        with tf.name_scope('biases_init'):
            self.biases_ph = tf.placeholder_with_default(
                self.biases,
                shape=self.biases.shape,
                name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

        # Check for need of ei_mask
        if np.sum(self.ei_mask) < len(self.ei_mask):
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None
    # END Layer._define_layer_variables

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            elif self.normalize_weights < 0:
                w_pn = tf.divide(w_p, tf.maximum(tf.norm(w_p, axis=0), 1))
            else:
                w_pn = w_p

            _pre = tf.matmul(inputs, w_pn)
            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = self._apply_dropout(post, use_dropout=use_dropout,
                                               noise_shape=[1, self.num_filters])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END Layer.build_graph

    def set_nl_param(self, new_val):
        self.nl_param = new_val

    def _apply_act_func(self, pre):

        if self.act_func == 'relu':
            post = tf.nn.relu(pre)
        elif self.act_func == 'requ':
            post = tf.square(tf.nn.relu(pre))
        elif self.act_func == 'sigmoid':
            post = tf.sigmoid(pre)
        elif self.act_func == 'tanh':
            post = tf.tanh(pre)
        elif self.act_func == 'lin':
            post = pre
        elif self.act_func == 'softplus':
            post = tf.nn.softplus(pre)
        elif self.act_func == 'quad':
            post = tf.square(pre)
        elif self.act_func == 'elu':
            post = tf.nn.elu(pre)
        elif self.act_func == 'exp':
            post = tf.exp(pre)
        elif self.act_func == 'leaky_relu':
            post = tf.nn.leaky_relu(pre, self.nl_param)
        else:
            raise ValueError('act func not defined')

        return post

    def _apply_dropout(self, x, use_dropout, noise_shape):
        if use_dropout == True and self.reg.vals['dropout'] is not None:
            return tf.nn.dropout(x, keep_prob=self.reg.vals['dropout'], noise_shape=noise_shape)
        else:
            return x

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""

        # Adjust weights (pos and normalize) if appropriate
        if self.pos_constraint is not None:
            self.weights = np.maximum(self.weights, 0)
        if self.normalize_weights > 0:
            self.weights = sk_normalize(self.weights, axis=0)
        elif self.normalize_weights < 0:
            self.weights = np.divide(self.weights, np.maximum(np.sqrt(np.sum(np.square(self.weights), axis=0)), 1))

        sess.run(
            [self.weights_var.initializer, self.biases_var.initializer],
            feed_dict={self.weights_ph: self.weights,
                       self.biases_ph: self.biases})
    # END Layer.assign_layer_params

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        _tmp_weights = sess.run(self.weights_var)

        if self.pos_constraint is not None:
            w_p = np.maximum(_tmp_weights, 0)
        else:
            w_p = _tmp_weights

        if self.normalize_weights > 0:
            w_pn = sk_normalize(w_p, axis=0)
        elif self.normalize_weights < 0:
            w_pn = np.divide(w_p, np.maximum(np.sqrt(np.sum(np.square(w_p), axis=0)), 1))
        else:
            w_pn = w_p

        self.weights = w_pn
        self.biases = sess.run(self.biases_var)
    # END Layer.write_layer_params

    def copy_layer_params(self, origin_layer):
        """Copy layer parameters over to new layer (which is self in this case). 
        Only should be called by NDN.copy_network_params()."""

        self.weights = deepcopy(origin_layer.weights)
        self.biases = deepcopy(origin_layer.biases)
        self.weights = deepcopy(origin_layer.weights)
        self.reg = origin_layer.reg.reg_copy()
        self.normalize_weights = deepcopy(origin_layer.normalize_weights)

    def define_regularization_loss(self):
        """Wrapper function for building regularization portion of graph"""
        with tf.name_scope(self.scope):
            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            elif self.normalize_weights < 0:
                w_pn = tf.divide(w_p, tf.maximum(tf.norm(w_p, axis=0), 1))
            else:
                w_pn = w_p

            return self.reg.define_reg_loss(w_pn)

    def set_regularization(self, reg_type, reg_val):
        """Wrapper function for setting regularization"""
        return self.reg.set_reg_val(reg_type, reg_val)

    def assign_reg_vals(self, sess):
        """Wrapper function for assigning regularization values"""
        self.reg.assign_reg_vals(sess)

    def get_reg_pen(self, sess):
        """Wrapper function for returning regularization penalty dict"""
        return self.reg.get_reg_penalty(sess)


class ConvLayer(Layer):
    """Implementation of convolutional layer

    Attributes:
        shift_spacing (int): stride of convolution operation
        num_shifts (int): number of shifts in horizontal and vertical 
            directions for convolution operation
            
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,   # this can be a list up to 3-dimensions
            num_filters=None,
            filter_dims=None,  # this can be a list up to 3-dimensions
            shift_spacing=1,
            dilation=1,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ConvLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to  
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights, -1 to have maxnorm,  0 otherwise
                [0] | 1, -1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer 
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`
            
        """

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional (space)
            input_dims = [1, input_dims, 1]

        if filter_dims is None:
            filter_dims = input_dims
        else:
            if isinstance(filter_dims, list):
                while len(filter_dims) < 3:
                    filter_dims.extend(1)
            else:
                filter_dims = [filter_dims, 1, 1]

        # if nlags is not None:
        #    filter_dims[0] *= nlags

        # If output dimensions already established, just strip out num_filters
        if isinstance(num_filters, list):
            num_filters = num_filters[0]

        # Calculate number of shifts (for output)
        num_shifts = [1, 1]
        if input_dims[1] > 1:
            num_shifts[0] = int(np.ceil(input_dims[1]/shift_spacing))
        if input_dims[2] > 1:
            num_shifts[1] = int(np.ceil(input_dims[2]/shift_spacing))

        super(ConvLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=num_filters,   # Note difference from layer
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,  # note difference from layer (not anymore)
                log_activations=log_activations)

        # ConvLayer-specific properties
        self.shift_spacing = shift_spacing
        self.num_shifts = num_shifts
        self.dilation = dilation
        # Changes in properties from Layer - note this is implicitly multi-dimensional
        self.output_dims = [num_filters] + num_shifts[:]
    # END ConvLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        assert params_dict is not None, 'Incorrect ConvLayer initialization.'
        # Unfold siLayer-specific parameters for building graph

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Reshape of inputs (4-D):
            input_dims = [-1, self.input_dims[2], self.input_dims[1],
                          self.input_dims[0]]
            # this is reverse-order from Matlab:
            # [space-2, space-1, lags, and num_examples]
            shaped_input = tf.reshape(inputs, input_dims)

            # Reshape weights (4:D: [filter_height, filter_width, in_channels, out_channels])
            conv_filter_dims = [self.filter_dims[2], self.filter_dims[1], self.filter_dims[0],
                                self.num_filters]

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            elif self.normalize_weights < 0:
                w_pn = tf.divide(w_p, tf.maximum(tf.norm(w_p, axis=0), 1))
            else:
                w_pn = w_p

            ws_conv = tf.reshape(w_pn, conv_filter_dims)

            # Make strides and dilation lists, 2D lists were failing on TF:12.0.2 for some reason
            strides, dilation = [1, 1, 1, 1], [1, 1, 1, 1]
            if conv_filter_dims[0] > 1:             # Assumes data_format: NHWC
                strides[1] = self.shift_spacing
                dilation[1] = self.dilation
            if conv_filter_dims[1] > 1:
                strides[2] = self.shift_spacing
                dilation[2] = self.dilation

            _pre = tf.nn.conv2d(shaped_input, ws_conv, strides=strides, dilations=dilation, padding='SAME')
            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            post_drpd = self._apply_dropout(post, use_dropout=use_dropout,
                                            noise_shape=[1, 1, 1, self.num_filters])
            self.outputs = tf.reshape(
                post_drpd, [-1, self.num_filters * self.num_shifts[0] * self.num_shifts[1]])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END ConvLayer.build_graph


# class ClusterLayer(Layer):
#     """Implementation of 'Cluster' layer: combined multiple subunits into separate streams for fitting
#     non-shared subunits. This is non-convolutional. For now, just uses layer constructor, treated as nornal,
#     and have to set num subunits away from default (2) to ge
#
#     Attributes:
#         num_subs: how many subunits to associate with each output unit
#     """
#
#     def __init__(
#             self,
#             scope=None,
#             input_dims=None,  # this can be a list up to 3-dimensions
#             output_dims=None,
#             activation_func='relu',
#             normalize_weights=0,
#             weights_initializer='normal',
#             biases_initializer='zeros',
#             reg_initializer=None,
#             num_inh=0,
#             pos_constraint=None,
#             log_activations=False):
#
#         assert num_subs is not None, 'Must give number of subs associated with each output unit'
#
#         super(ClusterLayer, self).__init__(
#             scope=scope,
#             input_dims=input_dims,
#             output_dims=output_dims,  # Note difference from layer
#             activation_func=activation_func,
#             normalize_weights=normalize_weights,
#             weights_initializer=weights_initializer,
#             biases_initializer=biases_initializer,
#             reg_initializer=reg_initializer,
#             num_inh=num_inh,
#             pos_constraint=pos_constraint,  # note difference from layer (not anymore)
#             log_activations=log_activations)
#
#         # ConvLayer-specific properties
#         self.num_subs = num_subs
#     # END ClusterLayer.__init__
#
#     def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):


class ConvReadoutLayer(Layer):
    """Reduces convolutional input to single output (num_outputs) with each output corresponding
    to one spatial location within the conv input, specified by internal variable xy_out. Weights 
    are over filter_dimensions after spatial reduction

    Attributes:
        shift_spacing (int): stride of convolution operation
        num_shifts (int): number of shifts in horizontal and vertical
            directions for convolution operation
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            num_filters=None,
            shift_spacing=1,
            activation_func='relu',
            xy_out=None,
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ConvLayerXY class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            xy_out (int array): num_filters x 2 array for the spatial output of each filter
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights, -1 to apply maxnorm, 0 otherwise
                [0] | 1, -1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        Does not currently save indices -- have to be set manually, externally at the moment
        """

        # Process stim and filter dimensions
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional 
            input_dims = [input_dims, 1, 1]

        # Internal representation will be 3-dimensional, combining num_lags with input_dims[0]
        if len(input_dims) > 3:
            self.num_lags = input_dims[3]
            input_dims[0] *= input_dims[3]
        else:
            self.num_lags = 1
        self.input_dims = input_dims[:3].copy()
        assert np.prod(self.input_dims[1:]) > 1, 'ConvReadout must be reading out input with spatial dims.'

        filter_dims = [self.input_dims[0], 1, 1]

        # If output dimensions already established, just strip out num_filters
        if isinstance(num_filters, list):
            num_filters = num_filters[0]

        super(ConvReadoutLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            filter_dims=filter_dims,
            output_dims=num_filters,  # Note difference from layer
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,  # note difference from layer (not anymore)
            log_activations=log_activations)

        
        self.xy_out = xy_out
    # END ConvReadoutLayer.__init__

    def _space_collapse_tensor(self):

        assert self.xy_out is not None, 'xy_out in ConvReadout must be set.'
        num_space = np.prod(self.input_dims[1:])

        pysct = np.zeros([self.input_dims[2], self.input_dims[1], self.xy_out.shape[0]])
        pysct[self.xy_out] = 1.0
        return np.reshape(pysct, [num_space, self.xy_out.shape[0]])       

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        #assert params_dict is not None, 'Incorrect Layer initialization.'

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Collapse spatial-convolution across space
            sct = tf.constant(self._space_collapse_tensor(), dtype=tf.float32)

            shaped_input = tf.reshape(inputs, [-1, self.input_dims[2]*self.input_dims[1], self.input_dims[0]])
            spatial_collapse_stim = tf.tensordot( shaped_input, sct, [1, 0])

            # Prepare weights
            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            if self.normalize_weights < 0:
                w_pn = tf.divide(w_p, tf.maximum(tf.norm(w_p, axis=0), 1))
            else:
                w_pn = w_p

            _pre = tf.add(tf.reduce_sum(
                tf.multiply(spatial_collapse_stim, w_pn), axis=1), self.biases_var)

            if self.ei_mask_var is None:
                _post = self._apply_act_func(_pre)
            else:
                _post = tf.multiply(self._apply_act_func(_pre), self.ei_mask_var)

            self.outputs = self._apply_dropout(_post, use_dropout=use_dropout,
                            noise_shape=[1, self.num_filters])
                
        if self.log:
            tf.summary.histogram('act_pre', _pre)
            tf.summary.histogram('act_post', _post)
    # END ConvXYLayer.build_graph


class ConvXYLayer(Layer):
    """Implementation of convolutional layer with additional XY output

    Attributes:
        shift_spacing (int): stride of convolution operation
        num_shifts (int): number of shifts in horizontal and vertical
            directions for convolution operation
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            num_filters=None,
            filter_dims=None,  # this can be a list up to 3-dimensions
            shift_spacing=1,
            xy_out=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ConvLayerXY class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            xy_out (int array): num_filters x 2 array for the spatial output of each filter
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights, -1 to apply maxnorm, 0 otherwise
                [0] | 1, -1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        Does not currently save indices -- have to be set manually, externally at the moment
        """

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            # assume 1-dimensional 
            input_dims = [input_dims, 1, 1]

 #       if xy_out is not None:
  #          filter_dims = [input_dims[0], 1, 1]
   #     else:
    #        filter_dims = None

        # If output dimensions already established, just strip out num_filters
        if isinstance(num_filters, list):
            num_filters = num_filters[0]

        # Calculate number of shifts (for output)
        num_shifts = [1, 1]
        if input_dims[1] > 1:
            num_shifts[0] = int(np.floor(input_dims[1] / shift_spacing))
        if input_dims[2] > 1:
            num_shifts[1] = int(np.floor(input_dims[2] / shift_spacing))

        super(ConvXYLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            filter_dims=filter_dims,
            output_dims=num_filters,  # Note difference from layer
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,  # note difference from layer (not anymore)
            log_activations=log_activations)

        # ConvLayer-specific properties
        self.shift_spacing = shift_spacing
        self.num_shifts = num_shifts
        # Changes in properties from Layer - note this is implicitly
        # multi-dimensional
        self.xy_out = xy_out
        if self.xy_out is None:
            self.output_dims = [num_filters] + num_shifts[:]
        else:
            self.output_dims = [num_filters, 1, 1]

    # END ConvXYLayer.__init__

    def _get_indices(self, batch_sz):
        """Old function used for previous implementation"""
        nc = self.num_filters
        space_ind = zip(np.repeat(np.arange(batch_sz), nc),
                        np.tile(self.xy_out[:, 1], (batch_sz,)),
                        np.tile(self.xy_out[:, 0], (batch_sz,)),
                        np.tile(np.arange(nc), (batch_sz,)))
        return tf.constant(space_ind, dtype=tf.int32)

    def _space_collapse_tensor(self):
        num_space = np.prod(self.num_shifts)

        pysct = np.zeros([self.num_shifts[0], self.num_shifts[1], self.xy_out.shape[0]])
        pysct[self.xy_out] = 1.0
        return np.reshape(pysct, [num_space, self.xy_out.shape[0]])       

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        assert params_dict is not None, 'Incorrect Layer initialization.'
        # Unfold Layer-specific parameters for building graph

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Define spatial-collapse tensor
            sct = tf.constant(self._space_collapse_tensor(), dtype=tf.float32)

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            if self.normalize_weights < 0:
                w_pn = tf.divide(w_p, tf.maximum(tf.norm(w_p, axis=0), 1))
            else:
                w_pn = w_p

            ws_conv = tf.reshape(w_pn, [self.filter_dims[2], self.filter_dims[1], self.filter_dims[0], self.num_filters])
            shaped_input = tf.reshape(inputs, [-1, self.input_dims[2], self.input_dims[1], self.input_dims[0]])

            strides = [1, 1, 1, 1]
            if self.filter_dims[2] > 1:
                strides[1] = self.shift_spacing
            if self.filter_dims[1] > 1:
                strides[2] = self.shift_spacing

            _pre0 = tf.nn.conv2d(shaped_input, ws_conv, strides, padding='SAME')

            print(sct)
            print(_pre0)

            if self.xy_out is not None:
                #indices = self._get_indices(batch_size)
                #_pre = tf.reshape(tf.gather_nd(_pre0, indices), (-1, self.num_filters))
                _pre = tf.tensordot(tf.reshape(_pre0, [-1, self.num_filters, self.num_shifts[0]*self.num_shifts[1] ]),
                                    sct, [2, 0])
                print(_pre)
            else:
                _pre = _pre0

            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            # reminder: self.xy_out = centers[which_cell] = [ctr_x, ctr_y]
            if self.xy_out is not None:
                self.outputs = self._apply_dropout(post, use_dropout=use_dropout,
                                                   noise_shape=[1, self.num_filters])
            else:
                post_drpd = self._apply_dropout(post, use_dropout=use_dropout,
                                                noise_shape=[1, 1, 1, self.num_filters])
                self.outputs = tf.reshape(
                    post_drpd, [-1, self.num_filters * self.num_shifts[0] * self.num_shifts[1]])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END ConvXYLayer.build_graph


class SepLayer(Layer):
    """Implementation of separable neural network layer; see 
    http://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where
    for more info

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,    # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for SepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data: [num_filter_dims, NX, NY, num_lags]
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the 
                weights. Default [0] is to normalize across the first dimension 
                (time/filters), but '1' will normalize across spatial 
                dimensions instead, and '2' will normalize both
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        """

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [1, input_dims, 1]  # assume 1-dimensional (space)

        # Determine filter dimensions
        if len(input_dims) > 3:
            num_lags = input_dims[3]
        else:
            num_lags = 1
        num_space = input_dims[1]*input_dims[2]

        filter_dims = [input_dims[0]*num_lags + num_space, 1, 1]
        if normalize_weights < 0:
            print('WARNING: maxnorm not implemented for SepLayer')

        super(SepLayer, self).__init__(
                scope=scope,
                #nlags=nlags,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=output_dims,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        self.partial_fit = None

        # Redefine specialized Regularization object to overwrite default
        self.reg = SepRegularization(
            input_dims=input_dims,
            num_outputs=self.reg.num_outputs,
            vals=reg_initializer)
    # END SepLayer.__init_

    def _define_layer_variables(self):
        # Define tensor-flow versions of variables (placeholder and variables)

        if self.partial_fit == 0:
            wt = self.weights[:self.input_dims[0], :]
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    wt, shape=wt.shape, name='wt_ph')
                self.weights_var = tf.Variable(
                    self.weights_ph, dtype=tf.float32, name='wt_var')
        elif self.partial_fit == 1:
            ws = self.weights[self.input_dims[0]:, :]
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    ws, shape=ws.shape, name='ws_ph')
                self.weights_var = tf.Variable(
                    self.weights_ph, dtype=tf.float32, name='ws_var')
        else:
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    self.weights,
                    shape=self.weights.shape,
                    name='weights_ph')
                self.weights_var = tf.Variable(
                    self.weights_ph,
                    dtype=tf.float32,
                    name='weights_var')

        # Initialize biases placeholder/variable
        with tf.name_scope('biases_init'):
            self.biases_ph = tf.placeholder_with_default(
                self.biases,
                shape=self.biases.shape,
                name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

        # Check for need of ei_mask
        if np.sum(self.ei_mask) < len(self.ei_mask):
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None
    # END SepLayer._define_layer_variables

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        if self.partial_fit == 0:
            wt = self.weights[:self.input_dims[0], :]
            sess.run(
                [self.weights_var.initializer, self.biases_var.initializer],
                feed_dict={self.weights_ph: wt, self.biases_ph: self.biases})
        elif self.partial_fit == 1:
            ws = self.weights[self.input_dims[0]:, :]
            sess.run(
                [self.weights_var.initializer, self.biases_var.initializer],
                feed_dict={self.weights_ph: ws, self.biases_ph: self.biases})
        else:
            sess.run(
                [self.weights_var.initializer, self.biases_var.initializer],
                feed_dict={self.weights_ph: self.weights, self.biases_ph: self.biases})
    # END SepLayer.assign_layer_params

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        # rebuild self.weights
        if self.partial_fit == 0:
            wt = sess.run(self.weights_var)
            ws = deepcopy(self.weights[self.input_dims[0]:, :])
        elif self.partial_fit == 1:
            wt = deepcopy(self.weights[:self.input_dims[0], :])
            ws = sess.run(self.weights_var)
        else:
            _tmp = sess.run(self.weights_var)
            wt = _tmp[:self.input_dims[0], :]
            ws = _tmp[self.input_dims[0]:, :]

        if self.pos_constraint == 0:
            wt_p = np.maximum(0.0, wt)
            ws_p = ws
        elif self.pos_constraint == 1:
            ws_p = np.maximum(0.0, ws)
            wt_p = wt
        elif self.pos_constraint == 2:
            wt_p = np.maximum(0.0, wt)
            ws_p = np.maximum(0.0, ws)
        else:
            wt_p = wt
            ws_p = ws

        # Normalize weights (one or both dimensions)
        if self.normalize_weights == 0:
            wt_pn = sk_normalize(wt_p, axis=0)
            ws_pn = ws_p
        elif self.normalize_weights == 1:
            wt_pn = wt_p
            ws_pn = sk_normalize(ws_p, axis=0)
        elif self.normalize_weights == 2:
            wt_pn = sk_normalize(wt_p, axis=0)
            ws_pn = sk_normalize(ws_p, axis=0)
        else:
            wt_pn = wt_p
            ws_pn = ws_p

        self.weights[:self.input_dims[0], :] = wt_pn
        self.weights[self.input_dims[0]:, :] = ws_pn

        self.biases = sess.run(self.biases_var)
    # END SepLayer.write_layer_params

    def _separate_weights(self):
        # Section weights into first dimension and space
        if self.partial_fit == 0:
            kt = self.weights_var
            ks = tf.constant(self.weights[self.input_dims[0]:, :], dtype=tf.float32)
        elif self.partial_fit == 1:
            kt = tf.constant(self.weights[:self.input_dims[0], :], dtype=tf.float32)
            ks = self.weights_var
        else:
            kt = tf.slice(self.weights_var, [0, 0],
                          [self.input_dims[0], self.num_filters])
            ks = tf.slice(self.weights_var, [self.input_dims[0], 0],
                          [self.input_dims[1] * self.input_dims[2], self.num_filters])

        if self.pos_constraint == 0:
            kt_p = tf.maximum(0.0, kt)
            ks_p = ks
        elif self.pos_constraint == 1:
            kt_p = kt
            ks_p = tf.maximum(0.0, ks)
        elif self.pos_constraint == 2:
            kt_p = tf.maximum(0.0, kt)
            ks_p = tf.maximum(0.0, ks)
        else:
            kt_p = kt
            ks_p = ks

        # Normalize weights (one or both dimensions)
        if self.normalize_weights == 0:
            kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
            ks_pn = ks_p
        elif self.normalize_weights == 1:
            kt_pn = kt_p
            ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
        elif self.normalize_weights == 2:
            kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
            ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
        else:
            kt_pn = kt_p
            ks_pn = ks_p

        w_full = tf.transpose(tf.reshape(tf.matmul(
            tf.expand_dims(tf.transpose(ks_pn), 2),
            tf.expand_dims(tf.transpose(kt_pn), 1)),
            [self.num_filters, np.prod(self.input_dims)]))

        return w_full

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            weights_full = self._separate_weights()

            _pre = tf.matmul(inputs, weights_full)
            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = self._apply_dropout(post, use_dropout=use_dropout,
                                               noise_shape=[1, self.num_filters])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END sepLayer._build_layer

    def define_regularization_loss(self):
        """overloaded function to handle different normalization in SepLayer"""
        with tf.name_scope(self.scope):
            # Section weights into first dimension and space
            if self.partial_fit == 0:
                kt = self.weights_var
                ks = tf.constant(self.weights[self.input_dims[0]:, :], dtype=tf.float32)
            elif self.partial_fit == 1:
                kt = tf.constant(self.weights[:self.input_dims[0], :], dtype=tf.float32)
                ks = self.weights_var
            else:
                kt = tf.slice(self.weights_var, [0, 0],
                              [self.input_dims[0], self.num_filters])
                ks = tf.slice(self.weights_var, [self.input_dims[0], 0],
                              [self.input_dims[1] * self.input_dims[2], self.num_filters])

            if self.pos_constraint == 0:
                kt_p = tf.maximum(0.0, kt)
                ks_p = ks
            elif self.pos_constraint == 1:
                kt_p = kt
                ks_p = tf.maximum(0.0, ks)
            elif self.pos_constraint == 2:
                kt_p = tf.maximum(0.0, kt)
                ks_p = tf.maximum(0.0, ks)
            else:
                kt_p = kt
                ks_p = ks

            # Normalize weights (one or both dimensions)
            if self.normalize_weights == 0:
                kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
                ks_pn = ks_p
            elif self.normalize_weights == 1:
                kt_pn = kt_p
                ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
            elif self.normalize_weights == 2:
                kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
                ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
            else:
                kt_pn = kt_p
                ks_pn = ks_p

            if self.partial_fit == 0:
                return self.reg.define_reg_loss(kt_pn)
            elif self.partial_fit == 1:
                return self.reg.define_reg_loss(ks_pn)
            else:
                return self.reg.define_reg_loss(tf.concat([kt_pn, ks_pn], 0))


class ConvSepLayer(Layer):
    """Implementation of separable neural network layer; see
    http://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where
    for more info
    """

    def __init__(
            self,
            scope=None,
            #nlags=None,
            input_dims=None,    # this can be a list up to 3-dimensions
            num_filters=None,
            filter_dims=None,  # this can be a list up to 3-dimensions
            shift_spacing=1,
            # output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, Valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        """

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [input_dims, 1, 1]  # assume 1-dimensional (filter-dim)

        if filter_dims is None:
            filter_dims = input_dims
        else:
            if isinstance(filter_dims, list):
                while len(filter_dims) < 3:
                    filter_dims.extend(1)
            else:
                filter_dims = [filter_dims, 1, 1]

        #if nlags is not None:
        #    filter_dims[0] *= nlags
        # Determine filter dimensions (first dim + space_dims)
        num_space = filter_dims[1]*filter_dims[2]
        num_input_dims_convsep = filter_dims[0]+num_space

        # If output dimensions already established, just strip out num_filters
        if isinstance(num_filters, list):
            num_filters = num_filters[0]

        # Calculate number of shifts (for output)
        num_shifts = [1, 1]
        if input_dims[1] > 1:
            num_shifts[0] = int(np.floor(input_dims[1]/shift_spacing))
        if input_dims[2] > 1:
            num_shifts[1] = int(np.floor(input_dims[2]/shift_spacing))
        if normalize_weights < 0:
            print('WARNING: maxnorm not implemented for ConvSepLayer')

        super(ConvSepLayer, self).__init__(
                scope=scope,
                #nlags=nlags,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=num_filters,     # ! check this out... output_dims=num_filters?
                my_num_inputs=num_input_dims_convsep,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        self.partial_fit = None

        # Redefine specialized Regularization object to overwrite default
        self.reg = SepRegularization(
            input_dims=filter_dims,
            num_outputs=self.num_filters,
            vals=reg_initializer)

        # ConvLayer-specific properties
        self.shift_spacing = shift_spacing
        self.num_shifts = num_shifts
        # Changes in properties from Layer - note this is implicitly
        # multi-dimensional
        self.output_dims = [num_filters] + num_shifts[:]
    # END ConvSepLayer.__init__

    #############################################################################################

    def _define_layer_variables(self):
        # Define tensor-flow versions of variables (placeholder and variables)

        if self.partial_fit == 0:
            wt = self.weights[:self.input_dims[0], :]
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    wt, shape=wt.shape, name='wt_ph')
                self.weights_var = tf.Variable(
                    self.weights_ph, dtype=tf.float32, name='wt_var')
        elif self.partial_fit == 1:
            ws = self.weights[self.input_dims[0]:, :]
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    ws, shape=ws.shape, name='ws_ph')
                self.weights_var = tf.Variable(
                    self.weights_ph, dtype=tf.float32, name='ws_var')
        else:
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    self.weights,
                    shape=self.weights.shape,
                    name='weights_ph')
                self.weights_var = tf.Variable(
                    self.weights_ph,
                    dtype=tf.float32,
                    name='weights_var')

        # Initialize biases placeholder/variable
        with tf.name_scope('biases_init'):
            self.biases_ph = tf.placeholder_with_default(
                self.biases,
                shape=self.biases.shape,
                name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

        # Check for need of ei_mask
        if np.sum(self.ei_mask) < len(self.ei_mask):
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None
    # END ConvSepLayer._define_layer_variables

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        if self.partial_fit == 0:
            wt = self.weights[:self.input_dims[0], :]
            sess.run(
                [self.weights_var.initializer, self.biases_var.initializer],
                feed_dict={self.weights_ph: wt, self.biases_ph: self.biases})
        elif self.partial_fit == 1:
            ws = self.weights[self.input_dims[0]:, :]
            sess.run(
                [self.weights_var.initializer, self.biases_var.initializer],
                feed_dict={self.weights_ph: ws, self.biases_ph: self.biases})
        else:
            sess.run(
                [self.weights_var.initializer, self.biases_var.initializer],
                feed_dict={self.weights_ph: self.weights, self.biases_ph: self.biases})
    # END ConvSepLayer.assign_layer_params

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        # rebuild self.weights
        if self.partial_fit == 0:
            wt = sess.run(self.weights_var)
            ws = deepcopy(self.weights[self.input_dims[0]:, :])
        elif self.partial_fit == 1:
            wt = deepcopy(self.weights[:self.input_dims[0], :])
            ws = sess.run(self.weights_var)
        else:
            _tmp = sess.run(self.weights_var)
            wt = _tmp[:self.input_dims[0], :]
            ws = _tmp[self.input_dims[0]:, :]

        if self.pos_constraint == 0:
            wt_p = np.maximum(0.0, wt)
            ws_p = ws
        elif self.pos_constraint == 1:
            ws_p = np.maximum(0.0, ws)
            wt_p = wt
        elif self.pos_constraint == 2:
            wt_p = np.maximum(0.0, wt)
            ws_p = np.maximum(0.0, ws)
        else:
            wt_p = wt
            ws_p = ws

        # Normalize weights (one or both dimensions)
        if self.normalize_weights == 0:
            wt_pn = sk_normalize(wt_p, axis=0)
            ws_pn = ws_p
        elif self.normalize_weights == 1:
            wt_pn = wt_p
            ws_pn = sk_normalize(ws_p, axis=0)
        elif self.normalize_weights == 2:
            wt_pn = sk_normalize(wt_p, axis=0)
            ws_pn = sk_normalize(ws_p, axis=0)
        else:
            wt_pn = wt_p
            ws_pn = ws_p

        self.weights[:self.input_dims[0], :] = wt_pn
        self.weights[self.input_dims[0]:, :] = ws_pn

        self.biases = sess.run(self.biases_var)
    # END ConvSepLayer.write_layer_params

    def define_regularization_loss(self):
        """overloaded function to handle different normalization in SepLayer"""
        with tf.name_scope(self.scope):
            # Section weights into first dimension and space
            if self.partial_fit == 0:
                kt = self.weights_var
                ks = tf.constant(self.weights[self.input_dims[0]:, :], dtype=tf.float32)
            elif self.partial_fit == 1:
                kt = tf.constant(self.weights[:self.input_dims[0], :], dtype=tf.float32)
                ks = self.weights_var
            else:
                kt = tf.slice(self.weights_var, [0, 0],
                              [self.input_dims[0], self.num_filters])
                ks = tf.slice(self.weights_var, [self.input_dims[0], 0],
                              [self.filter_dims[1] * self.filter_dims[2], self.num_filters])

            if self.pos_constraint == 0:
                kt_p = tf.maximum(0.0, kt)
                ks_p = ks
            elif self.pos_constraint == 1:
                kt_p = kt
                ks_p = tf.maximum(0.0, ks)
            elif self.pos_constraint == 2:
                kt_p = tf.maximum(0.0, kt)
                ks_p = tf.maximum(0.0, ks)
            else:
                kt_p = kt
                ks_p = ks

            # Normalize weights (one or both dimensions)
            if self.normalize_weights == 0:
                kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
                ks_pn = ks_p
            elif self.normalize_weights == 1:
                kt_pn = kt_p
                ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
            elif self.normalize_weights == 2:
                kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
                ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
            else:
                kt_pn = kt_p
                ks_pn = ks_p

            if self.partial_fit == 0:
                return self.reg.define_reg_loss(kt_pn)
            elif self.partial_fit == 1:
                return self.reg.define_reg_loss(ks_pn)
            else:
                return self.reg.define_reg_loss(tf.concat([kt_pn, ks_pn], 0))

    def _separate_weights(self):
        # Section weights into first dimension and space
        if self.partial_fit == 0:
            kt = self.weights_var
            ks = tf.constant(self.weights[self.input_dims[0]:, :], dtype=tf.float32)
        elif self.partial_fit == 1:
            kt = tf.constant(self.weights[:self.input_dims[0], :], dtype=tf.float32)
            ks = self.weights_var
        else:
            kt = tf.slice(self.weights_var, [0, 0],
                          [self.input_dims[0], self.num_filters])
            ks = tf.slice(self.weights_var, [self.input_dims[0], 0],
                          [self.filter_dims[1] * self.filter_dims[2], self.num_filters])

        if self.pos_constraint == 0:
            kt_p = tf.maximum(0.0, kt)
            ks_p = ks
        elif self.pos_constraint == 1:
            kt_p = kt
            ks_p = tf.maximum(0.0, ks)
        elif self.pos_constraint == 2:
            kt_p = tf.maximum(0.0, kt)
            ks_p = tf.maximum(0.0, ks)
        else:
            kt_p = kt
            ks_p = ks

        # Normalize weights (one or both dimensions)
        if self.normalize_weights == 0:
            kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
            ks_pn = ks_p
        elif self.normalize_weights == 1:
            kt_pn = kt_p
            ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
        elif self.normalize_weights == 2:
            kt_pn = tf.nn.l2_normalize(kt_p, axis=0)
            ks_pn = tf.nn.l2_normalize(ks_p, axis=0)
        else:
            kt_pn = kt_p
            ks_pn = ks_p

        w_full = tf.transpose(tf.reshape(tf.matmul(
            tf.expand_dims(tf.transpose(ks_pn), 2),
            tf.expand_dims(tf.transpose(kt_pn), 1)),
            [self.num_filters, np.prod(self.filter_dims)]))

        return w_full

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        with tf.name_scope(self.scope):
            self._define_layer_variables()

            weights_full = self._separate_weights()

            # now conv part of the computation begins:
            # Reshape of inputs (4-D):
            input_dims = [-1, self.input_dims[2], self.input_dims[1],
                          self.input_dims[0]]
            shaped_input = tf.reshape(inputs, input_dims)

            # Reshape weights (4:D:
            conv_filter_dims = [self.filter_dims[2], self.filter_dims[1],
                                self.filter_dims[0], self.num_filters]
            ws_conv = tf.reshape(weights_full, conv_filter_dims)

            # Make strides list
            strides = [1, 1, 1, 1]
            if conv_filter_dims[0] > 1:
                strides[1] = self.shift_spacing
            if conv_filter_dims[1] > 1:
                strides[2] = self.shift_spacing

            _pre = tf.nn.conv2d(shaped_input, ws_conv, strides, padding='SAME')
            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            post_drpd = self._apply_dropout(post, use_dropout=use_dropout,
                                            noise_shape=[1, 1, 1, self.num_filters])
            self.outputs = tf.reshape(
                post_drpd, [-1, self.num_filters * self.num_shifts[0] * self.num_shifts[1]])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END ConvSepLayer._build_layer


class AddLayer(Layer):
    """Implementation of a simple additive layer that combines several input streams additively.
    This has a number of [output] units, and number of input streams, each which the exact same
    size as the number of output units. Each output unit then does a weighted sum over its matching
    inputs (with a weight for each input stream)

    """

    def __init__(
            self,
            scope=None,
            #nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations
        """

        # check for required inputs
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify input and output dimensions')

        num_outputs = np.prod( output_dims )
        num_input_streams = int(np.prod(input_dims) / num_outputs)

        # Input dims is just number of input streams
        input_dims = [num_input_streams, 1, 1]

        if normalize_weights < 0:
            print('WARNING: maxnorm not implemented for SepLayer')

        super(AddLayer, self).__init__(
                scope=scope,
                #nlags=nlags,
                input_dims=input_dims,
                filter_dims=input_dims,
                output_dims=num_outputs,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer='zeros',
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Initialize all weights to 1, which is the default combination
        self.weights[:, :] = 1.0/np.sqrt(num_input_streams)
        self.biases[:] = 1e-8

    # END AddLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        """By definition, the inputs will be composed of a number of input streams, given by
        the first dimension of input_dims, and each stream will have the same number of inputs
        as the number of output units."""

        num_input_streams = self.input_dims[0]
        num_outputs = self.output_dims[0]
        # inputs will be NTx(num_input_streamsxnum_outputs)

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if num_input_streams == 1:
                _pre = tf.multiply(inputs, w_p)
            else:
                if self.normalize_weights > 0:
                    w_pn = tf.nn.l2_normalize(w_p, axis=0)
                else:
                    w_pn = w_p

                flattened_weights = tf.reshape(w_pn, [1, num_input_streams*num_outputs])
                # Define computation -- different from layer in that this is a broadcast-multiply
                # rather than  matmul
                # Sum over input streams for given output
                _pre = tf.reduce_sum(tf.reshape(tf.multiply(inputs, flattened_weights),
                                                [-1, num_input_streams, num_outputs]), axis=1)

            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = self._apply_dropout(post, use_dropout=use_dropout,
                                               noise_shape=[1, self.num_filters])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END AddLayer._build_graph

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        num_input_streams = self.input_dims[0]
        num_outputs = self.output_dims[1]
        _tmp_weights = sess.run(self.weights_var)

        if self.pos_constraint is not None:
            w_p = np.maximum(_tmp_weights, 0.0)
        else:
            w_p = _tmp_weights

        if (self.normalize_weights > 0) and (num_input_streams != 1):
            w_pn = sk_normalize(w_p, axis=0)
        else:
            w_pn = w_p

        #flattened_weights = np.reshape(w_pn, [num_input_streams, num_outputs])
        self.weights = w_pn #flattened_weights
        self.biases = sess.run(self.biases_var)


class MultLayer(Layer):
    """Implementation of a simple multiplicative layer that combines two input streams multiplicatively,
    with the output of each pair of streams (x_n, y_n) given by z_n = x_n * (1+y_n)
    This has a number of [output] units, and number of input streams, each which the exact same
    size as the number of output units. 
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations
        """

        # check for required inputs
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify input and output dimensions')

        num_outputs = np.prod(output_dims)
        num_input_streams = int(np.prod(input_dims) / num_outputs)
        # Enforce number of input streams is 2
        assert num_input_streams == 2, 'Number of input streams for MultLayer must be 2.'

        # Input dims is just number of input streams
        input_dims = [num_input_streams, 1, 1]
        # can have overal weights, but not otherwise fit (so weight dims is output dims

        super(MultLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=[1, 1, 1],
                output_dims=num_outputs,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer='normal',
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Initialize all weights to very small deviations about 0, which would have minimal mult power
        self.weights[:, :] = 1.0 # default is multplication is transparent

    # END MultLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        """By definition, the inputs will be composed of a number of input streams, given by
        the first dimension of input_dims, and each stream will have the same number of inputs
        as the number of output units."""

        num_outputs = self.output_dims[0]
        # inputs will be NTx2

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                if self.pos_constraint > 0:
                    w_p = tf.maximum(self.weights_var, 0.0)
                else:  
                    w_p = tf.maximum(self.weights_var, tf.minimum(self.weights_var, 0.0), -1)
            else:
                w_p = self.weights_var

            multipliers = tf.maximum( tf.add(
                tf.multiply( tf.slice( inputs, [0, num_outputs], [-1, num_outputs]), w_p),
                1.0), 0.0)

            #if self.normalize_weights > 0:
            #    w_pn = tf.nn.l2_normalize(w_p, axis=0)
            #else:
            #    w_pn = w_p

            pre = tf.add(tf.multiply(
                tf.slice( inputs, [0, 0], [-1, num_outputs]), 
                multipliers), self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = self._apply_dropout(post, use_dropout=use_dropout,
                                               noise_shape=[1, self.num_filters])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END MultLayer._build_graph

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""

        num_input_streams = self.input_dims[0]
        #num_outputs = self.output_dims[1]
        _tmp_weights = sess.run(self.weights_var)

        if self.pos_constraint is not None:
            if self.pos_constraint > 0:
                w_p = np.maximum(_tmp_weights, 0.0)
            else:
                w_p = np.minimum(_tmp_weights, 0.0)
        else:
            w_p = _tmp_weights

        #if (self.normalize_weights > 0) and (num_input_streams != 1):
        #    w_pn = sk_normalize(w_p, axis=0)
        #else:
        #    w_pn = w_p

        self.weights = w_p
        self.biases = sess.run(self.biases_var)
    # END MultLayer.write_layer_params


class FilterLayer(Layer):
    """This layer has a weight vector to be applied to dim-0 (# filters) of input stimulus, with
    a different filter for each of the other (spatial) dimensions. Thus, there needs to be a filter
    for each spatial location
    STILL EXPERIMENTAL (UNTESTED) -- not updated for current version yet
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            activation_func='lin',
            normalize_weights=1,
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations
        """

        # check for required inputs
        if input_dims is None:
            raise TypeError('Must specify input dimensions')

        super(FilterLayer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=[input_dims[0], 1, 1],
                output_dims=[1]+input_dims[1:3],
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer='zeros',
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        self.include_biases = False
    # END FilterLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        """By definition, the inputs will be composed of a number of input streams, given by
        the first dimension of input_dims, and each stream will have the same number of inputs
        as the number of output units."""

        num_outputs = np.prod(self.output_dims)

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            # reshape inputs and multiply
            mult_inputs = tf.reshape(inputs, [-1, num_outputs, self.filter_dims[0]])

            if self.include_biases:
                pre = tf.add(
                    tf.reduce_sum(tf.multiply(mult_inputs, tf.transpose(w_pn)), axis=2),
                    self.biases_var)
            else:
                pre = tf.reduce_sum(tf.multiply(mult_inputs, tf.transpose(w_pn)), axis=2)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = self._apply_dropout(post, use_dropout=use_dropout,
                                               noise_shape=[1, self.num_filters])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END FilterLayer._build_graph


class SpkNL_Layer(Layer):
    """Implementation of separable neural network layer; see
    http://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where
    for more info
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,    # this can be a list up to 3-dimensions
            log_activations=False):
        """Constructor for SepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        """

        # Process stim and filter dimensions
        # (potentially both passed in as num_inputs list)
        if isinstance(input_dims, list):
            while len(input_dims) < 3:
                input_dims.append(1)
        else:
            input_dims = [1, input_dims, 1]  # assume 1-dimensional (space)


        super(SpkNL_Layer, self).__init__(
                scope=scope,
                input_dims=input_dims,
                filter_dims=[2, 1, 1],
                output_dims=input_dims,
                activation_func='lin',
                normalize_weights=0,
                weights_initializer='ones',
                biases_initializer='zeros',
                reg_initializer=None,
                num_inh=0,
                pos_constraint=False,
                log_activations=log_activations)

        self.three_param_fit = False
    # END SpkNLLayer.__init_

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        """"""

        _MAX_Y_3PAR = 5
        if self.three_param_fit:
            self.weights[0, :] = np.minimum(self.weights[:, 0], _MAX_Y_3PAR-0.2)

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.three_param_fit:
                _pre = tf.add(tf.multiply(inputs, _MAX_Y_3PAR * tf.sigmoid(self.weights_var[:, 0])),
                              self.biases_var)
                post = tf.multiply(self.weights_var[1,:], tf.log( tf.add( tf.exp(_pre), 1.0)))
            else:
                #_pre = tf.add( tf.multiply( inputs, self.weights_var[:, 0]), self.biases_var)
                #post = tf.log( tf.add( tf.exp(_pre), 1.0) )
                post = tf.nn.softplus(tf.add(tf.multiply(inputs, self.weights_var[:, 0]), self.biases_var))

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', _pre)
            tf.summary.histogram('act_post', post)
    # END SpkNL._build_graph


class SpikeHistoryLayer(Layer):
    """Implementation of a simple additive layer that combines several input streams additively.
    This has a number of [output] units, and number of input streams, each which the exact same
    size as the number of output units. Each output unit then does a weighted sum over its matching
    inputs (with a weight for each input stream)

    """

    def __init__(
            self,
            scope=None,
            #nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for sepLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations
        """

        # check for required inputs
        if input_dims is None or output_dims is None:
            raise TypeError('Must specify input and output dimensions')
        filter_dims = input_dims[:]
        filter_dims[1] = 1

        super(SpikeHistoryLayer, self).__init__(
                scope=scope,
                #nlags=nlags,
                input_dims=input_dims,
                filter_dims=filter_dims,
                output_dims=output_dims,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer='trunc_normal',
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Initialize all weights to be positive (and will be multiplied by -1
        self.biases[:] = 0

    # END SpikeHistorylayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        """By definition, the inputs will be composed of a number of input streams, given by
        the first dimension of input_dims, and each stream will have the same number of inputs
        as the number of output units."""

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                ws_flat = tf.reshape(tf.maximum(0.0, tf.transpose(self.weights_var)),
                                     [1, self.input_dims[0]*self.input_dims[1]])
            else:
                ws_flat = tf.reshape(tf.transpose(self.weights_var),
                                     [1, self.input_dims[0]*self.input_dims[1]])

            pre = tf.reduce_sum(tf.reshape(tf.multiply(inputs, ws_flat),
                                           [-1, self.input_dims[1], self.input_dims[0]]),
                                axis=2)

            # Dont put in any biases: pre = tf.add( pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END SpikeHistoryLayer._build_graph


class BiConvLayer(ConvLayer):
    """Implementation of binocular convolutional layer

    Attributes:
        shift_spacing (int): stride of convolution operation
        num_shifts (int): number of shifts in horizontal and vertical
            directions for convolution operation

    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            num_filters=None,
            filter_dims=None,  # this can be a list up to 3-dimensions
            include_reflect=False,
            shift_spacing=1,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for convLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            include_reflect (boolean): doubles number of filters -> includes reflection filters
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        super(BiConvLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_dims,
            shift_spacing=shift_spacing,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,  # note difference from layer (not anymore)
            log_activations=log_activations)

        # BiConvLayer-specific modifications
        self.include_reflect = include_reflect
        self.num_shifts[0] = self.num_shifts[0]
        self.output_dims[0] = self.num_filters*2
        self.output_dims[1] = int(self.num_shifts[0]/2)
        if include_reflect:
            self.output_dims[0] *= 2
            print('Reflecting:', self.output_dims)
    # END BiConvLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        assert params_dict is not None, 'Incorrect layer initialization.'

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Computation performed in the layer
            # Reshape of inputs (4-D):
            input_dims = [-1, self.input_dims[2],
                          self.input_dims[1], self.input_dims[0]]
            # this is reverse-order from Matlab:
            # [space-2, space-1, lags, and num_examples]
            shaped_input = tf.reshape(inputs, input_dims)

            # Reshape weight-dims (4:D):
            conv_filter_dims = [self.filter_dims[2], self.filter_dims[1],
                                self.filter_dims[0], self.num_filters]

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            if self.include_reflect:
                transform_mat = np.kron(np.flipud(np.eye(self.filter_dims[1])), np.eye(self.filter_dims[0]))
                flip_sp_dims = tf.constant(transform_mat, dtype='float32')
                w_c = tf.concat([w_pn, tf.matmul(flip_sp_dims, w_pn)], 1)

                biases = tf.concat([self.biases_var, self.biases_var], 1)
                ei_mask = tf.concat([self.ei_mask_var, self.ei_mask_var], 0)
                conv_filter_dims[3] *= 2
            else:
                w_c = w_pn
                biases = self.biases_var
                ei_mask = self.ei_mask_var
            ws_conv = tf.reshape(w_c, conv_filter_dims)

            # Make strides list
            strides = [1, 1, 1, 1]
            if conv_filter_dims[0] > 1:
                strides[1] = self.shift_spacing
            if conv_filter_dims[1] > 1:
                strides[2] = self.shift_spacing

            _pre = tf.nn.conv2d(shaped_input, ws_conv, strides, padding='SAME')
            pre = tf.add(_pre, biases)

            if self.ei_mask_var is not None:
                post = tf.multiply(self._apply_act_func(pre), ei_mask)
            else:
                post = self._apply_act_func(pre)

            # cut into left and right processing and reattach
            left_post = tf.slice(post, [0, 0, 0, 0], [-1, -1, self.output_dims[1], -1])
            right_post = tf.slice(post, [0, 0, self.output_dims[1], 0],
                             [-1, -1, self.output_dims[1], -1])

            self.outputs = tf.reshape(tf.concat([left_post, right_post], axis=3), [-1, np.prod(self.output_dims)])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END BiConvLayer.build_graph


class ReadoutLayer(Layer):
    """Implementation of readout layer, with main difference from Layer being regularization
    on neuron-by-neuron basis"""

    def __init__(
            self,
            scope=None,
            input_dims=None,    # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ReadoutLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): type of normalization to apply to the
                weights. Default [0] is to normalize across the first dimension
                (time/filters), but '1' will normalize across spatial
                dimensions instead, and '2' will normalize both
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        """
        # Implement layer almost verbatim 
        super(ReadoutLayer, self).__init__(
                scope=scope,
                #nlags=nlags,
                input_dims=input_dims,
                filter_dims=None,
                output_dims=output_dims,
                activation_func=activation_func,
                normalize_weights=normalize_weights,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations)

        # Redefine specialized Regularization object to overwrite default
        self.reg = UnitRegularization(
            input_dims=self.input_dims,
            num_outputs=self.reg.num_outputs,
            vals=reg_initializer)

    # END ReadoutLayer.__init_


class ConvLayerLNL(ConvLayer):
    """Implementation of convolutional layer with both linear and nonlinear outputs

    Attributes:
        shift_spacing (int): stride of convolution operation
        num_shifts (int): number of shifts in horizontal and vertical
            directions for convolution operation

    """
    def __init__(
            self,
            scope=None,
            #nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            num_filters=None,
            filter_dims=None,  # this can be a list up to 3-dimensions
            shift_spacing=1,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ConvLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        super(ConvLayerLNL, self).__init__(
            scope=scope,
            #nlags=nlags,
            input_dims=input_dims,
            filter_dims=filter_dims,
            output_dims=num_filters,  # Note difference from layer
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,  # note difference from layer (not anymore)
            log_activations=log_activations)

        # LNL-ConvLayer-specific properties -- additional set of outputs corresponding to linear
        # output of filters
        self.output_dims[0] = self.output_dims[0]*2

    # END ConvLayerLNL.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        assert params_dict is not None, 'Incorrect siLayer initialization.'
        # Unfold siLayer-specific parameters for building graph

        with tf.name_scope(self.scope):
            self._define_layer_variables()

            # Computation performed in the layer
            # Reshape of inputs (4-D):
            input_dims = [-1, self.input_dims[2], self.input_dims[1],
                          self.input_dims[0]]
            # this is reverse-order from Matlab:
            # [space-2, space-1, lags, and num_examples]
            shaped_input = tf.reshape(inputs, input_dims)

            # Reshape weights (4:D:
            conv_filter_dims = [self.filter_dims[2], self.filter_dims[1], self.filter_dims[0],
                                self.num_filters]

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            ws_conv = tf.reshape(w_pn, conv_filter_dims)

            # Make strides list
            # check back later (this seems to not match with conv_filter_dims)
            strides = [1, 1, 1, 1]
            if conv_filter_dims[1] > 1:
                strides[1] = self.shift_spacing
            if conv_filter_dims[2] > 1:
                strides[2] = self.shift_spacing

            _pre = tf.nn.conv2d(shaped_input, ws_conv, strides, padding='SAME')
            pre = tf.add(_pre, self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            pre_post = tf.concat(tf.multiply(pre, self.ei_mask_var), post, axis=-1)

            post_drpd = self._apply_dropout(pre_post, use_dropout=use_dropout,
                                            noise_shape=[1, 1, 1, 2*self.num_filters])
            self.outputs = tf.reshape(
                post_drpd, [-1, 2*self.num_filters * self.num_shifts[0] * self.num_shifts[1]])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END ConvLayerLNL.build_graph


# TODO: deal with this later (if useful at all)
class HadiReadoutLayer(Layer):
    """Implementation of a readout layer compatible with convolutional layers

    Attributes:


    """

    def __init__(
            self,
            scope=None,
            #nlags=None,
            input_dims=None,  # this can be a list up to 3-dimensions
            num_filters=None,
            xy_out=None,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ConvLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            num_filters (int): number of convolutional filters in layer
            filter_dims (int or list of ints): dimensions of input data
            shift_spacing (int): stride of convolution operation
            activation_func (str, optional): pointwise function applied to
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            normalize_weights (int): 1 to normalize weights 0 otherwise
                [0] | 1
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (None, valued): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        if xy_out is not None:
            filter_dims = [input_dims[0], 1, 1]
        else:
            filter_dims = None

        super(HadiReadoutLayer, self).__init__(
            scope=scope,
            #nlags=nlags,
            input_dims=input_dims,
            filter_dims=filter_dims,
            output_dims=num_filters,  # Note difference from layer
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,  # note difference from layer (not anymore)
            log_activations=log_activations)

        self.xy_out = xy_out
        if self.xy_out is None:
            self.output_dims = [1, num_filters, 1]
        else:
            self.output_dims = [num_filters, 1, 1]

        # Unit Reg
        self.reg = UnitRegularization(
            input_dims=[input_dims[0], 1, 1],
            num_outputs=self.reg.num_outputs,
            vals=reg_initializer)
    # END HadiReadoutLayer.__init__

    def _get_indices(self, batch_sz):
        nc = self.num_filters
        space_pos = self.xy_out[:, 1] * self.input_dims[1] + self.xy_out[:, 0]
        space_ind = zip(np.repeat(np.arange(batch_sz), nc),
                        np.tile(space_pos, (batch_sz,)),
                        np.tile(np.arange(nc), (batch_sz,)))
        return tf.constant(space_ind, dtype=tf.int32)

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                k_p = tf.maximum(self.weights_var, 0.0)
            else:
                k_p = self.weights_var

            if self.normalize_weights > 0:
                k_pn = tf.nn.l2_normalize(k_p, axis=0)
            else:
                k_pn = k_p

            # shapes:
            # shaped_inputs -> (b, ny*nx, nf)
            # k_pn -> (nf, nc)
            # inputs_dot_w -> (b, ny*nx, nc)
            # pre0/pre -> (b, nc)

            shaped_inputs = tf.reshape(
                inputs, (-1, self.input_dims[2] * self.input_dims[1], self.input_dims[0]))

            indices = self._get_indices(int(shaped_inputs.shape[0]))

            if self.xy_out is not None:
                inputs_dot_w = tf.tensordot(shaped_inputs, k_pn, [-1, 0])
                pre0 = tf.reshape(tf.gather_nd(inputs_dot_w, indices), (-1, self.num_filters))
                pre = tf.add(pre0, self.biases_var)
            else:
                pre = tf.add(tf.matmul(inputs, k_pn), self.biases_var)

            if self.ei_mask_var is None:
                post = self._apply_act_func(pre)
            else:
                post = tf.multiply(self._apply_act_func(pre), self.ei_mask_var)

            self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END ConvReadoutLayer.build_graph


class GridSampleLayer(Layer):
    """Implementation of a readout layer compatible with convolutional layers

    Attributes:

    """

    def __init__(
            self,
            scope=None, # this can be a list up to 3-dimensions
            input_dims=None,
            n_neurons=1,
            activation_func='relu',
            normalize_weights=0,
            weights_initializer='normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=None,
            log_activations=False):
        """Constructor for ConvLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data in format [CHW], number of batches are not included
            num_filters (int): number of convolutional filters in layer

        Raises:
            ValueError: If `pos_constraint` is `True`

        """
        self.n_neurons = n_neurons
        super(GridSampleLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            filter_dims=[n_neurons],
            output_dims=input_dims[0],
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,  # note difference from layer (not anymore)
            log_activations=log_activations)
        #Correct output dims
        self.output_dims = [1,n_neurons,1]
        self.same_points_in_batch  = True
        bias_dims = (1, n_neurons)
        if biases_initializer == 'trunc_normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'normal':
            init_biases = np.random.normal(size=bias_dims, scale=0.1)
        elif biases_initializer == 'zeros':
            init_biases = np.zeros(shape=bias_dims, dtype='float32')
        else:
            raise ValueError('Invalid biases_initializer ''%s''' %
                             biases_initializer)
        # Initialize numpy array that will feed placeholder
        self.biases = init_biases.astype('float32')

    # END GridSampleLayer.__init__

    def _split_input(self, input): 
        field_len = np.prod(self.input_dims)
        field,grid_points = tf.split(input,[field_len,self.n_neurons*2],axis=1)
        field = tf.reshape(field, (-1, self.input_dims[2], self.input_dims[1], self.input_dims[0]))
        grid_points = tf.reshape(grid_points,(-1,self.n_neurons,2))
        
        return field,grid_points
 

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        with tf.name_scope(self.scope):
            self._define_layer_variables()

            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            if self.normalize_weights < 0:
                w_pn = tf.divide(w_p, tf.maximum(tf.norm(w_p, axis=0), 1))
            else:
                w_pn = w_p

            assert len(inputs)==2, 'Wrong inputs'
            field, grid_points = inputs
            field = tf.reshape(field, (-1, self.input_dims[2], self.input_dims[1], self.input_dims[0]))
            grid_points = tf.reshape(grid_points,(-1,self.n_neurons,2))
            h,w = self.input_dims[2],self.input_dims[1]
            
            # If grid_points are same for whole batch
            if self.same_points_in_batch:
                tiling = tf.concat([[tf.shape(field)[0]],[1,1]],axis=0)
                grid_points = tf.tile(grid_points,tiling)

            # Find interpolation sides
            i, j = grid_points[..., 0], grid_points[..., 1]
            i = tf.cast(h - 1, grid_points.dtype) * (i + 1) / 2
            j = tf.cast(w - 1, grid_points.dtype) * (j + 1) / 2
            i_1 = tf.maximum(tf.cast(tf.floor(i), tf.int32), 0)
            i_2 = tf.minimum(i_1 + 1, h - 1)
            j_1 = tf.maximum(tf.cast(tf.floor(j), tf.int32), 0)
            j_2 = tf.minimum(j_1 + 1, w - 1)
                
            #Batch indicies
            tiling = tf.concat([[1],tf.shape(i)[1:]],axis=0)
            ind = tf.tile(tf.range(tf.shape(field)[0])[:,tf.newaxis],tiling)
            ind_11 = tf.stack([ind,i_1,j_1],axis=-1)
            ind_12 = tf.stack([ind,i_1,j_2],axis=-1)
            ind_21 = tf.stack([ind,i_2,j_1],axis=-1)
            ind_22 = tf.stack([ind,i_2,j_2],axis=-1)

            # Gather pixels
            q_11 = tf.gather_nd(field,ind_11)
            q_12 = tf.gather_nd(field,ind_12)
            q_21 = tf.gather_nd(field,ind_21)
            q_22 = tf.gather_nd(field,ind_22)

            # Interpolation coefficients
            di = tf.cast(i, field.dtype) - tf.cast(i_1, field.dtype)
            di = tf.expand_dims(di, -1)
            dj = tf.cast(j, field.dtype) - tf.cast(j_1, field.dtype)
            dj = tf.expand_dims(dj, -1)

            # Final interpolation
            q_i1 = q_11 * (1 - di) + q_21 * di
            q_i2 = q_12 * (1 - di) + q_22 * di
            outputs_shaped = q_i1 * (1 - dj) + q_i2 * dj
            

            weighted_features = tf.multiply(w_pn,outputs_shaped,name='weighting')
            weighted_sum = tf.reduce_sum(weighted_features,axis=-1,name='sum_along_channels')
            biased_activation = tf.add(weighted_sum,self.biases_var,name='add_biases')
            neuron_activations = self._apply_act_func(biased_activation)

            # Shape output to shape [Batch, Neurons]
            self.outputs = tf.reshape(neuron_activations,(-1,self.n_neurons))
  
        # END GridSampleLayer.build_graph


class GridShifterLayer(Layer):
    """Implementation of a layer, which remembers grid points in bias variables. As input it recieves 2 values (shift), and will shift the grid points with this vector. Output and grid point coordinates are clipped between -1 and 1

    """

    def __init__(
            self,
            scope=None, # this can be a list up to 3-dimensions
            input_dims=None,
            biases_initializer='zeros',
            reg_initializer=None,
            n_neurons=1 ):
        """Constructor for GridShifterLayer class

        Args:
            scope (str): name scope for variables and operations in layer
            input_dims (int or list of ints): dimensions of input data
            n_neurons (int): number of neuron coordinates to remember. It will return (x,y) coordinates for each neuron
        """

        super(GridShifterLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            output_dims = n_neurons*2)

        self.n_neurons = n_neurons

    # END GridSampleLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):
        with tf.name_scope(self.scope):
            self._define_layer_variables() 
            
            tiled_shift = tf.tile(inputs,(1,self.n_neurons),name='Copy_shift')        
            shifted = tf.add(tiled_shift, self.biases_var, name='Shift_grid_points')
            clipped = tf.clip_by_value(shifted,-1,1)
            
            # Reshape to NDN interface shape
            self.outputs = tf.reshape(clipped,[-1,2*self.n_neurons])

    def _define_layer_variables(self):
        # Define tensor-flow versions of variables (placeholder and variables)
        with tf.name_scope('weights_init'):
            self.weights_ph = tf.placeholder_with_default(
                self.weights,
                shape=self.weights.shape,
                name='weights_ph')
            self.weights_var = tf.Variable(
                self.weights_ph,
                dtype=tf.float32,
                name='weights_var')
        # Initialize biases placeholder/variable
        with tf.name_scope('biases_init'):
            self.biases_ph = tf.placeholder_with_default(
                self.biases,
                shape=self.biases.shape,
                name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, -1., 1.),
                name='biases_var')
    # END ShiftGraidLayer._define_layer_variables

            

    # END GridSampleLayer.build_graph
