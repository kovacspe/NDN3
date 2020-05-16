"""Basic temporal-layer definitions"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from .regularization import Regularization
from .regularization import SepRegularization
from .NDNutils import tent_basis_generate

from .layer import Layer
from copy import deepcopy


class TLayer(Layer):
    """Implementation of a temporal layer, which uses temporal (and causal) convolutions to handle num_lags, without having it
    present in input.

    Attributes:
        Alterations from Layer: now num_lags is explicitly kept as input_dims[0] so temporal regularization can
            by applied. All other dims are kept as-is.
        Note the build_graph explicitly uses batch_size (which is passed in then)
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,  # this can be a list up to 3-dimensions (note should not include lags)
            num_lags=1,
            num_filters=None,
            activation_func='lin',
            dilation=1,
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
            input_dims (int or list of ints): size (and dimensionality) of the inputs to the layer. Should be in the
                form of: [# filts, # space dim1, # space dim2]. Note that num_lags is not a property of the input
                and is handled separately (see below).
                Default is none, which means will be determined at the build time.
            num_lags (int): number of lags to be generated within the TLayer (and operated on by filters).
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
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            ValueError: If `pos_constraint` is `True`

        """

        # First format information for standard processing by Layer (parent)
        if len(input_dims) > 3:
            assert input_dims[3] == num_lags, 'No lags can be included in input for temporal layer.'
            input_dims = input_dims[:3]
        assert num_lags > 1, 'time_expand must be greater than 1 if using a temporal layer'

        # output_dims are preserved (from input_dims, with extra filters from different temporal convolutions
        output_dims = input_dims.copy()
        output_dims[0] *= num_filters

        # Incorporate num_input_filters into first spatial dimension (all lumped together anyway), so that
        # temporal regularization can be applied to first dimension
        #input_dims[1] *= input_dims[0]
        #input_dims[0] = num_lags

        super(TLayer, self).__init__(
            scope=scope,
            input_dims=input_dims,
            filter_dims=[num_lags, 1, 1],
            output_dims=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        # Note that num_lags is set explicitly here, and is not part of input_dims
        self.num_lags = num_lags
        self.dilation = dilation
        self.output_dims = output_dims

        self.reg = Regularization(
            input_dims=[num_lags, 1, 1],
            num_outputs=num_filters,
            vals=reg_initializer)

        if activation_func == 'lin':
            self.include_biases = False
        else:
            self.include_biases = True

        # Add filter-basis dummy (that can be filled in later)
        self.filter_basis = None
    # END TLayer.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        num_inputs = np.prod(self.input_dims)
        with tf.name_scope(self.scope):
            self._define_layer_variables()
            
            # make shaped input
            shaped_input = tf.reshape(tf.transpose(inputs), [num_inputs, -1, 1, 1]) 
            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            if self.filter_basis is not None:
                # make constant-tensor if using temporal basis functions 
                filter_basis = tf.constant( self.filter_basis, name='filter_basis')
                ks = tf.matmul( filter_basis, w_pn )
            else:
                ks = w_pn

            padding = tf.constant([[0, self.num_lags], [0, 0]])
            padded_filt = tf.pad(tf.reverse(ks, [0]), padding)  # note the time-reversal

            shaped_padded_filt = tf.reshape(padded_filt, [2*self.num_lags, 1, 1, self.num_filters])

            # Temporal convolution
            strides = [1, 1, 1, 1]
            dilations = [1, self.dilation, 1, 1]
            _pre = tf.nn.conv2d(shaped_input, shaped_padded_filt, strides, dilations=dilations, padding='SAME')

            if self.include_biases:
                _post = self._apply_act_func(tf.add(_pre, self.biases_var))
            else:
                _post = self._apply_act_func(_pre)

            if self.ei_mask_var is None:
                _post2 = _post
            else:
                _post2 = tf.multiply(_post, self.ei_mask_var)

            self.outputs = tf.reshape(tf.transpose(_post2, [1, 0, 2, 3]), (-1, np.prod(self.output_dims)))

        if self.log:
            tf.summary.histogram('act_pre', _pre)
            tf.summary.histogram('act_post', _post)
    # END TLayer.build_graph

    def init_temporal_basis( 
        self, 
        filter_basis=None,
        xs=None, 
        num_params=None, doubling_time=None, init_spacing=1,
        first_lag=0, end_at_zero=False
        ):
        """Initializes temporal layer with tent-bases, calling the NDNutils function tent_basis_generate.
        It will make tent_basis over the range of 'xs', with center points at each value of 'xs'
        Alternatively (if xs=None), will generate a list with init_space and doubling_time up to
        the total number of parameters. Must specify xs OR num_param.
        
        Note that this basis must extend up to the number of lags initialized. Will adjust number of
        weights to the number of parameters specified here."""
 
        if filter_basis is not None:
            self.filter_basis = filter_basis
        else:
            self.filter_basis = tent_basis_generate(xs=xs, num_params=num_params, first_lag=first_lag,
                                        doubling_time=doubling_time, init_spacing=init_spacing)
        [NTbasis, num_params] = self.filter_basis.shape
        # Find anchor in last basis element
        last_anchor = np.argmax(self.filter_basis[:,-1])
        
        if last_anchor >= self.num_lags:
            print('Warning: basis elements will be truncated or limited by num_lags set for temporal basis.')

        # Truncate or grow to fit number of lags specified
        if NTbasis > self.num_lags:
            print("  Temporal layer: must truncate temporal basis from %d to %d."%(NTbasis, self.num_lags))
            self.filter_basis = self.filter_basis[range(self.num_lags), :]
            if end_at_zero:
                # go smoothly to zero
                dx = self.num_lags-last_anchor-1
                self.filter_basis[range(last_anchor, self.num_lags), -1] = 1-np.array(list(range(dx+1)))/dx
        elif NTbasis < self.num_lags:
            print("  Temporal layer: must expand temporal basis from %d to %d."%(NTbasis, self.num_lags))
            self.filter_basis = np.concatenate(
                (self.filter_basis, np.zeros([self.num_lags-NTbasis, num_params], dtype='float32')), 
                axis=0)
            if end_at_zero:
                # go smoothly to zero
                dx = self.num_lags-last_anchor-1
                self.filter_basis[range(last_anchor, self.num_lags), -1] = 1-np.array(list(range(dx+1)))/dx
            else:
                # extend last basis element over the whole range
                self.filter_basis[NTbasis:,-1] = 1  

        # Adjust number of weights in layer to reflect parameterization with basis
        if self.filter_dims[0] < num_params:
            print("Weird. Less parameters in the layer than should be. Error likely.")
        else:
            print("  Temporal layer: updating number of weights in temporal layer from %d to %d."
                    %(self.filter_dims[0], num_params))
        self.filter_dims[0] = num_params
        self.weights = self.weights[range(num_params), :]
        self.reg.input_dims[0] = num_params
    # END TLayer.init_temporal_basis

    def copy_layer_params(self, origin_layer):
        """Copy layer parameters over to new layer (which is self in this case) -- overloaded for TLayer
        to also copy temporal_basis if relevant."""

        super(TLayer, self).copy_layer_params(origin_layer)
        #self.weights = deepcopy(origin_layer.weights)
        #self.biases = deepcopy(origin_layer.biases)
        #self.weights = deepcopy(origin_layer.weights)
        #self.reg = origin_layer.reg.reg_copy()
        #self.normalize_weights = deepcopy(origin_layer.normalize_weights)
        # new addition
        self.filter_basis = deepcopy(origin_layer.filter_basis)
    # END TLayer.copy_layer_params


class TLayerSpecific(TLayer):
    """Implementation of a temporal layer where each temporal kernel processes a different input: input dims
    must match the number of outputs, and collapses with resulting temporal function across time lag

    Attributes:
        Alterations from Layer: now num_lags is explicitly kept as input_dims[0] so temporal regularization can
            by applied. All other dims are kept as-is.
        Note the build_graph explicitly uses batch_size (which is passed in then)
    """

    def __init__(
            self,
            scope=None,
            input_dims=None,
            num_lags=1,
            num_filters=None,
            activation_func='lin',
            dilation=1,
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
            input_dims (int or list of ints): size (and dimensionality) of the inputs to the layer. Should be in the
                form of: [# filts, # space dim1, # space dim2]. Note that num_lags is not a property of the input
                and is handled separately (see below).
                Default is none, which means will be determined at the build time.
            num_lags (int): number of lags to be generated within the TLayer (and operated on by filters).
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
            pos_constraint (bool, optional): True to constrain layer weights to
                be positive
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:

        """
        # number of filters must match number of input dims
        if num_filters is None:
            num_filters = np.prod(input_dims[:3])
        assert np.prod(input_dims[:3]) == num_filters, 'For specific temporal layer, num_filters must match input dims.'

        super(TLayerSpecific, self).__init__(
            scope=scope,
            input_dims=input_dims,
            num_lags = num_lags,
            num_filters=num_filters,
            activation_func=activation_func,
            normalize_weights=normalize_weights,
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            reg_initializer=reg_initializer,
            num_inh=num_inh,
            pos_constraint=pos_constraint,
            log_activations=log_activations)

        self.output_dims = [num_filters, 1, 1]  # collapse over input_dims = num_filters
    # END TLayerSpecific.__init__

    def build_graph(self, inputs, params_dict=None, batch_size=None, use_dropout=False):

        num_inputs = np.prod(self.input_dims)

        with tf.name_scope(self.scope):

            self._define_layer_variables()

            # make shaped input
            #shaped_input = tf.reshape(tf.transpose(inputs), [num_inputs, -1, 1, 1])  # from TLayer
            #shaped_input = tf.reshape(tf.transpose(inputs), [1, -1, 1, num_inputs])  # weird alternative
            shaped_input = tf.expand_dims( tf.expand_dims(inputs, axis=0), axis=2)
            
            #shaped_input = tf.reshape(inputs, [1, -1, 1, num_inputs]) 
            if self.pos_constraint is not None:
                w_p = tf.maximum(self.weights_var, 0.0)
            else:
                w_p = self.weights_var

            if self.normalize_weights > 0:
                w_pn = tf.nn.l2_normalize(w_p, axis=0)
            else:
                w_pn = w_p

            if self.filter_basis is not None:
                # make constant-tensor if using temporal basis functions 
                filter_basis = tf.constant( self.filter_basis, name='filter_basis')
                ks = tf.matmul( filter_basis, w_pn )
            else:
                ks = w_pn

            # ks is currently num_lags x NC
            padding = tf.constant([[0, self.num_lags], [0, 0]])
            padded_filt = tf.pad(tf.reverse(ks, [0]), padding)  # note the time-reversal

            tile_dims = tf.constant([1, 1, num_inputs], tf.int32)
            tile_filt = tf.expand_dims(tf.eye(num_inputs), axis=0)
            # Generates 4-d weight tensor: [num_lags, 1, num_inputs, num_inputs]) 
            #   this treats num_lags x 1 as spatial filter (to be convolved)
            shaped_padded_filt = tf.expand_dims(
                                    tf.multiply(
                                        #tf.repeat(padded_filt, num_inputs, axis=2), # only in tf-2.1
                                        tf.tile(tf.expand_dims(padded_filt, axis=2), tile_dims),
                                        tile_filt),
                                    axis=1)

            # Temporal convolution
            strides = [1, 1, 1, 1]
            dilations = [1, self.dilation, 1, 1]
            _pre = tf.nn.conv2d(shaped_input, shaped_padded_filt, strides, dilations=dilations, padding='SAME')

            if self.include_biases:
                _post = self._apply_act_func(tf.add(_pre, self.biases_var))
            else:
                _post = self._apply_act_func(_pre)

            if self.ei_mask_var is None:
                _post2 = _post
            else:
                _post2 = tf.multiply(_post, self.ei_mask_var)

            #self.outputs = tf.reshape(tf.transpose(_post, [1, 0, 2, 3]), (-1, np.prod(self.output_dims)))
            self.outputs = tf.squeeze(_post2, [0, 2])

        if self.log:
            tf.summary.histogram('act_pre', _pre)
            tf.summary.histogram('act_post', _post)
    # END TLayerSpecific.build_graph

