"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from __future__ import print_function

import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA


# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.matrix('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.2),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')
        learning_rate = T.fscalar('learning_rate')        
        # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
        
        # compute list of fine-tuning updates
       

        train_fn = theano.function(
            inputs=[index,learning_rate],
            outputs=[self.finetune_cost ],
             updates = [
             
                (param, param - ((gparam * learning_rate) ))
                for param, gparam in zip(self.params, gparams)
                
                
                
                 ],
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            [self.errors],
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            [self.errors],
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(finetune_lr=0.2, pretraining_epochs=30,
             pretrain_lr=0.1, training_epochs=180,
             dataset='mnist.pkl.gz', batch_size=30):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """


#    ##########################################################    
    #a = [100907,103342,105542,112311,114345]
    numpy_rng = numpy.random.RandomState(10)
    npix=4000
    blockx=26
    blocky=26
    n_training_ex=100000
    n_valid_ex=2000
    n_test_ex=2000  
    print('Loading the image data')
    def read_filename(string , n , n_ex ):
        s = string
        i=0
        t=0
        while i < n:
            
            j=0
        #    bad_regiojns_count=0
            tr_intermx = numpy.fromfile(s[i],numpy.float32)
            tr_intermy = numpy.fromfile(s[i]+'deblurreddenoised',numpy.float32)
            i=i+1
            scn_x = len(tr_intermx)/npix
            part = scn_x/8
            scn_y = len(tr_intermy)/npix
            train_x = numpy.reshape(tr_intermx , (scn_x,npix))
            train_y = numpy.reshape(tr_intermy , (scn_y,npix))
            tr_intermx = None
            tr_intermy = None
            del tr_intermx
            del tr_intermy
                ###############
                
                ############### 
            while j < n_ex:
                  x_tr_x = numpy.random.randint(t*part,(t+1)*part-blockx-1)
                  y_tr_x = numpy.random.randint(0,npix-blocky-1)
                  x_tr_y = x_tr_x
                  y_tr_y = y_tr_x
                  if j%12500==0 and j!=0:
                      t=t+1
                  j=j+1
                  random_trx_batch = train_x[x_tr_x : x_tr_x +blockx , y_tr_x : y_tr_x + blocky]
#                  print('Shape of:')
#                  print(random_trx_batch.shape)
                      
                  random_trx_batch = numpy.reshape(random_trx_batch , (1,blocky*blocky))
                  random_try_batch = train_y[x_tr_y : x_tr_y +blockx , y_tr_y : y_tr_y + blocky]
                  random_try_batch = numpy.reshape(random_try_batch , (1,blocky*blocky))
                  if j == 1 and i == 1:
                      trx_batch = random_trx_batch
                      try_batch = random_try_batch
                  else:
                      trx_batch = numpy.vstack([trx_batch , random_trx_batch]) 
                      try_batch = numpy.vstack([try_batch , random_try_batch])
            print(i)                
           # print('i=')
            #print(i)
            # random_trx_batch = None
             # random_try_batch = None
             # del random_trx_batch
             # del random_try_batch
        min_a = trx_batch.min(axis=1)
        max_a = trx_batch.max(axis=1)  
        dif = max_a - min_a 
        trx_batch = (trx_batch.transpose() - min_a).transpose()
        trx_batch = (trx_batch.transpose() / dif).transpose()    
        min_a = try_batch.min(axis = 1)
        max_a = try_batch.max(axis = 1)  
        dif = max_a - min_a 
        try_batch = (try_batch.transpose() - min_a).transpose()
        try_batch = (try_batch.transpose() / dif).transpose()   
        return trx_batch , try_batch     
    def read_filename2(string , n , n_ex ):
        s = string
        for i in range(n):
            
            tr_intermx = numpy.fromfile(s[i],numpy.float32)
            tr_intermy = numpy.fromfile(s[i]+'deblurreddenoised',numpy.float32)
            
            scn_x = len(tr_intermx)/npix
            scn_y = len(tr_intermy)/npix
            train_x = numpy.reshape(tr_intermx , (scn_x,npix))
            train_y = numpy.reshape(tr_intermy , (scn_y,npix))
            tr_intermx = None
            tr_intermy = None
            del tr_intermx
            del tr_intermy
            ###############
            
            ############### 
            for j in range(n_ex):
              x_tr_x = numpy.random.randint(0,scn_x-blockx-1)
              y_tr_x = numpy.random.randint(0,npix-blocky-1)
              x_tr_y = x_tr_x
              y_tr_y = y_tr_x
              random_trx_batch = train_x[x_tr_x : x_tr_x +blockx , y_tr_x : y_tr_x + blocky]
              random_trx_batch = numpy.reshape(random_trx_batch , (1,blocky*blocky))
              random_try_batch = train_y[x_tr_y : x_tr_y +blockx , y_tr_y : y_tr_y + blocky]
              random_try_batch = numpy.reshape(random_try_batch , (1,blocky*blocky))
              if j == 0 and i == 0:
                 trx_batch = random_trx_batch
                 try_batch = random_try_batch
              else:
                 trx_batch = numpy.vstack([trx_batch , random_trx_batch]) 
                 try_batch = numpy.vstack([try_batch , random_try_batch])
            print(i)
            # random_trx_batch = None
             # random_try_batch = None
             # del random_trx_batch
             # del random_try_batch
        min_a = trx_batch.min(axis=1)
        max_a = trx_batch.max(axis=1)  
        dif = max_a - min_a 
        trx_batch = (trx_batch.transpose() - min_a).transpose()
        trx_batch = (trx_batch.transpose() / dif).transpose()    
        min_a = try_batch.min(axis = 1)
        max_a = try_batch.max(axis = 1)  
        dif = max_a - min_a 
        try_batch = (try_batch.transpose() - min_a).transpose()
        try_batch = (try_batch.transpose() / dif).transpose()   
        return trx_batch , try_batch 
    

    train_string_set1=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-100907-R2-1_b1.rad_boost']
    train_string_set2=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-103342-R3-1_b1.rad_boost']
    train_string_set3=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-105542-R4-1_b1.rad_boost']
    train_string_set4=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-112311-R5-0_b1.rad_boost']
    train_string_set5=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-094440-R1-1_b1.rad_boost']
    train_string_set6=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-095721-R11-1_b1.rad_boost']
    train_string_set7=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-102039-R12-1_b1.rad_boost']
    train_string_set8=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-104314-R13-1_b1.rad_boost']
    train_string_set9=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-113233-R15-0_b1.rad_boost']
    train_string_set10=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-115821-R16-0_b1.rad_boost']
    train_string_set11=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-120945-R7-0_b1.rad_boost']
    train_string_set12=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-122040-R17-0_b1.rad_boost']
  
    
    trx_batch,try_batch=read_filename(train_string_set1,1,n_training_ex)    
    tra_batch,trb_batch=read_filename(train_string_set2,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set3,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set4,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])    
    tra_batch,trb_batch=read_filename(train_string_set5,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set6,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set7,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set8,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set9,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set10,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set11,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])
    tra_batch,trb_batch=read_filename(train_string_set12,1,n_training_ex)
    trx_batch = numpy.vstack([trx_batch,tra_batch])
    try_batch = numpy.vstack([try_batch,trb_batch])  
#    trx_batch,try_batch=read_filename(train_string_set,12,n_training_ex)
    print('Training set x shape')
    print(trx_batch.shape)
    print('Training set y shape')
    print(try_batch.shape)
    train_set_x = theano.shared(numpy.asarray(trx_batch,
                                               dtype=theano.config.floatX),
                                 borrow=True)      
    train_set_y = theano.shared(numpy.asarray(try_batch,
                                               dtype=theano.config.floatX),
                                 borrow=True)
    print('Training data loaded..')
    valid_string_set=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-114345-R6-0_b1.rad_boost']
    vax_batch,vay_batch=read_filename2(valid_string_set,1,n_valid_ex)
    
    valid_set_x = theano.shared(numpy.asarray(vax_batch,
                                               dtype=theano.config.floatX),
                                 borrow=True)      
    valid_set_y = theano.shared(numpy.asarray(vay_batch,
                                               dtype=theano.config.floatX),
                                 borrow=True)
    print('Validating data loaded..')
    test_string_set=['/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-114345-R6-0_b1.rad_boost']
    tex_batch,tey_batch=read_filename2(test_string_set,1,n_test_ex)
    
    test_set_x = theano.shared(numpy.asarray(tex_batch,
                                               dtype=theano.config.floatX),
                                 borrow=True)      
    test_set_y = theano.shared(numpy.asarray(tey_batch,
                                               dtype=theano.config.floatX),
                                 borrow=True) 
    print('Test data loaded..')    
    datasets = ([train_set_x,train_set_y],[valid_set_x,valid_set_y],[test_set_x,test_set_y])
    
    
    ################################################################
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # numpy random generator
    # start-snippet-3
    print('... building the model')
    # construct the stacked denoising autoencoder class
#    sda = SdA(
#        numpy_rng=numpy_rng,
#        n_ins=28 * 28,
#        hidden_layers_sizes=[1000, 1000, 1000],
#        n_outs=10
#    )
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=blockx*blocky,
        hidden_layers_sizes=[2620,2620],
        n_outs=blockx*blocky
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    import time
    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels =[0.0,0.0,0.0]
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            tic=time.time()
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            toc=time.time()            
            print('Pre-training layer %i, epoch %d, cost %f , Epochtime %f' % (i, epoch, numpy.mean(c),(toc-tic)))

    end_time = timeit.default_timer()

    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
         
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
       
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 35 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
    previous_cost=numpy.inf                              # check every epoch
  #  beta=0.1
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()
    alpha=0.1
    done_looping = False
    epoch = 0
    train_plt=[]
    valid_plt=[]
   # test_plt=[]
    epoch_plt=[]
    import matplotlib.pyplot as plt
    while (epoch < training_epochs ) :
        epoch = epoch + 1
        #alpha=finetune_lr/(1+(finetune_lr*beta*epoch))
        
        epoch_plt.append(epoch)
        minibatch_avg_cost=[]
        for minibatch_index in range(n_train_batches):
            minibatch_cost = train_fn(minibatch_index,alpha)
            minibatch_avg_cost.append(minibatch_cost)	
            iter = (epoch - 1) * n_train_batches + minibatch_index
           
           
            if (iter + 1) % validation_frequency == 0:
               # sys.exit(1)
                validation_losses  = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                valid_plt.append(this_validation_loss)
                if this_validation_loss < previous_cost:
                    alpha = alpha + (0.05*alpha)
                    previous_cost=this_validation_loss
                else:
                    minibatch_cost = train_fn(minibatch_index,-alpha)
                    #minibatch_avg_cost.append(minibatch_cost)	                        
                    alpha = alpha - (0.5*alpha)
                train_avg_cost=numpy.mean(minibatch_avg_cost)
                train_plt.append(train_avg_cost)  
                plt.plot(epoch_plt,valid_plt)
                plt.ylabel('Validation error')
                plt.show()
                
                
               # plt.plot(epoch,train_plt)
               # plt.show()
                print('epoch %i, minibatch %i/%i, validation MSE error %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epochs %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score ))
                    with open('best_model_sda(band1_2).pkl', 'wb') as f:
                     pickle.dump(sda, f) 
                 
            if patience <= iter:
                done_looping = True
    
        train_plt.append(numpy.mean(minibatch_avg_cost)) 
#        plt.plot(epoch_plt[:],train_plt , 'ro')
 #       plt.show()
              
                #break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation MSE core of %f , '
            'on iteration %i, '
            'with test performance %f'
        )
        % (best_validation_loss, best_iter + 1, test_score )
    )
    print(('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    print('The last layer of the out put is')
  #  h = theano.tensor.cast(sda.logLayer.p_y_given_x , 'float32')     
    print(sda.logLayer.p_y_given_x.dtype)
    print(type(sda.logLayer.p_y_given_x))
def predict_denoised_image():
    """
    An example of how to load a trained model and use it
    to predict a denoised image.
    """
    import numpy 
    import matplotlib.pyplot as plt
     
    sda = pickle.load(open('best_model_sda(band1_2).pkl'))
    print('Model loaded')     
    s = '/AKASHDP_DATA3/ankur/train/A1-NAO-25-FEB-2016-114345-R6-0_b1.rad_boost'

    t =s+'deblurreddenoised'
    ex_ar =   numpy.fromfile(t , numpy.float32)  
    test_ar = numpy.fromfile(s , numpy.float32)
    
    npix=4000
    test_scn = 4000#len(test_ar)/npix    
    stride_given=14
    blockx=26
    blocky=26
   # n_blocks = (test_scn*npix)/(blockx*blocky)
    
    test_ar = test_ar[:test_scn*npix]
    ex_ar = ex_ar[:test_scn*npix]
    #print('Image loaded as 1d matrix')   
    i_matrix = numpy.reshape(test_ar , (test_scn,npix))
    x = numpy.reshape(ex_ar , (test_scn,npix))
    x.tofile('/AKASHDP_DATA3/ankur/train/actual_denoised_band1')
    i_matrix.tofile('/AKASHDP_DATA3/ankur/train/noisy_band1')
    mean_original=i_matrix.mean()
    var_original=i_matrix.var()
    i=7
    
  
    
   


    
        
   # sys.exit(1)

    
    print('Image loaded as 2d matrix')  
    def patchify(img, patch_shape):
	    img = numpy.ascontiguousarray(img)  # won't make a copy if not needed
	    X, Y = img.shape
	    x, y = patch_shape
	    stride=stride_given
	    shape = (((X-x)/stride+1), ((Y-y)/stride+1), x, y)
         # number of patches, patch_shape
	    #shape = (2, 2, x, y) 
	    # The right strides can be thought by:
	    # 1) Thinking of `img` as a chunk of memory in C order
	    # 2) Asking how many items through that chunk of memory are needed when indices
	    #    i,j,k,l are incremented by one
	    strides = img.itemsize*numpy.array([Y*stride, stride, Y, 1])
	    return numpy.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


    patches = patchify(i_matrix, (blockx,blocky))
		#print(patches)
    test_set_x = numpy.ascontiguousarray(patches)
    test_set_x.shape = (-1, blockx**2)
#    print('Shape of the input matrix')
#    print(test_set_x.shape)
#    print(test_set_x[248003])
    #print(test_set_x[248004])
    #sys.exit(0)
#    print('len=')
#    print(len(test_set_x))
#    print('This is the first 10x10 patch of the input')
#    print(test_set_x[0])
#    print('This is the second 10x10 patch in the input')
#    print(test_set_x[1])
#    print('This is the lower 10x10 patch')
#    print((npix-blockx)/stride_given + 2)
#    print(test_set_x[572])
#    print('This is the first patch of the third row')
#    print(test_set_x[326040])
#    sys.exit(1)
	#	contiguous_patches.shape = (-1, 4**2)

    min_a = test_set_x.min(axis=1)
    max_a = test_set_x.max(axis=1)
    dif = max_a - min_a
    abc=test_set_x
    test_set_x = (test_set_x.transpose() - min_a).transpose()
    test_set_x = (test_set_x.transpose() / dif).transpose()
  #  print('This is the first block')    
  #  print(test_set_x[0])
   # print('************************')
   # print('This is the second block')
   # print(test_set_x[1])
   

    test_set_x = theano.shared(numpy.asarray(test_set_x,
                                               dtype=theano.config.floatX),
                                 borrow=True) 
    print('Done loading the image as test_set_x')  
   # print('Dimension of the test_set_x is ')
   # print(test_set_x.get_value().shape)
    predict = theano.function(inputs = [sda.x] , outputs = [sda.logLayer.p_y_given_x])
    test_set_y = predict(test_set_x.get_value())
    
 #   check=test_set_y
    print('Got the output sa the denoised image')

   
    denoised_img = i_matrix ######modified
    #denoised_img = numpy.zeros((test_scn,npix))    
    #print(denoised_img.shape)
    #sys.exit(1)
    print('Done transformation')    
    i=0
    j=0
    k=0
    c=0
    y_overlapped=[]
#    a=blockx/2
#    b=blocky/2
    stride=stride_given
    while i <(test_scn-blockx):
       while j < (npix-blocky):
         
          temp_matrix = abc[k]
          #print('This is the first block of the image matrix')
         # print(numpy.reshape(temp_matrix,(1,100)))            
          min_a = temp_matrix.min()
        #  print('This is the minimum of the first block')
       #   print(min_a)
          max_a = temp_matrix.max()
      #    print('This is the maximim of the first block' )
     #     print(max_a)
          dif = max_a - min_a 
         # print('This the difference of the min and max' )
        #  print(dif)
          #print('The dimesion of check is ')
          #print(check.shape)
          temp_matrix1 = test_set_y[0][k]
          #print('This is the matrix read from test_set_y') 
         # print(temp_matrix1)
        #  print(temp_matrix1.shape)
          #print('The dimension of the temp1_matrix is')
          
          #print(temp_matrix1.shape)
          
          k=k+1
          temp_matrix1 = (temp_matrix1 * dif) + min_a
        
         # print('This is the first blck of the output image normalized')
          #print('The dimension of the temp1_matrix is')
          #print(temp_matrix1.shape)
          
          temp_matrix1=numpy.reshape(temp_matrix1,(blockx,blocky))
#          if i==0 and j==0:
#              print('This is the actual op')
#              print(test_set_y[0][0])
       
          
          
          
          
         # print('This is the reshaped block')
        #  print(temp_matrix1)
       
#          print('Shape of denoised portion is ')
#          print(denoised_img[p:p+stride,q:q+stride].shape)
#          print('Shape of assigning matrix')
#          print(temp_matrix1[(blockx-stride)/2:(blockx-stride)/2+stride,(blocky- stride)/2:(blocky-stride)/2+stride].shape)            
          #denoised_img[p:p+stride,q:q+stride]=temp_matrix1[(blockx-stride)/2:(blockx-stride)/2+stride,(blocky- stride)/2:(blocky-stride)/2+stride] 
         # print('The first block of the denoised image is')
         # print(denoised_img[i:i+blockx,j:j+blocky])
         # sys.exit(1)
          if i==0 and j==0:
              denoised_img[i:i+blockx,j:j+blocky]=temp_matrix1
              overlappedx=temp_matrix1[: , stride:]
              y_overlapped.append(temp_matrix1[stride: , :])
              c=c+1
#              print('This is the first block of 10x10')
#              print(temp_matrix1)
#              print('This is the overlapping block of 10x4 in the x direction')
#              print(overlappedx)
#              print('This is the overlapping block of 4x10 in the y direction')
#              print(overlappedy)
          elif i==0 and j>0:
#              print('This is the second block in the x direction')
#              print(temp_matrix1)
#              print('This is the overlapped block of 10x4 in the x direction')
#              print(temp_matrix1[: , :blockx-stride])  
            
              temp_matrix1[:,:blockx-stride]=numpy.mean(numpy.array([overlappedx,temp_matrix1[:,:blockx-stride]]),axis=0)              
#              print('This is the averaged block')
#              print(temp_matrix1[: , :blockx-stride])  
#              sys.exit(1) 
              y_overlapped.append(temp_matrix1[stride: , :])
#              print('Shape of denoised img here is ')
#              print(denoised_img[i:i+blockx,j:j+blocky].shape)
#              print(j)                   
              denoised_img[i:i+blockx,j:j+blocky]=temp_matrix1
          elif j==0 and i>0:  
             # x=numpy.mean(numpy.array([overlappedy,temp_matrix1[:blocky-stride,:]]),axis=0)
              #print('yo done')                
                     
#              print('This is the second block in the y direction')
#              print(temp_matrix1)
#              print('This is the second block in y direction according to testsety')
#              print
#              print('This is the overlapped block of 4x10 in the y direction')
#              print(temp_matrix1[:blocky-stride , :])                  
#               temp_matrix1[:,:blockx-stride]=numpy.mean(numpy.array([overlappedx,temp_matrix1[:,:blockx-stride]]),axis=0)              
#              print('This is the averaged block')
#               
#              #sys.exit(1)                
               temp_matrix1[:blocky-stride,:]=numpy.mean(numpy.array([y_overlapped[j/stride],temp_matrix1[:blocky-stride,:]]),axis=0)
               y_overlapped[j/stride]=temp_matrix1[stride: , :]
#              print(temp_matrix1[:blocky-stride , :])
#              sys.exit(1)
#                              
               denoised_img[i:i+blockx,j:j+blocky]=temp_matrix1
          else:
              temp_matrix1[:,:blockx-stride]=numpy.mean(numpy.array([overlappedx,temp_matrix1[:,:blockx-stride]]),axis=0)
#                                       
              temp_matrix1[:blocky-stride,:]=numpy.mean(numpy.array([y_overlapped[j/stride],temp_matrix1[:blocky-stride,:]]),axis=0)
              denoised_img[i:i+blockx,j:j+blocky]=temp_matrix1
              overlappedx=temp_matrix1[: , stride:]
              y_overlapped[j/stride]=temp_matrix1[stride: , :]
          #print('******')    
          j=j+stride
          overlappedx=temp_matrix1[: , stride:]

          
       
       #print('hoho')        
       i=i+stride
      # a=a+stride
       j=0  
      # b=blockx/2
   #    print('hi')        
    #   print('i= '+str(i) + ', j = '+str(j))  
     #  print('i= '+str(i) + ', j = '+str(j))
    # compile a predictor function
   # print('The first block of the denoized image is ')
   # print(denoised_img[0:10,0:10])
   # print('The dimension of the original image matrix was')
   # print(i_matrix.shape)
    #print('The dimension of the output denoised image is ')
    #print(denoised_img.shape)
#    print('This is the block starting from (0,0) in image matrix')
#    print(x[3220:3230 , 1050:1060])
#    print('This is the correspondingblock satrting at (0,0) in the denoised image')
#    print(denoised_img[3220:3230 , 1050:1060])
    #final_denoised_img = numpy.reshape(denoised_img , (test_scn,4000))
    denoised_img.tofile('/AKASHDP_DATA3/ankur/train/mydenoised_band1')
    print('Done saving the image')
    new_mean=denoised_img.mean()
    var_new=denoised_img.var()
    print(mean_original)
    print(new_mean)
    print('**********')
    print(numpy.sqrt(var_original))
    print(numpy.sqrt(var_new))
    print('**************')
    from sklearn.metrics import mean_squared_error
    import numpy as np
    k = 20*np.log10(4095)
    mse = mean_squared_error(x,denoised_img)
    psnr = k - (10*np.log10(mse))
 
    print('PSNR is')
    print(psnr)
    
if __name__ == '__main__':
    predict_denoised_image()
