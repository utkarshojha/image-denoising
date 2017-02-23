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


class SdA(object):
  
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
       
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
      
        for i in range(self.n_layers):
           
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]


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
     
        # I now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
 
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
   
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
      
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
             dataset=None, batch_size=30):
  


#    ##########################################################    
    #a = [100907,103342,105542,112311,114345]
    numpy_rng = numpy.random.RandomState(10)
    npix=4000
    #setting the dimension of the patch size	
    blockx=26
    blocky=26
    n_training_ex=100000 #number of random patches to be sampled from each image in the training set
    n_valid_ex=2000  #number of random patches sampled from the validation set
    n_test_ex=2000  #number of random patches sampled from the test set
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
    
#Following are the list of paths to the images of training set which I used when I worked at my computer
#My training set consisted of 12 multispectral images (I know it will sound too less considering we're dealing with deep learning models, but each of those images has dimension 4000 x 80000.
# So I sampled 1000000 random patches (described later), resulting in my training set consisting 12000000 noisy and desnoied patches of size 26 x 26. 
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
  
#Forming the training set    
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
    
    

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # numpy random generator
    # start-snippet-3
    print('... building the model')
    # construct the stacked denoising autoencoder class

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
    test_scn = 4000  
    stride_given=14
    blockx=26
    blocky=26
    
    test_ar = test_ar[:test_scn*npix]
    ex_ar = ex_ar[:test_scn*npix]
    i_matrix = numpy.reshape(test_ar , (test_scn,npix))
    x = numpy.reshape(ex_ar , (test_scn,npix))
    x.tofile('/AKASHDP_DATA3/ankur/train/actual_denoised_band1')
    i_matrix.tofile('/AKASHDP_DATA3/ankur/train/noisy_band1')
    mean_original=i_matrix.mean()
    var_original=i_matrix.var()
    i=7
	
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
    test_set_x = numpy.ascontiguousarray(patches)
    test_set_x.shape = (-1, blockx**2)

    min_a = test_set_x.min(axis=1)
    max_a = test_set_x.max(axis=1)
    dif = max_a - min_a
    abc=test_set_x
    test_set_x = (test_set_x.transpose() - min_a).transpose()
    test_set_x = (test_set_x.transpose() / dif).transpose()

    test_set_x = theano.shared(numpy.asarray(test_set_x,
                                               dtype=theano.config.floatX),
                                 borrow=True) 
    print('Done loading the image as test_set_x')  
    predict = theano.function(inputs = [sda.x] , outputs = [sda.logLayer.p_y_given_x])
    test_set_y = predict(test_set_x.get_value())
    
    print('Got the output sa the denoised image')

   
    denoised_img = i_matrix ######modified
    print('Done transformation')    
    i=0
    j=0
    k=0
    c=0
    y_overlapped=[]

    stride=stride_given
    while i <(test_scn-blockx):
       while j < (npix-blocky):
         
          temp_matrix = abc[k]
          min_a = temp_matrix.min()
          max_a = temp_matrix.max()
          dif = max_a - min_a 

          temp_matrix1 = test_set_y[0][k]

          k=k+1
          temp_matrix1 = (temp_matrix1 * dif) + min_a
                  
          temp_matrix1=numpy.reshape(temp_matrix1,(blockx,blocky))

       
#          print('Shape of denoised portion is ')


          if i==0 and j==0:
              denoised_img[i:i+blockx,j:j+blocky]=temp_matrix1
              overlappedx=temp_matrix1[: , stride:]
              y_overlapped.append(temp_matrix1[stride: , :])
              c=c+1

          elif i==0 and j>0:

              temp_matrix1[:,:blockx-stride]=numpy.mean(numpy.array([overlappedx,temp_matrix1[:,:blockx-stride]]),axis=0)              

              y_overlapped.append(temp_matrix1[stride: , :])
                  
              denoised_img[i:i+blockx,j:j+blocky]=temp_matrix1
          elif j==0 and i>0:  
             # x=numpy.mean(numpy.array([overlappedy,temp_matrix1[:blocky-stride,:]]),axis=0)
                           
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
             
          j=j+stride
          overlappedx=temp_matrix1[: , stride:]               
       i=i+stride
      # a=a+stride
       j=0  
 
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
