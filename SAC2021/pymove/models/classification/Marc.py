import pandas as pd
import numpy as np
import time
from os import path
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Dense, LSTM, GRU, Bidirectional, Concatenate, Add, Average, Embedding, Dropout, Input
from keras.initializers import he_normal, he_uniform
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1
from keras import backend as K
from pymove.models import metrics
from pymove.processing import geoutils, trajutils

class Marc(object):
    def __init__(self, 
            max_lenght,
            num_classes,
            vocab_size = {},
            rnn='lstm',
            rnn_units=100,
            merge_type = 'concat',
            dropout=0.5,
            embedding_size = 100,
            geohash_precision=8):

        
        assert rnn in ['lstm', 'gru'], 'ERRO: rnn is invalid'
        assert merge_type in ['add', 'average', 'concat'], 'ERRO: merge_type is invalid'

        start_time = time.time()

        print('\n\n##########           CREATE A MARC MODEL       #########\n')
        print('... max_lenght: {}\n... vocab_size: {}\n... classes: {}'.format(max_lenght, vocab_size, num_classes))
        self.vocab_size = vocab_size
        self.col_name = list(vocab_size.keys())
        self.max_lenght = max_lenght

    
        
        input_model = [] 
        embedding_layers = []

        print('\n\n###########      Building Input and Embedding Layers      ###########') 
        for c in tqdm(self.col_name):
            print('... creating layer to column : {}'.format(c))
            if c == 'geohash':
                print('... vocab_size to column {}: {}'.format(c, self.vocab_size[c]))
                i_model= Input(shape=(self.max_lenght, self.vocab_size[c]), 
                            name='Input_{}'.format(c)) 
                e_output_ = Dense(units=embedding_size, 
                                kernel_initializer=he_uniform(seed=1), 
                                name='Embedding_{}'.format(c))(i_model)            
            else:
                print('... vocab_size to column {}: {}'.format(c, self.vocab_size[c]))
                i_model= Input(shape=(self.max_lenght,), 
                                name='Input_{}'.format(c)) 

                e_output_ = Embedding(input_dim = self.vocab_size[c], 
                                    output_dim = embedding_size, 
                                    name='Embedding_{}'.format(c), 
                                    input_length=self.max_lenght)(i_model)

            input_model.append(i_model)  
            embedding_layers.append(e_output_)             
        
        if len(embedding_layers) == 1:
            hidden_input = embedding_layers[0]
        elif merge_type == 'add':
            hidden_input = Add()(embedding_layers)
        elif merge_type == 'average':
            hidden_input = Average()(embedding_layers)
        else:
            hidden_input = Concatenate(axis=2)(embedding_layers)

        hidden_dropout = Dropout(dropout)(hidden_input)

        if rnn == 'lstm':
            rnn_cell = LSTM(units=rnn_units,
                            recurrent_regularizer=l1(0.02))(hidden_dropout)
        else:
            rnn_cell = GRU(units=rnn_units,
                        recurrent_regularizer=l1(0.02))(hidden_dropout)

        rnn_dropout = Dropout(dropout)(rnn_cell)

        softmax = Dense(units=num_classes,
                        kernel_initializer=he_uniform(),
                        activation='softmax')(rnn_dropout)

        self.model = Model(inputs=input_model, outputs=softmax)

        print('\n--------------------------------------\n')
        end_time = time.time()
        print('total Time: {}'.format(end_time - start_time))
    
    def fit(self, 
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=64,
            epochs=1000,
            monitor='val_acc',
            min_delta=0, 
            patience=30, 
            verbose=0,
            baseline=0.5,
            learning_rate = 0.001,
            mode = 'max',
            save_model=False,
            modelname='',
            save_best_only=True,
            save_weights_only=False,
            log_dir=None):
            
            print('\n\n##########      FIT MARC MODEL       ##########')
                       
            assert (y_train.ndim == 2), "ERRO: y_train dimension is incorrect"            
            assert (y_val.ndim == 2), "ERRO: y_test dimension is incorrect"
            assert (y_train.ndim == y_val.ndim), "ERRO: y_train and y_test have differents dimension"
          

            opt = Adam(lr=learning_rate)

            self.model.compile(optimizer=opt,
                   loss='categorical_crossentropy',
                   metrics=['acc', 'top_k_categorical_accuracy'])

        

            early_stop = EarlyStopping(monitor=monitor,
                                        min_delta=min_delta, 
                                        patience=patience, 
                                        verbose=verbose, # without print 
                                        mode=mode,
                                        baseline=baseline,
                                        restore_best_weights=True)
        
       
            print('... Defining checkpoint')
            if save_model == True:
                if (not modelname) | (modelname == None):
                    modelname = 'MARC_model_'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'.h5'         
                ck = ModelCheckpoint(modelname, 
                            monitor=monitor, 
                            save_best_only=save_best_only,
                            save_weights_only=save_weights_only)   
                my_callbacks = [early_stop, ck]    
            else:
                my_callbacks= [early_stop]    

            print('... Starting training')
            self.history = self.model.fit(X_train, y_train,
                                        epochs=epochs,
                                        callbacks=my_callbacks,
                                        validation_data=(X_val, y_val),
                                        verbose=1,
                                        shuffle=True,
                                        use_multiprocessing=True,          
                                        batch_size=batch_size)


    def predict(self, 
                X_test,
                y_test, 
                verbose=0,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        
                   
        print('\n\n##########      PREDICT MARC MODEL       ##########\n')
        assert (y_test.ndim == 2), "ERRO: y_train dimension is incorrect"       

        y_pred = np.array(self.model.predict(X_test))

        argmax = np.argmax(y_pred, axis=1)
        y_pred = np.zeros(y_pred.shape)
        for row, col in enumerate(argmax):
            y_pred[row][col] = 1
        
        print('... generate classification Report')  
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report
       

    def summary(self):
        if self.model is None:
           print('Erro: model is not exist') 
        else:
            self.model.summary()

    def get_params(self):
        print('get parameterns')
    
    def score(self, X, y):
        print('Score')
    
    def free(self):
        print('\n\n#######     Cleaning DeepeST model      #######')
        print('... Free memory')
        start_time = time.time()
        K.clear_session()
        print('... total_time: {}'.format(time.time()-start_time))



#from core.utils.metrics import compute_acc_acc5_f1_prec_rec

# class EpochLogger(EarlyStopping):

#     def __init__(self, monitor='val_acc', mode='max', baseline=0, patience=30):
#         super(EpochLogger, self).__init__(monitor=monitor,
#                                           mode=mode,
#                                           patience=patience)
#         self._metric = monitor
#         self._baseline = baseline
#         self._baseline_met = False

#     def on_epoch_begin(self, epoch, logs={}):
#         print("===== Training Epoch %d =====" % (epoch + 1))

#         if self._baseline_met:
#             super(EpochLogger, self).on_epoch_begin(epoch, logs)

#     def on_epoch_end(self, epoch, logs={}):
#         pred_y_train = np.array(self.model.predict(cls_x_train))
#         (train_acc,
#          train_acc5,
#          train_f1_macro,
#          train_prec_macro,
#          train_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_train,
#                                                          pred_y_train,
#                                                          print_metrics=True,
#                                                          print_pfx='TRAIN')

#         pred_y_test = np.array(self.model.predict(cls_x_test))
#         (test_acc,
#          test_acc5,
#          test_f1_macro,
#          test_prec_macro,
#          test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
#                                                         pred_y_test,
#                                                         print_metrics=True,
#                                                         print_pfx='TEST')
#         metrics.log(METHOD, int(epoch + 1), DATASET,
#                     logs['loss'], train_acc, train_acc5,
#                     train_f1_macro, train_prec_macro, train_rec_macro,
#                     logs['val_loss'], test_acc, test_acc5,
#                     test_f1_macro, test_prec_macro, test_rec_macro)
#         metrics.save(METRICS_FILE)

#         if self._baseline_met:
#             super(EpochLogger, self).on_epoch_end(epoch, logs)

#         if not self._baseline_met \
#            and logs[self._metric] >= self._baseline:
#             self._baseline_met = True

#     def on_train_begin(self, logs=None):
#         super(EpochLogger, self).on_train_begin(logs)

#     def on_train_end(self, logs=None):
#         if self._baseline_met:
#             super(EpochLogger, self).on_train_end(logs)

