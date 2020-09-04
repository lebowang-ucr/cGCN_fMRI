import imp
import os
import h5py
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.regularizers import l2
from keras import optimizers
import keras.backend as K
import matplotlib.pyplot as plt
from time import gmtime, strftime
from model import *

ROI_N = 200
frames = 315

def save_best_model(model_history, val_acc, folder='tmp', file_name='model', loss_name='loss', acc_name='acc', tmp_name='tmp/tmp_weights.hdf5'):
    # model.save('tmp/model_' + file_name + '_' + str(logs_to_save['val_acc'][-1]) + '.hdf5')
    best_model_name = file_name+'_valAcc_%.3f_.hdf5'%(val_acc) 
    os.system('mv ' + tmp_name + ' ' + best_model_name)
    print('Save model file to', best_model_name)


def save_logs_models(model_history0, val_acc, folder='tmp', lr_hist=None, file_name='model', loss_name='loss', acc_name='acc', tmp_name='tmp/tmp_weights.hdf5'):
    model_history = model_history0.history.copy()
    if lr_hist:
        model_history['lr'] = lr_hist

    best_log_name = file_name+'_valAcc_%.3f_.txt'%(val_acc) 

    with open(best_log_name,'w') as f:
        header = sorted(model_history.keys())
        f.write(",".join(header))

        for i in range(len(model_history[header[0]])):
            f.write('\n')
            for j in range(len(header)-1):
                f.write(str(model_history[header[j]][i])+',')
            f.write(str(model_history[header[-1]][i]))

    print('Save log file to', best_log_name)
    save_best_model(model_history, val_acc=val_acc, folder=folder, file_name=file_name, tmp_name=tmp_name)
    return best_log_name

fold = 10

os.system('mkdir tmp')
os.system('mkdir FC')
for fold_i in range(fold):
    print('--- fold %d:'%fold_i)
    os.system('mkdir tmp/%d'%fold_i)
    ########################################## Load data ########################################
    # Please download from: https://drive.google.com/file/d/1RhMRzDRT2vAkXDiW4t55Wbt8XRi6f9_x/view?usp=sharing
    with h5py.File('ABIDE_I_10_fold.h5', 'r') as f:
        x_train, y_train = [], []
        for fold_ii in range(fold):
            if fold_ii == fold_i:
                x_val = f[str(fold_ii)]['X'][()]
                y_val = f[str(fold_ii)]['Y'][()]
            else:
                x_train.append(f[str(fold_ii)]['X'][()])
                y_train.append(f[str(fold_ii)]['Y'][()])
    y_train = np.concatenate(y_train, 0)
    
    graph_path = 'FC/FC_no_fold_%d.npy'%fold_i
    if not os.path.exists(graph_path):
        FC = []
        for x_fold in x_train:
            for x in x_fold:
                idx = np.where(x.sum(1) == 0) # find non-zero frames
                if not idx[0].size:
                    tmp = x
                else:
                    tmp = x[:idx[0][0]]
                FC.append(np.corrcoef(tmp.T))
        FC = np.stack(FC, 0)
        FC = np.nan_to_num(FC)
        np.save(graph_path, FC.mean(0))

    x_train = np.expand_dims(np.concatenate(x_train, 0), -1) # (None, 315, 200, 1)
    x_val = np.expand_dims(x_val, -1)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_train.shape[0] + x_val.shape[0] == 1057
    print (x_train.shape)
    print (x_val.shape)
    ################################ Set parameter ###############################
    print()
    weight_name = None

    k = 5
    print('k:', k)
    batch_size = 1
    epochs = 50
    l2_reg = 1e-3
    kernels = [8,16,32,64,128]
    lr = 5e-4

    print('kernels:', kernels)
    print('l2:', l2_reg)
    print('batch_size:', batch_size)
    print('epochs:', epochs)
    print('lr:', lr)

    folder = 'tmp/%d/'%(fold_i)

    file_name=folder+'k_%d_l2_%g'%(k, l2_reg)
    print('file_name:', file_name)

    tmp_name = file_name + '_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.tmp'
    print('output tmp name:', tmp_name)

    ############################################### Get pre-trained model  ############################
    weight_name = None

    # # Find and load best pre-trained model
    # weight_path = 'tmp/%s/'%(site)
    # all_weights = os.listdir(weight_path)
    # all_right_models = {}
    # for n in all_weights:
    #     if '.hdf5' in n:
    #         n_split = n.split('_')
    #         if int(n_split[1+n_split.index('k')]) == k:
    #         # if int(n_split[1+n_split.index('k')]) == k and \
    #         #     float(n_split[1+n_split.index('l2')]) == l2_reg:
    #             all_right_models[float(n_split[1+n_split.index('valAcc')])] = n

    # if all_right_models:
    #     best_acc = np.max(list(all_right_models.keys()))
    #     print('-------best acc %f, model name: %s'%(best_acc, all_right_models[best_acc]))
    #     weight_name = weight_path+all_right_models[best_acc]

    ################################ get model  ######################################################
    model = get_model(
      graph_path=graph_path, 
      ROI_N=ROI_N,
      frames=frames,
      kernels=kernels, 
      k=k, l2_reg=l2_reg,  
      weight_path=weight_name, skip=[0,0])
    # model.summary()

    ######################################## Training ####################################################

    model.compile(loss=['binary_crossentropy'], 
                optimizer=optimizers.Adam(lr=lr),
                metrics=['accuracy'])

    print('Train...')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                                patience=10, min_lr=1e-6)
    lr_hist = []
    class Lr_record(keras.callbacks.Callback):
      def on_epoch_begin(self, epoch, logs={}):
          tmp = K.eval(model.optimizer.lr)
          lr_hist.append(tmp)
          print('Ir:', tmp)
    lr_record = Lr_record()
    earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)

    checkpointer = keras.callbacks.ModelCheckpoint(monitor='val_acc', filepath=tmp_name, 
                                                  verbose=1, save_best_only=True)
    model_history  = model.fit(x_train, y_train,
                            shuffle=True,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[checkpointer, lr_record, reduce_lr, earlystop])
                            # callbacks=[checkpointer, lr_record, reduce_lr])

    # ######################################## Val and Test ####################################################
    print('Val and Test...')
    del model
    model = get_model(
      graph_path=graph_path, 
      ROI_N=ROI_N,
      frames=frames,    
      kernels=kernels, 
      k=k, l2_reg=l2_reg,  
      weight_path=tmp_name, skip=[0,0])

    model.compile(loss=['binary_crossentropy'], 
                optimizer=optimizers.Adam(lr=0),
                metrics=['accuracy'])

    loss, val_acc = model.evaluate(x=x_val, y=y_val, batch_size=batch_size)
    print('------- val_acc:', val_acc)

    ######################################## save log and model #######################################

    save_logs_models(model_history, 
      folder=folder, 
      lr_hist=lr_hist, file_name=file_name,
      val_acc = val_acc,
      tmp_name=tmp_name)