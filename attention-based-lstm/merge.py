from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.layers import LSTM, SimpleRNN, GRU, Masking #Merge, merge
from keras.layers import concatenate, Concatenate
from keras.models import Model
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.wrappers import Bidirectional

import numpy as np
import pandas as pd
from numpy.random import RandomState
from random import shuffle
import datetime, itertools, warnings

np.random.seed(1024)
warnings.filterwarnings(action='ignore', message='Setting attributes')

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


def train_Bi_LSTM(X, Y, epochs = 30, validation_split = 0.2, patience=20):
    speed_input = Input(shape = (X.shape[1], X.shape[2]), name = 'speed')
    
    main_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=False), merge_mode='ave')(speed_input)
    
    final_model = Model(input = [speed_input], output = [main_output])
    
    final_model.summary()
    
    final_model.compile(loss='mse', optimizer='rmsprop')
    
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split = 0.2, nb_epoch = epochs, callbacks=[history, earlyStopping])
    
    return final_model, history

def train_2_Bi_LSTM_mask(X, Y, epochs = 30, validation_split = 0.2, patience=20):
    
    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(output_dim = X.shape[2], return_sequences=True, input_shape = (X.shape[1], X.shape[2])))
    model.add(LSTM(output_dim = X.shape[2], return_sequences=False, input_shape = (X.shape[1], X.shape[2])))

    model.add(Dense(X.shape[2]))
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split = 0.2, nb_epoch = epochs, callbacks=[history, earlyStopping])

    return model, history

def train_2_Bi_LSTM(X, Y, epochs = 30, validation_split = 0.2, patience=20):
    speed_input = Input(shape = (X.shape[1], X.shape[2]), name = 'speed')
    
    lstm_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2])(lstm_output)
    
    final_model = Model(input = [speed_input], output = [main_output])
    
    final_model.summary()
    
    final_model.compile(loss='mse', optimizer='rmsprop')
    
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split = 0.2, nb_epoch = epochs, callbacks=[history, earlyStopping])
    
    return final_model, history

def Get_Data_Label_Aux_Set2(speedMatrix, steps1, steps2):
    cabinets = speedMatrix.columns.values
    stamps = speedMatrix.index.values
    x_dim = len(cabinets)
    time_dim = len(stamps)
    
    speedMatrix = speedMatrix.iloc[:,:].values
    
    data_set = []
    label_set = []
    hour_set = []
    dayofweek_set = []

    for i in range(time_dim - (steps1+steps2) ):
        data_set.append(speedMatrix[i : i + steps1])
        label_set.append(speedMatrix[i + steps1 + steps2])
        stamp = stamps[i + steps1 + steps2]
        hour_set.append(float(stamp[11:13]))
        dayofweek = datetime.datetime.strptime(stamp[0:10], '%Y-%M-%d').strftime('%w')
        dayofweek_set.append(float(dayofweek))

    data_set = np.array(data_set)
    label_set = np.array(label_set)
    hour_set = np.array(hour_set)
    dayofweek_set = np.array(dayofweek_set)
    return data_set, label_set, hour_set, dayofweek_set

def Get_Data_Label_Aux_Set(speedMatrix, steps):
    cabinets = speedMatrix.columns.values
    stamps = speedMatrix.index.values
    x_dim = len(cabinets)
    time_dim = len(stamps)
    
    speedMatrix = speedMatrix.iloc[:,:].values
    
    data_set = []
    label_set = []
    hour_set = []
    dayofweek_set = []

    for i in range(time_dim - steps ):
        data_set.append(speedMatrix[i : i + steps])
        label_set.append(speedMatrix[i + steps])
        stamp = stamps[i + steps]
        hour_set.append(float(stamp[11:13]))
        dayofweek = datetime.datetime.strptime(stamp[0:10], '%Y-%M-%d').strftime('%w')
        dayofweek_set.append(float(dayofweek))

    data_set = np.array(data_set)
    label_set = np.array(label_set)
    hour_set = np.array(hour_set)
    dayofweek_set = np.array(dayofweek_set)
    return data_set, label_set, hour_set, dayofweek_set

def SplitData(X_full, Y_full, hour_full, dayofweek_full, horizon, window_size , RA):
    n = Y_full.shape[0]
    indices = np.arange(n)
    #RS = RandomState(1024)
    #RS.shuffle(indices)
    sep_1 = 5842 - horizon - window_size - RA
    sep_2 = 7303 - horizon - window_size - RA

    # sep_1 = int(float(n) * train_prop)
    # sep_2 = int(float(n) * (train_prop + valid_prop))
    # print ('train : valid : test = ', train_prop, valid_prop, test_prop)
    train_indices = indices[:sep_1]
    valid_indices = indices[sep_1:sep_2]
    test_indices = indices[sep_2:]
    X_train = X_full[train_indices]
    X_valid = X_full[valid_indices]
    X_test = X_full[test_indices]
    Y_train = Y_full[train_indices]
    Y_valid = Y_full[valid_indices]
    Y_test = Y_full[test_indices]
    hour_train = hour_full[train_indices]
    hour_valid = hour_full[valid_indices]
    hour_test = hour_full[test_indices]
    dayofweek_train = dayofweek_full[train_indices]
    dayofweek_valid = dayofweek_full[valid_indices]
    dayofweek_test = dayofweek_full[test_indices]
    return X_train, X_valid, X_test, \
            Y_train, Y_valid, Y_test, \
            hour_train, hour_valid, hour_test, \
            dayofweek_train, dayofweek_valid, dayofweek_test
            
def MeasurePerformance(Y_test_scale, Y_pred, X_max, model_name = 'default', epochs = 30, model_time_lag = 10):

    time_num = Y_test_scale.shape[0]
    loop_num = Y_test_scale.shape[1]

    difference_sum = np.zeros(time_num)
    diff_frac_sum = np.zeros(time_num)

    for loop_idx in range(loop_num):
        true_speed = Y_test_scale[:,loop_idx] * X_max
        predicted_speed = Y_pred[:,loop_idx] * X_max
        diff = np.abs( true_speed - predicted_speed )
        diff_frac = diff / true_speed
        difference_sum += diff
        diff_frac_sum += diff_frac
        
    difference_avg = difference_sum / loop_num
    MAPE = diff_frac_sum / loop_num * 100
    
    print('MAE :', round(np.mean(difference_avg),3), 'MAPE :', round(np.mean(MAPE),3), 'STD of MAE:', round(np.std(difference_avg),3))
    print('Epoch : ' , epochs)

def stacked_lstm (file_name = 'pre_price',  model_epoch = 100, horizon = 1, window_size = 30) :

    speedMatrix = pd.read_csv(file_name).iloc[:,:-1]
    if file_name=='/content/drive/MyDrive/workspace/data/pre_price.csv':
        RA_para=0
    elif file_name=='/content/drive/MyDrive/workspace/data/pre_price_s_7d.csv':
        RA_para=7-1
    elif file_name=='/content/drive/MyDrive/workspace/data/pre_price_s_14d.csv':
        RA_para=14-1
    elif file_name=='/content/drive/MyDrive/workspace/data/pre_price_s_21d.csv':
        RA_para=21-1
    elif file_name=='/content/drive/MyDrive/workspace/data/pre_price_s_28d.csv':
        RA_para=28-1

    base = datetime.datetime(2000, 1, 1, 0, 0)
    date_list = [(base + datetime.timedelta(minutes=x)).strftime("%Y-%m-%d %H:%M:%S") for x in range(0, len(speedMatrix)*5, 5)] # date_list.reverse()
    speedMatrix.index = date_list

    loopgroups_full = speedMatrix.columns.values
    X_full, Y_full, hour_full, dayofweek_full = Get_Data_Label_Aux_Set2(speedMatrix, window_size, horizon)

    X_train, X_valid, X_test, \
        Y_train, Y_valid, Y_test, \
        hour_train, hour_valid, hour_test, \
        dayofweek_train, dayofweek_valid, dayofweek_test \
                    = SplitData(X_full, Y_full, hour_full, dayofweek_full,
                                horizon = horizon, window_size = window_size ,RA = RA_para)
                                # train_prop = train_ratio, valid_prop = vali_ratio, test_prop = test_ratio)

    X_max = np.max([np.max(X_train), np.max(X_test)])
    X_min = np.min([np.min(X_train), np.min(X_test)])

    X_train_scale = X_train / X_max
    X_test_scale = X_test / X_max

    Y_train_scale = Y_train / X_max
    Y_test_scale = Y_test / X_max

    model_2_Bi_LSTM, history_2_Bi_LSTM = train_2_Bi_LSTM_mask(X_train_scale, Y_train_scale, epochs = model_epoch)
    Y_pred_test = model_2_Bi_LSTM.predict(X_test_scale)

    return (Y_test_scale, Y_pred_test, X_max)