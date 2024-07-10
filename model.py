import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,concatenate,Multiply,Permute,Reshape,LSTM,Flatten,Lambda,RepeatVector
from tensorflow.keras.layers import Dense,concatenate,Multiply,Permute,Reshape,LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import math
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf

input_1 = Input(shape=(indices_length,3))
input_2 = Input(shape=(3,))
input_dim=3
time_stpes=indices_length
def attention_method(inputs):
    input_dim = int(inputs.shape[2])
    a1 = Permute((2, 1))(inputs)
    a1 = Dense(indices_length, activation='softmax')(a1)
    a1 = Lambda(lambda x: K.mean(x, axis=1))(a1)
    a1 = RepeatVector(input_dim)(a1)
    a1 = Permute((2,1),name='attention_vec1')(a1)
    a2 = Dense(input_dim, activation='softmax',name='attention_vec2')(inputs)
    a2 = Lambda(lambda x: K.mean(x, axis=1))(a2)
    a2 = RepeatVector(indices_length)(a2)
    a_probs = Multiply()([a1,a2])
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
attention_mul = attention_method(input_1)
Layer_1 = LSTM(64,activation='tanh',return_sequences=True)(attention_mul, training = True)
Layer_2 = LSTM(32,activation='tanh',return_sequences=True)(Layer_1, training = True)
Layer_3 = LSTM(16,activation='tanh',return_sequences=True)(Layer_2, training = True)
Layer_4 = Flatten()(Layer_3)
concat_layer = concatenate([Layer_4,input_2])
dense_layer_1 = Dense(units=64, activation='relu')(concat_layer)
dense_layer_2 = Dense(32, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(16, activation='relu')(dense_layer_2)
dense_layer_4 = Dense(8, activation='relu')(dense_layer_3)
dense_layer_5 = Dense(4, activation='relu')(dense_layer_4)
output = Dense(1)(dense_layer_5)
my_model = Model(inputs=[input_1, input_2], outputs=output)
