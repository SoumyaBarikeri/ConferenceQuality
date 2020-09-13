"""
This file trains a model for classifying conference as Computer Science/Non Computer Science
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K
from sklearn.metrics import precision_score


# Read conference data downloaded from WikiCfp as Dataframe and merge it with Labeled conferences
df = pd.read_csv('data/wCfP_data_full_new.csv')
wCfpLabeled = pd.read_csv('data/conf_topics_420.csv')

df_reduced = df.loc[:, ['eventID', 'title', 'text']]
wCfpLabeled = pd.merge(wCfpLabeled, df_reduced, on='eventID')

wCfpLabeled.set_index('eventID', inplace=True)

# Get topics of conferences
X = wCfpLabeled.semantic + wCfpLabeled.syntactic
X = X.astype(str)

print(X.shape)
print(X.head())

Y = wCfpLabeled.csLabel

print(Y.head())

# Encode target column
Y = Y.replace("Non Computer science", 0)
Y = Y.replace("Computer science", 1)
print('Y after replace' + format(Y.shape))
print(Y.head())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(Y_test)

max_words = 50000
max_len = 250
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# print(sequences_matrix)


def RNN():
    """
    Function builds Recurrent Neural Network model

    Returns
    -------
    model
        The RNN model with different layers

    """
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(layer)
    layer = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(layer)
    layer = Dense(64,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


def recall(y_true, y_pred):
    """
    Function calculates recall of the model

    Parameters
    ----------
    y_true : pd.Series or np.array
        True labels(Computer Science or Non Computer Science) of conferences
    y_pred : pd.Series or np.array
        Labels predicted by the classifier

    Returns
    -------
    float
        Recall score of the model

    """

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return recall


opt = Adam()
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[recall])

model.fit(sequences_matrix,Y_train,batch_size=32,epochs=30, validation_split=0.3, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0005)])

# Predict CS labels for unseen test sample
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

score = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Recall: {:0.3f}'.format(score[0], score[1]))

y_pred = model.predict(test_sequences_matrix)

precision = precision_score(Y_test, y_pred.round())
print('Precision: %f' % precision)


model.save('models/modelRnn_wCfp_full_390.h5')

