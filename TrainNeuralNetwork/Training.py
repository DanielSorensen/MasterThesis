from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.constraints import max_norm
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam as adam
from keras.optimizers import Adamax as adamax
from keras.callbacks import EarlyStopping
from keras import regularizers

from matplotlib import pyplot

import numpy as np
import os.path

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

xFile = "Data/SC2TvZImperfectClean"
yFile = "Data/SC2TvZPerfectClean"

if not os.path.isfile(xFile + ".npy"):
    x_data = np.loadtxt(xFile + '.csv', delimiter=',', dtype=np.float32)
    np.save(xFile + '.npy', x_data);
else:
    x_data = np.load(xFile + '.npy')

if not os.path.isfile(yFile + ".npy"):
    y_data = np.loadtxt(yFile + '.csv', delimiter=',', dtype=np.float32)
    np.save(yFile + '.npy', y_data);
else:
    y_data = np.load(yFile + '.npy')

print("Before split: ", x_data.shape)

#Removing data points
#x_data = x_data[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 93, 94, 95]]
#x_data = np.delete(x_data, [0, 1, 2, 3, 5, 11, 12, 18, 45, 107, 108, 109], 1)

print("After split: ", x_data.shape)

split_at = len(x_data) - len(x_data) // 5

(x_train, x_val) = x_data[:split_at], x_data[split_at:]

(y_train, y_val) = y_data[:split_at], y_data[split_at:]

print("training data: ", x_train.shape[0])
print("perfect data: ", y_train.shape[0])

earlyStop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.00005, patience=10, verbose=0, mode='auto')
callbacks_list = [earlyStop]

HIDDEN_SIZE = 500
HIDDEN_SIZE_2 = 500
HIDDEN_SIZE_3 = 500

model = Sequential()

#model.add(Dropout(0.2, input_shape=(x_train.shape[1],)))
model.add(Dense(output_dim=HIDDEN_SIZE, input_dim=x_train.shape[1]))
model.add(Activation("relu"))
model.add(Dropout(0.2, input_shape=(HIDDEN_SIZE,)))
model.add(Dense(output_dim=HIDDEN_SIZE_2))
model.add(Activation("relu"))
model.add(Dropout(0.2, input_shape=(HIDDEN_SIZE_2,)))
model.add(Dense(output_dim=HIDDEN_SIZE_3))
model.add(Activation("relu"))
#model.add(Dropout(0.2, input_shape=(HIDDEN_SIZE_2,)))
#model.add(Dense(output_dim=HIDDEN_SIZE_3))
#model.add(Activation("relu"))
model.add(Dropout(0.2, input_shape=(HIDDEN_SIZE_3,)))
model.add(Dense(output_dim=y_train.shape[1]))
model.add(Activation("relu"))

model.compile(optimizer=adam(lr=0.0002, beta_1=0.9, beta_2=0.999), loss='mse', metrics=['mse'])
history = model.fit(x_train, y_train, nb_epoch=10, validation_data=(x_val, y_val), batch_size=128, verbose=2)

ind = np.random.randint(0, len(x_val))
rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
preds = model.predict(rowx, verbose=0)
q = rowx[0]
correct = rowy[0]
guess = preds[0]

print('Q', q)
print('T', correct)
print('Guess', guess)

score = model.evaluate(x_val, y_val, batch_size=16)
print('Score: ')
print(score)

model.save("Models/TvZ_BadTopology.h5")

pyplot.plot(history.history['val_mean_squared_error'])
pyplot.plot(history.history['mean_squared_error'])
pyplot.title('Mean Squared Error')
pyplot.legend(['Validation MSE', 'Training MSE'], loc='upper right')
pyplot.show()


#pyplot.plot(history.history['mean_absolute_percentage_error'])
#pyplot.show()

print("")
print("")
