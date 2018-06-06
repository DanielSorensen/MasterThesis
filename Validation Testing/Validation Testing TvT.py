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
import math
import copy

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

#Load data
xFile = "Data/SC2TvTImperfectClean"
yFile = "Data/SC2TvTPerfectClean"

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

#Zero input
x_data_zero = copy.deepcopy(x_data)
#x_data_zero = np.delete(x_data_zero, [0, 1, 2, 3, 5, 11, 12, 18, 45, 107, 108, 109], 1)
#x_data_zero = x_data_zero[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 107, 108, 109]]
#for x in range(0, len(x_data_zero)):
#	for y in range(1, 5):
#		x_data_zero[x][y] = 0
#	for y in range(6, len(x_data_zero[x])):
#		x_data_zero[x][y] = 0

#print("After split: ", x_data_zero.shape)

split_at_zero = len(x_data_zero) - len(x_data_zero) // 5

(x_train_zero, x_val_zero) = x_data_zero[:split_at_zero], x_data_zero[split_at_zero:]


split_at = len(x_data) - len(x_data) // 5

(x_train, x_val) = x_data[:split_at], x_data[split_at:]

(y_train, y_val) = y_data[:split_at], y_data[split_at:]

print("training data: ", x_train.shape[0])
print("perfect data: ", y_train.shape[0])

#Load model
model = load_model('Models/TvT.h5')

#Evaluate model
score = model.evaluate(x_val, y_val, batch_size=16)
print('Score: ')
print(score)

#SCVs baseline
SCVsAverageBaseline = []

if not os.path.isfile("TvTWorkerBaseline.npy"):
	SCVsAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		SCVsAverageList[index].append(rowy[0][1]*100)
	for x in range(0, 35):
		SCVsAverageBaseline.append(np.mean(SCVsAverageList[x]))

	np.save('TvTWorkerBaseline.npy', SCVsAverageBaseline);
else:
	SCVsAverageBaseline = np.load('TvTWorkerBaseline.npy')

#Own SCVs baseline
ownWorkersAverageBaseline = []

if not os.path.isfile("TvTOwnWorkerBaseline.npy"):
	ownWorkersAverageList = [[] * 1 for i in range(35)]

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)	
		ownWorkersAverageList[index].append(rowx[0][42-37]*100)

	for x in range(0, 35):
		ownWorkersAverageBaseline.append(np.mean(ownWorkersAverageList[x]))

	np.save('TvTOwnWorkerBaseline.npy', ownWorkersAverageBaseline);
else:
	ownWorkersAverageBaseline = np.load('TvTOwnWorkerBaseline.npy')

#Tanks baseline
tanksAverageBaseline = []

if not os.path.isfile("TvTTanksBaseline.npy"):
	tanksAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		tanksAverageList[index].append((rowy[0][78-37]*100 + rowy[0][51-37]*100))
	for x in range(0, 35):
		tanksAverageBaseline.append(np.mean(tanksAverageList[x]))

	np.save('TvTTanksBaseline.npy', tanksAverageBaseline);
else:
	tanksAverageBaseline = np.load('TvTTanksBaseline.npy')


#Bases baseline
basesAverageBaseline = []

if not os.path.isfile("TvTBasesBaseline.npy"):
	basesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		basesAverageList[index].append(rowy[0][54]*10)
	for x in range(0, 35):
		basesAverageBaseline.append(np.mean(basesAverageList[x]))

	np.save('TvTBasesBaseline.npy', basesAverageBaseline);
else:
	basesAverageBaseline = np.load('TvTBasesBaseline.npy')

#Own bases baseline
ownBasesAverageBaseline = []

if not os.path.isfile("TvTOwnBasesBaseline.npy"):
	ownBasesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownBasesAverageList[index].append(rowx[0][109]*10)
	for x in range(0, 35):
		ownBasesAverageBaseline.append(np.mean(ownBasesAverageList[x]))

	np.save('TvTOwnBasesBaseline.npy', ownBasesAverageBaseline);
else:
	ownBasesAverageBaseline = np.load('TvTOwnBasesBaseline.npy')

#ArmyCount baseline
armyCountAverageBaseline = []

if not os.path.isfile("TvTArmyCountBaseline.npy"):
	armyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		armyCountAverageList[index].append(rowy[0][51]*100)
	for x in range(0, 35):
		armyCountAverageBaseline.append(np.mean(armyCountAverageList[x]))

	np.save('TvTArmyCountBaseline.npy', armyCountAverageBaseline);
else:
	armyCountAverageBaseline = np.load('TvTArmyCountBaseline.npy')

#Own armyCount baseline
ownArmyCountAverageBaseline = []

if not os.path.isfile("TvTOwnArmyCountBaseline.npy"):
	ownArmyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownArmyCountAverageList[index].append(rowx[0][106]*100)
	for x in range(0, 35):
		ownArmyCountAverageBaseline.append(np.mean(ownArmyCountAverageList[x]))

	np.save('TvTOwnArmyCountBaseline.npy', ownArmyCountAverageBaseline);
else:
	ownArmyCountAverageBaseline = np.load('TvTOwnArmyCountBaseline.npy')

def oneGuess():

	print()

	print('Time: ' , int(round(rowx[0][0]*25000/24)), 'seconds')

	print()

	print('Own units and bases:')

	basesOwn = int(round(rowx[0][109]*10))

	armyCountOwn = int(round(rowx[0][106]*100))

	SCVsOwn = int(round(rowx[0][42-37]*100))
	marinesOwn = int(round(rowx[0][49-37]*100))
	maraudersOwn = int(round(rowx[0][48-37]*100))
	tanksOwn = int(round(rowx[0][82-37]*100)) + int(round(rowx[0][55-37]*100))

	print('Bases: ', basesOwn)

	print('ArmyCount: ', armyCountOwn)

	print('SCVs: ', SCVsOwn)
	print('Marines: ', marinesOwn)
	print('Marauders: ', maraudersOwn)
	print('Tanks: ', tanksOwn)

	print()

	print('Enemy units and bases:')

	basesGuess = int(round(preds[0][54]*10))
	basesCorrect = int(round(rowy[0][54]*10))

	print('BasesGuess: ', basesGuess)
	print('BasesCorrect: ', basesCorrect)

	armyCountGuess = int(round(preds[0][51]*100))
	armyCountCorrect = int(round(rowy[0][51]*100))

	print('ArmyCountGuess: ', armyCountGuess)
	print('ArmyCountCorrect: ', armyCountCorrect)

	print()

	print('Terran:')

	SCVsGuess = int(round(preds[0][38-37]*100))
	SCVsCorrect = int(round(rowy[0][38-37]*100))
	marinesGuess = int(round(preds[0][45-37]*100))
	marinesCorrect = int(round(rowy[0][45-37]*100))
	maraudersGuess = int(round(preds[0][44-37]*100))
	maraudersCorrect = int(round(rowy[0][44-37]*100))
	tanksGuess = int(round(preds[0][78-37]*100)) + int(round(preds[0][51-37]*100))
	tanksCorrect = int(round(rowy[0][78-37]*100)) + int(round(rowy[0][51-37]*100))


	print('SCVsGuess: ', SCVsGuess)
	print('SCVsCorrect: ', SCVsCorrect)
	print('MarinesGuess: ', marinesGuess)
	print('MarinesCorrect: ', marinesCorrect)
	print('MaraudersGuess: ', maraudersGuess)
	print('MaraudersCorrect: ', maraudersCorrect)
	print('TanksGuess: ', tanksGuess)
	print('TanksCorrect: ', tanksCorrect)

	print("")
	print("")

def averageSingleUnit(unitIndex, name):
	unitDifferenceList = [[] * 1 for i in range(35)]
	unitActualList = [[] * 1 for i in range(35)]
	unitGuessList = [[] * 1 for i in range(35)]
	unitDifferenceAvg = []
	unitActualAvg = []
	unitGuessAvg = []

	unitSquaredDifferences = [[] * 1 for i in range(35)]
	unitStandardDeviations = []
	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		unitDifferenceList[index].append(math.fabs(preds[0][unitIndex]*100 - rowy[0][unitIndex]*100))
		unitActualList[index].append(rowy[0][unitIndex]*100)
		unitGuessList[index].append(preds[0][unitIndex]*100)
	for x in range(0, 35):
		unitDifferenceAvg.append(np.mean(unitDifferenceList[x]))
		unitActualAvg.append(np.mean(unitActualList[x]))
		unitGuessAvg.append(np.mean(unitGuessList[x]))

	for x in range(0, 35):
		for y in range(0, len(unitDifferenceList[x])):
			unitSquaredDifferences[x].append(math.pow(unitDifferenceAvg[x] - unitDifferenceList[x][y], 2))
		unitStandardDeviations.append(math.sqrt(np.mean(unitSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(unitDifferenceAvg[x]+unitStandardDeviations[x])
	    c1.append(unitDifferenceAvg[x]-unitStandardDeviations[x])

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	if name == 'ArmyCount':
		pyplot.ylabel(name, **axis_font)
	else:
		pyplot.ylabel('Number of '+ name, **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, unitDifferenceAvg, color='blue')
	pyplot.plot(x, unitActualAvg, color='red')
	pyplot.plot(x, unitGuessAvg, color='green')
	if name == 'ArmyCount':
		pyplot.legend(['Absolute Error', 'Actual ' + name, 'Predicted ' + name, 'Standard Deviation'], loc='upper left')
	else:
		pyplot.legend(['Absolute Error', 'Actual Number of ' + name, 'Predicted Number of ' + name, 'Standard Deviation'], loc='upper left')
	pyplot.title('Absolute Error, Prediction and Actual Value of ' + name + ' (TvT)', **title_font)
	pyplot.show()

def averageSingleTanks(name):
	unitDifferenceList = [[] * 1 for i in range(35)]
	unitActualList = [[] * 1 for i in range(35)]
	unitGuessList = [[] * 1 for i in range(35)]
	unitDifferenceAvg = []
	unitActualAvg = []
	unitGuessAvg = []

	unitSquaredDifferences = [[] * 1 for i in range(35)]
	unitStandardDeviations = []
	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		unitDifferenceList[index].append(math.fabs((preds[0][78-37]*100 + preds[0][51-37]*100) - (rowy[0][78-37]*100 + rowy[0][51-37]*100)))
		unitActualList[index].append((rowy[0][78-37]*100 + rowy[0][51-37]*100))
		unitGuessList[index].append((preds[0][78-37]*100 + preds[0][51-37]*100))
	for x in range(0, 35):
		unitDifferenceAvg.append(np.mean(unitDifferenceList[x]))
		unitActualAvg.append(np.mean(unitActualList[x]))
		unitGuessAvg.append(np.mean(unitGuessList[x]))

	for x in range(0, 35):
		for y in range(0, len(unitDifferenceList[x])):
			unitSquaredDifferences[x].append(math.pow(unitDifferenceAvg[x] - unitDifferenceList[x][y], 2))
		unitStandardDeviations.append(math.sqrt(np.mean(unitSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(unitDifferenceAvg[x]+unitStandardDeviations[x])
	    c1.append(unitDifferenceAvg[x]-unitStandardDeviations[x])
	
	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Number of '+ name, **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, unitDifferenceAvg, color='blue')
	pyplot.plot(x, unitActualAvg, color='red')
	pyplot.plot(x, unitGuessAvg, color='green')
	pyplot.legend(['Absolute Error', 'Actual Number of ' + name, 'Predicted Number of ' + name, 'Standard Deviation'], loc='upper left')
	pyplot.title('Absolute Error, Prediction and Actual Value of ' + name + ' (TvT)', **title_font)
	pyplot.show()

def averageSingleBases():
	basesDifferenceList = [[] * 1 for i in range(35)]
	basesActualList = [[] * 1 for i in range(35)]
	basesGuessList = [[] * 1 for i in range(35)]
	basesDifferenceAvg = []
	basesActualAvg = []
	basesGuessAvg = []

	basesSquaredDifferences = [[] * 1 for i in range(35)]
	basesStandardDeviations = []
	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		basesDifferenceList[index].append(math.fabs(preds[0][54]*10 - rowy[0][54]*10))
		basesActualList[index].append(rowy[0][54]*10)
		basesGuessList[index].append(preds[0][54]*10)
	for x in range(0, 35):
		basesDifferenceAvg.append(np.mean(basesDifferenceList[x]))
		basesActualAvg.append(np.mean(basesActualList[x]))
		basesGuessAvg.append(np.mean(basesGuessList[x]))

	for x in range(0, 35):
		for y in range(0, len(basesDifferenceList[x])):
			basesSquaredDifferences[x].append(math.pow(basesDifferenceAvg[x] - basesDifferenceList[x][y], 2))
		basesStandardDeviations.append(math.sqrt(np.mean(basesSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(basesDifferenceAvg[x]+basesStandardDeviations[x])
	    c1.append(basesDifferenceAvg[x]-basesStandardDeviations[x])
	
	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Number of Bases', **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, basesDifferenceAvg, color='blue')
	pyplot.plot(x, basesActualAvg, color='red')
	pyplot.plot(x, basesGuessAvg, color='green')
	pyplot.legend(['Absolute Error', 'Actual Number of Bases', 'Predicted Number of Bases', 'Standard Deviation'], loc='upper left')
	pyplot.title('Absolute Error, Prediction and Actual Value of Bases (TvT)', **title_font)
	pyplot.show()

def averageMostImportant():
	SCVsDifferenceList = [[] * 1 for i in range(35)]
	SCVsDifferenceAvg = []
	marinesDifferenceList = [[] * 1 for i in range(35)]
	marinesDifferenceAvg = []
	maraudersDifferenceList = [[] * 1 for i in range(35)]
	maraudersDifferenceAvg = []
	tanksDifferenceList = [[] * 1 for i in range(35)]
	tanksDifferenceAvg = []
	armyCountDifferenceList = [[] * 1 for i in range(35)]
	armyCountDifferenceAvg = []
	basesDifferenceList = [[] * 1 for i in range(35)]
	basesDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		armyCountDifferenceList[index].append(math.fabs(preds[0][51]*100 - rowy[0][51]*100))
		tanksDifferenceList[index].append(math.fabs((preds[0][78-37]*100 + preds[0][51-37]*100) - (rowy[0][78-37]*100 + rowy[0][51-37]*100)))
		marinesDifferenceList[index].append(math.fabs(preds[0][45-37]*100 - rowy[0][45-37]*100))
		maraudersDifferenceList[index].append(math.fabs(preds[0][44-37]*100 - rowy[0][44-37]*100))
		SCVsDifferenceList[index].append(math.fabs(preds[0][38-37]*100 - rowy[0][38-37]*100))
		basesDifferenceList[index].append(math.fabs(preds[0][54]*10 - rowy[0][54]*10))
		
	for x in range(0, 35):
		armyCountDifferenceAvg.append(np.mean(armyCountDifferenceList[x]))
		tanksDifferenceAvg.append(np.mean(tanksDifferenceList[x]))	
		marinesDifferenceAvg.append(np.mean(marinesDifferenceList[x]))
		maraudersDifferenceAvg.append(np.mean(maraudersDifferenceList[x]))
		SCVsDifferenceAvg.append(np.mean(SCVsDifferenceList[x]))
		basesDifferenceAvg.append(np.mean(basesDifferenceList[x]))
	
	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Absolute error', **axis_font)
	pyplot.grid(True)
	pyplot.plot(x, armyCountDifferenceAvg, color='blue')
	pyplot.plot(x, SCVsDifferenceAvg, color='red')
	pyplot.plot(x, basesDifferenceAvg, color='green')
	pyplot.plot(x, marinesDifferenceAvg, color='black')
	pyplot.plot(x, maraudersDifferenceAvg, color='yellow')
	pyplot.plot(x, tanksDifferenceAvg, color='purple')	
	pyplot.legend(['ArmyCount', 'SCVs', 'Bases', 'Marines', 'Marauders', 'Tanks'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Essential Information (TvT)', **title_font)
	pyplot.show()

def averageMostImportantInOne():
	allUnitsDifferenceAverages = [[] * 1 for i in range(35)]
	allUnitsTotalAvg = []
	indices = [41, 14, 8, 7, 1]
	counter = 0

	for x in indices:
		print("Counter:", counter)
		currentUnitDifferenceList = [[] * 1 for i in range(35)]		

		for y in range(0, len(x_val)):		
			rowx, rowy = x_val[np.array([y])], y_val[np.array([y])]
			preds = model.predict(rowx, verbose=0)
			index = int(int(round(rowx[0][0]*25000/24))/30)
			currentUnitDifferenceList[index].append(math.fabs(preds[0][x]*100 - rowy[0][x]*100))
		for z in range(0, 35):
			allUnitsDifferenceAverages[z].append(np.mean(currentUnitDifferenceList[z]))
		counter = counter + 1
		
	for x in range(0, 35):
		allUnitsTotalAvg.append(np.mean(allUnitsDifferenceAverages[x]))

	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Units')
	pyplot.grid(True)
	pyplot.plot(x, allUnitsTotalAvg)
	pyplot.legend(['AllUnitsAverageDifference'], loc='upper left')
	pyplot.show()

def averageAll():
	allUnitsDifferenceAverages = [[] * 1 for i in range(35)]
	allUnitsTotalAvg = []
	counter = 0

	allUnitsSquaredDifferences = [[] * 1 for i in range(35)]
	allUnitsStandardDeviations = []

	for x in range(0, 51):
		print("Counter:", counter)
		currentUnitDifferenceList = [[] * 1 for i in range(35)]		

		for y in range(0, len(x_val)):		
			rowx, rowy = x_val[np.array([y])], y_val[np.array([y])]
			preds = model.predict(rowx, verbose=0)
			index = int(int(round(rowx[0][0]*25000/24))/30)
			currentUnitDifferenceList[index].append(math.fabs(preds[0][x]*100 - rowy[0][x]*100))
		for z in range(0, 35):
			allUnitsDifferenceAverages[z].append(np.mean(currentUnitDifferenceList[z]))
		counter = counter + 1
		
	for x in range(0, 35):
		allUnitsTotalAvg.append(np.mean(allUnitsDifferenceAverages[x]))

	for x in range(0, 35):
		for y in range(0, len(allUnitsDifferenceAverages[x])):
			allUnitsSquaredDifferences[x].append(math.pow(allUnitsTotalAvg[x] - allUnitsDifferenceAverages[x][y], 2))
		allUnitsStandardDeviations.append(math.sqrt(np.mean(allUnitsSquaredDifferences[x])))

	b = []
	c = []
	for x in range(0, 35):
	    b.append(allUnitsTotalAvg[x]+allUnitsStandardDeviations[x])
	    c.append(allUnitsTotalAvg[x]-allUnitsStandardDeviations[x])

	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Units')
	pyplot.grid(True)

	pyplot.fill_between(np.arange(0, 35*30, 30), c, b, color='xkcd:blue blue', alpha=0.3)
	pyplot.plot(x, allUnitsTotalAvg, color='xkcd:blue with a hint of purple')
	pyplot.legend(['AllUnitsAverageDifference', 'AllUnitsAverageDifference Std'], loc='upper left')
	pyplot.show()

def averageSCVsPercentage():
	SCVsPercentageDifferenceList = [[] * 1 for i in range(35)]
	SCVsPercentageDifferenceAvg = []
	SCVsBaselinePercentageDifferenceList = [[] * 1 for i in range(35)]
	SCVsBaselinePercentageDifferenceAvg = []
	SCVsBaseline2PercentageDifferenceList = [[] * 1 for i in range(35)]
	SCVsBaseline2PercentageDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if(not rowy[0][1] == 0):
			difference = math.fabs(preds[0][1]*100 - rowy[0][1]*100)
			actual = rowy[0][1]*100
			baselineDifference = math.fabs(rowx[0][42-37]*100 - rowy[0][1]*100)
			baseline2Difference = math.fabs(SCVsAverageBaseline[index] - rowy[0][1]*100)
		
			SCVsPercentageDifferenceList[index].append((difference/actual)*100)
			SCVsBaselinePercentageDifferenceList[index].append((baselineDifference/actual)*100)
			SCVsBaseline2PercentageDifferenceList[index].append((baseline2Difference/actual)*100)
	for x in range(0, 35):
		SCVsPercentageDifferenceAvg.append(np.mean(SCVsPercentageDifferenceList[x]))
		SCVsBaselinePercentageDifferenceAvg.append(np.mean(SCVsBaselinePercentageDifferenceList[x]))
		SCVsBaseline2PercentageDifferenceAvg.append(np.mean(SCVsBaseline2PercentageDifferenceList[x]))

	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, SCVsPercentageDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, SCVsBaselinePercentageDifferenceAvg, color='xkcd:gold')
	pyplot.plot(x, SCVsBaseline2PercentageDifferenceAvg, color='green')
	pyplot.legend(['SCVs Guess Percent Difference', 'Own Workers Baseline Percent Difference', 'Average SCVs Baseline Percent Difference'], loc='upper left')
	pyplot.title('Average percentage worker difference TvT')
	pyplot.show()

def averageTanksPercentage():
	tanksPercentageDifferenceList = [[] * 1 for i in range(35)]
	tanksPercentageDifferenceAvg = []
	tanksBaselinePercentageDifferenceList = [[] * 1 for i in range(35)]
	tanksBaselinePercentageDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if(not (rowy[0][78-37] + rowy[0][51-37]) == 0):
			difference = math.fabs((preds[0][78-37]*100 + preds[0][51-37]*100) - (rowy[0][78-37]*100 + rowy[0][51-37]*100))
			actual = (rowy[0][78-37]*100 + rowy[0][51-37]*100)
			baselineDifference = math.fabs(tanksAverageBaseline[index] - (rowy[0][78-37]*100 + rowy[0][51-37]*100))
		
			tanksPercentageDifferenceList[index].append((difference/actual)*100)
			tanksBaselinePercentageDifferenceList[index].append((baselineDifference/actual)*100)

	for x in range(0, 35):
		tanksPercentageDifferenceAvg.append(np.mean(tanksPercentageDifferenceList[x]))
		tanksBaselinePercentageDifferenceAvg.append(np.mean(tanksBaselinePercentageDifferenceList[x]))
	
	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	axes = pyplot.gca()
	axes.set_ylim([0,120])
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, tanksPercentageDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, tanksBaselinePercentageDifferenceAvg, color='xkcd:gold')
	pyplot.legend(['Tanks Guess Percent Difference', 'Average Tanks Baseline Percent Difference'], loc='upper left')
	pyplot.title('Average percentage tank difference TvT')
	pyplot.show()

def workersBaselineComparison():
	workersGuessDifferenceList = [[] * 1 for i in range(35)]
	workersGuessDifferenceAvg = []
	workersBaselineDifferenceList = [[] * 1 for i in range(35)]
	workersBaselineDifferenceAvg = []	
	workersBaseline2DifferenceList = [[] * 1 for i in range(35)]
	workersBaseline2DifferenceAvg = []
	
	workersGuessSquaredDifferences = [[] * 1 for i in range(35)]
	workersGuessStandardDeviations = []	
	workersBaselineSquaredDifferences = [[] * 1 for i in range(35)]
	workersBaselineStandardDeviations = []

	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		workersGuessDifferenceList[index].append(math.fabs(preds[0][1]*100 - rowy[0][1]*100))
		workersBaselineDifferenceList[index].append(math.fabs(rowy[0][1]*100 - rowx[0][42-37]*100))
		workersBaseline2DifferenceList[index].append(math.fabs(rowy[0][1]*100 - SCVsAverageBaseline[index]))
		
	for x in range(0, 35):
		workersGuessDifferenceAvg.append(np.mean(workersGuessDifferenceList[x]))
		workersBaselineDifferenceAvg.append(np.mean(workersBaselineDifferenceList[x]))
		workersBaseline2DifferenceAvg.append(np.mean(workersBaseline2DifferenceList[x]))
	
	for x in range(0, 35):
		for y in range(0, len(workersGuessDifferenceList[x])):
			workersGuessSquaredDifferences[x].append(math.pow(workersGuessDifferenceAvg[x] - workersGuessDifferenceList[x][y], 2))
			workersBaselineSquaredDifferences[x].append(math.pow(workersBaselineDifferenceAvg[x] - workersBaselineDifferenceList[x][y], 2))
		workersGuessStandardDeviations.append(math.sqrt(np.mean(workersGuessSquaredDifferences[x])))
		workersBaselineStandardDeviations.append(math.sqrt(np.mean(workersBaselineSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(workersGuessDifferenceAvg[x]+workersGuessStandardDeviations[x])
	    c1.append(workersGuessDifferenceAvg[x]-workersGuessStandardDeviations[x])

	b2 = []
	c2 = []
	for x in range(0, 35):
	    b2.append(workersBaselineDifferenceAvg[x]+workersBaselineStandardDeviations[x])
	    c2.append(workersBaselineDifferenceAvg[x]-workersBaselineStandardDeviations[x])

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Absolute error', **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, workersGuessDifferenceAvg, color='blue')
	pyplot.plot(x, workersBaselineDifferenceAvg, color='green')
	pyplot.plot(x, workersBaseline2DifferenceAvg, color='red')
	pyplot.legend(['SCVs', 'Baseline (Own Units)', 'Baseline (Average for matchup)', 'SCVs Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of SCVs (TvT)', **title_font)
	pyplot.show()

def basesBaselineComparison():
	basesGuessDifferenceList = [[] * 1 for i in range(35)]
	basesGuessDifferenceAvg = []
	basesBaselineDifferenceList = [[] * 1 for i in range(35)]
	basesBaselineDifferenceAvg = []
	basesBaseline2DifferenceList = [[] * 1 for i in range(35)]
	basesBaseline2DifferenceAvg = []
	
	basesGuessSquaredDifferences = [[] * 1 for i in range(35)]
	basesGuessStandardDeviations = []	
	basesBaselineSquaredDifferences = [[] * 1 for i in range(35)]
	basesBaselineStandardDeviations = []

	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		basesGuessDifferenceList[index].append(math.fabs(preds[0][54]*10 - rowy[0][54]*10))
		basesBaselineDifferenceList[index].append(math.fabs(rowy[0][54]*10 - rowx[0][109]*10))
		basesBaseline2DifferenceList[index].append(math.fabs(rowy[0][54]*10 - basesAverageBaseline[index]))
		
	for x in range(0, 35):
		basesGuessDifferenceAvg.append(np.mean(basesGuessDifferenceList[x]))
		basesBaselineDifferenceAvg.append(np.mean(basesBaselineDifferenceList[x]))
		basesBaseline2DifferenceAvg.append(np.mean(basesBaseline2DifferenceList[x]))

	for x in range(0, 35):
		for y in range(0, len(basesGuessDifferenceList[x])):
			basesGuessSquaredDifferences[x].append(math.pow(basesGuessDifferenceAvg[x] - basesGuessDifferenceList[x][y], 2))
			basesBaselineSquaredDifferences[x].append(math.pow(basesBaselineDifferenceAvg[x] - basesBaselineDifferenceList[x][y], 2))
		basesGuessStandardDeviations.append(math.sqrt(np.mean(basesGuessSquaredDifferences[x])))
		basesBaselineStandardDeviations.append(math.sqrt(np.mean(basesBaselineSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(basesGuessDifferenceAvg[x]+basesGuessStandardDeviations[x])
	    c1.append(basesGuessDifferenceAvg[x]-basesGuessStandardDeviations[x])

	b2 = []
	c2 = []
	for x in range(0, 35):
	    b2.append(basesBaselineDifferenceAvg[x]+basesBaselineStandardDeviations[x])
	    c2.append(basesBaselineDifferenceAvg[x]-basesBaselineStandardDeviations[x])

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Absolute error', **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, basesGuessDifferenceAvg, color='blue')
	pyplot.plot(x, basesBaselineDifferenceAvg, color='green')
	pyplot.plot(x, basesBaseline2DifferenceAvg, color='red')
	pyplot.legend(['Bases', 'Baseline (Own Units)', 'Baseline (Average for matchup)', 'Bases Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Bases (TvT)', **title_font)
	pyplot.show()

def armyCountBaselineComparison():
	armyCountGuessDifferenceList = [[] * 1 for i in range(35)]
	armyCountGuessDifferenceAvg = []
	armyCountBaselineDifferenceList = [[] * 1 for i in range(35)]
	armyCountBaselineDifferenceAvg = []
	armyCountBaseline2DifferenceList = [[] * 1 for i in range(35)]
	armyCountBaseline2DifferenceAvg = []
	
	armyCountGuessSquaredDifferences = [[] * 1 for i in range(35)]
	armyCountGuessStandardDeviations = []	
	armyCountBaselineSquaredDifferences = [[] * 1 for i in range(35)]
	armyCountBaselineStandardDeviations = []

	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		armyCountGuessDifferenceList[index].append(math.fabs(preds[0][51]*100 - rowy[0][51]*100))
		armyCountBaselineDifferenceList[index].append(math.fabs(rowy[0][51]*100 - rowx[0][106]*100))
		armyCountBaseline2DifferenceList[index].append(math.fabs(rowy[0][51]*100 - armyCountAverageBaseline[index]))
		
	for x in range(0, 35):
		armyCountGuessDifferenceAvg.append(np.mean(armyCountGuessDifferenceList[x]))
		armyCountBaselineDifferenceAvg.append(np.mean(armyCountBaselineDifferenceList[x]))
		armyCountBaseline2DifferenceAvg.append(np.mean(armyCountBaseline2DifferenceList[x]))
	
	for x in range(0, 35):
		for y in range(0, len(armyCountGuessDifferenceList[x])):
			armyCountGuessSquaredDifferences[x].append(math.pow(armyCountGuessDifferenceAvg[x] - armyCountGuessDifferenceList[x][y], 2))
			armyCountBaselineSquaredDifferences[x].append(math.pow(armyCountBaselineDifferenceAvg[x] - armyCountBaselineDifferenceList[x][y], 2))
		armyCountGuessStandardDeviations.append(math.sqrt(np.mean(armyCountGuessSquaredDifferences[x])))
		armyCountBaselineStandardDeviations.append(math.sqrt(np.mean(armyCountBaselineSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(armyCountGuessDifferenceAvg[x]+armyCountGuessStandardDeviations[x])
	    c1.append(armyCountGuessDifferenceAvg[x]-armyCountGuessStandardDeviations[x])

	b2 = []
	c2 = []
	for x in range(0, 35):
	    b2.append(armyCountBaselineDifferenceAvg[x]+armyCountBaselineStandardDeviations[x])
	    c2.append(armyCountBaselineDifferenceAvg[x]-armyCountBaselineStandardDeviations[x])

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Absolute error', **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, armyCountGuessDifferenceAvg, color='blue')
	pyplot.plot(x, armyCountBaselineDifferenceAvg, color='green')
	pyplot.plot(x, armyCountBaseline2DifferenceAvg, color='red')
	pyplot.legend(['ArmyCount', 'Baseline (Own ArmyCount)', 'Baseline (Average for matchup)', 'ArmyCount Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of ArmyCount (TvT)', **title_font)
	pyplot.show()

#averageSingleUnit(1, "SCVs")
#averageSingleUnit(45-37, "Marines")
#averageSingleUnit(44-37, "Marauders")
averageSingleUnit(51, "ArmyCount")
#averageSingleTanks("Tanks")
#averageSingleBases()
#workersBaselineComparison()
#armyCountBaselineComparison()
#basesBaselineComparison()
#averageMostImportant()

