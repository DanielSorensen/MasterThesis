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

xFile = "Data/SC2TvPImperfectClean"
yFile = "Data/SC2TvPPerfectClean"

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
#x_data_zero = np.delete(x_data_zero, [0, 1, 2, 3, 5, 11, 12, 18, 45, 93, 94, 95], 1)
#x_data_zero = x_data_zero[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 93, 94, 95]]
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

model = load_model('Models/TvP.h5')

#Evaluate model
#score = model.evaluate(x_val, y_val, batch_size=16)
#print('Score: ')
#print(score)


#Probes baseline
probesAverageBaseline = []

if not os.path.isfile("TvPWorkerBaseline.npy"):
	probesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		probesAverageList[index].append(rowy[0][9]*100)
	for x in range(0, 35):
		probesAverageBaseline.append(np.mean(probesAverageList[x]))

	np.save('TvPWorkerBaseline.npy', probesAverageBaseline);
else:
	probesAverageBaseline = np.load('TvPWorkerBaseline.npy')

#Own SCVs baseline
ownSCVsAverageBaseline = []

if not os.path.isfile("TvPOwnWorkerBaseline.npy"):
	ownSCVsAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownSCVsAverageList[index].append(rowx[0][42-37]*100)
	for x in range(0, 35):
		ownSCVsAverageBaseline.append(np.mean(ownSCVsAverageList[x]))

	np.save('TvPOwnWorkerBaseline.npy', ownSCVsAverageBaseline);
else:
	ownSCVsAverageBaseline = np.load('TvPOwnWorkerBaseline.npy')

#Immortals baseline
immortalsAverageBaseline = []

if not os.path.isfile("TvPImmortalsBaseline.npy"):
	immortalsAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		immortalsAverageList[index].append(rowy[0][5]*100)
	for x in range(0, 35):
		immortalsAverageBaseline.append(np.mean(immortalsAverageList[x]))

	np.save('TvPImmortalsBaseline.npy', immortalsAverageBaseline);
else:
	immortalsAverageBaseline = np.load('TvPImmortalsBaseline.npy')

#Bases baseline
basesAverageBaseline = []

if not os.path.isfile("TvPBasesBaseline.npy"):
	basesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		basesAverageList[index].append(rowy[0][40]*10)
	for x in range(0, 35):
		basesAverageBaseline.append(np.mean(basesAverageList[x]))

	np.save('TvPBasesBaseline.npy', basesAverageBaseline);
else:
	basesAverageBaseline = np.load('TvPBasesBaseline.npy')

#Own bases baseline
ownBasesAverageBaseline = []

if not os.path.isfile("TvPOwnBasesBaseline.npy"):
	ownBasesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownBasesAverageList[index].append(rowx[0][95]*10)
	for x in range(0, 35):
		ownBasesAverageBaseline.append(np.mean(ownBasesAverageList[x]))

	np.save('TvPOwnBasesBaseline.npy', ownBasesAverageBaseline);
else:
	ownBasesAverageBaseline = np.load('TvPOwnBasesBaseline.npy')

#ArmyCount baseline
armyCountAverageBaseline = []

if not os.path.isfile("TvPArmyCountBaseline.npy"):
	armyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		armyCountAverageList[index].append(rowy[0][37]*100)
	for x in range(0, 35):
		armyCountAverageBaseline.append(np.mean(armyCountAverageList[x]))

	np.save('TvPArmyCountBaseline.npy', armyCountAverageBaseline);
else:
	armyCountAverageBaseline = np.load('TvPArmyCountBaseline.npy')

#Own armyCount baseline
ownArmyCountAverageBaseline = []

if not os.path.isfile("TvPOwnArmyCountBaseline.npy"):
	ownArmyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownArmyCountAverageList[index].append(rowx[0][92]*100)
	for x in range(0, 35):
		ownArmyCountAverageBaseline.append(np.mean(ownArmyCountAverageList[x]))

	np.save('TvPOwnArmyCountBaseline.npy', ownArmyCountAverageBaseline);
else:
	ownArmyCountAverageBaseline = np.load('TvPOwnArmyCountBaseline.npy')

def oneGuess():

	print()

	print('Time: ' , int(round(rowx[0][0]*25000/24)), 'seconds')

	print()

	print('Own units and bases:')

	basesOwn = int(round(rowx[0][95]*10))

	armyCountOwn = int(round(rowx[0][92]*100))

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

	basesGuess = int(round(preds[0][40]*10))
	basesCorrect = int(round(rowy[0][40]*10))

	print('BasesGuess: ', basesGuess)
	print('BasesCorrect: ', basesCorrect)

	armyCountGuess = int(round(preds[0][37]*100))
	armyCountCorrect = int(round(rowy[0][37]*100))

	print('ArmyCountGuess: ', armyCountGuess)
	print('ArmyCountCorrect: ', armyCountCorrect)

	print()

	print('Protoss:')

	probesGuess = int(round(preds[0][9]*100))
	probesCorrect = int(round(rowy[0][9]*100))
	immortalsGuess = int(round(preds[0][5]*100))
	immortalsCorrect = int(round(rowy[0][5]*100))
	stalkersGuess = int(round(preds[0][11]*100))
	stalkersCorrect = int(round(rowy[0][11]*100))
	zealotsGuess = int(round(preds[0][14]*100))
	zealotsCorrect = int(round(rowy[0][14]*100))


	print('ProbesGuess: ', probesGuess)
	print('ProbesCorrect: ', probesCorrect)
	print('ImmortalsGuess: ', immortalsGuess)
	print('ImmortalsCorrect: ', immortalsCorrect)
	print('StalkersGuess: ', stalkersGuess)
	print('StalkersCorrect: ', stalkersCorrect)
	print('ZealotsGuess: ', zealotsGuess)
	print('ZealotsCorrect: ', zealotsCorrect)

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
	pyplot.title('Absolute Error, Prediction and Actual Value of ' + name + ' (TvP)', **title_font)
	pyplot.show()

def averageSingleBases():
	basesDifferenceList = [[] * 1 for i in range(35)]
	basesActualList = [[] * 1 for i in range(35)]
	basesGuessList = [[] * 1 for i in range(35)]
	basesDifferenceAvg = []
	basesActualAvg = []
	basesGuessAvg = []

	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	basesSquaredDifferences = [[] * 1 for i in range(35)]
	basesStandardDeviations = []
	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		basesDifferenceList[index].append(math.fabs(preds[0][40]*10 - rowy[0][40]*10))
		basesActualList[index].append(rowy[0][40]*10)
		basesGuessList[index].append(preds[0][40]*10)
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
	pyplot.title('Absolute Error, Prediction and Actual Value of Bases (TvP)', **title_font)
	pyplot.show()

def averageMostImportant():
	probesDifferenceList = [[] * 1 for i in range(35)]
	probesDifferenceAvg = []
	immortalsDifferenceList = [[] * 1 for i in range(35)]
	immortalsDifferenceAvg = []
	stalkersDifferenceList = [[] * 1 for i in range(35)]
	stalkersDifferenceAvg = []
	zealotsDifferenceList = [[] * 1 for i in range(35)]
	zealotsDifferenceAvg = []
	armyCountDifferenceList = [[] * 1 for i in range(35)]
	armyCountDifferenceAvg = []
	basesDifferenceList = [[] * 1 for i in range(35)]
	basesDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		armyCountDifferenceList[index].append(math.fabs(preds[0][37]*100 - rowy[0][37]*100))
		probesDifferenceList[index].append(math.fabs(preds[0][9]*100 - rowy[0][9]*100))
		immortalsDifferenceList[index].append(math.fabs(preds[0][5]*100 - rowy[0][5]*100))
		stalkersDifferenceList[index].append(math.fabs(preds[0][11]*100 - rowy[0][11]*100))
		zealotsDifferenceList[index].append(math.fabs(preds[0][14]*100 - rowy[0][14]*100))
		basesDifferenceList[index].append(math.fabs(preds[0][40]*10 - rowy[0][40]*10))
		
	for x in range(0, 35):
		armyCountDifferenceAvg.append(np.mean(armyCountDifferenceList[x]))
		probesDifferenceAvg.append(np.mean(probesDifferenceList[x]))	
		immortalsDifferenceAvg.append(np.mean(immortalsDifferenceList[x]))
		stalkersDifferenceAvg.append(np.mean(stalkersDifferenceList[x]))
		zealotsDifferenceAvg.append(np.mean(zealotsDifferenceList[x]))
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
	pyplot.plot(x, probesDifferenceAvg, color='red')
	pyplot.plot(x, basesDifferenceAvg, color='green')
	pyplot.plot(x, zealotsDifferenceAvg, color='black')
	pyplot.plot(x, stalkersDifferenceAvg, color='yellow')
	pyplot.plot(x, immortalsDifferenceAvg, color='purple')	
	pyplot.legend(['ArmyCount', 'Probes', 'Bases', 'Zealots', 'Stalkers', 'Immortals'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Essential Information (TvP)', **title_font)
	pyplot.show()

def averageMostImportantInOne():
	allUnitsDifferenceAverages = [[] * 1 for i in range(35)]
	allUnitsTotalAvg = []
	indices = [9, 5, 11, 14]
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

	for x in range(0, 37):
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

def averageProbesPercentage():
	probesPercentageDifferenceList = [[] * 1 for i in range(35)]
	probesPercentageDifferenceAvg = []
	probesBaselinePercentageDifferenceList = [[] * 1 for i in range(35)]
	probesBaselinePercentageDifferenceAvg = []
	probesBaseline2PercentageDifferenceList = [[] * 1 for i in range(35)]
	probesBaseline2PercentageDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if(not rowy[0][9] == 0):
			difference = math.fabs(preds[0][9]*100 - rowy[0][9]*100)
			actual = rowy[0][9]*100
			baselineDifference = math.fabs(rowx[0][42-37]*100 - rowy[0][9]*100)
			baseline2Difference = math.fabs(probesAverageBaseline[index] - rowy[0][9]*100)
		
			probesPercentageDifferenceList[index].append((difference/actual)*100)
			probesBaselinePercentageDifferenceList[index].append((baselineDifference/actual)*100)
			probesBaseline2PercentageDifferenceList[index].append((baseline2Difference/actual)*100)
	for x in range(0, 35):
		probesPercentageDifferenceAvg.append(np.mean(probesPercentageDifferenceList[x]))
		probesBaselinePercentageDifferenceAvg.append(np.mean(probesBaselinePercentageDifferenceList[x]))
		probesBaseline2PercentageDifferenceAvg.append(np.mean(probesBaseline2PercentageDifferenceList[x]))


	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	axes = pyplot.gca()
	axes.set_ylim([0,100])
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, probesPercentageDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, probesBaselinePercentageDifferenceAvg, color='xkcd:gold')
	pyplot.plot(x, probesBaseline2PercentageDifferenceAvg, color='green')
	pyplot.legend(['Probes Guess Percent Difference', 'Own Workers Baseline Percent Difference', 'Average Probes Baseline Percent Difference'], loc='upper left')
	pyplot.title('Average percentage worker difference TvP')
	pyplot.show()

def averageImmortalsPercentage():
	immortalsPercentageDifferenceList = [[] * 1 for i in range(35)]
	immortalsPercentageDifferenceAvg = []
	immortalsBaselinePercentageDifferenceList = [[] * 1 for i in range(35)]
	immortalsBaselinePercentageDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if(not rowy[0][5] == 0):
			difference = math.fabs(preds[0][5]*100 - rowy[0][5]*100)
			actual = rowy[0][5]*100
			baselineDifference = math.fabs(immortalsAverageBaseline[index] - rowy[0][5]*100)
		
			immortalsPercentageDifferenceList[index].append((difference/actual)*100)
			immortalsBaselinePercentageDifferenceList[index].append((baselineDifference/actual)*100)
	for x in range(0, 35):
		immortalsPercentageDifferenceAvg.append(np.mean(immortalsPercentageDifferenceList[x]))
		immortalsBaselinePercentageDifferenceAvg.append(np.mean(immortalsBaselinePercentageDifferenceList[x]))

	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, immortalsPercentageDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, immortalsBaselinePercentageDifferenceAvg, color='xkcd:gold')
	pyplot.legend(['Immortals Guess Percent Difference', 'Average Immortals Baseline Percent Difference'], loc='upper left')
	pyplot.title('Average percentage immortals difference TvP')
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
		basesGuessDifferenceList[index].append(math.fabs(preds[0][40]*10 - rowy[0][40]*10))
		basesBaselineDifferenceList[index].append(math.fabs(rowy[0][40]*10 - rowx[0][95]*10))
		basesBaseline2DifferenceList[index].append(math.fabs(rowy[0][40]*10 - basesAverageBaseline[index]))
		
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
	pyplot.title('Absolute Error in Prediction of Bases (TvP)', **title_font)
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
		workersGuessDifferenceList[index].append(math.fabs(preds[0][9]*100 - rowy[0][9]*100))
		workersBaselineDifferenceList[index].append(math.fabs(rowy[0][9]*100 - rowx[0][42-37]*100))
		workersBaseline2DifferenceList[index].append(math.fabs(rowy[0][9]*100 - probesAverageBaseline[index]))
		
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
	pyplot.legend(['Probes', 'Baseline (Own Units)', 'Baseline (Average for matchup)', 'Probes Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Probes (TvP)', **title_font)
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
		armyCountGuessDifferenceList[index].append(math.fabs(preds[0][37]*100 - rowy[0][37]*100))
		armyCountBaselineDifferenceList[index].append(math.fabs(rowy[0][37]*100 - rowx[0][92]*100))
		armyCountBaseline2DifferenceList[index].append(math.fabs(rowy[0][37]*100 - armyCountAverageBaseline[index]))
		
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
	pyplot.title('Absolute Error in Prediction of ArmyCount (TvP)', **title_font)
	pyplot.show()

averageSingleUnit(9, "Probes")
#averageSingleUnit(14, "Zealots")
#averageSingleUnit(11, "Stalkers")
#averageSingleUnit(5, "Immortals")
#averageSingleUnit(37, "ArmyCount")
#averageSingleBases()
#workersBaselineComparison()
#armyCountBaselineComparison()
#basesBaselineComparison()
#averageMostImportant()


