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

#Zero input
x_data_zero = copy.deepcopy(x_data)
#x_data_zero = np.delete(x_data_zero, [0, 1, 2, 3, 5, 11, 12, 18, 45, 140, 141, 142], 1)
x_data_zero = x_data_zero[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 140, 141, 142]]
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

model = load_model('Models/TvZ_OnlyEssentials.h5')

#Drones baseline
dronesAverageBaseline = []

if not os.path.isfile("TvZWorkerBaseline.npy"):
	dronesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		dronesAverageList[index].append(rowy[0][90-88]*100)
	for x in range(0, 35):
		dronesAverageBaseline.append(np.mean(dronesAverageList[x]))

	np.save('TvZWorkerBaseline.npy', dronesAverageBaseline);
else:
	dronesAverageBaseline = np.load('TvZWorkerBaseline.npy')

#Own SCVs baseline
ownSCVsAverageBaseline = []

if not os.path.isfile("TvZOwnWorkerBaseline.npy"):
	ownSCVsAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownSCVsAverageList[index].append(rowx[0][42-37]*100)
	for x in range(0, 35):
		ownSCVsAverageBaseline.append(np.mean(ownSCVsAverageList[x]))

	np.save('TvZOwnWorkerBaseline.npy', ownSCVsAverageBaseline);
else:
	ownSCVsAverageBaseline = np.load('TvZOwnWorkerBaseline.npy')

#Mutalisks baseline
mutalisksAverageBaseline = []

if not os.path.isfile("TvZMutaliskBaseline.npy"):
	mutaliskAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		mutaliskAverageList[index].append(rowy[0][94-88]*100)
	for x in range(0, 35):
		mutalisksAverageBaseline.append(np.mean(mutaliskAverageList[x]))

	np.save('TvZMutaliskBaseline.npy', mutalisksAverageBaseline);
else:
	mutalisksAverageBaseline = np.load('TvZMutaliskBaseline.npy')

#Bases baseline
basesAverageBaseline = []

if not os.path.isfile("TvZBasesBaseline.npy"):
	basesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		basesAverageList[index].append(rowy[0][87]*10)
	for x in range(0, 35):
		basesAverageBaseline.append(np.mean(basesAverageList[x]))

	np.save('TvZBasesBaseline.npy', basesAverageBaseline);
else:
	basesAverageBaseline = np.load('TvZBasesBaseline.npy')

#Own bases baseline
ownBasesAverageBaseline = []

if not os.path.isfile("TvZOwnBasesBaseline.npy"):
	ownBasesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownBasesAverageList[index].append(rowx[0][142]*10)
	for x in range(0, 35):
		ownBasesAverageBaseline.append(np.mean(ownBasesAverageList[x]))

	np.save('TvZOwnBasesBaseline.npy', ownBasesAverageBaseline);
else:
	ownBasesAverageBaseline = np.load('TvZOwnBasesBaseline.npy')

#ArmyCount baseline
armyCountAverageBaseline = []

if not os.path.isfile("TvZArmyCountBaseline.npy"):
	armyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		armyCountAverageList[index].append(rowy[0][84]*100)
	for x in range(0, 35):
		armyCountAverageBaseline.append(np.mean(armyCountAverageList[x]))

	np.save('TvZArmyCountBaseline.npy', armyCountAverageBaseline);
else:
	armyCountAverageBaseline = np.load('TvZArmyCountBaseline.npy')

#Own armyCount baseline
ownArmyCountAverageBaseline = []

if not os.path.isfile("TvZOwnArmyCountBaseline.npy"):
	ownArmyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		ownArmyCountAverageList[index].append(rowx[0][139]*100)
	for x in range(0, 35):
		ownArmyCountAverageBaseline.append(np.mean(ownArmyCountAverageList[x]))

	np.save('TvZOwnArmyCountBaseline.npy', ownArmyCountAverageBaseline);
else:
	ownArmyCountAverageBaseline = np.load('TvZOwnArmyCountBaseline.npy')


	

def oneGuess():

	index = np.random.randint(0, len(x_val))
	rowx, rowy = x_val[np.array([index])], y_val[np.array([index])]
	preds = model.predict(rowx, verbose=0)

	print()

	print('Time: ' , int(round(rowx[0][0]*25000/24)), 'seconds')

	print()

	print('Own units and bases:')

	basesOwn = int(round(rowx[0][142]*10))

	armyCountOwn = int(round(rowx[0][139]*100))

	SCVsOwn = int(round(rowx[0][42-37]*100))
	marinesOwn = int(round(rowx[0][49-37]*100))
	maraudersOwn = int(round(rowx[0][48-37]*100))
	tanksOwn = int(round(rowx[0][82-37]*100)) + int(round(rowx[0][55-37]*100))

	gameMap = int(round(rowx[0][1]*6))
	selfRace = int(round(rowx[0][2]*2))
	enemyRace = int(round(rowx[0][3]*2))
	minerals = int(round(rowx[0][140]*5000))
	gas = int(round(rowx[0][141]*5000))
	baseXCoord = int(round(rowx[0][143]*200))
	baseYCoord = int(round(rowx[0][144]*200))

	print('Bases: ', basesOwn)

	print('ArmyCount: ', armyCountOwn)

	print('SCVs: ', SCVsOwn)
	print('Marines: ', marinesOwn)
	print('Marauders: ', maraudersOwn)
	print('Tanks: ', tanksOwn)

	print('Map: ', gameMap)
	print('Own Race: ', selfRace)
	print('Enemy Race: ', enemyRace)
	print('Minerals: ', minerals)
	print('Gas: ', gas)
	print('Base X Coord: ', baseXCoord)
	print('Base Y Coord: ', baseYCoord)

	print()

	print('Enemy units and bases:')

	basesGuess = int(round(preds[0][87]*10))
	basesCorrect = int(round(rowy[0][87]*10))

	print('BasesGuess: ', basesGuess)
	print('BasesCorrect: ', basesCorrect)

	armyCountGuess = int(round(preds[0][84]*100))
	armyCountCorrect = int(round(rowy[0][84]*100))

	print('ArmyCountGuess: ', armyCountGuess)
	print('ArmyCountCorrect: ', armyCountCorrect)

	mineralsGuess = int(round(preds[0][85]*5000))
	mineralsCorrect = int(round(rowy[0][85]*5000))

	print('MineralsGuess: ', mineralsGuess)
	print('MineralsCorrect: ', mineralsCorrect)

	gasGuess = int(round(preds[0][86]*5000))
	gasCorrect = int(round(rowy[0][86]*5000))

	print('GasGuess: ', gasGuess)
	print('GasCorrect: ', gasCorrect)

	baseXCoordGuess = int(round(preds[0][88]*200))
	baseXCoordCorrect = int(round(rowy[0][88]*200))

	print('Base X Coord Guess: ', baseXCoordGuess)
	print('Base X Coord Correct: ', baseXCoordCorrect)

	baseYCoordGuess = int(round(preds[0][89]*200))
	baseYCoordCorrect = int(round(rowy[0][89]*200))

	print('Base Y Coord Guess: ', baseYCoordGuess)
	print('Base Y Coord Correct: ', baseYCoordCorrect)

	print()

	print('Zerg:')

	dronesGuess = int(round(preds[0][90-88]*100))
	dronesCorrect = int(round(rowy[0][90-88]*100))
	zerglingsGuess = int(round(preds[0][100-88]*100))
	zerglingsCorrect = int(round(rowy[0][100-88]*100))
	roachesGuess = int(round(preds[0][98-88]*100))
	roachesCorrect = int(round(rowy[0][98-88]*100))
	mutalisksGuess = int(round(preds[0][94-88]*100))
	mutalisksCorrect = int(round(rowy[0][94-88]*100))


	print('DronesGuess: ', dronesGuess)
	print('DronesCorrect: ', dronesCorrect)
	print('ZerglingsGuess: ', zerglingsGuess)
	print('ZerglingsCorrect: ', zerglingsCorrect)
	print('RoachesGuess: ', roachesGuess)
	print('RoachesCorrect: ', roachesCorrect)
	print('MutalisksGuess: ', mutalisksGuess)
	print('MutalisksCorrect: ', mutalisksCorrect)

	print("")
	print("")

def averageDronesPercentage():
	dronesPercentageDifferenceList = [[] * 1 for i in range(35)]
	dronesPercentageDifferenceAvg = []
	dronesBaselinePercentageDifferenceList = [[] * 1 for i in range(35)]
	dronesBaselinePercentageDifferenceAvg = []
	dronesBaseline2PercentageDifferenceList = [[] * 1 for i in range(35)]
	dronesBaseline2PercentageDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if(not rowy[0][90-88] == 0):
			difference = math.fabs(preds[0][90-88]*100 - rowy[0][90-88]*100)
			actual = rowy[0][90-88]*100
			baselineDifference = math.fabs(rowx[0][42-37]*100 - rowy[0][90-88]*100)
			baseline2Difference = math.fabs(dronesAverageBaseline[index] - rowy[0][90-88]*100)
		
			dronesPercentageDifferenceList[index].append((difference/actual)*100)
			dronesBaselinePercentageDifferenceList[index].append((baselineDifference/actual)*100)
			dronesBaseline2PercentageDifferenceList[index].append((baseline2Difference/actual)*100)
	for x in range(0, 35):
		dronesPercentageDifferenceAvg.append(np.mean(dronesPercentageDifferenceList[x]))
		dronesBaselinePercentageDifferenceAvg.append(np.mean(dronesBaselinePercentageDifferenceList[x]))
		dronesBaseline2PercentageDifferenceAvg.append(np.mean(dronesBaseline2PercentageDifferenceList[x]))

	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, dronesPercentageDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, dronesBaselinePercentageDifferenceAvg, color='xkcd:gold')
	pyplot.plot(x, dronesBaseline2PercentageDifferenceAvg, color='green')
	pyplot.legend(['Drones Guess Percent Difference', 'Own Workers Baseline Percent Difference', 'Average Drones Baseline Percent Difference'], loc='upper left')
	pyplot.title('Average percentage worker difference TvZ')
	pyplot.show()

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
	pyplot.title('Absolute Error, Prediction and Actual Value of ' + name + ' (TvZ)', **title_font)
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
		basesDifferenceList[index].append(math.fabs(preds[0][87]*10 - rowy[0][87]*10))
		basesActualList[index].append(rowy[0][87]*10)
		basesGuessList[index].append(preds[0][87]*10)
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
	pyplot.title('Absolute Error, Prediction and Actual Value of Bases (TvZ)', **title_font)
	pyplot.show()

def averageMutalisks():
	mutalisksDifferenceList = [[] * 1 for i in range(35)]
	mutalisksActualList = [[] * 1 for i in range(35)]
	mutalisksGuessList = [[] * 1 for i in range(35)]
	mutalisksDifferenceAvg = []
	mutalisksActualAvg = []
	mutalisksGuessAvg = []

	counter = 0
	flag = 0

	for x in range(np.random.randint(0, len(x_val)), len(x_val)):
		counter = counter + 1
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if index == 0 and counter > 5 and flag == 1:
			break
		if index == 0:
			counter = 0
			flag = 1
		if flag == 1:
			mutalisksDifferenceList[index].append(math.fabs(preds[0][94-88]*100 - rowy[0][94-88]*100))
			mutalisksActualList[index].append(rowy[0][94-88]*100)
			mutalisksGuessList[index].append(preds[0][94-88]*100)
	for x in range(0, 35):
		mutalisksDifferenceAvg.append(np.mean(mutalisksDifferenceList[x]))
		mutalisksActualAvg.append(np.mean(mutalisksActualList[x]))
		mutalisksGuessAvg.append(np.mean(mutalisksGuessList[x]))
	
	x = np.arange(0, 35*30, 30)
	pyplot.plot(x, mutalisksDifferenceAvg)
	pyplot.plot(x, mutalisksActualAvg)
	pyplot.plot(x, mutalisksGuessAvg)
	pyplot.legend(['MutalisksDifference', 'MutalisksActual', 'MutalisksGuess'], loc='upper left')
	pyplot.show()

def averageMutalisksPercentage():
	mutalisksPercentageDifferenceList = [[] * 1 for i in range(35)]
	mutalisksPercentageDifferenceAvg = []
	mutalisksBaselinePercentageDifferenceList = [[] * 1 for i in range(35)]
	mutalisksBaselinePercentageDifferenceAvg = []

	for x in range(0, len(x_val)):
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		if(not rowy[0][94-88] == 0):
			difference = math.fabs(preds[0][94-88]*100 - rowy[0][94-88]*100)
			actual = rowy[0][94-88]*100
			baselineDifference = math.fabs(mutalisksAverageBaseline[index] - rowy[0][94-88]*100)
		
			mutalisksPercentageDifferenceList[index].append((difference/actual)*100)
			mutalisksBaselinePercentageDifferenceList[index].append((baselineDifference/actual)*100)

	for x in range(0, 35):
		mutalisksPercentageDifferenceAvg.append(np.mean(mutalisksPercentageDifferenceList[x]))
		mutalisksBaselinePercentageDifferenceAvg.append(np.mean(mutalisksBaselinePercentageDifferenceList[x]))
	
	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	axes = pyplot.gca()
	axes.set_ylim([0,120])
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, mutalisksPercentageDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, mutalisksBaselinePercentageDifferenceAvg, color='xkcd:gold')
	pyplot.legend(['Mutalisks Guess Percent Difference', 'Average Mutalisks Baseline Percent Difference'], loc='upper left')
	pyplot.title('Average percentage mutalisk difference TvZ')
	pyplot.show()

def averageMostImportant():
	mutalisksDifferenceList = [[] * 1 for i in range(35)]
	mutalisksDifferenceAvg = []
	roachesDifferenceList = [[] * 1 for i in range(35)]
	roachesDifferenceAvg = []
	zerglingsDifferenceList = [[] * 1 for i in range(35)]
	zerglingsDifferenceAvg = []
	dronesDifferenceList = [[] * 1 for i in range(35)]
	dronesDifferenceAvg = []
	armyCountDifferenceList = [[] * 1 for i in range(35)]
	armyCountDifferenceAvg = []
	basesDifferenceList = [[] * 1 for i in range(35)]
	basesDifferenceAvg = []

	mutalisksStandardDeviations = []
	roachesStandardDeviations = []
	zerglingsStandardDeviations = []
	dronesStandardDeviations = []
	armyCountStandardDeviations = []
	basesStandardDeviations = []

	for x in range(0, len(x_val)):
		rowx_zero = x_val_zero[np.array([x])]
		rowx, rowy = x_val[np.array([x])], y_val[np.array([x])]
		preds = model.predict(rowx_zero, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		armyCountDifferenceList[index].append(math.fabs(preds[0][84]*100 - rowy[0][84]*100))
		dronesDifferenceList[index].append(math.fabs(preds[0][90-88]*100 - rowy[0][90-88]*100))
		zerglingsDifferenceList[index].append(math.fabs(preds[0][100-88]*100 - rowy[0][100-88]*100))
		roachesDifferenceList[index].append(math.fabs(preds[0][98-88]*100 - rowy[0][98-88]*100))
		mutalisksDifferenceList[index].append(math.fabs(preds[0][94-88]*100 - rowy[0][94-88]*100))
		basesDifferenceList[index].append(math.fabs(preds[0][87]*10 - rowy[0][87]*10))
		
	for x in range(0, 35):
		armyCountDifferenceAvg.append(np.mean(armyCountDifferenceList[x]))
		dronesDifferenceAvg.append(np.mean(dronesDifferenceList[x]))	
		zerglingsDifferenceAvg.append(np.mean(zerglingsDifferenceList[x]))
		roachesDifferenceAvg.append(np.mean(roachesDifferenceList[x]))
		mutalisksDifferenceAvg.append(np.mean(mutalisksDifferenceList[x]))
		basesDifferenceAvg.append(np.mean(basesDifferenceList[x]))

	for x in range(0, 35):
		mutalisksStandardDeviations.append(np.std(mutalisksDifferenceList[x]))
		zerglingsStandardDeviations.append(np.std(zerglingsDifferenceList[x]))
		roachesStandardDeviations.append(np.std(roachesDifferenceList[x]))
		dronesStandardDeviations.append(np.std(dronesDifferenceList[x]))
		armyCountStandardDeviations.append(np.std(armyCountDifferenceList[x]))
		basesStandardDeviations.append(np.std(basesDifferenceList[x]))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(mutalisksDifferenceAvg[x]+mutalisksStandardDeviations[x])
	    c1.append(mutalisksDifferenceAvg[x]-mutalisksStandardDeviations[x])

	b2 = []
	c2 = []
	for x in range(0, 35):
	    b2.append(zerglingsDifferenceAvg[x]+zerglingsStandardDeviations[x])
	    c2.append(zerglingsDifferenceAvg[x]-zerglingsStandardDeviations[x])

	b3 = []
	c3 = []
	for x in range(0, 35):
	    b3.append(roachesDifferenceAvg[x]+roachesStandardDeviations[x])
	    c3.append(roachesDifferenceAvg[x]-roachesStandardDeviations[x])

	b4 = []
	c4 = []
	for x in range(0, 35):
	    b4.append(dronesDifferenceAvg[x]+dronesStandardDeviations[x])
	    c4.append(dronesDifferenceAvg[x]-dronesStandardDeviations[x])

	b5 = []
	c5 = []
	for x in range(0, 35):
	    b5.append(armyCountDifferenceAvg[x]+armyCountStandardDeviations[x])
	    c5.append(armyCountDifferenceAvg[x]-armyCountStandardDeviations[x])

	b6 = []
	c6 = []
	for x in range(0, 35):
	    b6.append(basesDifferenceAvg[x]+basesStandardDeviations[x])
	    c6.append(basesDifferenceAvg[x]-basesStandardDeviations[x])
	
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
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='purple', alpha=0.1)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c2, b2, color='black', alpha=0.1)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c3, b3, color='yellow', alpha=0.1)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c4, b4, color='red', alpha=0.1)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c5, b5, color='blue', alpha=0.1)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c6, b6, color='green', alpha=0.1)
	pyplot.plot(x, armyCountDifferenceAvg, color='blue')
	pyplot.plot(x, dronesDifferenceAvg, color='red')
	pyplot.plot(x, basesDifferenceAvg, color='green')
	pyplot.plot(x, zerglingsDifferenceAvg, color='black')
	pyplot.plot(x, roachesDifferenceAvg, color='yellow')
	pyplot.plot(x, mutalisksDifferenceAvg, color='purple')	
	pyplot.legend(['ArmyCount', 'Drones', 'Bases', 'Zerglings', 'Roaches', 'Mutalisks'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Essential Information (TvZ)', **title_font)
	pyplot.show()

def averageAll():
	allUnitsDifferenceAverages = [[] * 1 for i in range(35)]
	allUnitsTotalAvg = []
	counter = 0

	allUnitsSquaredDifferences = [[] * 1 for i in range(35)]
	allUnitsStandardDeviations = []

	for x in range(0, 84):
		print("Counter:", counter)
		currentUnitDifferenceList = [[] * 1 for i in range(35)]		

		for y in range(0, len(x_val)):
			rowx_zero = x_val_zero[np.array([x])]		
			rowx, rowy = x_val[np.array([y])], y_val[np.array([y])]
			preds = model.predict(rowx_zero, verbose=0)
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

def averageMostImportantInOne():
	allUnitsDifferenceAverages = [[] * 1 for i in range(35)]
	allUnitsTotalAvg = []
	indices = [2, 12, 10, 6]
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
		workersGuessDifferenceList[index].append(math.fabs(preds[0][90-88]*100 - rowy[0][90-88]*100))
		workersBaselineDifferenceList[index].append(math.fabs(rowy[0][90-88]*100 - rowx[0][42-37]*100))
		workersBaseline2DifferenceList[index].append(math.fabs(rowy[0][90-88]*100 - dronesAverageBaseline[index]))
		
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
	pyplot.legend(['Drones', 'Baseline (Own Units)', 'Baseline (Average for matchup)', 'Drones Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Drones (TvZ)', **title_font)
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
		basesGuessDifferenceList[index].append(math.fabs(preds[0][87]*10 - rowy[0][87]*10))
		basesBaselineDifferenceList[index].append(math.fabs(rowy[0][87]*10 - rowx[0][142]*10))
		basesBaseline2DifferenceList[index].append(math.fabs(rowy[0][87]*10 - basesAverageBaseline[index]))
		
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
	pyplot.legend(['Bases', 'Baseline (Own Bases)', 'Baseline (Average for matchup)', 'Bases Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Bases (TvZ)', **title_font)
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
		armyCountGuessDifferenceList[index].append(math.fabs(preds[0][84]*100 - rowy[0][84]*100))
		armyCountBaselineDifferenceList[index].append(math.fabs(rowy[0][84]*100 - rowx[0][139]*100))		
		armyCountBaseline2DifferenceList[index].append(math.fabs(rowy[0][84]*100 - armyCountAverageBaseline[index]))
		
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
	pyplot.title('Absolute Error in Prediction of ArmyCount (TvZ)', **title_font)
	pyplot.show()


#oneGuess()

#averageSingleUnit(2, "Drones")
#averageSingleUnit(100-88, "Zerglings")
#averageSingleUnit(98-88, "Roaches")
#averageSingleUnit(94-88, "Mutalisks")
#averageSingleUnit(84, "ArmyCount")
#averageSingleBases()
#workersBaselineComparison()
#armyCountBaselineComparison()
#basesBaselineComparison()
averageMostImportant()


#score = model.evaluate(x_val, y_val, batch_size=16)
#print('Score: ')
#print(score)
