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

#Load data TvT
xFileTvT = "Data/SC2TvTImperfectClean"
yFileTvT = "Data/SC2TvTPerfectClean"

if not os.path.isfile(xFileTvT + ".npy"):
    x_data_TvT = np.loadtxt(xFileTvT + '.csv', delimiter=',', dtype=np.float32)
    np.save(xFileTvT + '.npy', x_data_TvT);
else:
    x_data_TvT = np.load(xFileTvT + '.npy')

if not os.path.isfile(yFileTvT + ".npy"):
    y_data_TvT = np.loadtxt(yFileTvT + '.csv', delimiter=',', dtype=np.float32)
    np.save(yFileTvT + '.npy', y_data_TvT);
else:
    y_data_TvT = np.load(yFileTvT + '.npy')

#Zero input TvT
x_data_TvT_zero = copy.deepcopy(x_data_TvT)
#x_data_TvT_zero = np.delete(x_data_TvT_zero, [0, 1, 2, 3, 5, 11, 12, 18, 45, 107, 108, 109], 1)
x_data_TvT_zero = x_data_TvT_zero[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 107, 108, 109]]
#for x in range(0, len(x_data_TvT_zero)):
#	for y in range(1, 5):
#		x_data_TvT_zero[x][y] = 0
#	for y in range(6, len(x_data_TvT_zero[x])):
#		x_data_TvT_zero[x][y] = 0

#print("After split: ", x_data_TvT_zero.shape)

split_at_TvT_zero = len(x_data_TvT_zero) - len(x_data_TvT_zero) // 5

(x_train_TvT_zero, x_val_TvT_zero) = x_data_TvT_zero[:split_at_TvT_zero], x_data_TvT_zero[split_at_TvT_zero:]


split_at_TvT = len(x_data_TvT) - len(x_data_TvT) // 5

(x_train_TvT, x_val_TvT) = x_data_TvT[:split_at_TvT], x_data_TvT[split_at_TvT:]

(y_train_TvT, y_val_TvT) = y_data_TvT[:split_at_TvT], y_data_TvT[split_at_TvT:]

model_TvT = load_model('Models/TvT_OnlyEssentials.h5')

#Load data TvZ
xFileTvZ = "Data/SC2TvZImperfectClean"
yFileTvZ = "Data/SC2TvZPerfectClean"

if not os.path.isfile(xFileTvZ + ".npy"):
    x_data_TvZ = np.loadtxt(xFileTvZ + '.csv', delimiter=',', dtype=np.float32)
    np.save(xFileTvZ + '.npy', x_data_TvZ);
else:
    x_data_TvZ = np.load(xFileTvZ + '.npy')

if not os.path.isfile(yFileTvZ + ".npy"):
    y_data_TvZ = np.loadtxt(yFileTvZ + '.csv', delimiter=',', dtype=np.float32)
    np.save(yFileTvZ + '.npy', y_data_TvZ);
else:
    y_data_TvZ = np.load(yFileTvZ + '.npy')

#Zero input TvZ
x_data_TvZ_zero = copy.deepcopy(x_data_TvZ)
#x_data_TvZ_zero = np.delete(x_data_TvZ_zero, [0, 1, 2, 3, 5, 11, 12, 18, 45, 140, 141, 142], 1)
x_data_TvZ_zero = x_data_TvZ_zero[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 140, 141, 142]]
#for x in range(0, len(x_data_TvZ_zero)):
#	for y in range(1, 5):
#		x_data_TvZ_zero[x][y] = 0
#	for y in range(6, len(x_data_TvZ_zero[x])):
#		x_data_TvZ_zero[x][y] = 0

#print("After split: ", x_data_TvZ_zero.shape)


split_at_TvZ_zero = len(x_data_TvZ_zero) - len(x_data_TvZ_zero) // 5

(x_train_TvZ_zero, x_val_TvZ_zero) = x_data_TvZ_zero[:split_at_TvZ_zero], x_data_TvZ_zero[split_at_TvZ_zero:]


split_at_TvZ = len(x_data_TvZ) - len(x_data_TvZ) // 5

(x_train_TvZ, x_val_TvZ) = x_data_TvZ[:split_at_TvZ], x_data_TvZ[split_at_TvZ:]

(y_train_TvZ, y_val_TvZ) = y_data_TvZ[:split_at_TvZ], y_data_TvZ[split_at_TvZ:]

model_TvZ = load_model('Models/TvZ_OnlyEssentials.h5')

#Load data TvP
xFileTvP = "Data/SC2TvPImperfectClean"
yFileTvP = "Data/SC2TvPPerfectClean"

if not os.path.isfile(xFileTvP + ".npy"):
    x_data_TvP = np.loadtxt(xFileTvP + '.csv', delimiter=',', dtype=np.float32)
    np.save(xFileTvP + '.npy', x_data_TvP);
else:
    x_data_TvP = np.load(xFileTvP + '.npy')

if not os.path.isfile(yFileTvP + ".npy"):
    y_data_TvP = np.loadtxt(yFileTvP + '.csv', delimiter=',', dtype=np.float32)
    np.save(yFileTvP + '.npy', y_data_TvP);
else:
    y_data_TvP = np.load(yFileTvP + '.npy')

#Zero input TvP
x_data_TvP_zero = copy.deepcopy(x_data_TvP)
#x_data_TvP_zero = np.delete(x_data_TvP_zero, [0, 1, 2, 3, 5, 11, 12, 18, 45, 93, 94, 95], 1)
x_data_TvP_zero = x_data_TvP_zero[:, [0, 1, 2, 3, 5, 11, 12, 18, 45, 93, 94, 95]]
#for x in range(0, len(x_data_TvP_zero)):
#	for y in range(1, 5):
#		x_data_TvP_zero[x][y] = 0
#	for y in range(6, len(x_data_TvP_zero[x])):
#		x_data_TvP_zero[x][y] = 0

#print("After split: ", x_data_TvP_zero.shape)


split_at_TvP_zero = len(x_data_TvP_zero) - len(x_data_TvP_zero) // 5

(x_train_TvP_zero, x_val_TvP_zero) = x_data_TvP_zero[:split_at_TvP_zero], x_data_TvP_zero[split_at_TvP_zero:]


split_at_TvP = len(x_data_TvP) - len(x_data_TvP) // 5

(x_train_TvP, x_val_TvP) = x_data_TvP[:split_at_TvP], x_data_TvP[split_at_TvP:]

(y_train_TvP, y_val_TvP) = y_data_TvP[:split_at_TvP], y_data_TvP[split_at_TvP:]

model_TvP = load_model('Models/TvP_OnlyEssentials.h5')

#Drones baseline
dronesAverageBaseline = []

if not os.path.isfile("TvZWorkerBaseline.npy"):
	dronesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		dronesAverageList[index].append(rowy[0][90-88]*100)
	for x in range(0, 35):
		dronesAverageBaseline.append(np.mean(dronesAverageList[x]))

	np.save('TvZWorkerBaseline.npy', dronesAverageBaseline);
else:
	dronesAverageBaseline = np.load('TvZWorkerBaseline.npy')

#SCVs baseline
SCVsAverageBaseline = []

if not os.path.isfile("TvTWorkerBaseline.npy"):
	SCVsAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		SCVsAverageList[index].append(rowy[0][1]*100)
	for x in range(0, 35):
		SCVsAverageBaseline.append(np.mean(SCVsAverageList[x]))

	np.save('TvTWorkerBaseline.npy', SCVsAverageBaseline);
else:
	SCVsAverageBaseline = np.load('TvTWorkerBaseline.npy')

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

#TvZ bases baseline
TvZBasesAverageBaseline = []

if not os.path.isfile("TvZBasesBaseline.npy"):
	TvZBasesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		TvZBasesAverageList[index].append(rowy[0][87]*10)
	for x in range(0, 35):
		TvZBasesAverageBaseline.append(np.mean(TvZBasesAverageList[x]))

	np.save('TvZBasesBaseline.npy', TvZBasesAverageBaseline);
else:
	TvZBasesAverageBaseline = np.load('TvZBasesBaseline.npy')

#TvT Bases baseline
TvTBasesAverageBaseline = []

if not os.path.isfile("TvTBasesBaseline.npy"):
	TvTBasesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		TvTBasesAverageList[index].append(rowy[0][54]*10)
	for x in range(0, 35):
		TvTBasesAverageBaseline.append(np.mean(TvTBasesAverageList[x]))

	np.save('TvTBasesBaseline.npy', TvTBasesAverageBaseline);
else:
	TvTBasesAverageBaseline = np.load('TvTBasesBaseline.npy')

#TvP Bases baseline
TvPBasesAverageBaseline = []

if not os.path.isfile("TvPBasesBaseline.npy"):
	TvPBasesAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		TvPBasesAverageList[index].append(rowy[0][40]*10)
	for x in range(0, 35):
		TvPBasesAverageBaseline.append(np.mean(TvPBasesAverageList[x]))

	np.save('TvPBasesBaseline.npy', TvPBasesAverageBaseline);
else:
	TvPBasesAverageBaseline = np.load('TvPBasesBaseline.npy')

#TvZ ArmyCount baseline
TvZArmyCountAverageBaseline = []

if not os.path.isfile("TvZArmyCountBaseline.npy"):
	TvZArmyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		TvZArmyCountAverageList[index].append(rowy[0][84]*100)
	for x in range(0, 35):
		TvZArmyCountAverageBaseline.append(np.mean(TvZArmyCountAverageList[x]))

	np.save('TvZArmyCountBaseline.npy', TvZArmyCountAverageBaseline);
else:
	TvZArmyCountAverageBaseline = np.load('TvZArmyCountBaseline.npy')

#TvT ArmyCount baseline
TvTArmyCountAverageBaseline = []

if not os.path.isfile("TvTArmyCountBaseline.npy"):
	TvTArmyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		TvTArmyCountAverageList[index].append(rowy[0][51]*100)
	for x in range(0, 35):
		TvTArmyCountAverageBaseline.append(np.mean(TvTArmyCountAverageList[x]))

	np.save('TvTArmyCountBaseline.npy', TvTArmyCountAverageBaseline);
else:
	TvTArmyCountAverageBaseline = np.load('TvTArmyCountBaseline.npy')

#TvP ArmyCount baseline
TvPArmyCountAverageBaseline = []

if not os.path.isfile("TvPArmyCountBaseline.npy"):
	TvPArmyCountAverageList = [[] * 1 for i in range(35)]		

	for x in range(0, len(x_train)):
		rowx, rowy = x_train[np.array([x])], y_train[np.array([x])]
		preds = model.predict(rowx, verbose=0)
		index = int(int(round(rowx[0][0]*25000/24))/30)
		
		TvPArmyCountAverageList[index].append(rowy[0][37]*100)
	for x in range(0, 35):
		TvPArmyCountAverageBaseline.append(np.mean(TvPArmyCountAverageList[x]))

	np.save('TvPArmyCountBaseline.npy', TvPArmyCountAverageBaseline);
else:
	TvPArmyCountAverageBaseline = np.load('TvPArmyCountBaseline.npy')


def workersBaselineComparisonAllTogether():
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

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		workersGuessDifferenceList[index_TvZ].append(math.fabs(preds_TvZ[0][90-88]*100 - rowy_TvZ[0][90-88]*100))
		workersBaselineDifferenceList[index_TvZ].append(math.fabs(rowy_TvZ[0][90-88]*100 - rowx_TvZ[0][42-37]*100))
		workersBaseline2DifferenceList[index_TvZ].append(math.fabs(rowy_TvZ[0][90-88]*100 - dronesAverageBaseline[index_TvZ]))

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		workersGuessDifferenceList[index_TvT].append(math.fabs(preds_TvT[0][1]*100 - rowy_TvT[0][1]*100))
		workersBaselineDifferenceList[index_TvT].append(math.fabs(rowy_TvT[0][1]*100 - rowx_TvT[0][42-37]*100))
		workersBaseline2DifferenceList[index_TvT].append(math.fabs(rowy_TvT[0][1]*100 - SCVsAverageBaseline[index_TvT]))

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		workersGuessDifferenceList[index_TvP].append(math.fabs(preds_TvP[0][9]*100 - rowy_TvP[0][9]*100))
		workersBaselineDifferenceList[index_TvP].append(math.fabs(rowy_TvP[0][9]*100 - rowx_TvP[0][42-37]*100))
		workersBaseline2DifferenceList[index_TvP].append(math.fabs(rowy_TvP[0][9]*100 - probesAverageBaseline[index_TvP]))
		
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
	pyplot.legend(['Workers', 'Baseline (Own Workers)', 'Baseline (Average for all matchups)', 'Workers Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Workers (All Terran Matchups)', **title_font)
	pyplot.show()

def workersBaselinePercentageComparisonAllTogether():
	workersGuessDifferenceList = [[] * 1 for i in range(35)]
	workersGuessDifferenceAvg = []
	workersBaselineDifferenceList = [[] * 1 for i in range(35)]
	workersBaselineDifferenceAvg = []
	workersBaseline2DifferenceList = [[] * 1 for i in range(35)]
	workersBaseline2DifferenceAvg = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)

		if(rowy_TvZ[0][90-88] == 0):
			rowy_TvZ[0][90-88] = 1
		difference_TvZ = math.fabs(preds_TvZ[0][90-88]*100 - rowy_TvZ[0][90-88]*100)
		actual_TvZ = rowy_TvZ[0][90-88]*100
		baselineDifference_TvZ = math.fabs(rowx_TvZ[0][42-37]*100 - rowy_TvZ[0][90-88]*100)
		baseline2Difference_TvZ = math.fabs(dronesAverageBaseline[index_TvZ] - rowy_TvZ[0][90-88]*100)

		workersGuessDifferenceList[index_TvZ].append((difference_TvZ/actual_TvZ)*100)
		workersBaselineDifferenceList[index_TvZ].append((baselineDifference_TvZ/actual_TvZ)*100)
		workersBaseline2DifferenceList[index_TvZ].append((baseline2Difference_TvZ/actual_TvZ)*100)

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)

		if(rowy_TvT[0][1] == 0):
			rowy_TvT[0][1] = 1
		difference_TvT = math.fabs(preds_TvT[0][1]*100 - rowy_TvT[0][1]*100)
		actual_TvT = rowy_TvT[0][1]*100
		baselineDifference_TvT = math.fabs(rowx_TvT[0][42-37]*100 - rowy_TvT[0][1]*100)
		baseline2Difference_TvT = math.fabs(SCVsAverageBaseline[index_TvT] - rowy_TvT[0][1]*100)

		workersGuessDifferenceList[index_TvT].append((difference_TvT/actual_TvT)*100)
		workersBaselineDifferenceList[index_TvT].append((baselineDifference_TvT/actual_TvT)*100)
		workersBaseline2DifferenceList[index_TvT].append((baseline2Difference_TvT/actual_TvT)*100)

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		
		if(rowy_TvP[0][9] == 0):
			rowy_TvP[0][9] = 1
		difference_TvP = math.fabs(preds_TvP[0][9]*100 - rowy_TvP[0][9]*100)
		actual_TvP = rowy_TvP[0][9]*100
		baselineDifference_TvP = math.fabs(rowx_TvP[0][42-37]*100 - rowy_TvP[0][9]*100)
		baseline2Difference_TvP = math.fabs(probesAverageBaseline[index_TvP] - rowy_TvP[0][9]*100)

		workersGuessDifferenceList[index_TvP].append((difference_TvP/actual_TvP)*100)
		workersBaselineDifferenceList[index_TvP].append((baselineDifference_TvP/actual_TvP)*100)
		workersBaseline2DifferenceList[index_TvP].append((baseline2Difference_TvP/actual_TvP)*100)
		
	for x in range(0, 35):
		workersGuessDifferenceAvg.append(np.mean(workersGuessDifferenceList[x]))
		workersBaselineDifferenceAvg.append(np.mean(workersBaselineDifferenceList[x]))
		workersBaseline2DifferenceAvg.append(np.mean(workersBaseline2DifferenceList[x]))
	
	fig, axes = pyplot.subplots(num=None, figsize = (8, 6))
	x = np.arange(0, 35*30, 30)
	pyplot.xlabel('Seconds')
	pyplot.ylabel('Percent')
	pyplot.grid(True)
	pyplot.plot(x, workersGuessDifferenceAvg, color='xkcd:blue with a hint of purple')
	pyplot.plot(x, workersBaselineDifferenceAvg, color='xkcd:gold')
	pyplot.plot(x, workersBaseline2DifferenceAvg, color='green')
	pyplot.legend(['WorkersGuess', 'OwnWorkersBaseline', 'AverageWorkersBaseline'], loc='upper left')
	pyplot.title('Average worker difference for all terran matchups')
	pyplot.show()

def basesBaselineComparisonAllTogether():
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

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		basesGuessDifferenceList[index_TvZ].append(math.fabs(preds_TvZ[0][87]*10 - rowy_TvZ[0][87]*10))
		basesBaselineDifferenceList[index_TvZ].append(math.fabs(rowy_TvZ[0][87]*10 - rowx_TvZ[0][142]*10))
		basesBaseline2DifferenceList[index_TvZ].append(math.fabs(rowy_TvZ[0][87]*10 - TvZBasesAverageBaseline[index_TvZ]))

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		basesGuessDifferenceList[index_TvT].append(math.fabs(preds_TvT[0][54]*10 - rowy_TvT[0][54]*10))
		basesBaselineDifferenceList[index_TvT].append(math.fabs(rowy_TvT[0][54]*10 - rowx_TvT[0][109]*10))
		basesBaseline2DifferenceList[index_TvT].append(math.fabs(rowy_TvT[0][54]*10 - TvTBasesAverageBaseline[index_TvT]))

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		basesGuessDifferenceList[index_TvP].append(math.fabs(preds_TvP[0][40]*10 - rowy_TvP[0][40]*10))
		basesBaselineDifferenceList[index_TvP].append(math.fabs(rowy_TvP[0][40]*10 - rowx_TvP[0][95]*10))
		basesBaseline2DifferenceList[index_TvP].append(math.fabs(rowy_TvP[0][40]*10 - TvPBasesAverageBaseline[index_TvP]))
		
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
	pyplot.legend(['Bases', 'Baseline (Own Bases)', 'Baseline (Average for all matchups)', 'Bases Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Bases (All Terran Matchups)', **title_font)
	pyplot.show()

def armyCountBaselineComparisonAllTogether():
	armyCountGuessDifferenceList = [[] * 1 for i in range(35)]
	armyCountGuessDifferenceAvg = []
	armyCountBaselineDifferenceList = [[] * 1 for i in range(35)]
	armyCountBaselineDifferenceAvg = []
	armyCountBaseline2DifferenceList = [[] * 1 for i in range(35)]
	armyCountBaseline2DifferenceAvg = []
	
	armyCountGuessSquaredDifferences = [[] * 1 for i in range(35)]
	armyCountGuessStandardDeviations = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		armyCountGuessDifferenceList[index_TvZ].append(math.fabs(preds_TvZ[0][84]*100 - rowy_TvZ[0][84]*100))
		armyCountBaselineDifferenceList[index_TvZ].append(math.fabs(rowy_TvZ[0][84]*100 - rowx_TvZ[0][139]*100))
		armyCountBaseline2DifferenceList[index_TvZ].append(math.fabs(rowy_TvZ[0][84]*100 - TvZArmyCountAverageBaseline[index_TvZ]))

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		armyCountGuessDifferenceList[index_TvT].append(math.fabs(preds_TvT[0][51]*100 - rowy_TvT[0][51]*100))
		armyCountBaselineDifferenceList[index_TvT].append(math.fabs(rowy_TvT[0][51]*100 - rowx_TvT[0][106]*100))
		armyCountBaseline2DifferenceList[index_TvT].append(math.fabs(rowy_TvT[0][51]*100 - TvTArmyCountAverageBaseline[index_TvT]))

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		armyCountGuessDifferenceList[index_TvP].append(math.fabs(preds_TvP[0][37]*100 - rowy_TvP[0][37]*100))
		armyCountBaselineDifferenceList[index_TvP].append(math.fabs(rowy_TvP[0][37]*100 - rowx_TvP[0][92]*100))
		armyCountBaseline2DifferenceList[index_TvP].append(math.fabs(rowy_TvP[0][37]*100 - TvPArmyCountAverageBaseline[index_TvP]))
		
	for x in range(0, 35):
		armyCountGuessDifferenceAvg.append(np.mean(armyCountGuessDifferenceList[x]))
		armyCountBaselineDifferenceAvg.append(np.mean(armyCountBaselineDifferenceList[x]))
		armyCountBaseline2DifferenceAvg.append(np.mean(armyCountBaseline2DifferenceList[x]))
	
	for x in range(0, 35):
		for y in range(0, len(armyCountGuessDifferenceList[x])):
			armyCountGuessSquaredDifferences[x].append(math.pow(armyCountGuessDifferenceAvg[x] - armyCountGuessDifferenceList[x][y], 2))
		armyCountGuessStandardDeviations.append(math.sqrt(np.mean(armyCountGuessSquaredDifferences[x])))

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
	pyplot.legend(['ArmyCount', 'Baseline (Own ArmyCount)', 'Baseline (Average for all matchups)', 'ArmyCount Std'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of ArmyCount (All Terran Matchups)', **title_font)
	pyplot.show()

def averageBases():
	basesDifferenceList = [[] * 1 for i in range(35)]
	basesActualList = [[] * 1 for i in range(35)]
	basesGuessList = [[] * 1 for i in range(35)]
	basesDifferenceAvg = []
	basesActualAvg = []
	basesGuessAvg = []

	basesSquaredDifferences = [[] * 1 for i in range(35)]
	basesStandardDeviations = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		basesDifferenceList[index_TvZ].append(math.fabs(preds_TvZ[0][87]*10 - rowy_TvZ[0][87]*10))
		basesActualList[index_TvZ].append(rowy_TvZ[0][87]*10)
		basesGuessList[index_TvZ].append(preds_TvZ[0][87]*10)

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		basesDifferenceList[index_TvT].append(math.fabs(preds_TvT[0][54]*10 - rowy_TvT[0][54]*10))
		basesActualList[index_TvT].append(rowy_TvT[0][54]*10)
		basesGuessList[index_TvT].append(preds_TvT[0][54]*10)

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		basesDifferenceList[index_TvP].append(math.fabs(preds_TvP[0][40]*10 - rowy_TvP[0][40]*10))
		basesActualList[index_TvP].append(rowy_TvP[0][40]*10)
		basesGuessList[index_TvP].append(preds_TvP[0][40]*10)
		
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
	pyplot.title('Absolute Error, Prediction and Actual Value of Bases (All Terran Matchups)', **title_font)
	pyplot.show()

def averageWorkers():
	workersDifferenceList = [[] * 1 for i in range(35)]
	workersActualList = [[] * 1 for i in range(35)]
	workersGuessList = [[] * 1 for i in range(35)]
	workersDifferenceAvg = []
	workersActualAvg = []
	workersGuessAvg = []

	workersSquaredDifferences = [[] * 1 for i in range(35)]
	workersStandardDeviations = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		workersDifferenceList[index_TvZ].append(math.fabs(preds_TvZ[0][2]*100 - rowy_TvZ[0][2]*100))
		workersActualList[index_TvZ].append(rowy_TvZ[0][2]*100)
		workersGuessList[index_TvZ].append(preds_TvZ[0][2]*100)

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		workersDifferenceList[index_TvT].append(math.fabs(preds_TvT[0][1]*100 - rowy_TvT[0][1]*100))
		workersActualList[index_TvT].append(rowy_TvT[0][1]*100)
		workersGuessList[index_TvT].append(preds_TvT[0][1]*100)

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		workersDifferenceList[index_TvP].append(math.fabs(preds_TvP[0][9]*100 - rowy_TvP[0][9]*100))
		workersActualList[index_TvP].append(rowy_TvP[0][9]*100)
		workersGuessList[index_TvP].append(preds_TvP[0][9]*100)
		
	for x in range(0, 35):
		workersDifferenceAvg.append(np.mean(workersDifferenceList[x]))
		workersActualAvg.append(np.mean(workersActualList[x]))
		workersGuessAvg.append(np.mean(workersGuessList[x]))
	
	for x in range(0, 35):
		for y in range(0, len(workersDifferenceList[x])):
			workersSquaredDifferences[x].append(math.pow(workersDifferenceAvg[x] - workersDifferenceList[x][y], 2))
		workersStandardDeviations.append(math.sqrt(np.mean(workersSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(workersDifferenceAvg[x]+workersStandardDeviations[x])
	    c1.append(workersDifferenceAvg[x]-workersStandardDeviations[x])

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Number of Workers', **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, workersDifferenceAvg, color='blue')
	pyplot.plot(x, workersActualAvg, color='red')
	pyplot.plot(x, workersGuessAvg, color='green')
	pyplot.legend(['Absolute Error', 'Actual Number of Workers', 'Predicted Number of Workers', 'Standard Deviation'], loc='upper left')
	pyplot.title('Absolute Error, Prediction and Actual Value of Workers (All Terran Matchups)', **title_font)
	pyplot.show()

def averageArmyCount():
	armyCountDifferenceList = [[] * 1 for i in range(35)]
	armyCountActualList = [[] * 1 for i in range(35)]
	armyCountGuessList = [[] * 1 for i in range(35)]
	armyCountDifferenceAvg = []
	armyCountActualAvg = []
	armyCountGuessAvg = []

	armyCountSquaredDifferences = [[] * 1 for i in range(35)]
	armyCountStandardDeviations = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		armyCountDifferenceList[index_TvZ].append(math.fabs(preds_TvZ[0][84]*100 - rowy_TvZ[0][84]*100))
		armyCountActualList[index_TvZ].append(rowy_TvZ[0][84]*100)
		armyCountGuessList[index_TvZ].append(preds_TvZ[0][84]*100)

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		armyCountDifferenceList[index_TvT].append(math.fabs(preds_TvT[0][51]*100 - rowy_TvT[0][51]*100))
		armyCountActualList[index_TvT].append(rowy_TvT[0][51]*100)
		armyCountGuessList[index_TvT].append(preds_TvT[0][51]*100)

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		armyCountDifferenceList[index_TvP].append(math.fabs(preds_TvP[0][37]*100 - rowy_TvP[0][37]*100))
		armyCountActualList[index_TvP].append(rowy_TvP[0][37]*100)
		armyCountGuessList[index_TvP].append(preds_TvP[0][37]*100)
		
	for x in range(0, 35):
		armyCountDifferenceAvg.append(np.mean(armyCountDifferenceList[x]))
		armyCountActualAvg.append(np.mean(armyCountActualList[x]))
		armyCountGuessAvg.append(np.mean(armyCountGuessList[x]))
	
	for x in range(0, 35):
		for y in range(0, len(armyCountDifferenceList[x])):
			armyCountSquaredDifferences[x].append(math.pow(armyCountDifferenceAvg[x] - armyCountDifferenceList[x][y], 2))
		armyCountStandardDeviations.append(math.sqrt(np.mean(armyCountSquaredDifferences[x])))

	b1 = []
	c1 = []
	for x in range(0, 35):
	    b1.append(armyCountDifferenceAvg[x]+armyCountStandardDeviations[x])
	    c1.append(armyCountDifferenceAvg[x]-armyCountStandardDeviations[x])

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('ArmyCount', **axis_font)
	pyplot.grid(True)
	pyplot.fill_between(np.arange(0, 35*0.5, 0.5), c1, b1, color='blue', alpha=0.3)
	pyplot.plot(x, armyCountDifferenceAvg, color='blue')
	pyplot.plot(x, armyCountActualAvg, color='red')
	pyplot.plot(x, armyCountGuessAvg, color='green')
	pyplot.legend(['Absolute Error', 'Actual ArmyCount', 'Predicted ArmyCount', 'Standard Deviation'], loc='upper left')
	pyplot.title('Absolute Error, Prediction and Actual Value of ArmyCount (All Terran Matchups)', **title_font)	
	pyplot.show()

def armyCountMatchupComparison():
	armyCountDifferenceList_TvZ = [[] * 1 for i in range(35)]
	armyCountDifferenceList_TvT = [[] * 1 for i in range(35)]
	armyCountDifferenceList_TvP = [[] * 1 for i in range(35)]
	armyCountDifferenceAvg_TvZ = []
	armyCountDifferenceAvg_TvT = []
	armyCountDifferenceAvg_TvP = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		armyCountDifferenceList_TvZ[index_TvZ].append(math.fabs(preds_TvZ[0][84]*100 - rowy_TvZ[0][84]*100))

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		armyCountDifferenceList_TvT[index_TvT].append(math.fabs(preds_TvT[0][51]*100 - rowy_TvT[0][51]*100))

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		armyCountDifferenceList_TvP[index_TvP].append(math.fabs(preds_TvP[0][37]*100 - rowy_TvP[0][37]*100))
		
	for x in range(0, 35):
		armyCountDifferenceAvg_TvZ.append(np.mean(armyCountDifferenceList_TvZ[x]))
		armyCountDifferenceAvg_TvT.append(np.mean(armyCountDifferenceList_TvT[x]))
		armyCountDifferenceAvg_TvP.append(np.mean(armyCountDifferenceList_TvP[x]))

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('ArmyCount', **axis_font)
	pyplot.grid(True)
	pyplot.plot(x, armyCountDifferenceAvg_TvP, color='blue')
	pyplot.plot(x, armyCountDifferenceAvg_TvT, color='red')
	pyplot.plot(x, armyCountDifferenceAvg_TvZ, color='green')
	pyplot.legend(['Protoss', 'Terran', 'Zerg'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of ArmyCount Across Matchups', **title_font)	
	pyplot.show()

def basesMatchupComparison():
	basesDifferenceList_TvZ = [[] * 1 for i in range(35)]
	basesDifferenceList_TvT = [[] * 1 for i in range(35)]
	basesDifferenceList_TvP = [[] * 1 for i in range(35)]
	basesDifferenceAvg_TvZ = []
	basesDifferenceAvg_TvT = []
	basesDifferenceAvg_TvP = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		basesDifferenceList_TvZ[index_TvZ].append(math.fabs(preds_TvZ[0][87]*10 - rowy_TvZ[0][87]*10))

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		basesDifferenceList_TvT[index_TvT].append(math.fabs(preds_TvT[0][54]*10 - rowy_TvT[0][54]*10))

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		basesDifferenceList_TvP[index_TvP].append(math.fabs(preds_TvP[0][40]*10 - rowy_TvP[0][40]*10))
		
	for x in range(0, 35):
		basesDifferenceAvg_TvZ.append(np.mean(basesDifferenceList_TvZ[x]))
		basesDifferenceAvg_TvT.append(np.mean(basesDifferenceList_TvT[x]))
		basesDifferenceAvg_TvP.append(np.mean(basesDifferenceList_TvP[x]))

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
	pyplot.plot(x, basesDifferenceAvg_TvP, color='blue')
	pyplot.plot(x, basesDifferenceAvg_TvT, color='red')
	pyplot.plot(x, basesDifferenceAvg_TvZ, color='green')
	pyplot.legend(['Protoss', 'Terran', 'Zerg'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Bases Across Matchups', **title_font)	
	pyplot.show()

def workersMatchupComparison():
	workersDifferenceList_TvZ = [[] * 1 for i in range(35)]
	workersDifferenceList_TvT = [[] * 1 for i in range(35)]
	workersDifferenceList_TvP = [[] * 1 for i in range(35)]
	workersDifferenceAvg_TvZ = []
	workersDifferenceAvg_TvT = []
	workersDifferenceAvg_TvP = []

	#TvZ
	for x in range(0, len(x_val_TvZ)):
		rowx_TvZ_zero = x_val_TvZ_zero[np.array([x])]
		rowx_TvZ, rowy_TvZ = x_val_TvZ[np.array([x])], y_val_TvZ[np.array([x])]
		preds_TvZ = model_TvZ.predict(rowx_TvZ_zero, verbose=0)
		index_TvZ = int(int(round(rowx_TvZ[0][0]*25000/24))/30)
		workersDifferenceList_TvZ[index_TvZ].append(math.fabs(preds_TvZ[0][2]*100 - rowy_TvZ[0][2]*100))

	#TvT
	for x in range(0, len(x_val_TvT)):
		rowx_TvT_zero = x_val_TvT_zero[np.array([x])]
		rowx_TvT, rowy_TvT = x_val_TvT[np.array([x])], y_val_TvT[np.array([x])]
		preds_TvT = model_TvT.predict(rowx_TvT_zero, verbose=0)
		index_TvT = int(int(round(rowx_TvT[0][0]*25000/24))/30)
		workersDifferenceList_TvT[index_TvT].append(math.fabs(preds_TvT[0][1]*100 - rowy_TvT[0][1]*100))

	#TvP
	for x in range(0, len(x_val_TvP)):
		rowx_TvP_zero = x_val_TvP_zero[np.array([x])]
		rowx_TvP, rowy_TvP = x_val_TvP[np.array([x])], y_val_TvP[np.array([x])]
		preds_TvP = model_TvP.predict(rowx_TvP_zero, verbose=0)
		index_TvP = int(int(round(rowx_TvP[0][0]*25000/24))/30)
		workersDifferenceList_TvP[index_TvP].append(math.fabs(preds_TvP[0][9]*100 - rowy_TvP[0][9]*100))
		
	for x in range(0, 35):
		workersDifferenceAvg_TvZ.append(np.mean(workersDifferenceList_TvZ[x]))
		workersDifferenceAvg_TvT.append(np.mean(workersDifferenceList_TvT[x]))
		workersDifferenceAvg_TvP.append(np.mean(workersDifferenceList_TvP[x]))

	axis_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	title_font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
	
	fig, axes = pyplot.subplots(num=None, figsize = (10, 6), dpi=150)
	x = np.arange(0, 35*0.5, 0.5)
	pyplot.xlabel('Minutes', **axis_font)
	pyplot.ylabel('Number of Workers', **axis_font)
	pyplot.grid(True)
	pyplot.plot(x, workersDifferenceAvg_TvP, color='blue')
	pyplot.plot(x, workersDifferenceAvg_TvT, color='red')
	pyplot.plot(x, workersDifferenceAvg_TvZ, color='green')
	pyplot.legend(['Protoss', 'Terran', 'Zerg'], loc='upper left')
	pyplot.title('Absolute Error in Prediction of Workers Across Matchups', **title_font)	
	pyplot.show()

averageBases()
#averageArmyCount()
#averageWorkers()
#workersBaselineComparisonAllTogether()
#armyCountBaselineComparisonAllTogether()
#basesBaselineComparisonAllTogether()
#armyCountMatchupComparison()
#basesMatchupComparison()
#workersMatchupComparison()


