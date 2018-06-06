#!/usr/bin/env python
import numpy as np
import os.path
import os
import io
import time
import json
from numpy import array
from keras.models import load_model

configPath = "BotConfig.txt"
with open(configPath) as jFile:
	data = json.load(jFile)
	ourRace = data["SC2API"]["BotRace"][0]
	enemyRace = data["SC2API"]["EnemyRace"][0]

modelToLoad = 'Models/' + ourRace + 'v' + enemyRace + '.h5'

model = load_model(modelToLoad)
filePath = "pyInput.txt"

while 1:
	if os.path.exists(filePath):
		with open(filePath, 'r') as f:
			lst = f.read()[:-1].split(",")
			if '' in lst:
				continue
			os.remove(filePath)
		npa = np.asarray([lst], dtype=np.float32)
		pred = model.predict(npa, verbose=0)
		with io.FileIO("prediction.txt", "w") as file:
			pred.tofile(file, ',', '%.4f')
		time.sleep(0.3)
