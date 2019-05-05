# -*- coding: utf-8 -*-

import numpy as np
import os, time

amount = 1

mu, sigma = -50, 20 # mean and standard deviation
acc_x = np.random.normal(mu, sigma, amount)

mu, sigma = +5, 0.3 # mean and standard deviation
acc_y = np.random.normal(mu, sigma, amount)

mu, sigma = +79, 8 # mean and standard deviation
acc_z = np.random.normal(mu, sigma, amount)

for i in range(amount) :
	print(acc_x[i],acc_y[i],acc_z[i])
	command = """curl -d "accelerometer acc_x={},acc_y={},acc_z={}" -X POST http://146.48.82.129:8086/write?db=mydb""".format(str(acc_x[i]), str(acc_y[i]), str(acc_z[i]))
	os.system(command)
	time.sleep(.300)

input("Press ENTER to exit.")