import matplotlib.pyplot as plt 

Info_cluster = []
random_path = './chrtest/bench/0_random_50_100_50/'
info_path = './chrtest/infomap_random1/0_random_50_100_50/'


import numpy as np
info_time = []
random_time = []

with open(info_path + 'train_time.txt') as fp:
    lines = fp.readlines()
    for t in lines:
        info_time.append(t)


    
with open(random_path + 'train_time.txt') as fp:
    lines = fp.readlines()
    for t in lines:
        random_time.append(t)


info_accu = np.load(info_path + 'train_accuracy.npy')
rand_Accu = np.load(random_path + 'train_accuracy.npy')

ROUND = 400
x = np.linspace(1, ROUND, num = ROUND)
plt.plot(x, rand_Accu[:ROUND])
plt.plot(x, info_accu[:ROUND])
plt.legend(['random', 'infomap'])
plt.show()

plt.plot(x, random_time[:ROUND])
plt.plot(x, info_time[:ROUND])
plt.legend(['random_time', 'info_time'])
plt.show()
