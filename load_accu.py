import numpy as np
import matplotlib.pyplot as plt 

dir = './chrtest/bench/0_random_50_100_50/train_accuracy.npy'

ROUND = 400 
accu = np.load(dir)
newaccu = []
itr = 0
for i in accu:
    if itr >= ROUND:
        break
    itr += 1
    newaccu.append(i)
# print(len(accu))

x = np.linspace(1, ROUND, num = ROUND)
plt.plot(x, newaccu)
plt.show()