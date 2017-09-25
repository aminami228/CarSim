import numpy as np
import matplotlib.pyplot as plt
import json

__author__ = 'qzq'


file_name = 'ddpg'
# file_name = 'hrl'
with open('../../' + file_name + '.txt', 'r') as json_file:
    results1 = json.load(json_file)

ep = len(results1['crash']) / 2
# results = {'crash': crash, 'non_stop': non_stop, 'unfinished': unfinished, 'overspeed': overspeed, 'stop': stop,
train_result = dict()
test_result = dict()
train_qun = dict()
test_qun = dict()

correct_key = {'crash', 'unfinished', 'overspeed', 'stop', 'succeess'} #, 'not_stop'}
for key in correct_key:
    train_result[key] = np.reshape(results1[key], (ep, 2))[:, 0]
    test_result[key] = np.reshape(results1[key], (ep, 2))[:, 1]

ep1 = len(results1['reward']) / 100
total_ep = len(results1['reward']) / 2
quan_key = {'reward', 'max_j'}
for key in quan_key:
    train_qun[key] = np.reshape(results1[key], (ep1, 100))[::2, :]
    train_qun[key] = np.reshape(train_qun[key], (1, total_ep))[0]
    test_qun[key] = np.reshape(results1[key], (ep1, 100))[1::2, :]
    test_qun[key] = np.reshape(test_qun[key], (1, total_ep))[0]

train_loss = np.reshape(results1['loss'], (total_ep, 1))[:, 0]

fig1 = plt.figure(1)
plt.subplot(211)
plt.title('train, total time: {0:.2f} hr'.format(results1['time'][-1] / 60.))
for key, value in train_result.iteritems():
    plt.plot(np.arange(ep), value, 'o-', label=key)
plt.legend(loc=1)

plt.subplot(212)
plt.title('test')
for key, value in test_result.iteritems():
    plt.plot(np.arange(ep), value, 'o-', label=key)
plt.legend(loc=1)
fig1.set_size_inches(24, 18)
fig1.savefig('../results/' + file_name + '_1.png', dpi=fig1.dpi)

fig2 = plt.figure(2)
plt.subplot(311)
plt.title('critic loss: {0:.2f}'.format(np.mean(train_loss[-100:])))
# plt.ylim([0, 50000])
plt.plot(np.arange(total_ep), train_loss, 'r', label='loss')
plt.legend(loc=1)
plt.subplot(312)
# plt.ylim([-5000, 1000])
plt.title('rewards: {0:.2f}'.format(np.mean(test_qun['reward'][-100:])))
plt.plot(np.arange(total_ep), train_qun['reward'], 'r', label='train reward')
plt.plot(np.arange(total_ep), test_qun['reward'], 'g', label='test reward')
plt.legend(loc=1)
plt.subplot(313)
plt.title('max jerk: {0:.2f}'.format(np.mean(test_qun['max_j'][-100:])))
plt.plot(np.arange(total_ep), train_qun['max_j'], 'r', label='train max jerk')
plt.plot(np.arange(total_ep), test_qun['max_j'], 'g', label='test max jerk')
plt.legend(loc=1)
fig2.set_size_inches(24, 18)
fig2.savefig('../results/' + file_name + '_2.png', dpi=fig2.dpi)

plt.show()