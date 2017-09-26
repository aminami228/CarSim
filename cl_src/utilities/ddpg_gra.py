import numpy as np
import matplotlib.pyplot as plt
import json

__author__ = 'qzq'

a = 2
if a == 0:
    file_name = 'cl_tra2'
    with open('../../cl_src/results/' + file_name + '.txt', 'r') as json_file:
        results = json.load(json_file)
    correct_key = {'crash', 'unfinished', 'overspeed', 'stop', 'succeess'}
else:
    file_name = 'ddpg' if (a == 1) else 'cl_hrl_tra'
    with open('../../' + file_name + '.txt', 'r') as json_file:
        results = json.load(json_file)
    correct_key = {'crash', 'unfinished', 'overspeed', 'stop', 'succeess', 'not_stop'}

ep = len(results['crash']) / 2
# results = {'crash': crash, 'non_stop': non_stop, 'unfinished': unfinished, 'overspeed': overspeed, 'stop': stop,
train_result = dict()
test_result = dict()
train_qun = dict()
train_qun1 = dict()
test_qun = dict()
test_qun1 = dict()


for key in correct_key:
    train_result[key] = np.reshape(results[key], (ep, 2))[:, 0]
    test_result[key] = np.reshape(results[key], (ep, 2))[:, 1]

ep1 = len(results['reward']) / 100
total_ep = len(results['reward']) / 2
quan_key = {'reward', 'max_j'}
for key in quan_key:
    train_qun[key] = np.reshape(results[key], (ep1, 100))[::2, :]
    train_qun1[key] = np.mean(train_qun[key], axis=1)
    train_qun[key] = np.reshape(train_qun[key], (1, total_ep))[0]
    test_qun[key] = np.reshape(results[key], (ep1, 100))[1::2, :]
    test_qun1[key] = np.mean(test_qun[key], axis=1)
    test_qun[key] = np.reshape(test_qun[key], (1, total_ep))[0]

train_loss = np.reshape(results['loss'], (total_ep, 1))[:, 0]

fig2 = plt.figure(1)
plt.subplot(211)
# plt.ylim([-5000, 1000])
plt.title('Rewards')
plt.plot(np.arange(total_ep), train_qun['reward'], '#FFCCCC')
plt.plot(np.arange(total_ep), test_qun['reward'], '#E5FFCC')
plt.plot(np.arange(0, total_ep, 100), train_qun1['reward'], 'r', label='train reward')
plt.plot(np.arange(0, total_ep, 100), test_qun1['reward'], 'g', label='test reward')
plt.legend(loc=1)
plt.subplot(212)
plt.title('Success rate')
# plt.ylim([0, 100])
plt.plot(np.arange(ep), train_result['succeess'], '.-', label='train')
plt.plot(np.arange(ep), test_result['succeess'], '.-', label='test')
plt.legend(loc=1)
fig2.set_size_inches(12, 9)
fig2.savefig('../results/' + file_name + '_2.eps', dpi=fig2.dpi)

plt.show()
