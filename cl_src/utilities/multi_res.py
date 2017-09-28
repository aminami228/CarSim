import numpy as np
import matplotlib.pyplot as plt
import json

__author__ = 'qzq'


with open('../../ddpg.txt', 'r') as json_file:
    ddpg = json.load(json_file)
with open('../../hrl_tra.txt', 'r') as json_file:
    hrl_tra = json.load(json_file)
with open('../../cl_hrl_loc2.txt', 'r') as json_file:
    cl_hrl_gpu1 = json.load(json_file)
with open('../../cl_hrl_gpu2.txt', 'r') as json_file:
    cl_hrl_gpu2 = json.load(json_file)
with open('../../cl_hrl_local/results/cl_hrl_loc1.txt', 'r') as json_file:
    cl_tra_local = json.load(json_file)
with open('../../cl_src/results/cl_tra3.txt', 'r') as json_file:
    cl_tra = json.load(json_file)

keys = ['ddpg', 'hddpg', 'cl+hddpg', 'cl_tra_local', 'gpu1']
results = [ddpg, hrl_tra, cl_hrl_gpu2, cl_tra_local, cl_hrl_gpu1]
# keys = ['ddpg', 'hddpg']
# results = [ddpg, hrl_tra]

train_succ = {}
test_succ = {}
train_reward = {}
train_reward1 = {}
test_reward = {}
test_reward1 = {}
train_loss = {}
for i, key in enumerate(keys):
    succ = results[i]['succeess']
    train_succ[key] = succ[::2]
    test_succ[key] = succ[1::2]
    reward = results[i]['reward']
    ep = len(reward) / 100
    train_reward[key] = np.reshape(reward, (ep, 100))[::2, :]
    train_reward1[key] = np.mean(train_reward[key], axis=1)
    train_reward[key] = np.reshape(train_reward[key], (1, ep*50))[0]
    test_reward[key] = np.reshape(reward, (ep, 100))[1::2, :]
    test_reward1[key] = np.mean(test_reward[key], axis=1)
    test_reward[key] = np.reshape(test_reward[key], (1, ep * 50))[0]
    train_loss[key] = results[i]['loss']

fig1 = plt.figure(1)
plt.subplot(211)
plt.title('Success rate: train')
# plt.ylim([0, 101])
for key, value in train_succ.iteritems():
    plt.plot(np.arange(len(value)), value, '.-', label=key)
plt.legend(loc=1)
plt.subplot(212)
plt.title('Success rate: test')
plt.xlabel('Learing epochs')
# plt.ylim([0, 101])
for key, value in test_succ.iteritems():
    plt.plot(np.arange(len(value)), value, '.-', label=key)
plt.legend(loc=1)
fig1.set_size_inches(12, 9)
fig1.savefig('../results/succ_1.eps', dpi=fig1.dpi)

fig2 = plt.figure(2)
# plt.subplot(311)
# plt.title('Critic loss')
# # plt.ylim([0, 50000])
# for key, value in train_loss.iteritems():
#     plt.plot(np.arange(len(value)), value, label=key)
# plt.legend(loc=1)
plt.subplot(211)
plt.ylim([-5000, 2000])
plt.title('Reward: train')
m_c = ['r', 'g', 'b', '#7F00FF', '#FF8000']
_c = ['#FFCCCC', '#E5FFCC', '#CCFFFF', '#E5CCFF', '#FFE5CC']
i, j = 0, 0
for key, value in train_reward.iteritems():
    plt.plot(np.arange(len(value)), value, _c[j])
    j += 1
for key, value in train_reward1.iteritems():
    plt.plot(np.arange(0, 100*len(value), 100), value, m_c[i], label=key)
    i += 1
plt.legend(loc=1)
plt.subplot(212)
plt.ylim([-5000, 2000])
plt.title('Reward: test')
plt.xlabel('Learing iteration')
i, j = 0, 0
for key, value in test_reward.iteritems():
    plt.plot(np.arange(len(value)), value, _c[j])
    j += 1
for key, value in test_reward1.iteritems():
    plt.plot(np.arange(0, 100*len(value), 100), value, m_c[i], label=key)
    i += 1
plt.legend(loc=1)
fig2.set_size_inches(12, 9)
fig2.savefig('../results/reward.eps', dpi=fig2.dpi)

plt.show()
