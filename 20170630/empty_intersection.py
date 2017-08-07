#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.engine.training import collect_trainable_weights
import json
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
# from pos import state_table
from state_space import current_state
from  state_space import reward_fun
from OU import OU
import timeit
from random import randint

OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.05  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic
    n_degree = 18

    # destination_pos = 7
    # av_pos = -30
    # av_xpos = 2
    # av_angle = np.pi / 2
    # l1 = 8
    # l2 = 8
    # l3 = 8
    # l4 = 8
    # v_t = 0
    # r_t_area = 1

    app_inter_ypos = -4
    sce = randint(0, 2)
    pos_x = sce * 48 + 2
    av_pos = -40
    fv_pos = random.random() * (app_inter_ypos - av_pos) + av_pos
    fv_pos_x = np.random.normal(pos_x, 0.1)
    no_v = 10
    hv_pos_f = np.random.uniform(-164, pos_x - 6, [1, no_v])
    hv_pos_fy = np.random.normal(-2, 0.1, no_v)
    hv_pos_b = np.random.uniform(pos_x + 2, 260, [1, no_v])
    hv_pos_by = np.random.normal(2, 0.1, no_v)
    n_degree = 30
    gamma = 0.9
    learning_rate = 0.01
    display_step = 30
    tau = 0.05
    vel_fv = 20
    vel_av = 20
    vel_hv = 15
    rx = 2
    lx = 6

    pass_inter_ypos = 4
    resol = 2 * np.pi / n_degree

    #initialize state and reward
    state_t0, reward_t0 = current_state(vel_av, av_pos, fv_pos, hv_pos_f, hv_pos_b)

    action_dim = 2  # Steering/Acceleration/Brake
    state_dim = n_degree + 5 #29  # of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Generate a Torcs environment
    # env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    # print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        # if np.mod(i, 3) == 0:
        #     ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        # else:
        #     ob = env.reset()

        state_t, reward_t = current_state(vel_av, av_pos, fv_pos, hv_pos_f, hv_pos_b)

        # s_t = np.hstack((np.array([l1, l2, l3, l4, av_xpos, av_pos, av_angle], ndmin=2), dist_t))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            # a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            a_t_original = actor.model.predict(state_t)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], -0.1, 1.00, 0.05)

            # The following code do the stochastic brake
            # if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            vel_ori = vel_av
            vel_av = np.maximum(vel_av + a_t[0][0] * TAU, 0)
            # a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]

            # ob, r_t, done, info = env.step(a_t[0])
            reward_ori = reward_t
            reward_t = reward_fun(state_t)
            av_pos += 0.5 * (vel_ori + vel_av) *TAU
            fv_pos += tau * vel_fv
            av_pos += tau * vel_av
            hv_pos_b -= tau * vel_hv
            hv_pos_f += tau * vel_hv
            # if r_t_area < r_t_ori:
            #     r_t = r_t_area - 2
            # else:
            #     r_t = r_t_area
            # if a_t[0][0] > 0:
            #     av_pos = pass_inter_ypos
            #     # r_t += 1
            # else:
            #     av_pos += 0.5 * 1 * (TAU ** 2)
            #     r_t -= 0.5

            if av_pos >= pass_inter_ypos:
                done = True

            # dist_t, area_t = current_state(av_pos)
            state_t1, reward_t1 = current_state(vel_av, av_pos, fv_pos, hv_pos_f, hv_pos_b)
            # s_t1 = np.hstack(
                # (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            # s_t1 = np.hstack((np.array([l1, l2, l3, l4, av_xpos, av_pos, av_angle], ndmin=2), dist_t))

            buff.add(state_t, a_t[0], reward_t, state_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            # y_t = rewards

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += reward_t
            state_t = state_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", reward_t, "Loss", loss)

            step += 1
            if done:
                sce = randint(0, 2)
                pos_x = sce * 48 + 2
                fv_pos = random.random() * 25 - 30
                av_pos = -30
                hv_pos_f = np.random.uniform(-164, pos_x - 6, [1, no_v])
                hv_pos_b = np.random.uniform(pos_x + 2, 260, [1, no_v])
                done = False
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    # env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()
