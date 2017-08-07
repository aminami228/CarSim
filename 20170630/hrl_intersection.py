#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
# from pos import state_table
from map_grid import IntersectionMap
from OU import OU

OU = OU()  # Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 100
    GAMMA = 0.99
    TAU = 0.0001  # Target Network HyperParameters
    LRA = 0.001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    #initialize state and reward
    Visual = False
    plt.ion()
    intersection = IntersectionMap(Visual)
    intersection.add_vehicle(0)
    state_t = intersection.get_state(0)
    total_correct = 0
    total_wrong = 0
    accuracy_all = []
    # if Visual:
    #     plt.pause(0.1)
    #     plt.clf()

    action_dim = 4  # Steering/Acceleration/Brake
    parameter_acc_dim = 2
    parameter_time_dim = action_dim
    action_size = action_dim + parameter_acc_dim + parameter_time_dim
    state_dim = intersection.state_dim
    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 20000
    max_steps = 2000
    done = False
    step = 0
    epsilon = 1

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

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

    for e in range(episode_count):
        print("Episode : " + str(e) + " Replay Buffer " + str(buff.count()), "Stop line: ", intersection.Stop_Line_X, "Sceneary", intersection.Scenary)

        total_reward = 0
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            noise_t = np.zeros([1, action_size])
            # a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            action_t_original = actor.model.predict(state_t)
            print("Action ", action_t_original)
            for i in range(action_dim):
                noise_t[0][i] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][i], 0.00, 0.10, 0.20)
            noise_t[0][4] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][4], -0.05, intersection.Max_Acc, 1.00)
            noise_t[0][5] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][5], 0.05, -intersection.Max_Acc, 1.00)
            # print(np.random.randn(1))
            for i in range(parameter_time_dim):
                ##########################################
                # Test 1 tau = 0.2
                # noise_t[0][i + action_dim + parameter_acc_dim] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][i + action_dim + parameter_acc_dim], 1.00, 0.20, 0.10)
                ##########################################
            # for i in range(parameter_steer_dim):
            #     ##########################################
            #     # Test 3 tau = 0.2
            #     noise_t[0][i + action_dim + parameter_acc_dim + parameter_time_dim] = train_indicator * max(epsilon, 0) * OU.function(
            #         action_t_original[0][i + action_dim + parameter_acc_dim + parameter_time_dim], - 0.20, 0.476, 0.20)
                ##########################################
                # Test 2 tau = 0.1
                noise_t[0][i + action_dim + parameter_acc_dim] = train_indicator * max(epsilon, 0) * OU.function(
                    action_t_original[0][i + action_dim + parameter_acc_dim], 0.01, 0.50, 0.10)

            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], -0.1, 1.00, 0.05)
            # The following code do the stochastic brake
            # if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            action_t = np.zeros([1, action_size])
            for i in range(action_size):
                action_t[0][i] = action_t_original[0][i] + noise_t[0][i]
            # print("Action ", action_t_original, "Action then ", action_t)

            choose_action = np.argmax(action_t[0][0:4])

            if choose_action == 0:
                approach_acc = action_t[0][4]
                approach_time = action_t[0][6]
                # approach_steer = action_t[0][10]
                # print("Approach Process (", approach_acc, ", ", approach_time, ") ", "AV = ", intersection.av_x)
                # acc = 2 * (intersection.av_x - approach_pos - intersection.av_velocity * approach_time) / (approach_time ** 2)
                ###########################################
                time_step = int(np.ceil(max(approach_time / intersection.Tau, 1.0)))
                for ts in range(time_step):
                    old_av_x = intersection.av_x
                    old_av_velocity = intersection.av_velocity
                    reward, collision = intersection.reward_function(approach_acc)
                    state_t1 = intersection.update_vehicle(approach_acc, choose_action)
                    if Visual:
                        plt.pause(0.1)
                        plt.clf()
                    # reward_t += reward
                    # collision_t += collision
                    ###########################################################3
                    buff.add(state_t, action_t[0], reward, state_t1, done)  # Add replay buffer

                    # Do the batch update
                    batch = buff.getBatch(BATCH_SIZE)
                    states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
                    actions = np.asarray([e[1] for e in batch])
                    rewards = np.asarray([e[2] for e in batch])
                    new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
                    dones = np.asarray([e[4] for e in batch])
                    y_t = np.asarray([e[2] for e in batch])

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

                    total_reward += reward

                    print("Approach Process (", approach_acc, ", ", approach_time, ") ", "AV = ", old_av_x,
                          "Velocity = ", old_av_velocity, "Episode", e,
                          "Step", step, "Reward", reward, "Loss", loss)

                    if state_t[0][0] <= 0:
                        state_t = state_t1
                        step += 1
                        break
                    ###########################################################
                    if old_av_x < intersection.Pass_Intersection or collision > 0 or step > max_steps:
                        if_pass = old_av_x < intersection.Pass_Intersection
                        done = True
                        break

                    state_t = state_t1
                    step += 1
                ###########################################
                # state_t1 = intersection.update_vehicle(acc)
                # ##plt.pause(0.01)
                # ##plt.clf()
                # reward_t, collision_t = intersection.reward_function(acc)
                # if intersection.av_x < intersection.Pass_Intersection or \
                #                         intersection.av_x < intersection.Pass_Intersection > 301 or collision_t > 0:
                #     done = True
                ###############################################
            else:
                if choose_action ==1:
                    observe_time = action_t[0][7]
                    # observe_steer = action_t[0][11]
                    # print("Observe Process (", observe_time, ") ", "AV = ", intersection.av_x)
                    ############################################
                    time_step = int(np.ceil(max(observe_time / intersection.Tau, 1.0)))
                    for ts in range(time_step):
                        old_av_x = intersection.av_x
                        old_av_velocity = intersection.av_velocity
                        acc = (intersection.observe_vel - intersection.av_velocity) / intersection.Tau
                        # if abs((intersection.observe_vel - intersection.av_velocity) / intersection.Tau) <= intersection.Max_Acc:
                        #     acc = (intersection.observe_vel - intersection.av_velocity) / intersection.Tau
                        # else:
                        #     if intersection.observe_vel > intersection.av_velocity:
                        #         acc = intersection.Max_Acc
                        #     else:
                        #         acc = -intersection.Max_Acc
                        reward, collision = intersection.reward_function(acc)
                        state_t1 = intersection.update_vehicle(acc, choose_action)
                        if Visual:
                            plt.pause(0.1)
                            plt.clf()
                        ###########################################################3
                        buff.add(state_t, action_t[0], reward, state_t1, done)  # Add replay buffer

                        # Do the batch update
                        batch = buff.getBatch(BATCH_SIZE)
                        states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
                        actions = np.asarray([e[1] for e in batch])
                        rewards = np.asarray([e[2] for e in batch])
                        new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
                        dones = np.asarray([e[4] for e in batch])
                        y_t = np.asarray([e[2] for e in batch])

                        target_q_values = critic.target_model.predict(
                            [new_states, actor.target_model.predict(new_states)])

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

                        total_reward += reward

                        print("Observe Process (", observe_time, ") ", "AV = ", old_av_x,
                              "Velocity = ", old_av_velocity, "Episode", e,
                              "Step", step, "Reward", reward, "Loss", loss)

                        ###########################################################
                        if old_av_x < intersection.Pass_Intersection or collision > 0 or step > max_steps:
                            if_pass = old_av_x < intersection.Pass_Intersection
                            done = True
                            break

                        state_t = state_t1
                        step += 1
                    #############################################
                    # if intersection.av_velocity > intersection.observe_vel:
                    #     acc = (intersection.observe_vel - intersection.av_velocity)/intersection.Tau
                    #     intersection.av_velocity = intersection.observe_vel
                    # else:
                    #     acc = 0
                    # state_t1 = intersection.update_vehicle(acc)
                    # ##plt.pause(0.01)
                    # ##plt.clf()
                    # reward_t, collision_t = intersection.reward_function(acc)
                    # if intersection.av_x < intersection.Pass_Intersection or \
                    #                         intersection.av_x < intersection.Pass_Intersection > 301 or collision_t > 0:
                    #     done = True
                    ##################################################
                else:
                    if choose_action ==2:
                        wait_time = action_t[0][8]
                        # print("Wait Process (", wait_time, ") ", "AV = ", intersection.av_x)
                        ########################################################333
                        time_step = int(np.ceil(max(wait_time / intersection.Tau, 1.0)))
                        for ts in range(time_step):
                            old_av_x = intersection.av_x
                            old_av_velocity = intersection.av_velocity
                            acc = (- intersection.av_velocity) / intersection.Tau
                            # if intersection.av_velocity > 0:
                            #     if abs((0 - intersection.av_velocity) / intersection.Tau) <= intersection.Max_Acc:
                            #         acc = (0 - intersection.av_velocity) / intersection.Tau
                            #     else:
                            #         acc = - intersection.Max_Acc
                            #     # state_t1 = intersection.update_vehicle(acc)
                            #     # intersection.av_velocity = 0
                            # else:
                            #     acc = 0
                            reward, collision = intersection.reward_function(acc)
                            state_t1 = intersection.update_vehicle(acc, choose_action)
                            if Visual:
                                plt.pause(0.1)
                                plt.clf()
                            ###########################################################3
                            buff.add(state_t, action_t[0], reward, state_t1, done)  # Add replay buffer

                            # Do the batch update
                            batch = buff.getBatch(BATCH_SIZE)
                            states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
                            actions = np.asarray([e[1] for e in batch])
                            rewards = np.asarray([e[2] for e in batch])
                            new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
                            dones = np.asarray([e[4] for e in batch])
                            y_t = np.asarray([e[2] for e in batch])

                            target_q_values = critic.target_model.predict(
                                [new_states, actor.target_model.predict(new_states)])

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

                            total_reward += reward

                            print("Wait Process (", wait_time, ") ", "AV = ", old_av_x,
                                  "Velocity = ", old_av_velocity, "Episode", e,
                                  "Step", step, "Reward", reward, "Loss", loss)

                            ###########################################################
                            if old_av_x < intersection.Pass_Intersection or collision > 0 or step > max_steps:
                                if_pass = old_av_x < intersection.Pass_Intersection
                                done = True
                                break

                            state_t = state_t1
                            step += 1
                        ###########################################################
                        # if intersection.av_velocity > 0:
                        #     acc = (0 - intersection.av_velocity) / intersection.Tau
                        #     intersection.av_velocity = 0
                        # else:
                        #     acc = 0
                        # state_t1 = intersection.update_vehicle(acc)
                        # ##plt.pause(0.01)
                        # ##plt.clf()
                        # reward_t, collision_t = intersection.reward_function(acc)
                        # if intersection.av_x < intersection.Pass_Intersection or \
                        #                         intersection.av_x < intersection.Pass_Intersection > 301 or collision_t > 0:
                        #     done = True
                        ##############################################################
                    else:
                        tranverse_acc = action_t[0][5]
                        tranverse_time = action_t[0][9]
                        # traverse_steer = action_t[0][12]
                        # print("Traverse Process (", tranverse_acc, ", ", tranverse_time, ") ", "AV = ", intersection.av_x)
                        # acc = 2 * (intersection.av_x - tranverse_goal - intersection.av_velocity * tranverse_time) / (tranverse_time ** 2)
                        ###################################################################################
                        time_step =  int(np.ceil(max(tranverse_time / intersection.Tau, 1.0)))
                        for ts in range(time_step):
                            old_av_x = intersection.av_x
                            old_av_velocity = intersection.av_velocity
                            reward, collision = intersection.reward_function(tranverse_acc)
                            state_t1 = intersection.update_vehicle(tranverse_acc, choose_action)
                            if Visual:
                                plt.pause(0.1)
                                plt.clf()
                            ###########################################################3
                            buff.add(state_t, action_t[0], reward, state_t1, done)  # Add replay buffer

                            # Do the batch update
                            batch = buff.getBatch(BATCH_SIZE)
                            states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
                            actions = np.asarray([e[1] for e in batch])
                            rewards = np.asarray([e[2] for e in batch])
                            new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
                            dones = np.asarray([e[4] for e in batch])
                            y_t = np.asarray([e[2] for e in batch])

                            target_q_values = critic.target_model.predict(
                                [new_states, actor.target_model.predict(new_states)])

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

                                total_reward += reward

                            print("Traverse Process (", tranverse_acc, ", ", tranverse_time, ") ", "AV = ", old_av_x,
                                  "Velocity = ", old_av_velocity, "Episode", e,
                                  "Step", step, "Reward", reward, "Loss", loss)

                            ###########################################################
                            if old_av_x < intersection.Pass_Intersection or collision > 0 or step > max_steps:
                                if_pass =  old_av_x < intersection.Pass_Intersection
                                done = True
                                break

                            state_t = state_t1
                            step += 1
                        ###################################################################################
                        # state_t1 = intersection.update_vehicle(acc)
                        # ##plt.pause(0.01)
                        # ##plt.clf()
                        # reward_t, collision_t = intersection.reward_function(acc)
                        # if intersection.av_x < intersection.Pass_Intersection or \
                        #                         intersection.av_x < intersection.Pass_Intersection > 301 or collision_t > 0:
                        #     done = True
                        #########################################################################

            # if intersection.av_x < intersection.Pass_Intersection or collision > 0:
            #     done = True

            # s_t1 = np.hstack(
                # (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            # s_t1 = np.hstack((np.array([l1, l2, l3, l4, av_xpos, av_pos, av_angle], ndmin=2), dist_t))

            ############################################################################
            # buff.add(state_t, action_t[0], reward_t, state_t1, done)  # Add replay buffer
            #
            # # Do the batch update
            # batch = buff.getBatch(BATCH_SIZE)
            # states = np.squeeze(np.asarray([e[0] for e in batch]), axis=1)
            # actions = np.asarray([e[1] for e in batch])
            # rewards = np.asarray([e[2] for e in batch])
            # new_states = np.squeeze(np.asarray([e[3] for e in batch]), axis=1)
            # dones = np.asarray([e[4] for e in batch])
            # y_t = np.asarray([e[2] for e in batch])
            #
            # target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            #
            # for k in range(len(batch)):
            #     if dones[k]:
            #         y_t[k] = rewards[k]
            #     else:
            #         y_t[k] = rewards[k] + GAMMA * target_q_values[k]
            #
            # if (train_indicator):
            #     loss += critic.model.train_on_batch([states, actions], y_t)
            #     a_for_grad = actor.model.predict(states)
            #     grads = critic.gradients(states, a_for_grad)
            #     actor.train(states, grads)
            #     actor.target_train()
            #     critic.target_train()
            #
            # total_reward += reward_t
            # state_t = state_t1
            #
            # print("Episode", e, "Step", step, "Reward", reward_t, "Loss", loss)
            #
            # step += 1
            ###############################################################################################3

            if done:
                plt.close("all")
                # if np.mod(e, 5) == 0:
                #     Visual = True
                # else:
                #     Visual = False
                Visual = False
                intersection = IntersectionMap(Visual)
                intersection.add_vehicle(0)
                state_t = intersection.get_state(0)
                step = 0
                if Visual:
                    plt.pause(0.1)
                    plt.clf()
                done = False
                break

        if np.mod(e, 5) == 0:
            if (train_indicator):
                # print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)


        total_correct += int(collision <= 0 and if_pass)
        total_wrong += int(collision > 0)
        accuracy = total_correct / (total_correct + total_wrong)

        if np.mod(e, 100) == 0:
            accuracy_all.append(accuracy)
            total_correct = 0
            total_wrong = 0

        print("TOTAL REWARD @ " + str(e) + "-th Episode  : Reward " + str(total_reward) + " Collision " + str(collision > 0) +
              " Accuracy " + str(accuracy) + " All Accuracy " + str(accuracy_all))

        print("")

    print("Finish.")


if __name__ == "__main__":
    playGame()


### DDPG
##No Front vehicle
## paper reference
## Reward function
## Prepare report
## Big story