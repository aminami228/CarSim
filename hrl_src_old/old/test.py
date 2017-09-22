#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from intersection import SimIntersection
from OU import OU

OU = OU()  # Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 100
    GAMMA = 0.99
    TAU = 0.0001  # Target Network HyperParameters
    LRA = 0.001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    #initialize state and reward
    intersection=SimIntersection()
    state_t = intersection.get_state()

    action_dim = 4  # Steering/Acceleration/Brake
    parameter_acc_dim = 2
    parameter_time_dim = action_dim
    action_size = action_dim + parameter_acc_dim + parameter_time_dim
    state_dim = intersection.state_dim
    np.random.seed(1337)

    EXPLORE = 100000.
    #episode_count = 20000
    #max_steps = 1000
    episode_count = 10
    max_steps = 10
    reward=0
    done = False
    step = 0
    epsilon = 1
    indicator=0

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
        print("Episode : " + str(e) + " Replay Buffer " + str(buff.count()), "Stop line: ", intersection.Stop_Line_Y)

        total_reward = 0
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            noise_t = np.zeros([1, action_size])
            action_t_original = actor.model.predict(state_t)
            print("Action ", action_t_original)
            for i in range(action_dim):
                noise_t[0][i] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][i], 0.00, 0.10, 0.20)
            noise_t[0][4] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][4], -0.05, intersection.Max_Acc, 1.00)
            noise_t[0][5] = train_indicator * max(epsilon, 0) * OU.function(action_t_original[0][5], 0.05, -intersection.Max_Acc, 1.00)
            for i in range(parameter_time_dim):
                noise_t[0][i + action_dim + parameter_acc_dim] = train_indicator * max(epsilon, 0) * OU.function(
                    action_t_original[0][i + action_dim + parameter_acc_dim], 0.01, 0.50, 0.10)

            action_t = np.zeros([1, action_size])
            for i in range(action_size):
                action_t[0][i] = action_t_original[0][i] + noise_t[0][i]
            choose_action = np.argmax(action_t[0][0:4])

            if choose_action == 0:
                approach_acc = action_t[0][4]
                approach_time = action_t[0][6]
                time_step = int(np.ceil(max(approach_time / intersection.Tau, 1.0)))
                for ts in range(time_step):
                    reward, collision = intersection.reward_function(approach_acc)
                    state_t1 = intersection.update_vehicle(approach_acc)
                    buff.add(state_t, action_t[0], reward, state_t1, done) 

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
                    state_t=state_t1
                    step+=1

                    print("Approach Process (", approach_acc, ", ", approach_time, ") ", "AV = ", intersection.av_y,
                          "Velocity = ", intersection.av_velocity, "Episode", e,
                          "Step", step, "Reward", reward, "Loss", loss)

                    if intersection.av_velocity <= 0:
                        break
                    ###########################################################
                    if intersection.av_y > intersection.Pass_Intersection or collision > 0:
                        done = True
                        break

            else:
                if choose_action ==1:
                    observe_time = action_t[0][7]
                    time_step = int(np.ceil(max(observe_time / intersection.Tau, 1.0)))
                    for ts in range(time_step):
                        acc = (intersection.observe_vel -intersection.av_velocity) / intersection.Tau
                        reward, collision = intersection.reward_function(acc)
                        state_t1 = intersection.update_vehicle(acc)
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
                        state_t = state_t1
                        step += 1

                        print("Observe Process (", observe_time, ") ", "AV = ", intersection.av_y,
                              "Velocity = ", intersection.av_velocity, "Episode", e,
                              "Step", step, "Reward", reward, "Loss", loss)

                        ###########################################################
                        if intersection.av_y > intersection.Pass_Intersection or collision > 0:
                            done = True
                            break

                else:
                    if choose_action ==2:
                        wait_time = action_t[0][8]
                        time_step = int(np.ceil(max(wait_time / intersection.Tau, 1.0)))
                        for ts in range(time_step):
                            if intersection.av_velocity>0:
                                acc = (- intersection.av_velocity) / intersection.Tau
                            else:
                                acc=0
                            reward, collision = intersection.reward_function(acc)
                            state_t1 = intersection.update_vehicle(acc)
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
                            state_t = state_t1
                            step += 1

                            print("Wait Process (", wait_time, ") ", "AV = ", intersection.av_y,
                                  "Velocity = ", intersection.av_velocity, "Episode", e,
                                  "Step", step, "Reward", reward, "Loss", loss)

                            ###########################################################
                            if intersection.av_y > intersection.Pass_Intersection or collision > 0:
                                done = True
                                break

                    else:
                        tranverse_acc = action_t[0][5]
                        tranverse_time = action_t[0][9]
                        time_step =  int(np.ceil(max(tranverse_time / intersection.Tau, 1.0)))
                        for ts in range(time_step):
                            reward, collision = intersection.reward_function(tranverse_acc)
                            state_t1 = intersection.update_vehicle(tranverse_acc)
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
                                state_t = state_t1
                                step += 1

                            print("Traverse Process (", tranverse_acc, ", ", tranverse_time, ") ", "AV = ", intersection.av_y,
                                  "Velocity = ", intersection.av_velocity, "Episode", e,
                                  "Step", step, "Reward", reward, "Loss", loss)

                            if intersection.av_y > intersection.Pass_Intersection or collision > 0:
                                done = True
                                break
            if intersection.av_y > intersection.Pass_Intersection or collision > 0:
                done = True
            
            if done:
                intersection=SimIntersection()
                state_t = intersection.get_state()
                step = 0
                done = False
                break

        if np.mod(e, 10) == 0:
            if (train_indicator):
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(e) + "-th Episode  : Reward " + str(total_reward) + " Collision " + str(collision > 0))

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
