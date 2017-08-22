# README.md

**Table of Contents**

[TOCM]

[TOC]

#HRL source code

> **approach_src:**
> - hrl_vires.py: Main training process for hierachical reinforcement learning.
> - reward_vires.py: Construct reward functions
> - Run with vires start

```sh
$ ./vtdStart.sh
$ python hrl_vires.py
```

#Vires interface

> **interface:**
> - vires_sim: Main simulation interface with Vires. Get states and send actions. One thread for reciving ego vehicle info, one for receive other vehicles info and another one thread for sending actions to vires. Rught now, we use rdb to recive info and use scp to send actions (speed).
> - rdb_comm.py: recive rdb info
> - scp_comm.py: send actions

#Deep learning network

> **newwork:**
> - ActorNetwork.py: Construc actor network of DDPG.
> - CriticNetwork.py: Construc critic network of DDPG
> - ReplayBuffer.py: Construc buffers for training critic network

#Utilities

> **utilities:**
> - toolfunc.py: Some useful math functions for the project

