# ELo-Rainbow: Evolving Losses + Rainbow DQN

This repository is the official implementation of ELo-Rainbow as a part of our [RL+SSL paper](https://openreview.net/forum?id=fVslVNBfjd8) at NeurIPS 2022. 
Our implementation is based on [CURL](https://github.com/aravindsrinivas/curl_rainbow) by Aravind Srinivas.

DMControl experiments were done in a separate codebase (Check [ELo-SAC](https://github.com/LostXine/elo-sac)). 


## Installation 

All of the dependencies are in `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

After the installation, you may need to import game ROMs to enable training on certain games. Please follow the instruction at [Atari_py](https://github.com/openai/atari-py).

Change the server IP and port in the `search-server-ip` file if necessary.

## Instructions

First, start the search server using `bash start-server.sh` or the following command:

```
python3 server.py --port 61889 --timeout 24
```

This will start a HTTP server listening at the given port. 
The server runs a PSO (Particle Swarm Optimization) algorithm and distributes tasks to the clients.
Timeout means how many hours the server will wait for the client to report results before it assigns the same task to another client.
Our optimization status is stored at `server/evol_rl.npy` and will be automatically loaded.
One could start a new search by assigning `--path` to a new file.

To start the parameter search on clients, run `bash search.sh`. 
The client will connect to the HTTP server and request the hyper-parameters for training.
When the training completes, the client will report the evaluation results to the server and requests a new task.

Run `bash check_status.sh` or `bash check_full_status.sh` to check the search status.

To stop the search, **stop** the current server and **restart** the search server with `--stop True`. All the clients will stop searching after finishing the current search.

To evaluate the optimal combination, run `bash eval-s0019.sh` and it will start to train ELo-Rainbow agents in 7 Atari environments with 20 random seeds.

Check `main.py` file for hyper-parameters.


## Contact

1. Issue
2. email: xiangli8@cs.stonybrook.edu
