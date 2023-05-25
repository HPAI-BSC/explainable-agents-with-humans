<div align="center">
<a href="https://hpai.bsc.es/"> 
      <img src="images/bsc.svg" width="45%"  alt="Barcelona Supercomputing Center" />
</a>
<a href="https://kemlg.upc.edu"> 
      <img src="images/upc_logo.png" width="45%"  alt="Barcelona Supercomputing Center" />
</a>
</div>
<h1 align="center">
  Explainable agents adapt to Human behaviour
</h1>
<h2 align="center">
  Adrian Tormos<sup>1</sup>, Victor Gimenez-Abalos<sup>1</sup>, Marc Domenech i Vila<sup>1</sup>, Dmitry Gnatyshak<sup>2</sup>, Sergio Alvarez-Napagao<sup>1</sup> and Javier Vazquez-Salceda<sup>2</sup>
</h2>
<h3 align="center">
  <sup>1</sup>Barcelona Supercomputing Center (BSC), <sup>2</sup>Universitat Politecnica de Catalunya (UPC)
</h3>

This repository contains the code implementation for the experiments in the paper [Explainable agents adapt to Human behaviour](), which was published in the [Citizen-Centric Multiagent Systems (C-MAS) '23 workshop](https://sites.google.com/view/cmas23/home) of the [22nd International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2023)](https://aamas2023.soton.ac.uk/). 

> When integrating artificial agents into physical or digital environments that are shared with humans, agents are often equipped with opaque Machine Learning methods to enable adapting their behaviour to dynamic human needs and environment. This brings about  agents that are also opaque and therefore hard to explain.
> In previous work, we show that we can reduce an opaque agent into an explainable Policy Graph (PG) which works accurately in multi-agent environments. Policy Graphs are based on a discretisation of the world into propositional logic to identify states, and the choice of which discretiser to apply is key to the performance of the reduced agent.
> In this work, we explore this further by 1) reducing a single agent into an explainable PG, and 2) enforcing collaboration between this agent and an agent trained from human behaviour. The human agent is computed by using GAIL from a series of human-played episodes, and kept unchanged. We show that an opaque agent created and trained to collaborate with the human agent can be reduced to an explainable, non-opaque PG, so long as predicates regarding collaboration are included in the state representation, by showing the difference in reward between the agent and its PG.

# Folder structure

- `train_with_human_agent.py`: Script that uses PPO to train an agent alongside a human GAIL agent. 
- `extract_pg.py`: Script that extracts a Policy Graph from an agent.
- `test_pg.py`: Script that runs an agent for a number of episodes and tracks its obtained rewards.
- `src/`: Contains the source code for the project. `src/utils` contains auxiliary, slightly edited code from other libraries.
- `data/`: Folder containing checkpoints, generated Policy Graphs and experiment artifacts. 
  - `policy_graphs/`: Folder that will contain the generated Policy Graphs after using `extract_pg.py`.
  - `ppo_runs/`: Folder containing the checkpoints of the agents that we trained alongside GAIL agents.
  - `rewards/`: Folder in which the results of `test_pg.py` runs will be saved.
- `images/`: Auxiliary images for the `README.md`.

# Running the code

## Cloning the repository

```shell
git clone https://github.com/HPAI-BSC/explainable-agents-with-humans.git
```

## Installing dependencies

First of all, clone the [human aware RL repository](https://github.com/HumanCompatibleAI/human_aware_rl), which contains the human agents from the paper and code to run them:

```shell
git clone --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

conda create -n harl python=3.7
conda activate harl

cd human_aware_rl
pip install versioneer
./install.sh
cd ..
```

Install the appropiate tensorflow library, depending on whether you will use GPU or not:

```shell
pip install tensorflow==1.13.1
conda install mpi4py -y
```
```shell
pip install tensorflow-gpu==1.13.1
conda install mpi4py -y
```

You can verify that the HARL repo and dependencies have been installed correctly running:

```shell
cd human_aware_rl/human_aware_rl
python3 run_tests.py
```

A note from its [repository README.md](https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/README.md):

> Note that most of the DRL tests rely on having the exact randomness settings that were used to generate the tests (and thus will not pass on a GPU-enabled device).
> On OSX, you may run into an error saying that Python must be installed as a framework. You can fix it by [telling Matplotlib to use a different backend](https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/).
> We recommend to first install tensorflow (non-GPU), run the tests, and then install tensorflow-gpu.

We also need to install the [overcooked-explainability](https://github.com/MarcDV1999/overcooked-explainability) repo:

```shell
git clone https://github.com/MarcDV1999/overcooked-explainability.git

cd overcooked-explainability
bash install.sh
cd ..
```

## Training an agent alongside a human GAIL agent

Before training, copy the folder containing the GAIL agent checkpoints to the root folder with the following command:

```shell
mkdir data/bc_runs
cp -r human_aware_rl/human_aware_rl/data/bc_runs/ data/
```

To train an agent, run the `train_with_human_agent.py` file. The `--map` flag indicates which map to train an agent on. The following maps are available: `simple`, `unident_s`, `random0`, `random1`, `random3`.

```shell
python3 train_with_human_agent.py [-h]  --map {simple,unident_s,random0,random1,random3}
```

The new agent's checkpoint will be saved in `data/ppo_runs`.

## Generating a Policy Graph

Run `extract_pg.py` if you want to generate a Policy Graph of a trained agent. As before, use `--map` to select the layout to use. The flag `--discretizer` is used to select one of the four available discretizers. The process will run for `EPISODES` amount of episodes.

```shell
python3 extract_pg.py [-h] [--normalize] [--verbose] --discretizer {11,12,13,14}
                       --map {simple,unident_s,random0,random1,random3}
                       --episodes EPISODES 
```

The resulting Policy Graph will be stored in `data/policy_graphs` as `pg_MAP_DISCRETIZER.pickle`.

## Evaluating a policy

Run `test_pg.py` to get the rewards of an agent, be it PPO or PG-based, on a map for a certain number of episodes. With the `--policy-mode` flag you can select one of the two PG-based policies given a Policy Graph (`greedy` or `stochastic`), or the agent the PG is based on  (`original`). The process will run for `EPISODES` amount of episodes.

```shell
python3 test_pg.py [-h] [--discretizer {11,12,13,14}] --episodes EPISODES
                   --map {simple,unident_s,random0,random1,random3}
                   --policy-mode {original,greedy,stochastic}
```

The list of obtained rewards (total reward per episode) will be stored as a CSV file in `data/rewards` as `rewards_MAP_POLICYMODE[_DISCRETIZER].csv`.

# Citation

If you use the agents provided here, please use the following citation:

```

```

# License

GNU General Public License v3.0