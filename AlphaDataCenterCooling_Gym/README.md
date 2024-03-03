# AlphaDataCenterCooling-Gym
AlphaDataCenterCooling-Gym is the  [Farama-Foundation Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI-Gym ）environment for the AlphaDataCenterCooling Docker Service.
It adapts the RESTful API from AlphaDataCenterCooling Docker Service to conform with Gymnasium standards. By using Gym standards, 
researchers can leverage a common framework for algorithm development and benchmarking, streamlining the process of implementing and testing new control strategies in data center cooling optimization.

## Structure
- `/testing`:  This directory includes notebooks for gym interaction.
- `alphaDataCenterCoolingEnv.py`: Core file containing the Gym environment's functionality. 
Essential for simulation interactions and standard Gym operations.
- `environment.yml`:  Configuration file listing all dependencies required for the conda environment's setup.
- `reward_function.py`: Script for defining the reward function used in implementing reinforcement learning (RL) algorithms.
- `schema.json`: Configuration file for gym environment parameters, allowing customization of the simulation settings.
- `utils.py`: Auxiliary functions supporting the Gym environment's operations . 
## Code Usage
1) **Setting up the AlphaDataCenterCooling Docker Service:**

- Download and navigate to the repository:
```
git clone https://github.com/wfzheng/AlphaDataCenterCooling.git
cd AlphaDataCenterCooling
```
- Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/). 
- To construct and initiate the AlphaDataCenter, execute the subsequent command within the `/AlphaDataCenterCooling` root directory:
```
docker-compose up
```
2) **Installing the Conda Environment:**
- To formulate a conda environment with all required dependencies, employ the provided `environment.yml` file:
 ```
conda env create -f environment.yml
```
- Activate the newly established conda environment:
 ```
conda activate AlphaDataCenterCooling
```
3) **Navigating to the AlphaDataCenterCooling_Gym Directory:**
- Change directory to AlphaDataCenterCooling_Gym:
 ```
cd AlphaDataCenterCooling_Gym
 ```
4) **Environment Testing:**
- Validate the functionality of the environment using the Jupyter notebook located at `testing/test_gym.ipynb`
# Feedback

Feel free to send any questions/feedback to: [Zhe Wang](mailto:cezhewang@ust.hk) 

# Citation

If you use our code, please cite us as follows: