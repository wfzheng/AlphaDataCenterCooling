import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

from AlphaDataCenterCooling_Gym.reward_function import RewardFunction
from AlphaDataCenterCooling_Gym.utils import read_json


class AlphaDataCenterCoolingEnv(gym.Env):
    """ AlphaDataCenterCoolingEnv is a custom Gym Environment.

    Parameters
    ----------
    schema: Union[str, Path, Mapping[str, Any]]
        Filepath to JSON representation or `dict` object of a AlphaDataCenterCoolingEnv schema.
    url: string
         Rest API url for communication with the BOPTEST interface.If provided, will override the defined value read from the schema.json file.
    reward_function: RewardFunction, optional
        Reward function class instance. If provided, will override the default one.
    simulation_start_time: float, optional
        Simulation start time in seconds. If provided, will override the default value read from the schema.json file.
    simulation_end_time: float,optional
        Simulation end time in seconds. If provided, will override the default value read from the schema.json file.
    seconds_per_step: float,optional
        Number of seconds in 1 `time_step` and must be set to >= 1 and must be a multiple of 300.
        For example, valid values include 300, 600, 900, etc. If provided, will override the defined value read from the schema.json file. "
    active_observations:  List[str], optional
       Specifies which observations to actively monitor and record during the simulation. If provided, will override the defined value read from the schema.json file.

    Notes
    -----
    Action space:
    - 100 actions in total
        -act[0:6]: Valve of cooling tower OPEN/CLOSE ('U_CT1', 'U_CT2', 'U_CT3', 'U_CT4', 'U_CT5', 'U_CT6').
        -act[6: 18]: Cooling tower normalized speed ratio ('Ffan_CT1_02', 'Ffan_CT2_01', 'Ffan_CT2_02',
                    'Ffan_CT3_01', 'Ffan_CT3_02', 'Ffan_CT4_01', 'Ffan_CT4_02', 'Ffan_CT5_01', 'Ffan_CT5_02', 'Ffan_CT6_01','Ffan_CT6_02').
        -act[18:24]: Condenser water pump rotating speed ('CDWP01_rpm', 'CDWP02_rpm', 'CDWP03_rpm', 'CDWP04_rpm', 'CDWP05_rpm', 'CDWP06_rpm').
        -act[24:30]: Chilled water pump rotating speed ('CHWP01_rpm','CHWP02_rpm', 'CHWP03_rpm', 'CHWP04_rpm', 'CHWP05_rpm', 'CHWP06_rpm').
        -act[30:36]: Chiller ON/OFF ('CHI01', 'CHI02', 'CHI03', 'CHI04', 'CHI05','CHI06').
        -act[36:60]: Condenser water side valve OPEN/CLOSE ('CHI01_CW1', 'CHI01_CW2', 'CHI01_CW3', 'CHI01_CW4', 'CHI02_CW1', 'CHI02_CW2',
                     'CHI02_CW3', 'CHI02_CW4', 'CHI03_CW1', 'CHI03_CW2', 'CHI03_CW3', 'CHI03_CW4', 'CHI04_CW1', 'CHI04_CW2', 'CHI04_CW3',
                     'CHI04_CW4','CHI05_CW1', 'CHI05_CW2', 'CHI05_CW3', 'CHI05_CW4', 'CHI06_CW1', 'CHI06_CW2', 'CHI06_CW3', 'CHI06_CW4').
        -act[60:84]: Chilled water side valve OPEN/CLOSE ('CHI01_CHW1', 'CHI01_CHW2', 'CHI01_CHW3', 'CHI01_CHW4', 'CHI02_CHW1', 'CHI02_CHW2',
                     'CHI02_CHW3', 'CHI02_CHW4','CHI03_CHW1', 'CHI03_CHW2', 'CHI03_CHW3', 'CHI03_CHW4', 'CHI04_CHW1', 'CHI04_CHW2', 'CHI04_CHW3',
                     'CHI04_CHW4','CHI05_CHW1', 'CHI05_CHW2', 'CHI05_CHW3', 'CHI05_CHW4', 'CHI06_CHW1', 'CHI06_CHW2', 'CHI06_CHW3', 'CHI06_CHW4').
        -act[84:90]: Valve of condenser water pump OPEN/CLOSE ('CDWP01_ONOFF', 'CDWP02_ONOFF', 'CDWP03_ONOFF', 'CDWP04_ONOFF', 'CDWP05_ONOFF', 'CDWP06_ONOFF').
        -act[90:96]: Valve of chilled water pump OPEN/CLOSE ('CHWP01_ONOFF', 'CHWP02_ONOFF', 'CHWP03_ONOFF', 'CHWP04_ONOFF', 'CHWP05_ONOFF', 'CHWP06_ONOFF').
        -act[96]: Average speed of all condenser water pumps (CWP_speedInput).
        -act[97]: Chilled water supply temperature set point of chiller (Tchws_set_CHI).
        -act[98]: Chilled water supply temperature set point of heat exchanger (Tchws_set_HEX).
        -act[99]: Condenser water pump activated number (CWP_activatedNumber).
    """

    def __init__(self,schema: Union[str, Path, Mapping[str, Any]],url:str=None,reward_function: RewardFunction = None,
                 simulation_start_time:float=None,simulation_end_time:float=None,seconds_per_step:float=None,active_observations:List[str]=None):

        if reward_function is not None:
            self.reward_function=reward_function
        else:
            self.reward_function=RewardFunction()

        self.schema=schema

        url, seconds_per_time_step, simulation_start_time,simulation_end_time, active_observations=self._load(
            url=url,
            simulation_start_time=simulation_start_time,
            simulation_end_time=simulation_end_time,
            seconds_per_step=seconds_per_step,
            active_observations=active_observations)

        self.url=url
        self.seconds_per_time_step=seconds_per_time_step
        self.simulation_start_time=simulation_start_time
        self.simulation_end_time=simulation_end_time
        self.active_observations=active_observations

        self.action_names=['U_CT1', 'U_CT2', 'U_CT3', 'U_CT4', 'U_CT5', 'U_CT6', 'Ffan_CT1_01', 'Ffan_CT1_02', 'Ffan_CT2_01', 'Ffan_CT2_02',
                     'Ffan_CT3_01', 'Ffan_CT3_02', 'Ffan_CT4_01', 'Ffan_CT4_02', 'Ffan_CT5_01', 'Ffan_CT5_02', 'Ffan_CT6_01',
                     'Ffan_CT6_02', 'CDWP01_rpm', 'CDWP02_rpm', 'CDWP03_rpm', 'CDWP04_rpm', 'CDWP05_rpm', 'CDWP06_rpm', 'CHWP01_rpm',
                     'CHWP02_rpm', 'CHWP03_rpm', 'CHWP04_rpm', 'CHWP05_rpm', 'CHWP06_rpm', 'CHI01', 'CHI02', 'CHI03', 'CHI04', 'CHI05',
                     'CHI06', 'CHI01_CW1', 'CHI01_CW2', 'CHI01_CW3', 'CHI01_CW4', 'CHI02_CW1', 'CHI02_CW2', 'CHI02_CW3', 'CHI02_CW4',
                     'CHI03_CW1', 'CHI03_CW2', 'CHI03_CW3', 'CHI03_CW4', 'CHI04_CW1', 'CHI04_CW2', 'CHI04_CW3', 'CHI04_CW4',
                     'CHI05_CW1', 'CHI05_CW2', 'CHI05_CW3', 'CHI05_CW4', 'CHI06_CW1', 'CHI06_CW2', 'CHI06_CW3', 'CHI06_CW4',
                     'CHI01_CHW1', 'CHI01_CHW2', 'CHI01_CHW3', 'CHI01_CHW4', 'CHI02_CHW1', 'CHI02_CHW2', 'CHI02_CHW3', 'CHI02_CHW4',
                     'CHI03_CHW1', 'CHI03_CHW2', 'CHI03_CHW3', 'CHI03_CHW4', 'CHI04_CHW1', 'CHI04_CHW2', 'CHI04_CHW3', 'CHI04_CHW4',
                     'CHI05_CHW1', 'CHI05_CHW2', 'CHI05_CHW3', 'CHI05_CHW4', 'CHI06_CHW1', 'CHI06_CHW2', 'CHI06_CHW3', 'CHI06_CHW4',
                     'CDWP01_ONOFF', 'CDWP02_ONOFF', 'CDWP03_ONOFF', 'CDWP04_ONOFF', 'CDWP05_ONOFF', 'CDWP06_ONOFF', 'CHWP01_ONOFF',
                     'CHWP02_ONOFF', 'CHWP03_ONOFF', 'CHWP04_ONOFF', 'CHWP05_ONOFF', 'CHWP06_ONOFF', 'CWP_speedInput', 'Tchws_set_CHI',
                     'Tchws_set_HEX', 'CWP_activatedNumber']

        # Define the state and action space
        self.observation_space=self.get_observation_space()
        self.action_space=self.get_action_space()

        self.reset()




    def _load(self, **kwargs)-> Tuple[str, float, float, float, List[str]]:
        """Return `AlphaDataCenterCoolingEnv` objects as defined by the `schema`.
        Returns
        -------
        url: string
             Rest API url for communication with the AlphaDataCenterCooling_Gym interface

        seconds_per_time_step: float
            Number of seconds in 1 `time_step` and must be set to >= 1 and must be a multiple of 300.

        simulation_start_time: float
            Time to start the simulation

        active_observations: List[str]
            Names of active observations.
        """

        if isinstance(self.schema, (str, Path)) and os.path.isfile(self.schema):
            self.schema = read_json(self.schema)


        elif isinstance(self.schema, dict):
            self.schema = deepcopy(self.schema)
        else:
            raise ValueError('Unknown schema parsed into constructor. Schema must be a filepath to JSON representation '
                             'or `dict` object of the AlphaDataCenterCooling_Gym schema.')
        url = kwargs['url'] if kwargs.get('url') is not None else self.schema[
            'url']


        seconds_per_time_step = kwargs['seconds_per_step'] if kwargs.get('seconds_per_step') is not None else \
        self.schema['seconds_per_step']
        simulation_start_time = kwargs['simulation_start_time'] if kwargs.get(
            'simulation_start_time') is not None else \
            self.schema['simulation_start_time']
        simulation_end_time = kwargs['simulation_end_time'] if kwargs.get(
            'simulation_end_time') is not None else \
            self.schema['simulation_end_time']

        observations = self.schema['observations']
        active_observations = kwargs['active_observations'] if kwargs.get('active_observations') is not None else \
            [k for k, v in observations.items() if v['active']]

        return (
            url,seconds_per_time_step,simulation_start_time,simulation_end_time,active_observations
        )

    def reset(self)->np.array:
        """
        Reset AlphaDataCenterCoolingEnv to initial state

        Returns
        -------
        observations: numpy array
            Reformatted observations of the initial state which includes active measurements
            at the end of the initialization (beginning of the episode).
        """
        # Reset the time step index to self.simulation_start_time for the start of a new episode
        self.time_step_idx = self.simulation_start_time

        # Initialize the simulation
        res=requests.put('{0}/initialize'.format(self.url),
                     json={'start_time': self.simulation_start_time,'end_time':self.simulation_end_time}).json()['payload']

        # Set simulation step
        requests.put('{0}/step'.format(self.url), json={'step': self.seconds_per_time_step})

        # Get observations at the end of the initialization period
        observations = self._get_observations(res)

        return observations

    def step(self, act:List[Union[int, float]])->Tuple[np.ndarray, float, bool, dict]:
        """
        Advance the simulation one time step

        Parameters
        ----------
        action: list
            List of actions computed by the agent to be implemented in this step

        Returns
        -------
        observations: numpy array
            Observations at the end of this time step
        reward: float
            Reward for the state-action pair implemented
        done: boolean
            Whether a `terminal state` is reached
        info: dictionary
            Additional information for this step
        """

        # Initialize inputs to send through AlphaDataCenterCooling_Gym Rest API
        u = {}

        # Assign values to inputs if any
        for i, act_name in enumerate(self.action_names):
            # Assign value
            u[act_name]=float(act[i])

        # Advance a step of simulation
        res = requests.post('{0}/advance'.format(self.url), json=u).json()['payload']

        # Get observations at the end of this time step
        observations = self._get_observations(res)

        # Calculate the reward at the end of this time step
        reward = self.reward_function.calculate(observations=observations)

        # Increment the time step index by the number of seconds per time step
        self.time_step_idx += self.seconds_per_time_step

        # Check if the current time step is the last one based on the simulation end time
        # or if the next time step exceeds the end_time
        done = self.time_step_idx == self.simulation_end_time or (self.time_step_idx + self.seconds_per_time_step) > self.simulation_end_time

        # Optionally we can pass additional info
        info=self.get_info()

        return observations, reward, done, info

    def _get_observations(self, res: Dict[str,float])->np.array:
        """
        Get the active observations and reformat observations
        Parameters
        ----------
        res: dictionary
            Dictionary mapping simulation variables and their value at the
            end of the last time step.

        Returns
        -------
        observations: numpy array
            Reformatted observations that include active measurements.

        """

        # Initialize observations
        observations = []

        # Get measurements at the end of the simulation step
        for obs in self.active_observations:
            observations.append(res[obs])

        # Reformat observations
        observations = np.array(observations).astype(np.float32)

        return observations


    def get_info(self) -> Mapping[Any, Any]:
        """Other information to return from the `AlphaDataCenterCooling_Gym.AlphaDataCenterCoolingEnv.step` function，we leave it blank for now"""

        return {}

    def get_observation_space(self)-> spaces.Box:
        """
        "Get estimate of observation spaces. Find minimum and maximum possible values of all the observations.

        Returns
        -------
        observation_space : spaces.Box
        Observation low and high limits.
        """
        low_limit, high_limit = [], []
        for key in self.active_observations:

            if key.startswith('H_CDWP'):
                low_limit.append(0.0)
                high_limit.append(40.0)
            elif key.startswith('H_CHWP'):
                low_limit.append(0.0)
                high_limit.append(45.0)
            elif key in ['P_CDWPs_sum', 'P_CHWPs_sum']:
                low_limit.append(0.0)
                high_limit.append(792000.0)
            elif key =='P_CTfans_sum':
                low_limit.append(0.0)
                high_limit.append(444000.0)
            elif key =='P_Chillers_sum':
                low_limit.append(0.0)
                high_limit.append(3462000.0)
            elif key.startswith('Pchi'):
                low_limit.append(0.0)
                high_limit.append(577000.0)
            elif key.startswith('Pfan_CT'):
                low_limit.append(0.0)
                high_limit.append(37000.0)
            elif key=='Tchw_supply' or key.startswith('Tchws_'):
                low_limit.append(282.15)
                high_limit.append(295.15)
            elif key == 'Tcw_returnPipe' or key=='Tcw_supply':
                low_limit.append(273.15)
                high_limit.append(318.15)
            elif key.startswith('Tcwr_'):
                low_limit.append(283.15)
                high_limit.append(318.15)
            elif key.startswith('Tlvg_'):
                low_limit.append(273.15)
                high_limit.append(313.15)
            elif key == 'VolumeFlowRate_cw':
                low_limit.append(0.0)
                high_limit.append(1.5)
            elif key.startswith('eta_'):
                low_limit.append(0.0)
                high_limit.append(1.0)

        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def get_action_space(self) -> spaces.Box:
        """
        "Get estimate of action spaces. Find minimum and maximum possible values of all the actions.

        Returns
        -------
        action_space : spaces.Box
        Action low and high limits.
        """
        low_limit, high_limit = [], []
        for key in self.action_names:
            if key in ['U_CT1', 'U_CT2', 'U_CT3', 'U_CT4', 'U_CT5', 'U_CT6',
                       'CHI01', 'CHI02', 'CHI03', 'CHI04', 'CHI05','CHI06',
                       'CHI01_CW1', 'CHI01_CW2', 'CHI01_CW3', 'CHI01_CW4', 'CHI02_CW1', 'CHI02_CW2','CHI02_CW3', 'CHI02_CW4',
                       'CHI03_CW1', 'CHI03_CW2', 'CHI03_CW3', 'CHI03_CW4', 'CHI04_CW1','CHI04_CW2', 'CHI04_CW3','CHI04_CW4',
                       'CHI05_CW1', 'CHI05_CW2', 'CHI05_CW3', 'CHI05_CW4', 'CHI06_CW1', 'CHI06_CW2','CHI06_CW3', 'CHI06_CW4',
                       'CHI01_CHW1', 'CHI01_CHW2', 'CHI01_CHW3', 'CHI01_CHW4', 'CHI02_CHW1', 'CHI02_CHW2', 'CHI02_CHW3', 'CHI02_CHW4',
                       'CHI03_CHW1', 'CHI03_CHW2', 'CHI03_CHW3', 'CHI03_CHW4', 'CHI04_CHW1', 'CHI04_CHW2', 'CHI04_CHW3', 'CHI04_CHW4',
                       'CHI05_CHW1', 'CHI05_CHW2', 'CHI05_CHW3', 'CHI05_CHW4', 'CHI06_CHW1', 'CHI06_CHW2', 'CHI06_CHW3', 'CHI06_CHW4',
                       'CDWP01_ONOFF', 'CDWP02_ONOFF', 'CDWP03_ONOFF', 'CDWP04_ONOFF', 'CDWP05_ONOFF', 'CDWP06_ONOFF',
                       'CHWP01_ONOFF', 'CHWP02_ONOFF', 'CHWP03_ONOFF', 'CHWP04_ONOFF', 'CHWP05_ONOFF', 'CHWP06_ONOFF'
                       ]:
                low_limit.append(0.0)
                high_limit.append(1.0)
            elif key in ['Ffan_CT1_02', 'Ffan_CT2_01', 'Ffan_CT2_02',
                'Ffan_CT3_01', 'Ffan_CT3_02', 'Ffan_CT4_01', 'Ffan_CT4_02', 'Ffan_CT5_01', 'Ffan_CT5_02', 'Ffan_CT6_01','Ffan_CT6_02']:
                low_limit.append(0.0)
                high_limit.append(np.inf)
            elif key in ['CDWP01_rpm', 'CDWP02_rpm', 'CDWP03_rpm', 'CDWP04_rpm', 'CDWP05_rpm', 'CDWP06_rpm']:
                low_limit.append(0.0)
                high_limit.append(np.inf)
            elif key in ['CHWP01_rpm','CHWP02_rpm', 'CHWP03_rpm', 'CHWP04_rpm', 'CHWP05_rpm', 'CHWP06_rpm']:
                low_limit.append(0.0)
                high_limit.append(np.inf)
            elif key=='CWP_speedInput':
                low_limit.append(0.0)
                high_limit.append(np.inf)
            elif key=='Tchws_set_CHI':
                low_limit.append(0.0)
                high_limit.append(np.inf)
            elif key=='Tchws_set_HEX':
                low_limit.append(0.0)
                high_limit.append(np.inf)
            elif key=='CWP_activatedNumber':
                low_limit.append(0.0)
                high_limit.append(6.0)


        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))









