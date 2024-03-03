# -*- coding: utf-8 -*-
"""
This file is used to configure the test case.

"""
def get_config():
    '''Returns the configuration structure for the test case.
    
    Returns
    -------
    config : dict
    Dictionary contatinin configuration information.
    {
    'fmupath'  : string, location of model fmu
    'step'     : int, default control step size in seconds
    }
    
    '''
        
    config = {
    # Enter default configuration information
    'fmupath'  : 'Resources/AlphaDataCenterCooling_FMU.fmu',
    'name'  : 'AlphaDataCenterCooling_Gym',
    'step'     : 300
    }

    return config