#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers
@author: thomas
"""
import numpy as np
import random
import os
from shutil import copyfile
from gym import spaces


def check_space(space):
    """ Check the properties of an environment state or action space """
    if isinstance(space, spaces.Box):
        dim = space.shape
        discrete = False    
    elif isinstance(space, spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError('This type of space is not supported')
    return dim, discrete


def store_safely(to_store):
    """ to prevent losing information due to interruption of process"""
    working_dir = os.getcwd()
    folder = os.path.join(working_dir, 'bkp')
    os.makedirs(folder, exist_ok=True)
    new_name = os.path.join(folder, 'result.npy')
    old_name = os.path.join(folder, 'result_old.npy')
    if os.path.exists(new_name):
        copyfile(new_name, old_name)
    np.save(new_name, to_store)
    if os.path.exists(old_name):            
        os.remove(old_name)


# ----- Atari helpers -----------
    
def get_base_env(env):
    """ removes all wrappers """
    while hasattr(env, 'env'):
        env = env.env
    return env


def copy_atari_state(env):
    env = get_base_env(env)
    return env.clone_full_state()
#    return env.ale.cloneSystemState()


def restore_atari_state(env, snapshot):
    env = get_base_env(env)
    env.restore_full_state(snapshot)
#    env.ale.restoreSystemState(snapshot)


def is_atari_game(env):
    ''' Verify whether game uses the Arcade Learning Environment '''
    env = get_base_env(env)
    return hasattr(env, 'ale')


# ----- Visualization ----------

def symmetric_remove(x, n):
    ''' removes n items from beginning and end '''
    odd = is_odd(n)
    half = int(n/2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x

def is_odd(number):
    ''' checks whether number is odd, returns boolean '''
    return bool(number & 1)

def smooth(y, window, mode):
    ''' smooth 1D vectory y '''    
    return np.convolve(y, np.ones(window)/window, mode=mode)