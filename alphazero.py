"""
Adapted from:
https://github.com/tmoer/alphazero_singleplayer
https://tmoer.github.io/AlphaZero/
"""

import numpy as np

import argparse
import os
import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim

from teaching.env import TeachingEnv

from helpers import check_space, is_atari_game, copy_atari_state, \
    store_safely,  \
    restore_atari_state, smooth, symmetric_remove
from rl.make_game import make_game

from tqdm import tqdm

sns.set()


class Model(nn.Module):
    """
    Neural network for policy and value
    """
    
    def __init__(self, env, lr, n_hidden_layers, n_hidden_units):
        super().__init__()
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(env.action_space)
        self.state_dim, self.state_discrete = check_space(env.observation_space)

        if not self.action_discrete: 
            raise ValueError('Continuous action space not implemented')

        if len(self.state_dim) != 1:
            raise ValueError(f"`len(stade_dim)` is {len(self.state_dim)} but should be one")

        self.base_nn = nn.Sequential(*[
            nn.Linear(self.state_dim[0], n_hidden_units),
            nn.ReLU(), ] + [
                nn.Linear(n_hidden_units, n_hidden_units),
                nn.ReLU(), ] * n_hidden_layers)

        self.pi_hat = nn.Linear(n_hidden_units, self.action_dim)
        self.v_hat = nn.Linear(n_hidden_units, 1)

        self.v_loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def pi_loss(self, input, target):

        log_pi_hat = self.log_softmax(input)
        return torch.mean(-torch.sum(target * log_pi_hat, dim=-1))

    def train_on_example(self, sb, vb, pib):
        if isinstance(sb, np.ndarray):
            sb = torch.from_numpy(sb.astype(np.float32))
        if isinstance(vb, np.ndarray):
            vb = torch.from_numpy(vb.astype(np.float32))
        if isinstance(pib, np.ndarray):
            pib = torch.from_numpy(pib.astype(np.float32))
        x = self.base_nn(sb)
        v_hat = self.v_hat(x)
        raw_pi_hat = self.pi_hat(x)
        v_loss = self.v_loss(target=vb, input=v_hat)
        pi_loss = self.pi_loss(target=pib, input=raw_pi_hat)
        loss = v_loss + torch.mean(pi_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def predict_v(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s.astype(np.float32))
        with torch.no_grad():
            x = self.base_nn(s)
            v = self.v_hat(x)
            return v.numpy()
        
    def predict_pi(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s.astype(np.float32))
        with torch.no_grad():
            x = self.base_nn(s)
            pi = self.softmax(self.pi_hat(x))
            return pi.numpy()


# ----- MCTS functions -----
class Action:
    ''' Action object '''
    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init
                
    def add_child_state(self, s1, r, terminal, model):
        self.child_state = State(s1, r, terminal, self,
                                 self.parent_state.na, model)
        return self.child_state
        
    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W/self.n


class State:
    """
    State object
    """

    def __init__(self, index, r, terminal, parent_action, na, model):
        ''' Initialize a new state '''
        self.index = index        # state
        self.r = r                # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        
        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a, parent_state=self, Q_init=self.V) for a in range(na)]
        self.priors = model.predict_pi(index[None, ]).flatten()
    
    def select(self, c=1.5):
        """ Select one of the child actions based on UCT rule """
        n = len(self.child_actions)
        uct = np.zeros(n)
        for i in range(n):
            ca, prior = self.child_actions[i], self.priors[i]
            uct[i] = ca.Q + prior * c * (np.sqrt(self.n)/(ca.n + 1))
        winner = np.nanargmax(uct)   # is is possible to have nan here?
        return self.child_actions[winner]

    def evaluate(self):
        """ Bootstrap the state value """
        self.V = np.squeeze(self.model.predict_v(self.index[None,])) if not self.terminal else np.array(0.0)

    def update(self):
        """ Update count on backward pass """
        self.n += 1


class MCTS:
    """
    MCTS object
    """

    def __init__(self, root, root_index, model, na, gamma):
        self.root = root
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
    
    def search(self, n_mcts, c, Env, mcts_env):
        """
        Perform the MCTS search from the root
        """
        if self.root is None:
            self.root = State(self.root_index, r=0.0, terminal=False,
                              parent_action=None, na=self.na, model=self.model)  # initialize new root
        else:
            self.root.parent_action = None # continue from current root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env) # for Atari: snapshot the root at the beginning     
        
        for i in range(n_mcts):     
            state = self.root # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)
            
            while not state.terminal: 
                action = state.select(c=c)
                s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state   # select
                    continue
                else:
                    state = action.add_child_state(s1, r, t, self.model) # expand
                    break

            # Back-up 
            R = state.V         
            while state.parent_action is not None:   # loop back-up until root is reached
                R = state.r + self.gamma * R 
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()                
    
    def return_results(self, temp):
        """ Process the output at the root node """
        n = len(self.root.child_actions)
        counts = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            ca = self.root.child_actions[i]
            counts[i] = ca.n
            Q[i] = ca.Q

        pi_target = self.stable_normalizer(counts, temp)
        v_target = np.sum((counts/np.sum(counts))*Q)[None, ]
        return self.root.index, pi_target, v_target

    @staticmethod
    def stable_normalizer(x, temp):
        """ Computes x[i]**temp/sum_i(x[i]**temp) """
        x = (x / np.max(x))**temp
        return np.abs(x/np.sum(x))
    
    def forward(self, a, s1):
        """ Move the root forward """
        if not hasattr(self.root.child_actions[a], 'child_state'):
            self.root = None
            self.root_index = s1
        # elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
        #     print('Warning: this domain seems stochastic. Not re-using the subtree for next search. '+
        #           'To deal with stochastic environments, implement progressive widening.')
        #     self.root = None
        #     self.root_index = s1
        else:
            self.root = self.root.child_actions[a].child_state


class ReplayBuffer:

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.clear()
        self.sample_array = None
        self.sample_index = 0

    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0

    def store(self, experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size += 1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self, *args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)

    def shuffle(self):
        self.sample_array = np.arange(self.size)
        np.random.shuffle(self.sample_array)
        self.sample_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_index + self.batch_size > self.size \
                and self.sample_index != 0:
            self.shuffle()  # Reset for the next epoch
            raise StopIteration

        if self.sample_index + 2 * self.batch_size > self.size:
            indices = self.sample_array[self.sample_index:]
        else:
            indices = self.sample_array[self.sample_index:
                                        self.sample_index + self.batch_size]
        batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size

        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add)
        return arrays


def train(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size,
          temp, n_hidden_layers, n_hidden_units):
    ''' Outer training loop '''

    episode_returns = []   # storage
    timepoints = []
    # Environments
    if game == "teaching":
        env = TeachingEnv()
    else:
        env = make_game(game)
    is_atari = is_atari_game(env)
    mcts_env = make_game(game) if is_atari else None

    replay_buffer = ReplayBuffer(max_size=data_size, batch_size=batch_size)
    model = Model(env=env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
    t_total = 0 # total steps   
    R_best = -np.Inf

    for ep in range(n_ep):
        start = time.time()
        s = env.reset()
        R = 0.0   # Total return counter
        a_store = []
        seed = np.random.randint(1e7)  # draw some Env seed
        env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        mcts = MCTS(root_index=s, root=None,
                    model=model,
                    na=model.action_dim,
                    gamma=gamma)   # the object responsible for MCTS searches

        if game == "teaching":
            iterator = tqdm(range(env.t_max))
        else:
            iterator = range(max_ep_len)
        for _ in iterator:
            # MCTS step
            mcts.search(n_mcts=n_mcts, c=c, Env=env, mcts_env=mcts_env)  # perform a forward search
            state, pi, v = mcts.return_results(temp)                     # extract the root output
            replay_buffer.store((state, v, pi))

            # Make the true step
            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)
            s1, r, terminal, _ = env.step(a)
            R += r
            t_total += n_mcts                                           # total number of environment steps (counts the mcts steps)

            if terminal:
                break
            else:
                mcts.forward(a, s1)

        # Finished episode
        episode_returns.append(R)  # store the total episode return
        timepoints.append(t_total)  # store the timestep count of the episode return
        store_safely({'R': episode_returns, 't': timepoints})

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        print(f'Finished episode {ep}, total return: {np.round(R,2)}, '
              f'total time: {time.time()-start:.1f} sec')
        # Train
        replay_buffer.shuffle()
        for sb, vb, pib in replay_buffer:
            model.train_on_example(sb=sb, vb=vb, pib=pib)
    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='CartPole-v0',help='Training environment')
    parser.add_argument('--n_ep', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=25, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=10000, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.0007, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.0, help='UCT constant')   # 1.5
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount parameter')   # 1.0
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    args = parser.parse_args()
    episode_returns, timepoints, a_best, seed_best, R_best = train(
        game=args.game, n_ep=args.n_ep, n_mcts=args.n_mcts,
        max_ep_len=args.max_ep_len, lr=args.lr, c=args.c, gamma=args.gamma,
        data_size=args.data_size, batch_size=args.batch_size, temp=args.temp,
        n_hidden_layers=args.n_hidden_layers,
        n_hidden_units=args.n_hidden_units)

    # Finished training: Visualize
    fig, ax = plt.subplots(1, figsize=[7, 5])
    total_eps = len(episode_returns)
    episode_returns = smooth(episode_returns, args.window, mode='valid')
    ax.plot(symmetric_remove(np.arange(total_eps), args.window-1),
            episode_returns, linewidth=4, color='darkred')
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode', color='darkred')
    plt.savefig(os.getcwd()+'/learning_curve.png', bbox_inches="tight", dpi=300)
    
#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
