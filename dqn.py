__author__ = 'Guillaume'

import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *

class GridWorld():
    def __init__(self, H, W, N1, N2, T, observable=None, gamma=1.,
                 background_value=0, agent_value=3, green_value=2, red_value=1, p=0.):
        self.H = H   # height
        self.W = W   # width
        self.N1 = N1 # nb of green blocks
        self.N2 = N2 # nb of red blocks
        if observable is None:
            self.observable = [H, W]
        else:
            self.observable = observable
        self.T = T   # episode length
        self.grid = np.zeros((H,W))
        self.agent_position = np.array([int(H/2), int(W/2)])
        self.background_value = background_value
        self.agent_value = agent_value
        self.red_value = red_value
        self.green_value = green_value
        self.count = 0
        self.gamma = gamma
        self.last_decision = "None"
        self.last_reward = None
        self.cum_reward = 0.
        self.p = p
        self.savename = None
        self.reset()

    def reset(self):
        self.grid[:,:] = self.background_value
        # Reset agent position
        self.agent_position = np.array([int(self.H/2), int(self.W/2)])
        # Add green blocks (randomly)
        for i in range(self.N1):
            self.add_block(self.green_value)
        # Add red blocks (randomly)
        for i in range(self.N2):
            self.add_block(self.red_value)
        # Reset counter
        self.count = 0
        self.cum_reward = 0.
        self.last_reward = None
        self.last_decision = "None"
        # Update the grid
        self.update_grid()

    def add_block(self, value):
        pos = np.random.randint(0, (self.H)*(self.W))
        pos = np.array([int(pos//(self.H)), int(pos % (self.H))])
        # If the position is not empty, try again
        if self.grid[pos[0], pos[1]] != self.background_value:
            return self.add_block(value)
        # Else update the grid
        self.grid[pos[0], pos[1]] = value

    def update_grid(self):
        '''Clear and update positions.'''
        # Update the agent position
        h, w = self.agent_position
        # If the agent found a green/red block, add a new one
        if self.grid[h, w] == self.red_value:
            self.add_block(self.red_value)
        elif self.grid[h, w] == self.green_value:
            self.add_block(self.green_value)
        # Update the grid to the new agent position
        self.grid[self.grid == self.agent_value] = self.background_value
        self.grid[h ,w] = self.agent_value

    def clip_positions(self):
        '''Clip position to make sure objects are not outside the grid.'''
        self.agent_position[0] = np.clip(self.agent_position[0], 0, self.H-1)
        self.agent_position[1] = np.clip(self.agent_position[1], 0, self.W-1)

    def get_reward(self):
        h, w = self.agent_position
        if self.grid[h, w] == self.red_value:
            return -1.0
        elif self.grid[h, w] == self.green_value:
            return +0.5
        else:
            return -0.1

    def update_from_decision(self, decision, verbose=False):
        '''Update the environment based on a decision'''
        possible_actions = ['left', 'right', 'up', 'down']
        if not type(decision) is str:
            if np.ndim(decision)==0:
                decision = possible_actions[decision]
            else: # one-hot
                decision = possible_actions[np.argmax(decision)]
        # Map 'string' action to 'numerical' ones
        if decision == 'left':
            u = [0, -1]
        elif decision == 'right':
            u = [0, +1]
        elif decision == 'down':
            u = [+1, 0]
        elif decision == "up":
            u = [-1, 0]
        else:
            u = [0, 0]
        # Update the position of the agent
        self.agent_position += u
        # Make sure that we did not get out the grid
        self.clip_positions()
        # Get reward
        R = self.get_reward()
        # Print (if asked)
        self.last_decision = decision
        self.last_reward = R
        self.cum_reward += (self.gamma**self.count) * R
        if verbose:
            print "Decision: %s - Reward %.1f"%(decision, R)
        # Clear the grid
        self.update_grid()
        # Increment
        self.count += 1
        return R

    def get_dropout_mask(self):
        m = np.random.randn(self.observable[0]*self.observable[1]).reshape(self.observable)
        return (m>self.p).astype("float32")

    def return_state(self):
        window = [int(self.observable[0]/2), int(self.observable[1]/2)]
        up = max(self.agent_position[0]-window[0],0)-max(0, self.agent_position[0]+window[0]-self.H + 1)  + (self.observable[0]+1)%2
        left = max(self.agent_position[1]-window[1],0)-max(0,self.agent_position[1]+window[1]-self.W + 1) + (self.observable[1]+1)%2
        state = np.array(self.grid[up:up+2*window[0]+self.observable[0]%2, left:left+2*window[1]+self.observable[1]%2], "uint8")
        mask = self.get_dropout_mask()
        return state*mask

    def is_terminal(self):
        return self.count>=self.T

    def visualize(self, fig=True, figsize=(8,8)):
        '''Plot the current grid.'''
        g = np.copy(self.grid)[:,:,None]
        rgb_grid = np.concatenate([g==self.red_value, g==self.green_value, g==self.agent_value], axis=2)
        rgb_grid = 255*rgb_grid.astype("uint8")
        if fig:
            plt.figure(figsize=figsize)
        plt.imshow(rgb_grid, interpolation="nearest")
        plt.scatter(self.agent_position[1], self.agent_position[0], marker="x", s=150, c="k")
        # White rows
        for i in range(self.H):
            plt.plot([-0.5, self.W+0.5], [i-0.5, i-0.5], c="w")
        # White columns
        for i in range(self.W):
            plt.plot([i-0.5, i-0.5], [-0.5, self.H+0.5], c="w")
        plt.title("Last decision: %s - Reward %s - Cum reward = %.2f - T = %d"%(self.last_decision, str(self.last_reward),
                                                                                self.cum_reward, self.count))
        # Visible area
        window = [int(self.observable[0]/2), int(self.observable[1]/2)]
        up = max(self.agent_position[0]-window[0],0)-max(0, self.agent_position[0]+window[0]-self.H + 1)  + (self.observable[0]+1)%2
        down = up+2*window[0] + self.observable[0]%2
        left = max(self.agent_position[1]-window[1],0)-max(0,self.agent_position[1]+window[1]-self.W + 1) + (self.observable[1]+1)%2
        right = left+2*window[1] + self.observable[1]%2
        plt.plot([left-0.5, right-0.5], [up-0.5, up-0.5], c="m")
        plt.plot([left-0.5, right-0.5], [down-0.5, down-0.5], c="m")
        plt.plot([left-0.5, left-0.5], [down-0.5, up-0.5], c="m")
        plt.plot([right-0.5, right-0.5], [down-0.5, up-0.5], c="m")
        plt.axis('off')
        plt.ylim([self.H+0.5, -0.5])
        if fig:
            plt.show()


def demo_GridWorld():
    i = 0
    np.random.seed(123)
    environment = GridWorld(7, 7, 4, 2, 50)
    decisions = ["up"] + 2*["right"] + 2*["down"] + 1*["right"]+2*["left"]
    while True:
        if i==8:
            i = 0
            np.random.seed(123)
            environment = GridWorld(7, 7, 4, 2, 50)
        environment.visualize()
        yield environment.return_state()
        environment.update_from_decision(decisions[i])
        i += 1


class ExperienceReplay():
    def __init__(self, state_shape, max_queue_size, dtype="uint8"):
        self.max_queue_size = max_queue_size
        # Initiate S and S' memories
        sh = tuple([max_queue_size] + list(state_shape))
        self.S_memory = np.zeros(sh, dtype=dtype)
        self.S_prime_memory = np.zeros(sh, dtype=dtype)
        # Action memory
        self.A_memory = np.zeros(max_queue_size, "uint8")
        # Reward memory
        self.R_memory = np.zeros(max_queue_size, "float32")
        # Terminal memory : 1 if terminal, 0 otherwise
        # We need to store this because the target is
        # not the same whether the state is terminal or not
        self.T_memory = np.zeros(max_queue_size, "uint8")
        # Index of the next interaction to store
        self.idx = 0
        # Is the memory full ?
        self.full = False

    def store(self, S, A, R, S_prime, terminal):
        # Store <S, A, R, S'>
        self.S_memory[self.idx] = S
        self.A_memory[self.idx] = A
        self.R_memory[self.idx] = R
        self.S_prime_memory[self.idx] = S_prime
        self.T_memory[self.idx] = terminal
        # Increment self.idx
        self.idx += 1
        # Check if the memory is full
        if self.idx == self.max_queue_size:
            self.full = True
            self.idx = 0 # we will overwrite oldest interactions first

    def sample(self, B):
        # Return the last stored interaction + B-1 other interactions taken uniformly in the memory
        U = (1-int(self.full))*self.idx + int(self.full)*self.max_queue_size
        assert U > 0, "No interaction has been stored yet."
        indices = [self.idx-1] + list(np.random.randint(0, U, B-1))
        return (self.S_memory[indices], self.A_memory[indices], self.R_memory[indices],
                self.S_prime_memory[indices], self.T_memory[indices])


class TemporalExperienceReplay():
    def __init__(self, T, state_shape, max_queue_size, dtype="uint8"):
        self.max_queue_size = max_queue_size
        # Initiate S and S' memories
        self.T = T
        sh = tuple([max_queue_size, T] + list(state_shape))
        self.S_memory = np.zeros(sh, dtype=dtype)
        self.S_prime_memory = np.zeros(sh, dtype=dtype)
        # Action memory
        self.A_memory = np.zeros((max_queue_size, T), "uint8")
        # Reward memory
        self.R_memory = np.zeros((max_queue_size, T), "float32")
        # Terminal memory : 1 if terminal, 0 otherwise
        # We need to store this because the target is
        # not the same whether the state is terminal or not
        self.T_memory = np.zeros((max_queue_size, T), "uint8")
        # Index of the next interaction to store
        self.idx = 0
        self.temporal_idx = 0
        # Sampling indices
        self.sampling_indices = None
        self.sampling_timestep = 0
        # Is the memory full ?
        self.full = False

    def store(self, S, A, R, S_prime, terminal):
        # Store <S, A, R, S'>
        self.S_memory[self.idx, self.temporal_idx] = S
        self.S_prime_memory[self.idx, self.temporal_idx] = S_prime
        self.A_memory[self.idx, self.temporal_idx] = A
        self.R_memory[self.idx, self.temporal_idx] = R
        self.T_memory[self.idx, self.temporal_idx] = terminal
        # Increment self.idx
        self.temporal_idx += 1
        if self.temporal_idx == self.T:
            self.temporal_idx = 0
            self.idx += 1
        # Check if the memory is full
        if self.idx == self.max_queue_size:
            self.full = True
            self.idx = 0 # we will overwrite oldest interactions first

    def sample(self, B):
        if self.sampling_timestep == 0:
            if self.idx == 0 and not self.full:
                self.sampling_indices = [0 for i in range(B)]
            else:
                U = (1-int(self.full))*self.idx + int(self.full)*self.max_queue_size
                assert U > 0, "No interaction has been stored yet."
                self.sampling_indices = [self.idx] + list(np.random.randint(0, U, B-1))

        out =  (self.S_memory[self.sampling_indices, self.sampling_timestep],
                self.A_memory[self.sampling_indices, self.sampling_timestep],
                self.R_memory[self.sampling_indices, self.sampling_timestep],
                self.S_prime_memory[self.sampling_indices, self.sampling_timestep],
                self.T_memory[self.sampling_indices, self.sampling_timestep])

        self.sampling_timestep += 1
        self.sampling_timestep = self.sampling_timestep%self.T

        return out


class DQNAgent():

    def __init__(self, environment, net, replay, batch_size, target_net=None, gamma=0.95,
                 epsilon=0.8, target_net_patience=10, nitermax=10000, nb_of_actions=4):
        self.E = environment
        self.DQN = net
        self.replay = replay
        self.B = batch_size
        if target_net is None:
            self.target_DQN = self.DQN
            self.use_target_net = False # no need to update the weights of the target net
            self.target_DQN_patience = 1
        else:
            self.target_DQN = target_net
            self.target_DQN.set_weights(net.get_weights())
            self.use_target_net = True
            self.target_DQN_patience = target_net_patience
        self.gamma = gamma
        self.epsilon = epsilon
        self.nitermax = nitermax
        self.A = nb_of_actions
        self.cum_rewards = []
        self.train_generator = self.qlearning_generator()
        self.test_generator = self.demo_generator()

    def qlearning_generator(self, replay=None, batch_size=None, verbose=True, seed=123):
        if replay is None:
            replay = self.replay
        if batch_size is None:
            batch_size = self.B
        # Repeat (for each episode):
        episode = 0
        patience = 0
        best = - np.Inf
        running_cum_reward = None
        # Fix the seed
        np.random.seed(seed)
        while True:
            self.E.reset() # New episode : reset the environment
            cum_reward = 0.
            patience = self.on_epoch_begin(patience)
            episode_start = time.time()
            # Initial state and action of episode
            S = self.E.return_state()
            A = np.random.randint(0, self.A)
            # Repeat (for each step of episode)
            for it in range(self.nitermax):
                # Take action A, observe R, S_prime
                R = self.E.update_from_decision(A)
                S_prime = self.E.return_state()
                # Store <A, S, Q, S'> in the memory
                replay.store(S, A, R, S_prime, 1-int(self.E.is_terminal()))

                ### Instead of using only <S,A,R,S'> to update theta, we will use
                ### a batch of interactions that have been previously experienced
                ### Ideally, it would be great to parallelize this code :
                ###    - one generator which interacts with the environment and
                ###      fills the memory queue
                ###    - one generator which reads the memory queue and yield batches
                ###      to the neural net

                # Randomly pick interactions seen previously
                S_batch, A_batch, R_batch, S_prime_batch, T_batch = replay.sample(batch_size)

                self.on_batch_start(it, S_batch)

                # Preprocess states and get Q(S,:)
                S_batch = self.preprocess(S_batch)
                # Preprocess states and get Q(S',:)
                S_prime_batch = self.preprocess(S_prime_batch)
                # Get target
                target, Q_prime_batch = self.compute_target_and_Qprime(S_batch, S_prime_batch, A_batch, R_batch, T_batch)

                # Yield for SGD update
                yield S_batch, np.array(target, "float32")

                ### Back to the current episode
                cum_reward += (self.gamma**it)*R

                # If terminal:
                if self.E.is_terminal():
                    episode_end = time.time()
                    if running_cum_reward is None:
                        running_cum_reward = cum_reward
                    else:
                        running_cum_reward = 0.98*running_cum_reward + 0.02*cum_reward
                    # If best, save weights
                    if running_cum_reward > best and self.savename is not None:
                        self.save_weights(self.savename)
                    if verbose:
                        print "\rEpisode %d - Running cum reward = %.3f took %.1fs."%(episode, running_cum_reward,
                                                                                     episode_end-episode_start),
                    self.cum_rewards.append(cum_reward)
                    break
                else:
                    # Choose A' as a function of Q(S',*,theta)
                    A_prime =  self.sample_next_action(Q_prime_batch[0], it)    # Assumes that the first row of
                                                                                # the batch contains the last experienced
                                                                                # interaction
                    # Update A, S, Q
                    A = A_prime
                    S = S_prime
            episode += 1

    def on_epoch_begin(self, patience):
        # Update the weights of 'target_net'
        if patience == self.target_DQN_patience:
            self.target_DQN.set_weights(self.DQN.get_weights())
            return 0
        else:
            return patience+1

    def on_batch_start(self, it, S):
        pass

    def preprocess(self, S_batch):
        return preprocess_grids(S_batch)

    def compute_target_and_Qprime(self, S_batch, S_prime_batch, A_batch, R_batch, T_batch):
        Q_batch = self.DQN.predict(S_batch)
        Q_prime_batch = self.target_DQN.predict(S_prime_batch)
        # Build the target such that:
        #      - target[selected action] = R + gamma*max(Q(S',:))
        #      - target[other actions] = Q(S,:) (i.e. the loss is 0 for non-selected actions)
        A_batch_oh = one_hot(A_batch, self.A)
        target = (R_batch + self.gamma*T_batch*np.max(Q_prime_batch, axis=1))[:,None]*A_batch_oh + Q_batch*(1.-A_batch_oh)
        return target, Q_prime_batch

    def sample_next_action(self, Q, it):
        Q = Q.flatten()
        return epsilon_greedy(Q, self.epsilon)

    def demo_generator(self):
        return self.qlearning_generator(replay=ExperienceReplay(self.E.return_state().shape, 1),
                                        batch_size=1, verbose=False)

    def demo(self, fig=True, figsize=(8,8)):
        self.savename = None
        batch, target_batch = next(self.test_generator)
        self.E.visualize(figsize=figsize, fig=fig)
        return batch, target_batch

    def fit(self, nb_update_per_epoch, nb_epoch, max_q_size=1, savename=None):
        self.savename = savename
        h = self.DQN.fit_generator(self.train_generator, nb_update_per_epoch*self.B, nb_epoch=nb_epoch, verbose=0,
                                   max_q_size=max_q_size)
        return h

    def save_weights(self, path):
        w = self.DQN.get_weights()
        np.save(path, w)

    def save_curve(self, path):
        np.save(path, self.cum_rewards)

    def load_weights(self, path):
        w = np.load(path)
        self.DQN.set_weights(w)
        self.target_DQN.set_weights(w)


class StatefulRDQNAgent(DQNAgent):

    def __init__(self, environment, net, net_copy, replay, batch_size, target_net, gamma=0.95, epsilon=0.8,
                 target_net_patience=10, nitermax=10000, nb_of_actions=4):
        self.T = replay.T
        self.DQN_copy = net_copy
        DQNAgent.__init__(self, environment, net, replay, batch_size, target_net, gamma,
                          epsilon, target_net_patience, nitermax, nb_of_actions)

    def compute_target_and_Qprime(self, S_batch, S_prime_batch, A_batch, R_batch, T_batch):
        Q_batch = self.DQN_copy.predict(S_batch)
        #print(Q_batch[0].flatten())
        Q_prime_batch = self.target_DQN.predict(S_prime_batch)
        # Build the target such that:
        #      - target[selected action] = R + gamma*max(Q(S',:))
        #      - target[other actions] = Q(S,:) (i.e. the loss is 0 for non-selected actions)
        A_batch_oh = one_hot(A_batch, self.A)[:,None,:]
        target = (R_batch[:,None] + self.gamma*T_batch[:,None]*np.max(Q_prime_batch, axis=2))[:,:,None]*A_batch_oh + Q_batch*(1.-A_batch_oh)
        return target, Q_prime_batch

    def demo_generator(self):
        return self.qlearning_generator(replay=self.replay,
                                        batch_size=self.B, verbose=False)

    def preprocess(self, S_batch):
        S = preprocess_grids(S_batch)[:, None, :]
        return S.reshape(self.B, 1, -1)

    def on_batch_start(self, it, S):
        # Reset states (stateful rnn)
        if it == 0:
            self.DQN.reset_states()
            self.DQN_copy.reset_states()
            self.target_DQN.reset_states()
            # Set the target net one step ahead
            #print(self.target_DQN.predict(self.preprocess(S))[0])

        # Update the weights of DQN_copy
        self.DQN_copy.set_weights(self.DQN.get_weights())

    def load_weights(self, path):
        w = np.load(path)
        self.DQN.set_weights(w)
        self.DQN_copy.set_weights(w)
        self.target_DQN.set_weights(w)
