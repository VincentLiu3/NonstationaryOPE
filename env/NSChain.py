import numpy as np


class NSChain():
    def __init__(self):
        self.num_state = 3
        self.num_action = 2
        self.horizon = 10

        speed = 0.25
        np.random.seed(1687)
        self.mean_line = 0.5
        self.amplitude = np.random.uniform(0.2, 0.5, (self.num_state, self.num_action))
        self.frequency = np.random.uniform(0.1, 1.0, (self.num_state, self.num_action)) * speed
        self.stds = np.random.uniform(0.0, 0.01, (self.num_state, self.num_action))

        self.round_id = 0
        self.state = None
        self.t = None
        self.reward_vector = None

    def next_round(self):
        self.round_id += 1
        np.random.seed(1687 + self.round_id)
        print('environment round {}'.format(self.round_id))
        ns_reward = self.mean_line + self.amplitude * np.sin(self.round_id * self.frequency) + np.random.randn() * self.stds
        self.reward_vector = ns_reward.clip(min=0, max=1)

    def reset(self):
        assert self.reward_vector is not None
        self.state = 0
        self.t = 0
        return self.state

    def step(self, action):
        if action == 0:
            new_state = 1
        elif action == 1:
            new_state = 2
        else:
            raise ValueError('action must be 0 or 1.')

        r = self.reward_vector[self.state, action]
        self.t += 1
        if self.t == self.horizon:
            done = True
        else:
            done = False

        self.state = new_state
        return self.state, r, done, {}

    def evaluate_policy(self, target_policy):
        # target_policy has shape (num_state, num_action)
        visitation_prob = np.zeros([self.num_state])

        P = np.hstack([np.zeros([self.num_state, 1]), target_policy])
        mu = np.zeros([self.num_state])
        mu[0] = 1.0
        for h in range(self.horizon):
            visitation_prob += mu
            mu = np.matmul(mu, P)

        target_prob = np.expand_dims(visitation_prob, axis=1) * target_policy
        vpi = (target_prob * self.reward_vector).sum()
        return vpi


class NSBanditTree():
    def __init__(self):
        self.num_action = 2
        self.horizon = 5
        self.num_state = self.num_action**self.horizon - 1
        # np.random.seed(1687)
        self.rng = np.random.RandomState(1687)  # random seed for this env

        speed = 0.25
        self.mean_line = self.rng.uniform(0.25, 0.75, (self.num_state, self.num_action))
        self.amplitude = 0.25  # np.random.uniform(0.1, 0.25, (self.num_state, self.num_action))
        self.frequency = self.rng.uniform(0.1, 1.0, (self.num_state, self.num_action)) * speed
        self.stds = self.rng.uniform(0.0, 0.01, (self.num_state, self.num_action))

        self.round_id = 0
        self.base_state = None
        self.state = None
        self.t = None
        self.reward_vector = None

    def next_round(self):
        self.round_id += 1
        # np.random.seed(1687 + self.round_id)
        print('environment round {}'.format(self.round_id))
        ns_reward = self.mean_line + self.amplitude * np.sin(self.round_id * self.frequency) + self.rng.randn() * self.stds
        self.reward_vector = ns_reward.clip(min=0, max=1)

    def reset(self):
        assert self.reward_vector is not None
        self.base_state = 0
        self.state = 0
        self.t = 0
        return self.state

    def step(self, action):
        self.t += 1

        base_state = 2**self.t - 1
        increment_state = 2 * (self.state - self.base_state)
        if action == 0:
            new_state = base_state + increment_state + 0
        elif action == 1:
            new_state = base_state + increment_state + 1
        else:
            raise ValueError('action must be 0 or 1.')

        # print(self.t, self.state, base_state, increment_state, action, new_state)
        r = self.reward_vector[self.state, action]
        if self.t == self.horizon:
            done = True
        else:
            done = False

        self.base_state = base_state
        self.state = new_state
        return self.state, r, done, {}

    def evaluate_policy(self, target_policy):
        # TODO
        # target_policy has shape (num_state, num_action)
        visitation_prob = np.zeros([self.num_state])

        P = np.hstack([np.zeros([self.num_state, 1]), target_policy])
        mu = np.zeros([self.num_state])
        mu[0] = 1.0
        for h in range(self.horizon):
            visitation_prob += mu
            mu = np.matmul(mu, P)

        target_prob = np.expand_dims(visitation_prob, axis=1) * target_policy
        vpi = (target_prob * self.reward_vector).sum()
        return vpi


class Agent():
    def __init__(self, num_state, num_action):
        self.num_state = num_state
        self.num_action = num_action
        self.Q_value = None
        self.alpha = None

    def reset(self):
        self.Q_value = np.ones([self.num_state, self.num_action])
        H = int(np.log2(self.num_state+1))
        for i in range(H):
            self.Q_value[self.num_action ** i - 1:self.num_action ** (i + 1) - 1] = H - i
        self.alpha = 0.1

    def step(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.Q_value[next_state, :].max()
        td_error = target - self.Q_value[state, action]
        self.Q_value[state, action] += self.alpha * td_error

    def pick_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            a = np.random.choice(self.num_action)
        else:
            a = self.Q_value[state, :].argmax()
        return a

    def random_policy(self, state):
        a = np.random.choice(self.num_action)
        prob = 1 / self.num_action
        return a, prob


class NSRLEnv():
    def __init__(self, num_round):
        self.num_round = num_round
        self.target_policy = None
        self.dir_name = 'chain'

    def generate_data(self, num_round, temp):
        # generate target policy
        # env = NSChain()
        print('generating data.')
        self.num_round = num_round
        env = NSBanditTree()
        agent = Agent(env.num_state, env.num_action)
        agent.reset()
        num_episode_per_round = 10
        epsilon_schedule = 1.0
        for round_id in range(self.num_round):
            env.next_round()
            for episode_id in range(num_episode_per_round):
                state = env.reset()
                for t in range(env.horizon):
                    action = agent.pick_action(state, epsilon=epsilon_schedule)
                    next_state, reward, done, info = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    epsilon_schedule = max(epsilon_schedule - 1/20/env.horizon, 0.1)

        self.target_policy = np.exp(temp * agent.Q_value) / np.sum(np.exp(temp * agent.Q_value), axis=1, keepdims=True)
        np.save('data/{}/target_policy'.format(self.dir_name), self.target_policy)
        np.save('data/{}/Q_value'.format(self.dir_name), agent.Q_value)

    def read_data(self):
        print('read data')
        self.target_policy = np.load('data/{}/target_policy.npy'.format(self.dir_name), allow_pickle=True)

    def sample_data(self, sample_size, data_name, fix_seed):
        if self.target_policy is None:
            self.read_data()

        # env = NSChainEnv()
        env = NSBanditTree()
        agent = Agent(env.num_state, env.num_action)
        agent.reset()

        data_list = []
        for round_id in range(self.num_round):
            print('sample data round {}'.format(round_id))
            np.random.seed(fix_seed + round_id)

            env.next_round()
            # np.random.seed(fix_seed + round_id)
            # ns_reward = self.all_reward[round_id]
            # env.set_reward(ns_reward)
            sample_trajectory = []
            sample_feedbacks = np.zeros([sample_size])
            sample_pb = np.zeros([sample_size])
            sample_pt = np.zeros([sample_size])
            for episode_id in range(sample_size):
                trajectory = []
                state = env.reset()
                done = False
                epi_return = 0
                pb_ratio = 1.0
                pt_ratio = 1.0
                while done is False:
                    action, pb = agent.random_policy(state)
                    next_state, reward, done, info = env.step(action)
                    pb_ratio *= pb
                    pt = self.target_policy[state, action]
                    pt_ratio *= pt
                    trajectory.append((state, action, pb, pt, reward))
                    state = next_state
                    epi_return += reward

                sample_trajectory.append(np.array(trajectory))
                sample_feedbacks[episode_id] = epi_return
                sample_pb[episode_id] = pb_ratio
                sample_pt[episode_id] = pt_ratio

            # vpi = env.evaluate_policy(self.target_policy)
            # check if vpi is correct
            total_return = 0
            for _ in range(1000):
                state = env.reset()
                done = False
                epi_return = 0
                while done is False:
                    action = np.random.choice(env.num_action, p=self.target_policy[state])
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    epi_return += reward
                total_return += epi_return
            vpi = total_return/1000
            print(total_return/1000, vpi)

            data_dict = {
                'round_id': round_id,
                'target_policy': self.target_policy,
                'vpi': vpi,
                'trajectory': sample_trajectory,
                'feedbacks': sample_feedbacks,
                'pt': sample_pt,
                'pb': sample_pb,
                'n': sample_size,
                'num_state': env.num_state,
                'num_action': env.num_action
            }
            data_list.append(data_dict)

        # np.save('data/{}/ns_sample_data'.format(self.dir_name), data_list)
        np.save(data_name, data_list)
