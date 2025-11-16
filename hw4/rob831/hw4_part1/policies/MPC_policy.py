import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            random_action_sequences = np.random.uniform(
                self.low, self.high,
                size=(num_sequences, horizon, self.ac_dim)
            )
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            elite_mean = None
            elite_std = None

            for i in range(self.cem_iterations):
                if i == 0:
                    candidate_action_sequences = np.random.uniform(
                        self.low, self.high,
                        size=(num_sequences, horizon, self.ac_dim)
                    )
                else:
                    candidate_action_sequences = np.random.normal(
                        elite_mean, elite_std,
                        size=(num_sequences, horizon, self.ac_dim)
                    )
                    candidate_action_sequences = np.clip(
                        candidate_action_sequences, self.low, self.high
                    )

                predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
                elite_indices = np.argsort(predicted_rewards)[-self.cem_num_elites:]
                elites = candidate_action_sequences[elite_indices]

                new_elite_mean = np.mean(elites, axis=0)
                new_elite_std = np.std(elites, axis=0)

                if i == 0:
                    elite_mean = new_elite_mean
                    elite_std = new_elite_std
                else:
                    elite_mean = self.cem_alpha * new_elite_mean + (1 - self.cem_alpha) * elite_mean
                    elite_std = self.cem_alpha * new_elite_std + (1 - self.cem_alpha) * elite_std

            cem_action = elite_mean

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        sum_of_rewards_per_model = []
        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            sum_of_rewards_per_model.append(sum_of_rewards)

        mean_sum_of_rewards = np.mean(sum_of_rewards_per_model, axis=0)
        return mean_sum_of_rewards

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            best_index = np.argmax(predicted_rewards)
            best_action_sequence = candidate_action_sequences[best_index]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N = candidate_action_sequences.shape[0]
        sum_of_rewards = np.zeros(N)

        current_obs = np.tile(obs, (N, 1))

        for t in range(self.horizon):
            actions = candidate_action_sequences[:, t, :]
            rewards, _ = self.env.get_reward(current_obs, actions)
            sum_of_rewards += rewards
            current_obs = model.get_prediction(current_obs, actions, self.data_statistics)

        return sum_of_rewards
