from collections import defaultdict
import gymnasium as gym
import numpy as np


class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        bin_size: int = 10
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.bin_size = bin_size
        self.bins = self._create_bins_()


        self.training_error = []

    def get_action(self, obs: tuple[float, float, float, float]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            binned_obs = self._discretize_state_(obs, self.bins)
            return int(np.argmax(self.q_values[binned_obs]))
        
    def _create_bins_(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_bins = np.linspace(-4.8, 4.8, self.bin_size)
        vel_bins = np.linspace(-5, 5, self.bin_size)
        ang_bins = np.linspace(-6.28, 6.28, 2*self.bin_size) 
        ang_vel_bins = np.linspace(-5, 5, self.bin_size)
        return (pos_bins, vel_bins, ang_bins, ang_vel_bins)
            

    def _discretize_state_(self, state: tuple[float, float, float, float], bins: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        indices = []
        for value, bin_edges in zip(state, bins):
            bin_idx = np.digitize(value, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
            indices.append(bin_idx)
        
        return tuple(indices)

    def update(
        self,
        obs: tuple[float, float, float, float],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[float, float, float, float],
    ):
        """Updates the Q-value of an action."""

        # Try to force termination
        if abs(next_obs[2]) > 0.48:
            print("SHOULD TERMINATE HERE")
            terminated = True
            reward = -1.0

        binned_next_obs = self._discretize_state_(next_obs, self.bins)
        binned_obs = self._discretize_state_(obs, self.bins)
        future_q_value = (not terminated) * np.max(self.q_values[binned_next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[binned_obs][action]
        )

        self.q_values[binned_obs][action] = (
            self.q_values[binned_obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


#########################TRAINING
# hyperparameters
learning_rate = 0.1
n_episodes = 10_001
start_epsilon = 1.0
epsilon_decay = .995 # reduce the exploration over time
final_epsilon = .2

env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = CartPoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    bin_size=20
)

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # Only create render env when we want to watch
    if episode % 2000 == 0:
        render_env = gym.make("CartPole-v1", render_mode="human")
        render_obs, _ = render_env.reset()
        print(f"\nEpisode {episode}")


        
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if episode % 2000 == 0:
            render_env.step(action)
            if terminated:
                print("Episode terminated!")
                render_env.close()  # Close render window immediately
                break  # Exit the loop

            
        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()


from matplotlib import pyplot as plt
# visualize the episode rewards, episode length and training error in one figure
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# np.convolve will compute the rolling mean for 100 episodes

axs[0].plot(np.convolve(env.return_queue, np.ones(100)/100, mode='valid'))
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(np.convolve(env.return_queue, np.ones(100)/100, mode='valid'))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(env.return_queue, np.ones(100)/100, mode='valid'))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout()
plt.show()