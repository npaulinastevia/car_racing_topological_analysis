import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
#import pyvirtualdisplay
#import reverb

import tensorflow as tf
import gym
from tf_agents.agents.ppo import ppo_agent, ppo_policy
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.utils import common
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
#hyperparameters
num_iterations = 15000 # @param {type:"integer"}

#initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1000  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}
conv_params=[(16, 8, 4), (32, 3, 2)],
in_fc_params=(256,)
out_fc_params=(128,)
lstm_size=(256,256)
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

#loading env
gym.envs.register(
     id='MyCar-v0',
     entry_point='envs:CarRacing',
     max_episode_steps=200,
)

#env=train_py_env = suite_gym.load('MyCar-v0')
train_py_env = suite_gym.load('MyCar-v0')
eval_py_env = suite_gym.load('MyCar-v0')
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#obs_spec = train_env.observation_spec()
#neural network create
observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(train_env))

#actor_net = ActorDistributionNetwork(obs_spec, train_env.action_spec(),
#                                     fc_layer_params=(200, 100),
#                                     activation_fn=tf.keras.activations.tanh)

#value_net = ValueNetwork(obs_spec)

actor_net = ActorDistributionRnnNetwork(observation_tensor_spec, action_tensor_spec,activation_fn=tf.keras.activations.tanh,lstm_size=lstm_size)
        # Define value network eval_py_env
value_net = ValueRnnNetwork(observation_tensor_spec,lstm_size=lstm_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
#ts_spec =train_env.time_step_spec()

#ime_step_spec = TimeStep(discount=ts_spec.discount,
#                          observation=obs_spec,
#                          reward=ts_spec.reward,
#                          step_type=ts_spec.step_type)
agent = ppo_agent.PPOAgent(time_step_tensor_spec, action_tensor_spec,
                           actor_net=actor_net, value_net=value_net,
                           optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate))

agent.initialize()
eval_policy = agent.policy
collect_policy = agent.collect_policy
def compute_avg_return_coord_track(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):
    policy_state = policy.get_initial_state(train_env.batch_size)
    time_step = environment.reset()

    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step,policy_state)
      time_step = environment.step(action_step.action)
      print(environment.pyenv.get_info())
      episode_return += time_step.reward
      policy_state = action_step.state
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

# Rplay buffer



replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)
def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

# Train agent

# Dataset generates trajectories with shape [Bx2x...]
#dataset = replay_buffer.as_dataset(
#    num_parallel_calls=3,
 #   sample_batch_size=batch_size,
 #   num_steps=2).prefetch(3)
#iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training. time_step = self.train_env.current_time_step()
avg_return = compute_avg_return_coord_track(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_env.reset()


for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  #experience, unused_info = next(iterator)
  experience=replay_buffer.gather_all()
  train_loss = agent.train(experience)

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return_coord_track(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    returns.append(avg_return)

