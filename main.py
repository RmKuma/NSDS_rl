from stable_baselines import SAC, PPO2
#from custom_ppo import CPPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from environment.env import NetwEnv
from stable_baselines.common.vec_env import SubprocVecEnv
import tensorflow as tf
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.atari_wrappers import LazyFrames


if __name__ == "__main__":
    #env = [ lambda : NetwEnv(num_of_data=8, serverPort=(10056+i)) for i in range(16)]
    #env = [  lambda: NetwEnv(num_of_data=8, serverPort=10056,k=2)] + [lambda:  NetwEnv(num_of_data=8, serverPort=10057,k=2)]+ [lambda:  NetwEnv(num_of_data=8, serverPort=10058,k=2)]+ [lambda:  NetwEnv(num_of_data=8, serverPort=10059,k=2)]+ [lambda:  NetwEnv(num_of_data=8, serverPort=10060,k=2)]+ [lambda:  NetwEnv(num_of_data=8, serverPort=10061,k=2)]+ [lambda:  NetwEnv(num_of_data=8, serverPort=10062,k=2)]+ [lambda:  NetwEnv(num_of_data=8, serverPort=10063, k=2)]
    #env = SubprocVecEnv(env)
    #agent = PPO2(MlpPolicy, env, tensorboard_log="./log", verbose=1, n_steps=64, policy_kwargs={"act_fun":tf.nn.relu})

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./model/', name_prefix='SAC_model')
    env = NetwEnv(num_of_data=8, serverPort=10055, k=2)
    #env = FrameStack(env, 4)
    agent = SAC(LnMlpPolicy, env, 1, buffer_size=100000, tensorboard_log='./log', verbose=1)
    agent.learn(total_timesteps=10000000, log_interval=1, callback=checkpoint_callback)

    agent.save('./model/SAC_model')
