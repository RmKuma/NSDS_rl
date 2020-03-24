from stable_baselines import SAC, PPO2
#from custom_ppo import CPPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from environment.env import NetwEnv
from stable_baselines.common.vec_env import SubprocVecEnv
import tensorflow as tf

if __name__ == "__main__":
    #env = [ lambda : NetwEnv(8, serverPort=(5556+i)) for i in range(16)]
    env = [  lambda: NetwEnv(8, serverPort=5556)] + [lambda:  NetwEnv(8, serverPort=5557)]+ [lambda:  NetwEnv(8, serverPort=5558)]+ [lambda:  NetwEnv(8, serverPort=5559)]+ [lambda:  NetwEnv(8, serverPort=5560)]+ [lambda:  NetwEnv(8, serverPort=5561)]+ [lambda:  NetwEnv(8, serverPort=5562)]+ [lambda:  NetwEnv(8, serverPort=5563)]+ [lambda:  NetwEnv(8, serverPort=5564)]+ [lambda:  NetwEnv(8, serverPort=5565)]+ [lambda:  NetwEnv(8, serverPort=5566)]+ [lambda:  NetwEnv(8, serverPort=5567)]+ [lambda:  NetwEnv(8, serverPort=5568)]+ [lambda:  NetwEnv(8, serverPort=5569)]+ [lambda:  NetwEnv(8, serverPort=5570)]+ [lambda:  NetwEnv(8, serverPort=5571)]
    #env = [lambda : NetwEnv(8, serverPort=5556)]
    env = SubprocVecEnv(env)
    agent = PPO2(MlpPolicy, env, tensorboard_log="./log", verbose=1, n_steps=64, policy_kwargs={"act_fun":tf.nn.relu})

    #env = NetwEnv(20, serverPort=5556)
    #agent = SAC(LnMlpPolicy, env, buffer_size=100000, tensorboard_log='./log', verbose=1)
    agent.learn(total_timesteps=10000000, log_interval=1)

    agent.save('./model/PPO_model')
