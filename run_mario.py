import numpy as np
import torch
from core.constants import PRETRAINED_MODELS
from core.model import CNNDQN
from core.wrappers import wrap_environment
from os.path import join
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT 
import time

def test_with_render(environment, action_space, iteration):
    flag = False
    env = wrap_environment(environment, action_space, monitor=True,
                           iteration=iteration)
    net = CNNDQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(join(PRETRAINED_MODELS,
                                        '%s-powerfull.dat' % environment),
                                  map_location=torch.device('cpu')))

    total_reward = 0.0
    state = env.reset()
    try:
        while True:
            env.render()  # Display the game window
            time.sleep(0.1)  # Add a delay of 0.1 seconds to slow down the video
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if info['flag_get']:
                print('WE GOT THE FLAG!!!!!!!')
                flag = True
            if done:
                print(f"Total reward: {total_reward}")
                break
    finally:
        env.close()  # Make sure to close the environment
    return flag

# Run Mario on World 1-2
environment = 'SuperMarioBros-1-1-v0'
iteration = 0  # This will create a new recording folder 'run0'

# Run the test function which will load the model and play the game
flag = test_with_render(environment, COMPLEX_MOVEMENT, iteration)
if flag:
    print("Mario completed the level!")