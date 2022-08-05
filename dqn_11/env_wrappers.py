
import gym


def make_env(env_id, render_mode="rgb_array", new_step_api=True):
    # game holds the env
    # env = gym.make("ALE/Breakout-v5",
    env = gym.make(env_id,
                   #    render_mode="human",  # or rgb_array
                   render_mode=render_mode,  # or human
                   new_step_api=new_step_api)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)

    return env
