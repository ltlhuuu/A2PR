import gym
import d4rl
import numpy as np
import imageio

fps = 60


def eval_policy(
    policy,
    env_name,
    seed,
    mean=0,
    std=1,
    seed_offset=100,
    eval_episodes=10,
    save_gif=False,
    video_path=None,
):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    ep_obs = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            if save_gif and _ < 0:
            # if save_gif:
                obs = eval_env.render(mode="rgb_array")
                ep_obs.append(obs)
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            # action = policy.eval_step(state)
            state, reward, done, __ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    if save_gif:
        # with imageio.get_writer(video_path, duration=1000*1/fps) as writer:
        with imageio.get_writer(video_path, fps=fps) as writer:
            for obs in ep_obs:
                writer.append_data(obs)
    return avg_reward, d4rl_score
