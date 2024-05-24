import numpy as np
import time
import os
import d4rl

from utils.eval import eval_policy
from utils.config import get_config, save_config
from utils.logger import get_logger, get_writer
from utils.buffer import ReplayBuffer
from A2PR import A2PR
from utils import utils_spot


if __name__ == "__main__":
    start_time = time.time()

    # the result save path
    out = "./result"
    os.makedirs(out, exist_ok=True)

    # Introduce the information of this run
    info = 'A2PR'
    args, env, kwargs = get_config(info)

    # add the env save dir
    env_save_dir = os.path.join(out, args.env_id)
    os.makedirs(env_save_dir, exist_ok=True)

    result_dir = os.path.join(
        env_save_dir,
        time.strftime("%m-%d-%H:%M:%S")
        + "_"
        + args.policy
        + "_"
        + args.env_id
        + "-seed"
        + str(args.seed)
        + "-vae_weight"
        + str(args.vae_weight)
        + "-mask"
        + str(args.mask),
    )

    writer = get_writer(result_dir)

    file_name = f"{args.policy}_{args.env_id}_{args.seed}"
    logger = get_logger(os.path.join(result_dir, file_name + ".log"))
    logger.info(
        f"Policy: {args.policy}, Env: {args.env_id}, Seed: {args.seed}, Info: {args.info}"
    )

    # save configs
    save_config(args, os.path.join(result_dir, "config.txt"))

    # save src
    utils_spot.snapshot_src('.', os.path.join(result_dir, 'src'), '.gitignore')

    # load model
    if args.load_model != "default":
        model_name = args.load_model
    else:
        model_name = file_name
    ckpt_dir = os.path.join(result_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, model_name + ".pth")
    replay_buffer = ReplayBuffer(kwargs["state_dim"], kwargs["action_dim"], args.device, args.env_id, args.scale, args.shift)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    states = replay_buffer.state
    actions = replay_buffer.action

    # define policy
    policy = A2PR(**kwargs)
    evaluations = []
    evaluation_path = os.path.join(result_dir, file_name + ".npy")
    if os.path.exists(model_path):
        policy.load(model_path)
    for t in range(int(args.max_timesteps)):
        result = policy.train(replay_buffer, args.batch_size)
        for key, value in result.items():
            writer.add_scalar(key, value, global_step=t)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            model_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".pth")
            video_path = os.path.join(ckpt_dir, model_name + "_" + str(t + 1) + ".gif")

            if args.save_model and (t + 1) % args.save_model_freq == 0:
                avg_reward, d4rl_score = eval_policy(
                    policy,
                    args.env_id,
                    args.seed,
                    mean,
                    std,
                    save_gif=False,
                    video_path=video_path,
                )
                policy.save(model_path)
            else:
                avg_reward, d4rl_score = eval_policy(
                    policy, args.env_id, args.seed, mean, std
                )
            writer.add_scalar("avg_reward", avg_reward, global_step=t)
            writer.add_scalar("d4rl_score", d4rl_score, global_step=t)
            evaluations.append(d4rl_score)
            logger.info("---------------------------------------")
            logger.info(f"Time steps: {t + 1}, D4RL score: {d4rl_score}")

    np.save(evaluation_path, evaluations)
    end_time = time.time()
    logger.info(f"Total Time: {end_time - start_time}")
