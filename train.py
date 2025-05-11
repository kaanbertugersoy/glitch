import gymnasium as gym
import numpy as np
import sys
from datetime import datetime
from tqdm import tqdm

from glitch.buffer import ReplayBuffer
from glitch.nets.glitchNet import GlitchNet
from glitch.transformer import ImageTransformer
from glitch.config import Config
from glitch.episode import play_one


if __name__ == '__main__':

    config = Config()

    # Create the environment
    env = gym.make("ALE/Breakout-v5")
    print("Observation space:", env.observation_space)
    act_dim = env.action_space.n  # discrete action space

    # Instantiate the experience replay buffer
    replay_buffer = ReplayBuffer(
        frame_width=config.IMAGE_DIM, frame_height=config.IMAGE_DIM, size=config.MAX_BUFFER_SIZE)

    # Instantiate the models
    model = GlitchNet(K=act_dim)
    target_model = GlitchNet(K=act_dim)

    # Instantiate the image transformer
    transformer = ImageTransformer(
        config.IMAGE_DIM, config.IMAGE_RESIZE_MODE, config.IMAGE_CHANNEL_MODE)

    # Start populating the experience replay buffer
    print("Populating experience replay buffer...")
    s, _ = env.reset()
    s_trans = transformer.transform(s)

    for i in tqdm(range(config.MIN_BUFFER_SIZE), desc="Populating Buffer", unit="frames"):

        a = env.action_space.sample()
        s2, r, terminated, truncated, _ = env.step(a)
        done = int(terminated or truncated)
        s2_trans = transformer.transform(s2)
        # next_state = update_state(state, s2_trans)

        replay_buffer.store(s=s_trans, a=a,
                            s2=s2_trans, r=r, done=done)

        s_trans = s2_trans
        if done:
            s, _ = env.reset()
            s_trans = transformer.transform(s)

    print("Buffer filled with enough experience. Starting training.")

    # Train the model
    eps = 1.0
    total_t = 0
    episode_rewards = np.zeros(config.NUM_EPISODES)
    t0 = datetime.now()
    for i in range(config.NUM_EPISODES):
        total_t, episode_reward, duration, num_steps_in_episode, time_per_step, eps, avg_loss_of_episode = play_one(
            env, total_t, replay_buffer, model, target_model, transformer, config.GAMMA, config.BATCH_SIZE, eps, config.EPS_DECAY, config.EPS_MIN)
        episode_rewards[i] = episode_reward

        last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
        print(f"Episode: {i} | Duration: {duration} | Steps: {num_steps_in_episode} | Reward: {episode_reward} | Training time/Step: {time_per_step:.3f} | Avg Reward (100): {last_100_avg:.3f} | Epsilon: {eps:.3f} | Avg Loss: {avg_loss_of_episode:.3f}")
        sys.stdout.flush()
    print("Total duration:", datetime.now() - t0)

    # At the end of training, save the model
    model.save()
    print("Model saved.")
    env.close()
