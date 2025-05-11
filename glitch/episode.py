from datetime import datetime
import numpy as np
from glitch.utils import share_weights
from glitch.config import Config

config = Config()


def play_one(env, total_t, replay_buffer, model, target_model, transformer, gamma, batch_size, eps, eps_decay, eps_min):

    t0 = datetime.now()

    # Reset the environment
    s, _ = env.reset()
    s_trans = transformer.transform(s)

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0
    avg_loss_of_episode = 0

    done = False
    while not done:

        # Update target network
        if total_t % config.TARGET_NETWORK_UPDATE_PERIOD == 0:
            share_weights(src=model, dst=target_model)
            print("Copied model parameters to target network | Total steps taken = %s" % (
                total_t))

        # Take action
        s_trans_batched = np.expand_dims(s_trans, axis=0)
        a = model.sample_action(s_trans_batched, eps)
        s2, r, terminated, truncated, _ = env.step(a)
        done = int(terminated or truncated)

        # downsample and convert to grayscale
        s2_trans = transformer.transform(s2)
        # next_state = update_state(state, s2_trans)

        # Save the latest experience
        replay_buffer.store(s=s_trans, a=a,
                            s2=s2_trans, r=r, done=done)

        # Train the model, keep track of time
        t0_2 = datetime.now()
        loss = learn(model, target_model, replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2

        # Compute total reward
        episode_reward += r
        eps = max(eps - eps_decay, eps_min)
        total_t += 1

        # More debugging info
        avg_loss_of_episode = (
            loss + avg_loss_of_episode * num_steps_in_episode) / (num_steps_in_episode + 1)
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        s2_trans = s2_trans

    return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, eps, avg_loss_of_episode


def learn(model, target_model, replay_buffer, gamma, batch_size):

    states, actions, rewards, next_states, dones = replay_buffer.sample_batch(
        batch_size)

    # Calculate targets
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    inverted_dones = 1 - dones
    targets = rewards + inverted_dones.astype(np.float32) * gamma * next_Q

    # Train model
    loss = model.train_step(states, actions, targets)
    return loss.numpy()
