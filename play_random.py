import gymnasium as gym
from glitch.transformer import ImageTransformer

if __name__ == '__main__':

    env = gym.make("ALE/Pacman-v5", render_mode="human")
    transformer = ImageTransformer(84)

    s, _ = env.reset()
    done = False
    states = []
    rewards = 0

    while not done:
        env.render()
        s2, r, terminated, truncated, info = env.step(
            env.action_space.sample())
        done = terminated or truncated
        s2_trans = transformer.transform(s2)
        rewards += r
        states.append(s2_trans)

    print(rewards)
    print(len(states))
    env.close()
