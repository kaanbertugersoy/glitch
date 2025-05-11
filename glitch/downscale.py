import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt

DOWNSCALE_OFFSET_HEIGHT = 40
DOWNSCALE_OFFSET_WIDTH = 0
DOWNSCALE_TARGET_HEIGHT = 160
DOWNSCALE_TARGET_HEIGHT = 160


def downscale(frame):
    cropped_image = tf.image.crop_to_bounding_box(
        frame, DOWNSCALE_OFFSET_HEIGHT, DOWNSCALE_OFFSET_WIDTH, DOWNSCALE_TARGET_HEIGHT, DOWNSCALE_TARGET_HEIGHT)
    return cropped_image


if __name__ == '__main__':
    env = gym.make("ALE/Carnival-v5")

    observation, info = env.reset()
    print(observation.shape)
    downscaled = downscale(frame=observation)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(observation)
    axes[0].set_title("Original Image")
    axes[1].imshow(downscaled)
    axes[1].set_title("Downscaled Image")
    plt.tight_layout()
    plt.show()
