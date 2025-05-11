

class Config():

    MAX_BUFFER_SIZE = 500000
    MIN_BUFFER_SIZE = 5000

    GAMMA = 0.99
    BATCH_SIZE = 32

    EPS_MIN = 0.1
    EPS_DECAY = 1e-5

    NUM_EPISODES = 3000
    TARGET_NETWORK_UPDATE_PERIOD = 5000

    IMAGE_RESIZE_MODE = "square"
    IMAGE_CHANNEL_MODE = "grayscale"
    IMAGE_DIM = 84

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
