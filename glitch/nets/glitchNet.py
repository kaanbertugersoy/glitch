import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization  # type: ignore


class GlitchNet(tf.keras.Model):
    def __init__(self, K):
        super(GlitchNet, self).__init__()
        self.K = K
        self.conv1 = Conv2D(32, (8, 8), activation='relu')
        self.pool1 = MaxPooling2D((4, 4))
        self.conv2 = Conv2D(64, (4, 4), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.pool3 = MaxPooling2D((1, 1))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.output_layer = Dense(K)

        # Instantiate the loss function and optimizer
        self.loss_fn = losses.Huber()
        self.optimizer = optimizers.Adam(1e-3)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)

    # The default runtime in TensorFlow 2 is eager execution.
    # As such, our training loop executes eagerly.

    # This is great for debugging, but graph compilation has a
    # definite performance advantage. Describing our computation
    # as a static graph enables the framework to apply global
    # performance optimizations. This is impossible when the framework
    # is constrained to greedily execute one operation after another,
    # with no knowledge of what comes next.

    # We can compile into a static graph any function that takes tensors
    # as input. Just add a @tf.function decorator on it, like this:
    # Speed is => 3x
    @tf.function
    def train_step(self, states, actions, targets):
        with tf.GradientTape() as tape:
            preds = self(states, training=True)
            selected_action_values = tf.reduce_sum(
                preds * tf.one_hot(tf.cast(actions, dtype=tf.int64), depth=self.K), axis=1
            )
            loss = tf.reduce_mean(self.loss_fn(
                targets, selected_action_values))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables))

        return loss

    def predict(self, states):
        return self(states)

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict(x)[0])

    def save(self, filepath="tf_dqn_model.h5"):
        self.save_weights(filepath)

    def load(self, filepath="tf_dqn_model.h5"):
        self.load_weights(filepath)
