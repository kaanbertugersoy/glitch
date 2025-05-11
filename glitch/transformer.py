import tensorflow as tf

# Convert to grayscale
# Resize
# Crop


class ImageTransformer:
    def __init__(self, image_dim, resize_mode='square', channel_mode='grayscale', ):
        self.resize_mode = resize_mode
        self.channel_mode = channel_mode
        self.image_dim = image_dim

    def transform(self, image):
        # Normalize pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        if self.channel_mode == 'grayscale':
            image = self._convert_to_grayscale(image)

        if self.resize_mode == 'square':
            image = self._crop_to_bounding_box(image)

        if image.shape[1] > self.image_dim:
            image = self._resize(image)

        return image

    def _convert_to_grayscale(self, image):
        return tf.image.rgb_to_grayscale(image)

    def _crop_to_bounding_box(self, image):
        return tf.image.crop_to_bounding_box(
            image, 34, 0, 160, 160)

    def _resize(self, image):
        return tf.image.resize(image, [self.image_dim, self.image_dim], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
