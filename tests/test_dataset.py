"""Test datasets compatibility."""

import tensorflow as tf

from unet.dataset import tensorflow_dataset


class TestDataset(tf.test.TestCase):
    def test_tensorflow_dataset(self):
        """Test weather dataset is created correctly."""
        originals = ["original_0.jpg", "original_1.jpg"]
        masks = ["mask_0.png", "mask_1.png"]
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(originals, name="Originals"),
                tf.constant(masks, name="Masks"),
            )
        )
        test_function = tensorflow_dataset(originals_paths=originals, masks_paths=masks)

        self.assertAllEqual(
            a=list(test_function.as_numpy_iterator()),
            b=list(test_dataset.as_numpy_iterator()),
            msg="Content of dataset is not the same.",
        )


if __name__ == "__main__":
    tf.test.main()
