import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_decision_forests as tfdf
print(f'Running TF versions {tf.__version__}:')
print(f'\t tensorflow_hub {hub.__version__}')
print(f'\t tensorflow_text {text.__version__}')
print(f'\t tensorflow_decision_forests {tfdf.__version__}')
print(f'\nAvailable GPU devices: {len(tf.config.list_physical_devices("GPU"))}')