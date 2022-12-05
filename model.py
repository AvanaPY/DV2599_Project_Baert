import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import datetime

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

_MODEL = None
_PREPROCESSOR = None

def get_modle(output_size:int=16):
    preprocessor = get_preprocessing_modle()
    encoder = get_encoder()

    # Apply BERT preprocessing and encoding
    inp = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Input")
    enc_inp = preprocessor(inp)
    outputs = encoder(enc_inp)

    net = outputs["pooled_output"]
    net = tf.keras.layers.Dense(4096, activation='relu', name="l1")(net)
    net = tf.keras.layers.Dense(4096, activation='relu', name="l2")(net)
    net = tf.keras.layers.Dense(4096, activation='relu', name="l3")(net)
    net = tf.keras.layers.Dense(4096, activation='relu', name="l4")(net)
    net = tf.keras.layers.Dense(4096, activation='relu', name="l5")(net)
    net = tf.keras.layers.Dense(output_size, activation='softmax', name="Baert")(net)
    baert : tf.keras.Model = tf.keras.Model(inp, net), preprocessor, encoder
    return baert

def get_encoder():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
    encoder = hub.KerasLayer(tfhub_handle_encoder)
    _MODEL = encoder
    return encoder

def get_preprocessing_modle():
    global _PREPROCESSOR
    if _PREPROCESSOR is not None:
        return _PREPROCESSOR
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    _PREPROCESSOR = bert_preprocess_model
    return bert_preprocess_model

def train_model(train_ds : tf.data.Dataset, 
                val_ds   : tf.data.Dataset, 
                baert : tf.keras.Model, 
                epochs : int = 5):
    init_lr = 3e-5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer   = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')
    loss        = tf.keras.losses.CategoricalCrossentropy()
    metrics     = [
        tf.keras.metrics.CategoricalAccuracy()
    ]

    baert.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    m_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + m_name
    chkp_dir = "chkp/" + m_name
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
        tf.keras.callbacks.ModelCheckpoint(chkp_dir, 
                                            monitor = "val_loss",
                                            save_weights_only=True,
                                            save_freq='epoch')
    ]

    baert.fit(train_ds, 
            epochs=epochs, 
            validation_data=val_ds,
            callbacks=callbacks)