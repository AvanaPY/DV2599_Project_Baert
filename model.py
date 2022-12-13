import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from const import MBTI_CLASSES
import datetime

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

_MODEL = None
_PREPROCESSOR = None

def get_modle(output_size:int=MBTI_CLASSES):
    preprocessor = get_preprocessing_modle()
    encoder = get_encoder()

    # Apply BERT preprocessing and encoding
    inp = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Input")
    enc_inp = preprocessor(inp)
    outputs = encoder(enc_inp)

    # Apply very many Dense layers
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dense(256, activation='relu', name="l1")(net)
    net = tf.keras.layers.Dense(1024, activation='relu', name="l2")(net)
    net = tf.keras.layers.Dense(4096, activation='relu', name="l3")(net)
    net = tf.keras.layers.Dense(1024, activation='relu', name="l4")(net)
    net = tf.keras.layers.Dense(output_size, activation='softmax', name="output")(net)
    baert : tf.keras.Model = tf.keras.Model(inp, net, name="Baert")
    return baert, preprocessor, encoder

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

def get_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy()

def get_optimizer(init_lr : float, num_train_steps : int, num_warmup_steps):
    return optimization.create_optimizer(init_lr=init_lr,
                                        num_train_steps=num_train_steps,
                                        num_warmup_steps=num_warmup_steps,
                                        optimizer_type='adamw')
def get_metrics():
    return [                                                                   
        tf.keras.metrics.SparseCategoricalAccuracy()
    ]

def load_model(model_path : str):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def save_model(model : tf.keras.Model, model_path : str):
    model.save(model_path)

def compile_model(baert : tf.keras.Model, init_lr : float, num_train_steps : int, num_warmup_steps):

    optimizer   = get_optimizer(init_lr, num_train_steps, num_warmup_steps)
    loss        = get_loss()
    metrics     = get_metrics()

    baert.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

def train_model(train_ds : tf.data.Dataset, 
                val_ds   : tf.data.Dataset, 
                baert    : tf.keras.Model,
                epochs   : int,
                m_name   : str):

    log_dir = "logs/" + m_name
    chkp_dir = "chkp/" + m_name
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
        tf.keras.callbacks.ModelCheckpoint(chkp_dir, 
                                            monitor = "val_loss",
                                            save_weights_only=True,
                                            save_freq='epoch'),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            mode='auto',
            restore_best_weights=True
        )
    ]

    baert.fit(train_ds, 
            epochs=epochs, 
            validation_data=val_ds,
            callbacks=callbacks)