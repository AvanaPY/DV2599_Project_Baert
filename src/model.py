import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_decision_forests as tfdf
import numpy as np
import datetime
import json
from official.nlp import optimization
from .const import MBTI_CLASSES

tf.get_logger().setLevel('ERROR')

_MODEL = None
_PREPROCESSOR = None
_CONFIG_FILE_NAME = 'baert.config'

def get_modle(output_size:int=MBTI_CLASSES):
    nnbaert = BaertNN(output_size=output_size)
    return nnbaert, nnbaert.preprocessor, nnbaert.encoder

def get_modle_random_forest(num_trees:int=300):
    random_baert = RandomBaert(num_trees=num_trees)
    return random_baert, random_baert.preprocessor, random_baert.encoder

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
    try:
        config = BaertModel.load_config(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'No {_CONFIG_FILE_NAME} file associated with model {model_path}.')
        
    if config['type'] == 'DNN':
        model = BaertNN.load_model(model_path)
    elif config['type'] == 'RandomForest':
        model = RandomBaert.load_model(model_path)

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

    log_dir = "skip/logs/" + m_name
    chkp_dir = "skip/chkp/" + m_name
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

# Wrapper for the model
class BaertModel:
    def __init__(self):
        self._preprocessor = get_preprocessing_modle()
        self._encoder = get_encoder()
        self._model : tf.keras.Model = None
    
    @property
    def encoder(self):
        return self._encoder

    @property
    def preprocessor(self):
        return self._preprocessor

    def preprocess_data(self, x, y):
        return x, y

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self._model.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def compile(self, *args, **kwargs):
        return self._model.compile(*args, **kwargs)

    def save(self, model_path):
        save_model(self._model, model_path=model_path)
        self.save_config(model_path)

    def save_config(self, model_path : str):
        raise NotImplementedError
    
    @staticmethod
    def load_config(model_path : str):
        conf_path = os.path.join(model_path, _CONFIG_FILE_NAME)
        with open(conf_path, 'r') as f:
            config = json.load(f)
        return config

    @staticmethod
    def load_model(model_path : str):
        raise NotImplementedError

class BaertNN(BaertModel):
    def __init__(self, output_size:int=MBTI_CLASSES, model:tf.keras.Model=None):
        super(BaertNN, self).__init__()

        if model is None:
            inp = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Input")
            enc_inp = self.preprocessor(inp)
            outputs = self.encoder(enc_inp)

            # Apply very many Dense layers
            net = outputs["pooled_output"]
            net = tf.keras.layers.Dense(256, activation='relu', name="l1")(net)
            net = tf.keras.layers.Dense(1024, activation='relu', name="l2")(net)
            net = tf.keras.layers.Dense(4096, activation='relu', name="l3")(net)
            net = tf.keras.layers.Dense(1024, activation='relu', name="l4")(net)
            net = tf.keras.layers.Dense(output_size, activation='softmax', name="output")(net)
            self._model : tf.keras.Model = tf.keras.Model(inp, net, name="Baert")
        else:
            self._model = model

    def save_config(self, model_path : str):
        conf_path = os.path.join(model_path, _CONFIG_FILE_NAME)
        config = {
            'type' : 'DNN'
        }
        with open(conf_path, 'w') as f:
            json.dump(config, f)

    @staticmethod
    def load_model(model_path : str):
        model = tf.keras.models.load_model(model_path, compile=False)
        return BaertNN(model=model)

class RandomBaert(BaertModel):
    def __init__(self, num_trees : int = 300, model:tf.keras.Model=None):
        super(RandomBaert, self).__init__()

        if model is None:
            self._model = tfdf.keras.RandomForestModel(
                num_trees=num_trees,
                verbose=1
            )
        else:
            self._model = model

    def preprocess_data(self, x, y):
        # Apply the bert pre processor on the data in chunks
        print(f'Applying BERT to {len(x)} instances of data:')

        x_tmp = []
        BERT_MAX_ROWS = 256
        for i in range(0, len(x), BERT_MAX_ROWS):
            print(f'    BERTing {i:5d} to {min(len(x), i + BERT_MAX_ROWS):5d}...')
            _x = x[i:i+BERT_MAX_ROWS]
            _x = self.preprocessor(_x)
            _x = self.encoder(_x)['pooled_output']
            x_tmp.append(_x)
        x = np.concatenate(x_tmp)
        print(f'Bert applied: {len(x)} instances.')
        return x, y

    def predict(self, data : list, *args, **kwargs):
        data, _ = self.preprocess_data(data, None)
        return self._model.predict(data)

    def save_config(self, model_path : str):
        conf_path = os.path.join(model_path, _CONFIG_FILE_NAME)
        config = {
            'type' : 'RandomForest'
        }
        with open(conf_path, 'w') as f:
            json.dump(config, f)

    @staticmethod
    def load_model(model_path : str):
        model = tf.keras.models.load_model(model_path)
        return RandomBaert(model=model)