import json
import numpy as np
import os
import tensorflow as tf

from azureml.core.model import Model

def init():
    global model
    model_root = os.getenv('AZUREML_MODEL_DIR')
    model_folder = 'model/one_step_model'
    model = tf.saved_model.load(os.path.join(model_root, model_folder))
 
    
def run(input_):
    json_ = json.loads(input_)
    next_char = tf.constant([json_["data"]])
    n = json_['num']

    json_1 = { "data" : [] }
    for i in range(n):
        states = None
        next_char = tf.constant([json_["data"]])
        result = [ next_char ]
        for n in range(100):
            next_char, states = model.generate_one_step(next_char, states=states)
            result.append(next_char)
        json_1["data"].append(tf.strings.join(result)[0].numpy().decode("utf-8").replace('\r\n', " "))

    return json_1

