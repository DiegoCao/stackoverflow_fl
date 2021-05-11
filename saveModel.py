import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

def save(model, path):
    tf.keras.models.save_model(model, path)

def load(path):
    return tf.keras.models.load_model(path)

if __name__ == "__main__":
    print('save model test')
    

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
    save(model,'/tmp/model')
    loaded_model = load('/tmp/model')
    x = tf.random.uniform((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))