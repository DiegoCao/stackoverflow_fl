import tensorflow as tf
import tensorflow_federated as tff

def save(dataset, location='data/tf-records/'):
    dataset = dataset.map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter(location)
    writer.write(dataset)
    return location


def load(tf_record='data/tf-records/'):
    dataset = tf.data.TFRecordDataset(tf_record)
    dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.int64))
    return dataset


(train, raw_test) = tff.simulation.datasets.shakespeare.load_data()
  
trainpath = './data/save_shake1'

rawpath = './data/save_shake2'
save(train, trainpath)
save(raw_test, rawpath)
