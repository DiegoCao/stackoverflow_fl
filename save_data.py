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


(train, _, raw_test) = tff.simulation.datasets.stackoverflow.load_data()
  
# trainpath = './data/train_data'

# rawpath = './data/test_raw'
# tf.dataset.experimental.save(train, savepath)
# tf.dataset.experimental.save(raw_test, rawpath)
