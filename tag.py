import tensorflow as tf
import tensorflow_federated as tff

import os.path

import collections
import json

import matplotlib.pyplot as plt


def load_tag_counts(cache_dir=None):
  """Loads the tag counts for the Stack Overflow dataset.
  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.
  Returns:
    A collections.OrderedDict where the keys are string tags, and the values
    are the counts of unique users who have at least one example in the training
    set containing with that tag. The dictionary items are in decreasing order
    of tag frequency.
  """
  path = tf.keras.utils.get_file(
      'stackoverflow.tag_count.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/stackoverflow.tag_count.tar.bz2',
      file_hash='6fe281cec490d9384a290d560072438e7e2b377bbb823876ce7bd6f82696772d',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  file_path = os.path.join(dir_path, 'stackoverflow.tag_count')
  with open(file_path) as f:
    tag_counts = json.load(f)
  return collections.OrderedDict(
      sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))

res = load_tag_counts(
    cache_dir=None
)

keys = res.keys()
vals = res.values()
print('the length of tag: ', len(res))

small_dict = res.take(100)

# i = 0
# for key, val in res.items():
#     i += 1
#     if i > 10000:
#         break
#     print(key, val)
plt.plot(keys, vals)

# plt.show()
plt.savefig('tag_count.png')