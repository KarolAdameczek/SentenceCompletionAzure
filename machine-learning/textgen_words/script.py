import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from azureml.core import Run

import numpy as np
import os
import time
import glob
import string


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--model-folder', type=str, dest='model_folder', default=None, help='model to train')
parser.add_argument('--text-id', type=int, dest='text_id', default=1)
parser.add_argument('--text-size', type=int, dest='text_size', default=10000)
parser.add_argument('--epochs', type=int, dest='epochs', default=1)
args = parser.parse_args()

data_folder = args.data_folder
print('training dataset is stored here:', data_folder)
files = os.listdir(data_folder)

model = None
model_folder = args.model_folder
if model_folder is not None:
  print('model is stored here:', model_folder)
  model_files = os.listdir(model_folder)
  model = tf.saved_model.load(model_folder)


EPOCHS = args.epochs


# Wczytanie danych
print("---------- Wczytywanie danych ----------")

texts = []
for file_ in files:
    texts.append(open(os.path.join(data_folder, file_), 'rb').read().decode("utf-8"))
    l = (len(texts[-1]))
    print(f"Length of text {file_}: {l} characters")

print("---------- Zakończono wczytywanie danych ----------")

n = args.text_id if args.text_id < len(texts) else len(texts)
s = args.text_size if args.text_size < len(texts[n-1]) else len(texts[n-1])

texts = texts[n-1:n]


# Przetworzenie tekstu
print("---------- Przetwarzanie tekstu 1/2----------")

for i, text in enumerate(texts):
  text = text.lower()
  vocab = sorted(set(text))
  polish_alphabet = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż"

  unwanted_whitespaces = string.whitespace.translate(str.maketrans("", "", " "))
  allowed_chars = polish_alphabet + polish_alphabet.upper() + string.digits + "!?.,:%" + " "
  unwanted_chars = "".join(vocab).translate(str.maketrans("","",allowed_chars))
  
  print(f"--------- {i}/{len(texts)} ----------")
  text = text.translate(str.maketrans(unwanted_whitespaces, "".join([" " for _ in range(len(unwanted_whitespaces))])))
  text = text.translate(str.maketrans("", "", unwanted_chars))



# Przetworzenie tekstu
print("---------- Przetwarzanie teksty 2/2----------")
vocab = set()
for text in texts:
  vocab = vocab.union(set(text))
vocab = sorted(vocab)

ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)



print("---------- Tokenizacja tekstu ----------")

all_ids = []
ids_datasets = []
for i, text in enumerate(texts):
  print(f"Tokenizacja tekstu {i}/{len(texts)}")
  all_ids.append(ids_from_chars(tf.strings.unicode_split(text, 'UTF-8')))
  ids_datasets.append(tf.data.Dataset.from_tensor_slices(all_ids[-1]))

for ids in ids_datasets[0].take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

print("---------- Tokenizacja tekstu zakończona ----------")



print("---------- Tworzenie zbiorów sekwencji ----------")
seq_length = 100
examples_per_epoch = len(texts[0])//(seq_length+1)
sequences_d = []
for i, ids_dataset in enumerate(ids_datasets):
  print(f"Tworzenie zbiorów sekwencji {i}/{len(ids_datasets)}")
  sequences_d.append(ids_dataset.batch(seq_length+1, drop_remainder=True))

for seq in sequences_d[0].take(5):
  print((text_from_ids(seq).numpy()).decode("utf-8"))

print("---------- Tworzenie zbiorów wejść i wyjść ----------")
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

split_input_target(list("Korpus parlamentarny"))

datasets = []
for sequences in enumerate(sequences_d):
  print(f"{i}/{len(sequences_d)}")
  datasets.append(sequences_d[i].map(split_input_target))

for input_example, target_example in  datasets[0].take(2):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())


print("---------- Tworzenie paczek szkoleniowych ----------")
# Tworzenie paczek szkoleniowych
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset_train = datasets[0]
for dataset in datasets[1:]:
  dataset_train = dataset_train.concatenate(dataset)

dataset_train = (
    dataset_train
    .take(2000000)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))



# Budowanie modelu
print("---------- Budowanie modelu ---------")

vocab_size = len(vocab)

embedding_dim = 256

units = 1024


class MyModelRNN(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units):
    super(MyModelRNN, self).__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(units,
                                   return_sequences=True, 
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else: 
      return x


if model is None:
  model = MyModelRNN(
      vocab_size=len(ids_from_chars.get_vocabulary()),
      embedding_dim=embedding_dim,
      units=units)

for input_example_batch, target_example_batch in dataset_train.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)


tf.exp(mean_loss).numpy()


model.compile(optimizer='adam', loss=loss)

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        run.log('Loss', logs['loss'])



# Directory where the checkpoints will be saved
checkpoint_dir = './outputs/training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_freq=500)



print("---------- Trenowanie modelu ----------")

history = model.fit(dataset_train, epochs=EPOCHS, verbose=1, callbacks=[checkpoint_callback, LogRunMetrics()])



class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=0.5):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "" or "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['','[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices = skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())]) 
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')[:100]
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits] 
    predicted_logits, states =  self.model(inputs=input_ids, states=states, 
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states



one_step_model = OneStep(model, chars_from_ids, ids_from_chars)



states = None
next_char = tf.constant(['pan'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)


os.makedirs('./outputs/model', exist_ok=True)
tf.saved_model.save(one_step_model, './outputs/model/one_step_model')
one_step_reloaded = tf.saved_model.load('./outputs/model/one_step_model')

states = None
next_char = tf.constant(['pan:'])
result = [next_char]

for n in range(100):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode("utf-8"))




print("---------- Zapis modelu ----------")
# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)
tf.keras.models.save_model(model, './outputs/model/model')
print("model saved in ./outputs/model folder")

