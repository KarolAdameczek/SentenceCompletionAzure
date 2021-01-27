import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import normalize
from gensim import models

import numpy as np
import os
import string

from azureml.core import Run

import numpy as np
import os
import time
import glob
import string


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--model-folder', type=str, dest='model_folder', default=None, help='model to train')
parser.add_argument('--embed-folder', type=str, dest='embed_folder', default='embed', help='embed-folder mounting point')
parser.add_argument('--text-id', type=int, dest='text_id', default=1)
parser.add_argument('--text-size', type=int, dest='text_size', default=1000000000)
parser.add_argument('--epochs', type=int, dest='epochs', default=1)
args = parser.parse_args()

data_folder = args.data_folder
print('training dataset is stored here:', data_folder)
files = os.listdir(data_folder)

embed_folder = args.embed_folder
print('Pretrained word embedings is stored here:', embed_folder)
emb_file = embed_folder

model = None
model_folder = args.model_folder
if model_folder is not None:
  print('model is stored here:', model_folder)
  model_files = os.listdir(model_folder)
  model = tf.saved_model.load(model_folder)


EPOCHS = args.epochs
n = args.text_id if args.text_id < len(files) else len(files)
s = args.text_size


# Wczytanie danych
print("---------- Wczytywanie danych ----------")

texts = []
for file_ in files[:n]:
  texts.append(open(os.path.join(data_folder, file_), 'rb').read().decode("utf-8")[:s])
  l = (len(texts[-1]))
  print(f"Length of text {file_}: {l} characters")


word2vec_model = models.KeyedVectors.load_word2vec_format(emb_file)
embedding_matrix = word2vec_model.vectors
print('Shape of embedding matrix: ', embedding_matrix.shape)

print("---------- Zakończono wczytywanie danych ----------")

texts = texts[:n]


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
  text = text.replace("\r\n", " ")
  text = text.translate(str.maketrans(unwanted_whitespaces, "".join([" " for _ in range(len(unwanted_whitespaces))])))
  text = text.translate(str.maketrans("", "", unwanted_chars))
  texts[i] = text



# Przetworzenie tekstu
print("---------- Przetwarzanie tekstu 2/2----------")
vocab = set()
for text in texts:
  vocab = vocab.union(set(text.split()))
vocab = sorted(vocab)

vectorizer = TextVectorization(standardize=None)
text_ds = tf.data.Dataset.from_tensor_slices(vocab).batch(128)
vectorizer.adapt(text_ds)



tokens_from_words = preprocessing.StringLookup(
    vocabulary=vectorizer.get_vocabulary())

words_from_tokens = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=tokens_from_words.get_vocabulary(), invert=True)

def text_from_tokens(ids):
    return tf.strings.reduce_join(words_from_tokens(ids), axis=-1)

print(len(tokens_from_words.get_vocabulary()))


voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

test = ["pan", "jest", "tu", "i", "tam", "."]
[word_index[w] for w in test]

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = None
    try:
        embedding_vector = word2vec_model[word]
    except Exception:
        pass
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))



print("---------- Tokenizacja tekstu ----------")

all_tokens = []
tokens_datasets = []
for i, text in enumerate(texts):
  print(f"Tokenizacja tekstu {i}/{len(texts)}")
  tokens = tokens_from_words(text.split())
  all_tokens.append(tokens)
  tokens_datasets.append(tf.data.Dataset.from_tensor_slices(tokens))

for token in tokens_datasets[0].take(10):
    print(token)
    print(words_from_tokens(token).numpy().decode('utf-8'))

print("---------- Tokenizacja tekstu zakończona ----------")



print("---------- Tworzenie zbiorów sekwencji ----------")
seq_length = 5
examples_per_epoch = len(texts[0])//(seq_length+1)
sequences_d = []
for i, tokens_dataset in enumerate(tokens_datasets):
  print(f"Tworzenie zbiorów sekwencji {i}/{len(tokens_dataset)}")
  sequences_d.append(tokens_dataset.batch(seq_length+1, drop_remainder=True))

for seq in sequences_d[0].take(10):
  print(seq)
  print((text_from_tokens(seq).numpy()).decode("utf-8"))

del texts

print("---------- Tworzenie zbiorów wejść i wyjść ----------")

def split_input_target(sequence):
  input_text = sequence[:-1]
  target_text = sequence[1:]
  return input_text, target_text

datasets = []
for i, sequences in enumerate(sequences_d):
  print(f"{i}/{len(sequences_d)}")
  datasets.append(sequences_d[i].map(split_input_target))

for input_example, target_example in  datasets[0].take(20):
  print("Input :", (text_from_tokens(input_example).numpy()).decode('utf-8'))
  print("Target:", (text_from_tokens(target_example).numpy()).decode('utf-8'))


# Tworzenie paczek szkoleniowych

BATCH_SIZE = 64

BUFFER_SIZE = 10000

dataset_train = datasets[0]
for dataset in datasets[1:]:
  dataset_train = dataset_train.concatenate(dataset)

dataset_train = (
    dataset_train.shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))



# Budowanie modelu
print("---------- Budowanie modelu ---------")

vocab_size = len(vocab)
print(vocab_size)


units = 1024

class MyModelLSTM(tf.keras.Model):
  def __init__(self, embedding_matrix, units):
    super(MyModelLSTM, self).__init__()
    self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0],
                                                embedding_matrix.shape[1],
                                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                trainable=False)
    self.lstm = tf.keras.layers.LSTM(units,
                                      return_sequences=True,
                                      return_state=True,
                                      dropout=0.2)
    self.dense = tf.keras.layers.Dense(embedding_matrix.shape[0])

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x)
    if states is None:
      state1, state2 = self.lstm.get_initial_state(x)
      states = [state1, state2]
    x, state1, state2 = self.lstm(x, initial_state=states, training=training)
    states = [state1, state2]
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else: 
      return x




if model is None:
  model = MyModelLSTM(
    embedding_matrix=embedding_matrix,
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
  def __init__(self, model, words_from_tokens, tokens_from_words, temperature=1):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.words_from_tokens = words_from_tokens
    self.tokens_from_words = tokens_from_words

    # Create a mask to prevent "" or "[UNK]" from being generated.
    skip_ids = self.tokens_from_words(['','[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices = skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(tokens_from_words.get_vocabulary())+2]) 
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)


  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_words = tf.strings.split(inputs, sep=' ')
    print(input_words)
    input_tokens = self.tokens_from_words(input_words).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits] 
    predicted_logits, states =  self.model(inputs=input_tokens, states=states, 
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
    predicted_chars = self.words_from_tokens(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states



one_step_model = OneStep(model, words_from_tokens, tokens_from_words)

os.makedirs('./outputs/model', exist_ok=True)
tf.saved_model.save(one_step_model, './outputs/model/one_step_model')

states = None
next_char = tf.constant(['pan to jednak'])
result = [x+' ' for x in next_char]

for n in range(6):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char+' ')

result = tf.strings.join(result)

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)




print("---------- Zapis modelu ----------")
# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)
tf.keras.models.save_model(model, './outputs/model/model')
print("model saved in ./outputs/model folder")

