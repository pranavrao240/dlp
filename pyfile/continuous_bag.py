
# Importing libraries
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd

# Taking random sentences as data
data = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks"""

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

word2id = tokenizer.word_index

# Adding padding token
word2id['PAD'] = 0

id2word = {v: k for k, v in word2id.items()}

# Converting text into sequences of word IDs
wids = tokenizer.texts_to_sequences([data])

vocab_size = len(word2id)
embed_size = 100
window_size = 2

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])


# Generating (context word, target word) pairs
def generate_context_word_pairs(corpus, window_size, vocab_size):

    context_length = window_size * 2

    for words in corpus:

        sentence_length = len(words)

        for index, word in enumerate(words):

            context_words = []

            start = max(0, index - window_size)
            end = min(sentence_length, index + window_size + 1)

            for i in range(start, end):
                if i != index:
                    context_words.append(words[i])

            # Padding context words
            context_words = pad_sequences(
                [context_words],
                maxlen=context_length,
                padding='post',
                value=0
            )[0]

            # Converting target word into categorical format
            label_word = to_categorical(
                word,
                num_classes=vocab_size
            )

            yield np.array([context_words]), np.array([label_word])


# Creating CBOW model
cbow = Sequential()

cbow.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embed_size,
        mask_zero=True
    )
)

cbow.add(GlobalAveragePooling1D())

cbow.add(Dense(vocab_size, activation='softmax'))

cbow.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop'
)

print(cbow.summary())


# Training
for epoch in range(1, 6):

    loss = 0.0
    i = 0

    for x, y in generate_context_word_pairs(
        corpus=wids,
        window_size=window_size,
        vocab_size=vocab_size
    ):

        i += 1
        loss += float(cbow.train_on_batch(x, y))

    print('Epoch:', epoch, '\tLoss:', loss / i)


# Extracting embeddings
weights = cbow.get_weights()[0]

# Removing padding embedding
weights = weights[1:]

print('Embedding Shape:', weights.shape)

print(
    pd.DataFrame(
        weights,
        index=[id2word[i] for i in range(1, vocab_size)]
    ).head()
)


# Computing similarity
from sklearn.metrics.pairwise import euclidean_distances

distance_matrix = euclidean_distances(weights)

print('Distance Matrix Shape:', distance_matrix.shape)


# Finding similar words
similar_words = {}

for search_term in ['deep']:

    word_index = word2id[search_term] - 1

    similar_indices = distance_matrix[word_index].argsort()[1:6]

    similar_words[search_term] = [
        id2word[idx + 1] for idx in similar_indices
    ]

print(similar_words)

