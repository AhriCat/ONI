from sklearn.model_selection import train_test_split
from collections import defaultdict
train_corpus, val_corpus, train_labels, val_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)
tokenizer = tokenizer
tokenizer.fit_on_texts(train_corpus)
train_sequences = tokenizer.texts_to_sequences(train_corpus)
val_sequences = tokenizer.texts_to_sequences(val_corpus)

max_length = max(len(seq) for seq in train_sequences + val_sequences)
train_data = pad_sequences(train_sequences, maxlen=max_length)
val_data = pad_sequences(val_sequences, maxlen=max_length)

label_to_index = defaultdict(lambda: len(label_to_index))
for label in train_labels + val_labels:
    label_to_index[label]

# Convert string labels to integers
train_labels = [label_to_index[label] for label in train_labels]
val_labels = [label_to_index[label] for label in val_labels]

# Convert labels to one-hot encoded format
num_classes = len(label_to_index)

class LinguisticNN(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(LinguisticNN, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.bidirectional_lstm = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))
        self.context_dense = TimeDistributed(Dense(64, activation='relu'))
        self.sentiment_dense = Dense(32, activation='relu')
        self.final_dense = Dense(10, activation='softmax')  # Assuming classification task

    def call(self, inputs):
        embedded_inputs = self.embedding(inputs)
        lstm_output = self.bidirectional_lstm(embedded_inputs)
        context_vector = self.context_dense(lstm_output)
        sentiment_vector = self.sentiment_dense(lstm_output[:, -1, :])  # Extract sentiment from the last time step
        combined_vector = tf.concat([context_vector, sentiment_vector], axis=-1)
        return self.final_dense(combined_vector)

linguistics = LinguisticNN(vocab_size,embedding_dim)
linguistics.build(input_shape=(None, None)) 

linguistics.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

linguistics.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=32)