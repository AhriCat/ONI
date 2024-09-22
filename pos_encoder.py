class PositionalEncodingLayer(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def positional_encoding(self, max_len, embed_dim):
        pos_enc = np.zeros((max_len, embed_dim))
        for pos in range(max_len):
            for i in range(0, embed_dim, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                if i + 1 < embed_dim:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / embed_dim)))
        return tf.cast(pos_enc, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_len, :]