# code-clasification

This code can classify following sourcecodes:
- Java
- Python
- C++

This project was developed with pure TensorFlow and Keras.

## Tips
If you want include new kind of lenguage, you only must create new folder in dataset with you lenguage and set the code below:

```
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(vocab_size, embed_size),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3), # -> Increase output layer
    tf.keras.layers.Activation('sigmoid')
])
```

## Usage
### To train using dataset in this project
python3 train.py

### To predict some new code
python3 predict.py

Predictions is an array with conficence of all lenguages.
