import tensorflow as tf
import os

print(tf.__version__)

os.environ["TF_USE_LEGACY_KERAS"]="1"

batch_size = 32
seed = 42
vocab_size = 10000

raw_train_set = tf.keras.preprocessing.text_dataset_from_directory(
    directory="./dataset",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed
)

raw_test_set = tf.keras.preprocessing.text_dataset_from_directory(
    directory="./dataset",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed
)

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens = vocab_size)
vectorize_layer.adapt(raw_train_set.map(lambda x, y: x))

train_ds = raw_train_set.cache().prefetch(batch_size)
test_ds = raw_test_set.cache().prefetch(1)

embed_size = 128
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(vocab_size, embed_size),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Activation('sigmoid')
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer="nadam", metrics=["accuracy"])
model.load_weights("weight/cp.ckpt").expect_partial()

## PREDICT
inputs = [
'''
public class Persona {
    private int edad;
          
    public Persona(int edad) {
        this.edad = edad;
    }     
}
''', 
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import os

print(tf.__version__)
''',
'''
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "argc == " << argc << '\n';

    return 1;
}
'''
]

predictions = model.predict(inputs)

labels = ["C++", "Java", "Python"]

for idx, prediction in enumerate(predictions):
    print("==============================================================================================")
    print("Sentence: " + inputs[idx])
    print("==============================================================================================")
    print(" - Scoring: " + str(prediction))
    print(" - Prediction: " + labels[prediction.argmax()])
    print("==============================================================================================")
