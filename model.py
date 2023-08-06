import tensorflow as tf
import keras
import numpy as np
import pandas as pd

DATATYPE = np.float64
BATCH_SIZE = 256

def compile_model() -> keras.models.Sequential:
    model = keras.models.Sequential(
        [
            keras.layers.Dense(units=3, input_shape=(3,), activation='relu'),
            keras.layers.Dense(units=1, activation='sigmoid')
        ]
    )

    optimizer = keras.optimizers.Adam()
    loss_func = keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', keras.metrics.BinaryAccuracy(), keras.metrics.BinaryCrossentropy()]
    
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=metrics
    )

    return model

def data_prep() -> tf.data.Dataset:
    df = pd.read_csv('data/loan_data.csv')

    train_raw_data = (tf.convert_to_tensor(df[['Employed', 'Bank Balance', 'Annual Salary']], dtype=DATATYPE)[:8000], tf.convert_to_tensor(df['Default'], dtype=DATATYPE)[:8000])
    test_raw_data = (tf.convert_to_tensor(df[['Employed', 'Bank Balance', 'Annual Salary']], dtype=DATATYPE)[8000:], tf.convert_to_tensor(df['Default'], dtype=DATATYPE)[8000:])

    train_ds = tf.data.Dataset.from_tensor_slices(train_raw_data)
    test_ds = tf.data.Dataset.from_tensor_slices(test_raw_data)

    train_ds.shuffle(128)
    test_ds.shuffle(32)

    return train_ds, test_ds

def train(model: keras.models.Sequential, dataset: tf.data.Dataset):
    model.fit(x=dataset.batch(32), epochs=10)

def test(model: keras.models.Sequential, dataset: tf.data.Dataset):
    model.evaluate(x=dataset.batch(32))

def main():
    train_ds, test_ds = data_prep()
    model = compile_model()
    train(model=model, dataset=train_ds)
    test(model=model, dataset=test_ds)

    return 0

if __name__ == '__main__':
    main()


