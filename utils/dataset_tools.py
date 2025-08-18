import numpy as np
import tensorflow as tf

class DatasetTools:
    SEQUENCE_SIZE = 30
    FEATURE_SIZE = 63 # 21 landmarków * 3 współrzędne
    LANDMARKS_SIZE = 21
    DIMENSIONS = 3
    @staticmethod
    def normalize_wrist(sequence):
        """
        Normalizuje sekwencję względem nadgarstka z pierwszej klatki.
        """
        sequence = sequence.copy()
        wrist_origin = sequence[0].reshape(DatasetTools.LANDMARKS_SIZE, DatasetTools.DIMENSIONS)[0]
        return np.array([(frame.reshape(DatasetTools.LANDMARKS_SIZE, DatasetTools.DIMENSIONS) - wrist_origin).flatten() for frame in sequence])
    @staticmethod
    def divide_into_sequences(df):
        """
                Dzieli dane na sekwencje o określonej długości.
                Zwraca tablicę sekwencji oraz odpowiadających im etykiet.
        """
        sequences, labels = [], []
        for sample_id, group in df.groupby('sample_id'):
            group = group.sort_values('frame').reset_index(drop=True)
            label = group.loc[0, 'label']
            data = group.drop(columns=['sample_id', 'frame', 'label'], errors='ignore')

            for start in range(0, len(data) - DatasetTools.SEQUENCE_SIZE + 1):
                segment = data.iloc[start:start + DatasetTools.SEQUENCE_SIZE].values
                sequences.append(segment)
                labels.append(label)

        return np.array(sequences), np.array(labels)

    @staticmethod
    def augment(sequence: tf.Tensor, augment_config: dict) -> tf.Tensor:
        """
        Augmentacja sekwencji: flip, gaussian, speckle, salt & pepper.
        """
        seq = tf.identity(sequence)

        # Flip osi X
        if augment_config.get('flip', False):
            x_indices = tf.range(tf.shape(seq)[-1]) % 3 == 0
            flip_mask = tf.cast(x_indices, seq.dtype)
            seq = seq * (1 - 2 * flip_mask)

        # Gaussian noise
        if augment_config.get('gaussian_noise', False):
            noise = tf.random.normal(tf.shape(seq), mean=0.0,
                                     stddev=0.01, dtype=tf.float64)
            seq += noise

        # Speckle noise
        if augment_config.get('speckle_noise', False):
            noise = tf.random.normal(tf.shape(seq), mean=0.0,
                                     stddev=0.01, dtype=tf.float64)
            noise = tf.sqrt(tf.abs(seq)) * noise
            noise = tf.where(noise < 0.01, tf.zeros_like(noise), noise)
            seq += seq * noise

        # Salt & pepper noise
        if augment_config.get('salt_pepper', False):
            prob = 0.01
            rnd = tf.random.uniform(tf.shape(seq))
            seq = tf.where(rnd < prob / 2, tf.zeros_like(seq), seq)
            seq = tf.where(rnd > 1 - prob / 2, tf.ones_like(seq), seq)

        return seq