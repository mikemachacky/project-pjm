import numpy as np

class DatasetTools:
    SEQUENCE_SIZE = 30
    FEATURE_SIZE = 63  # 21 landmarków * 3 współrzędne
    LANDMARKS_SIZE = 21
    DIMENSIONS = 3

    @staticmethod
    def normalize_wrist(sequence):
        sequence = sequence.copy()
        wrist_origin = sequence[0].reshape(DatasetTools.LANDMARKS_SIZE, DatasetTools.DIMENSIONS)[0]
        return np.array([(frame.reshape(DatasetTools.LANDMARKS_SIZE, DatasetTools.DIMENSIONS) - wrist_origin).flatten()
                         for frame in sequence])

    @staticmethod
    def divide_into_sequences(df):
        sequences, labels = [], []
        for sample_id, group in df.groupby('sample_id'):
            group = group.sort_values('frame').reset_index(drop=True)
            label = group.loc[0, 'label']
            data = group.drop(columns=['sample_id', 'frame', 'label'], errors='ignore').values

            if len(data) != 30:
                print("Dropping: ", sample_id)
                continue  # pomiń próbki o niepoprawnej długości

            sequences.append(data)
            labels.append(label)

        return np.array(sequences), np.array(labels)

    @staticmethod
    def _augment_np(sequence_np: np.ndarray, augment_cfg: dict) -> np.ndarray:
        """
        Augmentacja danych z numpy. sequence_np shape = (seq_len, features)
        augment_cfg: dict z bool dla kluczy: 'flip', 'gaussian_noise', 'speckle_noise', 'salt_pepper'
        """
        seq = sequence_np.astype(np.float32).copy()
        seq_frames = seq.reshape(seq.shape[0], DatasetTools.LANDMARKS_SIZE, DatasetTools.DIMENSIONS)  # (T,21,3)

        # Odwrócenie osi X)
        if augment_cfg.get('flip', False):
            seq_frames[..., 0] = -seq_frames[..., 0]

        # Szum Gaussa
        if augment_cfg.get('gaussian_noise', False):
            std = augment_cfg.get('gaussian_std', 0.01)
            seq_frames += np.random.normal(0.0, std, size=seq_frames.shape).astype(np.float32)

        # Speckle noise (multiplikatywny)
        if augment_cfg.get('speckle_noise', False):
            std = augment_cfg.get('speckle_std', 0.01)
            noise = np.random.normal(0.0, std, size=seq_frames.shape).astype(np.float32)
            seq_frames += seq_frames * noise

        # Salt & pepper (dropout pojedynczych wartości do 0)
        if augment_cfg.get('salt_pepper', False):
            prob = augment_cfg.get('salt_pepper_prob', 0.01)
            rnd = np.random.rand(*seq_frames.shape)
            seq_frames[rnd < (prob / 2)] = 0.0  # pepper
            # for "salt" zamiast ustawiać na 1, dodamy losowy mały skok:
            salt_mask = rnd > (1 - prob / 2)
            seq_frames[salt_mask] += np.random.normal(0.0, 0.05, size=np.count_nonzero(salt_mask)).reshape(-1,)

        return seq_frames.reshape(seq.shape[0], -1)

    @staticmethod
    def augment_generator(sequences, labels, batch_size, augment_config,
                          multiplier: int = 1, include_original: bool = True, random_state: int = None):
        """
        Generator augmentacji:
        - include_original=True -> dodaje oryginały,
        - multiplier -> ile razy powielić każdy przykład (w tym oryginał, jeśli include_original=True)
        """
        if random_state is not None:
            np.random.seed(random_state)

        out_seqs = []
        out_labels = []

        for seq, lbl in zip(sequences, labels):
            examples = []
            if include_original:
                examples.append(seq)

            for _ in range(multiplier - (1 if include_original else 0)):
                examples.append(DatasetTools._augment_np(seq, augment_config))

            out_seqs.extend(examples)
            out_labels.extend([lbl] * len(examples))

        out_seqs = np.array(out_seqs, dtype=np.float32)
        out_labels = np.array(out_labels)

        for i in range(0, len(out_seqs), batch_size):
            yield out_seqs[i:i + batch_size], out_labels[i:i + batch_size]


