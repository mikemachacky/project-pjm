import numpy as np
import tensorflow as tf

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
            data = group.drop(columns=['sample_id', 'frame', 'label'], errors='ignore')

            for start in range(0, len(data) - DatasetTools.SEQUENCE_SIZE + 1):
                segment = data.iloc[start:start + DatasetTools.SEQUENCE_SIZE].values
                sequences.append(segment)
                labels.append(label)

        return np.array(sequences), np.array(labels)

    @staticmethod
    def _augment_np(sequence_np: np.ndarray, augment_cfg: dict) -> np.ndarray:
        """
        Bezpieczna augmentacja na numpy (łatwiej debugować). sequence_np shape = (seq_len, features)
        augment_cfg: dict z bool dla kluczy: 'flip', 'gaussian_noise', 'speckle_noise', 'salt_pepper'
        """
        seq = sequence_np.astype(np.float32).copy()
        seq_frames = seq.reshape(seq.shape[0], DatasetTools.LANDMARKS_SIZE, DatasetTools.DIMENSIONS)  # (T,21,3)

        # Flip X (odwrócenie osi X)
        if augment_cfg.get('flip', False):
            seq_frames[..., 0] = -seq_frames[..., 0]

        # Gaussian noise
        if augment_cfg.get('gaussian_noise', False):
            std = augment_cfg.get('gaussian_std', 0.01)
            seq_frames += np.random.normal(0.0, std, size=seq_frames.shape).astype(np.float32)

        # Speckle noise (multiplicative)
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
        Zwraca generator yielding batches. Dla każdej sekwencji:
          - jeśli include_original=True: zachowuje oryginał jako jeden z przykładów,
          - multiplier: ile łącznie przykładów (wliczając oryginał, jeśli include_original=True)
            ma się pojawić per oryginał. Przykłady:
              multiplier=1, include_original=True -> tylko oryginały
              multiplier=3, include_original=True -> 1 oryginał + 2 augmenty
              multiplier=3, include_original=False -> 3 augmenty na oryginał
        augment_config może zawierać dodatkowe parametry (np. gaussian_std).
        """
        if random_state is not None:
            np.random.seed(random_state)

        num_samples = len(sequences)
        augment_types = [k for k, v in augment_config.items() if isinstance(v, bool) and v]

        # sanity check
        if multiplier < 1:
            raise ValueError("multiplier musi być >= 1")

        # przygotuj listę (sequence, label) do zwrócenia
        out_seqs = []
        out_labels = []

        for i in range(num_samples):
            orig = sequences[i]
            lbl = labels[i]

            if include_original:
                out_seqs.append(orig)
                out_labels.append(lbl)

            # ile dodatkowych wariantów stworzyć
            extra = multiplier - (1 if include_original else 0)
            for _ in range(extra):
                # stwórz losową konfigurację augmentacji na podstawie augment_config
                # (dla każdego typu decydujemy losowo, czy zastosować; pozwala łączyć augmenty)
                cfg = {}
                for key, val in augment_config.items():
                    if isinstance(val, bool):
                        # jeśli włączone globalnie, stosuj z p=0.5 (możesz zmienić)
                        cfg[key] = val and (np.random.rand() < 0.8)  # 80% szansy użycia w wariancie
                    else:
                        # jeśli wartość parametryczna (np. std, prob), przekaż ją dalej
                        cfg[key] = val
                augmented = DatasetTools._augment_np(orig, cfg)
                out_seqs.append(augmented)
                out_labels.append(lbl)

        out_seqs = np.array(out_seqs, dtype=np.float32)
        out_labels = np.array(out_labels)

        # generator batchowy
        total = len(out_seqs)
        indices = np.arange(total)
        # losowe permutowanie — jeśli chcesz porządek zachować, usuń tę linię
        np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_idx = indices[start:end]
            yield out_seqs[batch_idx], out_labels[batch_idx]
