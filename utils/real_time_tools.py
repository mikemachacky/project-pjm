import numpy as np


class RealTimeTools:

    def extract_hand_landmarks(hand_landmarks):
        if not hand_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
