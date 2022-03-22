import numpy as np

def length_los(distance, h_tx, h_rx):
    return np.sqrt(distance**2 + (h_tx-h_rx)**2)

def length_ref(distance, h_tx, h_rx):
    return np.sqrt(distance**2 + (h_tx+h_rx)**2)
