import numpy as np
import pandas as pd

def to_decibel(value):
    return 10*np.log10(value)

def export_results(results, filename):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename, sep='\t', index=False)

def achievable_rate(rec_power, bw, noise_fig_db=3, noise_den_db=-174):
    noise_fig = 10**(noise_fig_db/10.)
    noise_den = 10**(noise_den_db/10.)
    noise_power = noise_fig*noise_den*bw
    snr = rec_power/noise_power
    return bw*np.log2(1 + snr)
