import numpy as np
import pandas as pd

def to_decibel(value):
    return 10*np.log10(value)

def export_results(results, filename):
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename, sep='\t', index=False)
