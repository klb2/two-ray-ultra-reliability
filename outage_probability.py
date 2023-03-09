import logging

import numpy as np
from scipy import constants
from scipy import stats
import matplotlib.pyplot as plt

from single_frequency import rec_power
from two_frequencies import sum_power_lower_envelope
from optimal_frequency_distance import find_optimal_delta_freq
from rate_comparison import rate_single_freq, rate_two_freq, rate_two_freq_lower
from util import export_results, to_decibel


LOGGER = logging.getLogger(__name__)

def outage_prob_mc(rates, threshold=None):
    if threshold is None:
        threshold = np.logspace(4, 10, 2000)
    rates = {k: np.expand_dims(_p, 1) for k, _p in rates.items()}
    LOGGER.debug("Estimate outage probabilities... (This might take a while...)")
    #results = {k: np.count_nonzero(_rate < threshold, axis=0)/num_samples
    results = {k: np.count_nonzero(_rate < threshold, axis=0)/len(_rate)
               for k, _rate in rates.items()}
    return results

def main_outage_prob_rate(d_min, d_max, freq, h_tx, h_rx, bw, df: float = None,
                          noise_fig_db: float = 3, noise_den_db: float = -174,
                          c=constants.c, num_samples=100000,
                          plot=False, export=False, **kwargs):
    LOGGER.info(f"Simulating outage probability with parameters: f1={freq:E}, h_tx={h_tx:.1f}, h_rx={h_rx:.1f}, dmin={d_min:.1f}, dmax={d_max:.1f}")
    LOGGER.info(f"Number of samples: {num_samples:E}")
    distance = (d_max-d_min)*np.random.rand(num_samples) + d_min
    if df is None:
        df = find_optimal_delta_freq(d_min, d_max, freq, h_tx, h_rx)

    rate_rv = _generate_rate_rv(distance, d_max, freq, h_tx, h_rx, bw, df,
                                noise_fig_db, noise_den_db)
    threshold = np.logspace(4, 10, 2000)
    results = {k: v.cdf(threshold) for k, v in rate_rv.items()}

    if plot:
        fig, axs = plt.subplots()
        for _name, _prob in results.items():
            axs.loglog(threshold, _prob, label=_name)
        axs.set_xlabel("Rate Threshold [bit/s]")
        axs.set_ylabel("Outage Probability $\\varepsilon$")
        axs.legend()

    results['threshold'] = threshold
    if export:
        LOGGER.info("Exporting results.")
        export_results(results, f"out_prob_rate-{freq:E}-dmin{d_min:.1f}-dmax{d_max:.1f}-t{h_tx:.1f}-r{h_rx:.1f}-bw{bw:E}.dat")
    return results

def _generate_rate_rv(distance, d_max, freq, h_tx, h_rx, bw, df,
                      noise_fig_db: float = 3, noise_den_db: float = -174,
                      c=constants.c):
    LOGGER.debug("Work on single frequency scenario...")
    rate_single = rate_single_freq(distance, freq, h_tx, h_rx, bw,
                                   noise_fig_db=noise_fig_db,
                                   noise_den_db=noise_den_db)
    
    LOGGER.info(f"Frequency spacing: {df:E}")
    LOGGER.debug("Work on two frequency scenario...")
    rate_two = rate_two_freq(distance, freq, df, h_tx, h_rx, bw,
                             noise_fig_db=noise_fig_db,
                             noise_den_db=noise_den_db)
    LOGGER.debug("Work on two frequency scenario (lower bound)...")
    rate_two_lower = rate_two_freq_lower(distance, freq, df, h_tx, h_rx,
                                         bw=bw, d_max=d_max,
                                         noise_fig_db=noise_fig_db,
                                         noise_den_db=noise_den_db)

    rates = {"singleActual": rate_single, "twoActual": rate_two,
             "twoLower": rate_two_lower}
    rates_hist = {k: np.histogram(v) for k, v in rates.items()}
    rates_rv = {k: stats.rv_histogram(v) for k, v in rates_hist.items()}
    return rates_rv
    #results = outage_prob_mc(rates, threshold)
    #return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.)
    parser.add_argument("-r", "--h_rx", type=float, default=1.)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-dmin", "--d_min", type=float, default=10.)
    parser.add_argument("-dmax", "--d_max", type=float, default=100.)
    parser.add_argument("-n", "--num_samples", type=int, default=int(1e6))
    parser.add_argument("-bw", type=float, default=None)
    parser.add_argument("-df", type=float, default=None)
    parser.add_argument("-F", "--noise_fig_db", type=float, default=3.)
    parser.add_argument("-N", "--noise_den_db", type=float, default=-174)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="Increase output verbosity")
    args = vars(parser.parse_args())
    verb = args.pop("verbosity")
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    loglevel = logging.WARNING - verb*10
    LOGGER.setLevel(loglevel)
    main_outage_prob_rate(**args)
    plt.show()
