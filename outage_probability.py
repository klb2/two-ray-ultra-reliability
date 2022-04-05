import logging

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

from single_frequency import rec_power
from two_frequencies import sum_power_lower_envelope
from optimal_frequency_distance import find_optimal_delta_freq
from util import export_results, to_decibel


LOGGER = logging.getLogger(__name__)

def main_outage_prob(d_min, d_max, freq, h_tx, h_rx, c=constants.c,
                     num_samples=100000, plot=False, export=False, **kwargs):
    LOGGER.info(f"Simulating outage probability with parameters: f1={freq:E}, h_tx={h_tx:.1f}, h_rx={h_rx:.1f}, dmin={d_min:.1f}, dmax={d_max:.1f}")
    LOGGER.info(f"Number of samples: {num_samples:E}")
    distance = (d_max-d_min)*np.random.rand(num_samples) + d_min
    pow_single_freq = rec_power(distance, freq, h_tx, h_rx)
    pow_single_bound = sum_power_lower_envelope(distance, 0., freq, h_tx, h_rx)
    
    opt_df = find_optimal_delta_freq(d_min, d_max, freq, h_tx, h_rx)
    LOGGER.info(f"Optimal frequency spacing: {opt_df:E}")
    pow_two_freq = .5*(rec_power(distance, freq, h_tx, h_rx)+rec_power(distance, freq+opt_df, h_tx, h_rx))
    pow_two_bound = sum_power_lower_envelope(distance, opt_df, freq, h_tx, h_rx)

    powers = {"singleActual": pow_single_freq, "singleBound": pow_single_bound,
              "twoActual": pow_two_freq, "twoBound": pow_two_bound}
    powers = {k: np.expand_dims(to_decibel(_p), 1) for k, _p in powers.items()}
    sensitivity = np.linspace(-120, -60, 1000)
    results = {k: np.count_nonzero(_pow < sensitivity, axis=0)/num_samples
                for k, _pow in powers.items()}

    if plot:
        fig, axs = plt.subplots()
        for _name, _prob in results.items():
            axs.semilogy(sensitivity, _prob, label=_name)
        axs.set_xlabel("Receiver Sensitivity [dB]")
        axs.set_ylabel("Outage Probability")
        axs.legend()

    results['sensitivity'] = sensitivity
    if export:
        LOGGER.info("Exporting results.")
        export_results(results, f"out_prob-{freq:E}-dmin{d_min:.1f}-dmax{d_max:.1f}-t{h_tx:.1f}-r{h_rx:.1f}.dat")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.)
    parser.add_argument("-r", "--h_rx", type=float, default=1.)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-dmin", "--d_min", type=float, default=10.)
    parser.add_argument("-dmax", "--d_max", type=float, default=100.)
    parser.add_argument("-n", "--num_samples", type=int, default=int(1e6))
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
    main_outage_prob(**args)
    plt.show()
