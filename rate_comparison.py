import logging

import numpy as np
from scipy import constants
from scipy import optimize
import matplotlib.pyplot as plt

from single_frequency import rec_power, min_rec_power_single_freq, length_los, length_ref
from two_frequencies import sum_power_lower_envelope, delta_freq_peak_approximation
from optimal_frequency_distance import find_optimal_delta_freq
from util import to_decibel, export_results, achievable_rate


LOGGER = logging.getLogger(__name__)

def main_rate_comparison(d_min: float, d_max: float, freq: float, 
                         h_tx: float, h_rx: float, bw: float = None,
                         df: float = None, c: float = constants.speed_of_light,
                         noise_fig_db: float = 3, noise_den_db: float = -174,
                         plot=False, export=False):
    noise_fig = 10**(noise_fig_db/10.)
    noise_den = 10**(noise_den_db/10.)

    distance = np.logspace(np.log10(d_min)-.1, np.log10(d_max)+.1, 3000)
    #distance = np.logspace(np.log10(d_min), np.log10(d_max), 10000)
    #distance = np.logspace(np.floor(np.log10(d_min)), np.ceil(np.log10(d_max)), 3000)

    power_rx_single = rec_power(distance, freq, h_tx, h_rx)
    power_rx_single_db = to_decibel(power_rx_single)
    _min_power_single = min_rec_power_single_freq(d_min, d_max, freq, h_tx, h_rx)

    if df is None:
        df = find_optimal_delta_freq(d_min, d_max, freq, h_tx, h_rx, c)
    LOGGER.info(f"Frequency spacing: {df:E}")
    power_rx_second = rec_power(distance, freq+df, h_tx, h_rx)
    power_rx_sum_lower = sum_power_lower_envelope(distance, df, freq, h_tx, h_rx)
    _w1 = 2*np.pi*freq
    _w2 = 2*np.pi*(freq+df)
    _power_offset_two = 1./(2*_w1*_w2)**2 * (c/4)**4 * (1/length_los(d_max, h_tx, h_rx) - 1/length_ref(d_max, h_tx, h_rx))**4
    _min_power_two_lower = sum_power_lower_envelope(d_max, df, freq, h_tx, h_rx)

    if bw is None:
        bw = df/2
    alpha = _power_offset_two/(noise_fig*noise_den*bw/2) # no square here!
    LOGGER.debug(f"Offset power due to product: {to_decibel(alpha):.2f} dB")

    rate_single = achievable_rate(power_rx_single, bw=bw,
                                  noise_fig_db=noise_fig_db,
                                  noise_den_db=noise_den_db)

    rate_two = (achievable_rate(0.5*power_rx_single, bw=bw/2,
                                noise_fig_db=noise_fig_db,
                                noise_den_db=noise_den_db) +
                achievable_rate(0.5*power_rx_second, bw=bw/2,
                                noise_fig_db=noise_fig_db,
                                noise_den_db=noise_den_db))
    rate_two_lower = achievable_rate(power_rx_sum_lower+alpha,
                                     bw=bw/2, noise_fig_db=noise_fig_db,
                                     noise_den_db=noise_den_db)
    results = {"distance": distance, "rateSingle": rate_single,
               "rateTwo": rate_two, "rateTwoLower": rate_two_lower}

    _min_rate_single = achievable_rate(_min_power_single, bw=bw,
                                       noise_fig_db=noise_fig_db,
                                       noise_den_db=noise_den_db)
    _min_rate_two_lower = achievable_rate(_min_power_two_lower+alpha, bw=bw/2,
                                          noise_fig_db=noise_fig_db,
                                          noise_den_db=noise_den_db)
    LOGGER.info(f"Minimum Rate (One Freq.): {_min_rate_single:E}")
    LOGGER.info(f"Minimum Rate (Two Freq.): {_min_rate_two_lower:E}")

    if plot:
        fig, axs = plt.subplots()
        axs.loglog(distance, rate_single, '-b', label="Single Frequency")
        axs.loglog(distance, rate_two, '-r', label="Two Freq. - Optimal $\Delta f$")
        axs.loglog(distance, rate_two_lower, '--r', alpha=.5, label="Two Freq. Lower Bound")
        axs.set_xlabel("Distance $d$ [m]")
        axs.set_ylabel("Achievable Rate [bit/s/Hz]")
        axs.set_title(f"Parameters: $f_1=${freq:E} Hz, $B=${bw:E}\n$h_{{tx}}={h_tx:.1f}$ m, $h_{{rx}}={h_rx:1f}$ m,\n$d_{{min}}={d_min:.1f}$ m, $d_{{max}}={d_max:.1f}$ m")
        axs.legend()
    if export:
        LOGGER.debug("Exporting results.")
        export_results(results, f"rate-{freq:E}-df{df:E}-t{h_tx:.1f}-r{h_rx:.1f}-dmin{d_min:.1f}-dmax{d_max:.1f}-bw{bw:E}.dat")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.)
    parser.add_argument("-r", "--h_rx", type=float, default=1.)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-dmin", "--d_min", type=float, default=10.)
    parser.add_argument("-dmax", "--d_max", type=float, default=100.)
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
    logging.basicConfig(format="%(asctime)s - %(module)s -- [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    loglevel = logging.WARNING - verb*10
    LOGGER.setLevel(loglevel)
    main_rate_comparison(**args)
    plt.show()
