import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sdeint

from optimal_frequency_distance import find_optimal_delta_freq
from outage_probability import _generate_rate_rv
from rate_comparison import rate_two_freq_lower
from util import export_results

LOGGER = logging.getLogger(__name__)

def main(freq, h_tx, h_rx, bw, df: float = None, radius=150, d_lake=30,
         noise_fig_db: float = 3, noise_den_db: float = -174,
         num_runs=1000, num_steps=2000, plot=False, export=False):
    pos_tx = (radius+d_lake)*np.exp(1j*np.pi/4)
    d_min = d_lake
    d_max = d_lake + 2*radius
    LOGGER.info(f"Distance is within [{d_min:.2f}, {d_max:.2f}] meter")
    if df is None:
        df = find_optimal_delta_freq(d_lake, d_lake+2*radius, freq, h_tx, h_rx)
        LOGGER.info(f"Optimal frequency spacing: {df:E}")

    a = np.array([[0, -1, 0], [3, 1, 3], [0, 0, 7]])
    b = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    timeline = np.linspace(0, 100, num_steps)
    distance = np.array([])
    _count = 0
    while len(distance) < num_runs*num_steps:
        positions = get_uav_positions(a, b, timeline)
        _positions = positions["x"] + 1j*positions["y"]
        if np.any(np.abs(_positions) > radius):
            continue
        _distances = np.abs(_positions - pos_tx)
        distance = np.append(distance, _distances)
        _count = _count + 1
        LOGGER.debug(f"Completed run {_count:d}/{num_runs:d}")
    LOGGER.info(f"Completed all {num_runs:d} runs with {num_steps:d} time samples each.")

    LOGGER.debug("Estimate outage probabilities... (This might take a while...)")
    rate_rv = _generate_rate_rv(distance, d_max, freq, h_tx, h_rx, bw, df,
                                noise_fig_db, noise_den_db)
    df_comparison = 100e6
    #df_comparison = 50e6
    rate_two_comparison = rate_two_freq_lower(distance, freq, df_comparison,
                                              h_tx, h_rx, bw=bw, d_max=d_max,
                                              noise_fig_db=noise_fig_db,
                                              noise_den_db=noise_den_db)
    _hist_two_comparison = np.histogram(rate_two_comparison)
    rate_rv["twoComparison"] = stats.rv_histogram(_hist_two_comparison)

    #threshold = np.logspace(3, 9, 2000)
    threshold = np.logspace(1, 7, 2000)
    results = {k: v.cdf(threshold) for k, v in rate_rv.items()}


    if plot:
        fig, axs = plt.subplots()
        for i in range(len(timeline)-1):
            axs.plot(positions['x'][i:i+2], positions['y'][i:i+2],
                     color=plt.colormaps.get_cmap("hot")(positions['c'][i]))
        axs.add_patch(plt.Circle((0, 0), radius=radius, alpha=.5))
        axs.plot(pos_tx.real, pos_tx.imag, 'ok')
        fig2, axs2 = plt.subplots()
        for _name, _prob in results.items():
            axs2.loglog(threshold, _prob, label=_name)
        axs2.set_xlabel("Rate Threshold [bit/s]")
        axs2.set_ylabel("Outage Probability $\\varepsilon$")
        axs2.legend()

    results['threshold'] = threshold
    if export:
        LOGGER.info("Exporting results.")
        export_results(positions, f"uav_positions.dat")
        export_results(results, f"out_prob_uav-{freq:E}-dmin{d_min:.1f}-dmax{d_max:.1f}-t{h_tx:.1f}-r{h_rx:.1f}-bw{bw:E}-df{df:E}.dat")
    return results

def get_uav_positions(a, b, timeline):
    def f(x, t):
        return -a.dot(x)
    def G(x, t):
        return b
    x0 = np.array([0, 0, 0])
    x = sdeint.itoint(f, G, x0, timeline)
    y = sdeint.itoint(f, G, x0, timeline)
    scale = 100
    positions = {"x": scale*x[:, 0], "y": scale*y[:, 0],
                 "c": timeline/max(timeline)}
    return positions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.)
    parser.add_argument("-r", "--h_rx", type=float, default=1.)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("--radius", type=float, default=150.)
    parser.add_argument("-l", "--d_lake", type=float, default=30.)
    parser.add_argument("-n", "--num_runs", type=int, default=int(1e3))
    parser.add_argument("-bw", type=float, default=100e6)
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
    main(**args)
    plt.show()
