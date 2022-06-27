import logging

import numpy as np
from scipy import constants
from scipy import optimize
import matplotlib.pyplot as plt

from model import length_los, length_ref
from util import to_decibel, export_results
from two_frequencies import sum_power_lower_envelope, delta_freq_peak_approximation


LOGGER = logging.getLogger(__name__)


def find_peak_delta_freq(freq, distance, h_tx, h_rx, c=constants.c):
    d_los = length_los(distance, h_tx, h_rx)
    d_ref = length_ref(distance, h_tx, h_rx)
    omega = 2*np.pi*freq
    a = (d_ref-d_los)/c
    #func = lambda tau, omega, a: np.tan(tau)+2/(omega*a+tau)
    #sol_max = optimize.root_scalar(func, args=(omega, a), bracket=[np.pi/2+np.finfo(float).eps, np.pi])
    #sol_min = optimize.root_scalar(func, args=(omega, a), bracket=[3*np.pi/2+np.finfo(float).eps, 2*np.pi])

    #func = lambda tau, omega, a: np.abs(np.tan(tau)+2/(omega*a+tau))
    #func = lambda tau, omega, a: np.abs(omega**2 + 2*np.cos(tau) + (a*omega+tau)*np.sin(tau))
    func = lambda tau, omega, a: np.log(c**2*np.abs((d_ref**2+d_los**2)/(d_los*d_ref)*np.sqrt(1/omega**4+1/(omega+tau/a)**4+2*np.cos(tau)/(omega**2*(omega+tau/a)**2))-(2/(omega+tau/a)**2+1/omega**2 * (2*np.cos(tau)+(a*omega+tau)*np.sin(tau))))**2)
    sol_min = optimize.minimize(func, args=(omega, a), x0=.99*np.pi, bounds=optimize.Bounds(np.pi/2, np.pi))
    sol_max = optimize.minimize(func, args=(omega, a), x0=1.99*np.pi, bounds=optimize.Bounds(3*np.pi/2, 2.2*np.pi))
    #print(sol_min, sol_max)
    freq_min = sol_min.x[0]/(a*2*np.pi)
    freq_max = sol_max.x[0]/(a*2*np.pi)
    return freq_min, freq_max


def _main_delta_freq_peaks(freq, distance, h_tx, h_rx, c=constants.c):
    #delta_freq = np.logspace(7, 10, 3000)
    #delta_freq = np.logspace(7, np.log10(3e9), 3000)
    delta_freq = np.logspace(7, 9, 1500)
    power_sum_lower = sum_power_lower_envelope(distance, delta_freq, freq, h_tx, h_rx)
    delta_freq_peak = find_peak_delta_freq(freq, distance, h_tx, h_rx)
    delta_freq_peak = np.array(delta_freq_peak)
    results = {"df": delta_freq, "power": power_sum_lower}
    return results, delta_freq_peak

def main_peaks_approximation(freq, distance, h_tx, h_rx, c=constants.c,
                             plot=False, export=False):
    results_f1, df_peak = _main_delta_freq_peaks(freq, distance, h_tx, h_rx, c=c)
    df = results_f1['df']
    results_f1['power'] = to_decibel(results_f1['power'])
    df_peak_approximation = delta_freq_peak_approximation(distance, h_tx, h_rx)
    LOGGER.info(f"Exact maximum/minimum locations: {df_peak}")
    LOGGER.info(f"Appr. maximum/minimum locations: {df_peak_approximation}")

    if plot:
        _ylim = [-120, -60]
        fig, axs = plt.subplots()
        axs.set_ylim(_ylim)
        axs.semilogx(df, results_f1['power'], 'b-', label="${P_r}$")
        axs.vlines(df_peak, *_ylim, color='g', label="Exact Peaks $\Delta f$") 
        axs.vlines(df_peak_approximation, *_ylim, color='r', linestyle='--', label="Approximation")
        axs.set_xlabel("Frequency Spacing $\\Delta f$ [Hz]")
        axs.set_ylabel("Receive Power Envelope ${P_r}$ [dB]")
        axs.legend()
    if export:
        LOGGER.debug("Exporting results.")
        export_results(results_f1, f"power_envelope_df-{freq:E}-dist{distance:.1f}-t{h_tx:.1f}-r{h_rx:.1f}.dat")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.)
    parser.add_argument("-r", "--h_rx", type=float, default=1.)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-d", "--distance", type=float, default=100.)
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
    main_peaks_approximation(**args)
    plt.show()
