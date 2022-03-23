import logging

import numpy as np
from scipy import constants
from scipy import optimize
import matplotlib.pyplot as plt

from util import export_results, to_decibel

from model import length_los, length_ref
from single_frequency import rec_power, crit_dist

#plt.rc('text', usetex=True)

LOGGER = logging.getLogger(__name__)


def sum_power_lower_envelope(distance, delta_freq, freq, 
                             h_tx, h_rx, G_los=1, G_ref=1,
                             c=constants.c, power_tx=1):
    d_los = length_los(distance, h_tx, h_rx)
    d_ref = length_ref(distance, h_tx, h_rx)
    freq2 = freq+delta_freq
    omega = 2*np.pi*freq
    omega2 = 2*np.pi*freq2
    delta_omega = omega2-omega
    _part1 = c**2/(4*d_los**2) * (1./omega**2 + 1./omega2**2)
    _part2 = c**2/(4*d_ref**2) * (1./omega**2 + 1./omega2**2)
    A = (c/(2*omega))**2
    B = (c/(2*omega2))**2
    _part3 = -2/(d_los*d_ref) * np.sqrt(A**2 + B**2 + 2*A*B*np.cos(delta_omega/c*(d_ref-d_los)))
    power_rx = power_tx/2 * (_part1 + _part2 + _part3)
    return power_rx



def delta_freq_for_dist(d, h_tx=10, h_rx=2, c=constants.speed_of_light):
    _factor = c*np.pi/(np.sqrt(2)*h_rx*h_tx)
    _part_inner = np.sqrt((d**2+h_rx**2)**2 + 2*(d-h_rx)*(d+h_rx)*h_tx**2 + h_tx**4)
    _part_outer = np.sqrt(d**2 + h_rx**2 + h_tx**2 + _part_inner)
    delta_omega = _factor*_part_outer
    delta_f = delta_omega/(2*np.pi)
    return delta_f


def main_power_two_freq(freq, delta_freq, h_tx, h_rx,
                        plot=False, export=False, **kwargs):
    distance = np.logspace(0, 3, 2000)
    power_rx = rec_power(distance, freq, h_tx, h_rx)
    power_rx_db = to_decibel(power_rx)
    freq2 = freq + delta_freq
    power_rx2 = rec_power(distance, freq2, h_tx, h_rx)
    power_rx2_db = to_decibel(power_rx2)
    power_sum = .5*(power_rx+power_rx2)
    power_sum_db = to_decibel(power_sum)
    power_sum_lower = sum_power_lower_envelope(distance, delta_freq, freq, h_tx, h_rx)
    power_sum_lower_db = to_decibel(power_sum_lower)
    results = {"distance": distance,
               "powerSum": power_sum_db,
               "envelope": power_sum_lower_db}

    if plot:
        fig, axs = plt.subplots()
        axs.semilogx(distance, power_sum_db)
        axs.semilogx(distance, power_sum_lower_db)
    if export:
        LOGGER.debug("Exporting single frequency power results.")
        export_results(results, f"power_sum-{freq:E}-df{delta_freq:E}-t{h_tx:.1f}-r{h_rx:.1f}.dat")
    return results

def main_optimization_problem(d_min, d_max, freq, h_tx, h_rx, c=constants.c, 
                              plot=False, export=False, **kwargs):
    if d_max <= d_min:
        raise ValueError("The maximum distance needs to be larger than the minimum distance.")

    df_min = delta_freq_for_dist(d_min, h_tx=h_tx, h_rx=h_rx)
    df_max = delta_freq_for_dist(d_max, h_tx=h_tx, h_rx=h_rx)
    df_lower_bound = c/(2*np.minimum(h_rx, h_tx))

    df1 = np.logspace(np.log10(df_lower_bound), np.log10(df_min), 500)
    df2 = np.logspace(np.log10(df_min), np.log10(df_max), 500)

    p_min = sum_power_lower_envelope(d_min, df1, freq, h_tx, h_rx)
    p_max1 = sum_power_lower_envelope(d_max, df1, freq, h_tx, h_rx)
    p_max2 = sum_power_lower_envelope(d_max, df2, freq, h_tx, h_rx)
    def p_dist1(df):
        dist1 = np.sqrt(0*1j+((c*np.pi)**2 - (2*np.pi*df*h_rx)**2)*((c*np.pi)**2 - (2*np.pi*df*h_tx)**2))/(c*np.pi*2*np.pi*df)
        dist1 = np.real(dist1)
        return sum_power_lower_envelope(dist1, df, freq, h_tx, h_rx)
    p_d1 = p_dist1(df2)
    p_min_db = to_decibel(p_min)
    p_max1_db = to_decibel(p_max1)
    p_max2_db = to_decibel(p_max2)
    p_d1_db = to_decibel(p_d1)
    results1 = {"df": df1, "dw": 2*np.pi*df1, "pmax": p_max1_db, "pmin": p_min_db}
    results2 = {"df": df2, "dw": 2*np.pi*df2, "pmax": p_max2_db, "pd1": p_d1_db}

    if plot:
        fig, axs = plt.subplots()
        axs.semilogx(df1, p_min_db, 'b-', label="${P_r}(d_{min})$")
        axs.semilogx(df1, p_max1_db, 'r-', label="${P_r}(d_{max})$")
        axs.semilogx(df2, p_max2_db, 'r-')
        axs.semilogx(df2, p_d1_db, 'g-', label="${P_r}(d_{1})$")
        axs.set_xlabel("Frequency Spacing $\\Delta f$ [Hz]")
        axs.set_ylabel("Individual Minimum Receive Powers ${P_r}$ [dB]")
        axs.legend()
    if export:
        LOGGER.debug("Exporting single frequency power results.")
        export_results(results1, f"power_min_parts1-{freq:E}-dmin{d_min:.1f}-dmax{d_max:.1f}-t{h_tx:.1f}-r{h_rx:.1f}.dat")
        export_results(results2, f"power_min_parts2-{freq:E}-dmin{d_min:.1f}-dmax{d_max:.1f}-t{h_tx:.1f}-r{h_rx:.1f}.dat")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.)
    parser.add_argument("-r", "--h_rx", type=float, default=1.)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-df", "--delta_freq", type=float, default=100e6)
    parser.add_argument("-dmin", "--d_min", type=float, default=10.)
    parser.add_argument("-dmax", "--d_max", type=float, default=100.)
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
    main_power_two_freq(**args)
    main_optimization_problem(**args)
    plt.show()
