#!/bin/sh

# This repository is accompanying the paper "A Simple Frequency Diversity Scheme
# for Ultra-Reliable Communications in Ground Reflection Scenarios" (K.-L.
# Besser, E. Jorswieck, J. Coon, Jun. 2022, arXiv:2206.13459)
#
# Copyright (C) 2022
# License: GPLv3

echo "Figures 2 and 3"
python3 single_frequency.py -t 10 -r 1.5 -f 477134516 --plot --export -v
python3 single_frequency.py -t 10 -r 1.5 -f "2.4e9" --plot --export -v

echo "Figure 4"
python3 single_frequency.py -t 10 -r 1.5 -f "2.4e9" --rho=1 --plot --export -v
python3 single_frequency.py -t 10 -r 1.5 -f "2.4e9" --rho=.5 --plot --export -v
python3 single_frequency.py -t 10 -r 1.5 -f "2.4e9" --rho=.1 --plot --export -v

echo "Figures 5 and 7"
python3 two_frequencies.py -f "2.4e9" -df "250e6" -t 10 -r 1.5 -dmin 10 -dmax 100 --plot --export -v
#python3 two_frequencies.py -f "2.4e9" -df "100e6" -t 10 -r 1.5 --plot --export -v

echo "Figure 6"
python3 approximation_min_max.py -f "100e6" -t 10 -r 1.5 -d 50 -v --plot --export
python3 approximation_min_max.py -f "2.4e9" -t 10 -r 1.5 -d 50 -v --plot --export

echo "Figure 8"
python3 optimal_frequency_distance.py -f "2.4e9" -t 10 -r 1.5 -dmin 10 -dmax 100 --plot -v --export

echo "Figure 9"
python3 rate_comparison.py -f "2.4e9" -t 10 -r 1.5 -bw "100e3" -dmin 10 -dmax 100 -v --plot --export

echo "Figure 10"
python3 outage_probability.py -f "2.4e9" -t 10 -r 1.5 -bw "100e3" -dmin 10 -dmax 100 -v --plot -n 10000000 --export

echo "Figures 11 and 12"
python3 uav_example.py -f "2.4e9" -bw "100e3" -t 10 -r 3 --radius 150 -vv -n 1000 --plot --export
