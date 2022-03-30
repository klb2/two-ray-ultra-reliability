#!/bin/sh

# ...
# Information about the paper...
# ...
#
# Copyright (C) 20XX ...
# License: GPLv3

#echo "Figures 1 and 2: ..."
#python3 single_frequency.py -t 10 -r 1 -f 477134516 --plot --export -v

echo "Figure X"
python3 approximation_min_max.py -f "100e6" -t 10 -r 1.5 -d 50 -v --plot --export
python3 approximation_min_max.py -f "2.4e9" -t 10 -r 1.5 -d 50 -v --plot --export


#echo "..."
#python3 two_frequencies.py -f "2.4e9" -df "250e6" -t 10 -r 1.5 --plot --export -v -dmin 10 -dmax 100
#python3 two_frequencies.py -f "2.4e9" -df "100e6" -t 10 -r 1.5 --plot --export -v
