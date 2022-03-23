#!/bin/sh

# ...
# Information about the paper...
# ...
#
# Copyright (C) 20XX ...
# License: GPLv3

#echo "Figure 1: ..."
#python3 single_frequency.py -t 10 -r 1 -f 477134516 --plot --export -v

echo "..."
python3 two_frequencies.py -f "2.4e9" -df "250e6" -t 10 -r 1.5 --plot --export -v -dmin 10 -dmax 100
#python3 two_frequencies.py -f "2.4e9" -df "100e6" -t 10 -r 1.5 --plot --export -v
