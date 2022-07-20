# Ultra-Reliable Communications in Two-Ray Ground Reflection Scenarios

This repository is accompanying the paper "A Simple Frequency Diversity Scheme
for Ultra-Reliable Communications in Ground Reflection Scenarios" (K.-L.
Besser, E. Jorswieck, J. Coon, Jun. 2022,
[arXiv:2206.13459](https://arxiv.org/abs/2206.13459)).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/two-ray-ultra-reliability/HEAD)


## File List
The following files are provided in this repository:

- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `Ultra-Reliability Two-Ray Ground Reflection.ipynb`: Jupyter notebook that
  contains interactive plots of most of the results shown in the paper.
- `util.py`: Python module that contains utility functions, e.g., for saving results.
- `model.py`: Python module that contains utility functions around the two-ray
  ground reflection model.
- `single_frequency.py`: Python module that contains the functions to calculate
  the receive power when a single frequency is used.
- `two_frequencies.py`: Python module that contains the functions to calculate
  the receive power when two frequencies are used in parallel.
- `optimal_frequency_distance.py`: Python module that contains the algorithm to
  calculate the optimal frequency spacing for worst-case design.
- `rate_comparison.py`: Python module that contains the functions to calculate
  the achievable rates for the different scenarios.
- `outage_probability.py`: Python module that contains the functions to
  estimate the outage probabilities.
- `uav_example.py`: Python module that contains the UAV example.

## Usage
### Running it online
The easiest way is to use services like [Binder](https://mybinder.org/) to run
the notebook online. Simply navigate to
[https://mybinder.org/v2/gh/klb2/two-ray-ultra-reliability/HEAD](https://mybinder.org/v2/gh/klb2/two-ray-ultra-reliability/HEAD)
to run the notebooks in your browser without setting everything up locally.

### Local Installation
If you want to run it locally on your machine, Python3 and Jupyter are needed.
The present code was developed and tested with the following versions:

- Python 3.10
- Jupyter 1.0
- numpy 1.22
- scipy 1.8
- sdeint 0.2.4

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages (including Jupyter) by running
```bash
pip3 install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
```
This will install all the needed packages which are listed in the requirements 
file. The second line enables the interactive controls in the Jupyter
notebooks.

Finally, you can run the Jupyter notebooks with
```bash
jupyter notebook
```

You can also recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported by the Federal	Ministry of Education and Research
Germany (BMBF) as part of the 6G Research and Innovation Cluster 6G-RIC under
Grant 16KISK020K and by the EPSRC under grant number EP/T02612X/1.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
