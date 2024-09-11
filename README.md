# BridgingTheGapFigures
Code to reproduce results in "A reliable set of equations for anisotropy in ice sheet models"

# Requirements

The code requires the following data sets:

- Greenland velocity data from https://doi.org/10.5067/QUA5Q9SVMSJG
- Greenland surface height data from https://doi.org/10.5067/FPSU0V1MWUB6

EGRIP eigenvalue data is available from https://doi.org/10.1594/PANGAEA.949248 and pole figure data from https://doi.org/10.5281/zenodo.8015759

The code has the following dependencies

- numpy
- scipy
- matplotlib
- cartopy
- pyproj
- netCDF4
- pandas
- tqdm
- pickle
- specfab: https://github.com/nicholasmr/specfab
- mcfab: https://github.com/dhrichards/mcfab


# Reproducing figures


The results of the paper can be reproduced by running 'simscript2.py' and 'streamages.py'



