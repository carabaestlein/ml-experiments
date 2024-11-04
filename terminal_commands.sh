# create and activate python virtual environment
python3 -m venv wine_recs_venv # first time only
source wine_recs_venv/bin/activate

# install dependencies
python3 -m pip install ipython numpy matplotlib # first time only

# start IPython console
ipython

# exits IPython
control + D

# deactivate virtual environment
deactivate