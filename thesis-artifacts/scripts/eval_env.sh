# Environment for the official CARLA leaderboard evaluation.
# The leaderboard and scenario_runner copies are the ones vendored in the
# TransFuser repository: they are the exact versions the published Longest6
# numbers were produced with, so the scorer matches.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egca
export CARLA_ROOT=$HOME/carla_api
export LEADERBOARD_ROOT=$HOME/transfuser/leaderboard
export SCENARIO_RUNNER_ROOT=$HOME/transfuser/scenario_runner
export ROUTES6=$LEADERBOARD_ROOT/data/longest6
export PYTHONPATH=$CARLA_ROOT/carla:$CARLA_ROOT/carla/agents:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$HOME/thesis/code:$PYTHONPATH
export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export DEBUG_CHALLENGE=0
