#!/bin/bash

session="nodes"
tmux new-session -d -s $session
tmux set -g mouse on

#!/bin/bash

session="WEI"
folder="~/workspace/polybot_workcell"
tmux new-session -d -s $session
tmux set -g mouse on

window=0
tmux new-window -t $session:$window -n 'redis'
tmux send-keys -t $session:$window 'cd ' $folder C-m
tmux send-keys -t $session:$window 'envsubst < redis.conf | redis-server -' C-m


window=1
tmux new-window -t $session:$window -n 'server'
tmux send-keys -t $session:$window 'cd ' $folder C-m
tmux send-keys -t $session:$window 'source ~/wei_ws/install/setup.bash' C-m
tmux send-keys -t $session:$window 'python3 -m rpl_wei.server --workcell ~/workspace/polybot_workcell/polybot_workcell.yaml' C-m

window=2
tmux new-window -t $session:$window -n 'worker'
tmux send-keys -t $session:$window 'cd ' $folder C-m
tmux send-keys -t $session:$window 'source ~/wei_ws/install/setup.bash' C-m
tmux send-keys -t $session:$window 'python3 -m rpl_wei.processing.worker' C-m


window=3
tmux new-window -t $session:$window -n 'ur5'
tmux send-keys -t $session:$window 'source ~/wei_ws/install/setup.bash' C-m
tmux send-keys -t $session:$window 'ros2 launch ur_client ur_client.launch.py ip:=146.139.48.76' C-m


tmux attach-session -t $session

