#!/bin/bash

run="$1"
[ -z "$1" ] && run='process.py'

if [ -x "$HOME/anaconda/bin/python" ]; then
    "$HOME/anaconda/bin/python" $run
else
    python $run
fi
