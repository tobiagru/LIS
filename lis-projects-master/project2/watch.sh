#!/bin/bash

fswatch *.py | while read f; do
    fname="/media/psf/${f:7}"
    dir=$(dirname "$fname")
    file=$(basename "$fname")
    echo -e "\r------------------- executing '$file' -------------------"
    ssh scipy "cd '$dir'; './$file'"
    echo -n "----- done $RANDOM -----"
done
