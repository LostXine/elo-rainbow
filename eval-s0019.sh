#!/bin/bash

F_END=${0: -5:2}
F_START=${0: -7:2}

echo "This script will eval code from random seed $F_START to $F_END"

for i in $(seq $F_START $F_END)
do
    echo "Seed $i"
    bash eval-backbone.sh $i
done

# python3 ../mail.py
