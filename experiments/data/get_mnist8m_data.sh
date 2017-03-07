#!/bin/bash

if [ -d infimnist ]
then
    exit
fi

wget http://leon.bottou.org/_media/projects/infimnist.tar.gz
tar -xvf infimnist.tar.gz
cd infimnist
make
./infimnist pat 10000 8109999 > "train-patterns"
./infimnist lab 10000 8109999 > "train-labels"
./infimnist pat 0 9999 > "test-patterns"
./infimnist lab 0 9999 > "test-labels"
gzip "train-patterns"
gzip "train-labels"
gzip "test-patterns"
gzip "test-labels"
cd ..
