#!/bin/bash

# Download FB15K, FB15K-237, WN, WN18RR and YAGO3-10

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

wget https://s3.amazonaws.com/kbcdata/data.tar.gz
tar -xvzf data.tar.gz
rm data.tar.gz
