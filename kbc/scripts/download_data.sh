# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Download FB15K, FB15K-237, WN, WN18RR and YAGO3-10

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

wget https://dl.fbaipublicfiles.com/kbc/data.tar.gz
tar -xvzf data.tar.gz
rm data.tar.gz
