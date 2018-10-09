# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.sparse import csr_matrix
import collections
import numpy as np


def loadSparse(fd, one_based=False, transpose=False, prop_kept=1):
    """
        Reads the lines in fd, strip, split("\t"), convert to int.
        Builds a list of csr frontal slices.
        The lines in fd are expected to be :
            lhs_id \t rel_id \t rhs_id
    """
    parse_line = None
    if one_based:
        parse_line = lambda line: [int(x) - 1 for x in line.strip().split("\t")]
    else:
        parse_line = lambda line: [int(x) for x in line.strip().split("\t")]
    frontal_slices = collections.defaultdict(
        lambda: {"data": [], "row": [], "col": []}
    )
    max_left = 0
    max_mid = 0  # SVO verbs are not contiguous -_-
    max_right = 0  # SVO verbs are not contiguous -_-
    for (l, rel, r) in (parse_line(line) for line in fd):
        if transpose:
            r, rel = rel, r
        max_left, max_mid, max_right = (
            max(max_left, l), max(max_mid, rel), max(max_right, r))
        if np.random.uniform() > prop_kept:
            continue
        frontal_slices[rel]["data"].append(1)
        frontal_slices[rel]["row"].append(l)
        frontal_slices[rel]["col"].append(r)

    to_return = []
    for rel in range(max_mid + 1):
        if rel in frontal_slices:
            to_return.append(csr_matrix(
                (frontal_slices[rel]["data"], (frontal_slices[rel]["row"],
                                               frontal_slices[rel]["col"])),
                shape=(max_left + 1, max_right + 1), dtype='d'
            ))
        else:
            to_return.append(csr_matrix(
                ([], ([], [])),
                shape=(max_left + 1, max_right + 1), dtype='d'
            ))
    return to_return


def loadSparseAsList(fd, one_based=False, rel2rel=None):
    """
        Same as above but returns a list
    """
    parse_line = None
    if one_based:
        parse_line = lambda line: [int(x) - 1 for x in line.strip().split("\t")]
    else:
        parse_line = lambda line: [int(x) for x in line.strip().split("\t")]

    res = [parse_line(line) for line in fd]

    if rel2rel is not None:
        return [(x, rel2rel[y], z) for (x, y, z) in res]

    return res
