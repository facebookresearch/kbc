# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pkg_resources
import os
import errno

from collections import defaultdict

DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    adjacency = {k: defaultdict(list) for k in files}
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        else:
            pass
    # write ent to id / rel to id
    for (d, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in d.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        to_write = open(os.path.join(DATA_PATH, name, f), 'w+')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            a_lhs, a_rhs = entities_to_id[lhs], entities_to_id[rhs]
            a_rel = relations_to_id[rel]
            to_write.write("{}\t{}\t{}\n".format(a_lhs, a_rel, a_rhs))
            adjacency[f][(str(a_lhs), str(a_rel), -1)].append(str(a_rhs))
            adjacency[f][(-1, str(a_rel), str(a_rhs))].append(str(a_lhs))
        to_write.close()

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        file_path = os.path.join(DATA_PATH, name, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            to_skip['lhs'][(rhs, rel)].add(lhs)
            to_skip['rhs'][(lhs, rel)].add(rhs)
        to_read.close()

    for k, v in to_skip.items():
        f = open(os.path.join(DATA_PATH, name, 'to_skip_{}'.format(k)), 'w+')
        for (a, b), value in v.items():
            f.write("{}\t{}\t{}\n".format(a, b, '\t'.join(sorted(value))))
        f.close()

if __name__ == "__main__":
    datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir, 'data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
