# BEAST .history.trees --> ete3 trees converter.
# Tested on python 3.9
#
# Usage:
#   python3 beast_to_ete3.py [myfile.history.trees]
#
# Outputs:
#   - myfile-ete3_trees.pickle, containing a dictionary of ete3 trees
#     indexed by their ID number
#   - beast_parser_temp/, a directory containing data files with relevant
#     tree properties inserted into the ete3 objects, as well as a copy of
#     the .trees file with some unused information removed (to keep file
#     size small)


import pandas as pd
import pickle
import re
import subprocess
import sys
from collections import namedtuple
from ete3 import Tree
from gcdyn.phenotype import DMSPhenotype
from pathlib import Path
from tqdm import tqdm


phenotype = DMSPhenotype(
    1,
    1,
    336,
    "https://raw.githubusercontent.com/jbloomlab/Ab-CGGnaive_DMS/main/data/CGGnaive_sites.csv",
    "tdms-linear.model", # from gcdyn 29-sampling branch
    ["delta_log10_KD", "delta_expression", "delta_psr"],
    -10.43,
)


def apply_site_change(sequence, site_change):
    index = site_change.where - 1
    assert sequence[index] == site_change.from_base

    return sequence[:index] + site_change.to_base + sequence[index + 1 :]


# Adds intermediate nodes between the parent of `child` & `child` itself
# to describe the given sequence of site changes.
def grow_branch(child, site_changes):
    if not site_changes:
        return
    
    parent = child.get_ancestors()[0]
    child.detach()

    intermediate = parent
    previous_distance = 0

    site_changes = sorted(site_changes,
        key = lambda site_change: site_change.when,
        reverse = True # remember time 0 is leaves, and time > 0 is distance to leaves
    )

    # Skip the last site change because we already have the resulting child...
    for site_change in site_changes[:-1]:
        new_sequence = apply_site_change(intermediate.sequence, site_change)
        intermediate = intermediate.add_child(dist = site_change.when - previous_distance)
        previous_distance = site_change.when

        intermediate.add_features(
            sequence = new_sequence,
            x = phenotype.calculate_KD([new_sequence])[0],
            event = "mutation"
        )

    # ...but make sure it'd be correct anyway
    # TODO: this made me notice that tree 14000 has a mutation unaccounted for by the history. What gives?
    # assert apply_site_change(intermediate.sequence, site_changes[-1]) == child.sequence

    intermediate.add_child(child = child)


if __name__ == "__main__":
    filename = Path(sys.argv[1])
    basename = filename.stem.split(".")[0]
    output_dir = Path("beast_parser_temp")

    output_dir.mkdir(exist_ok = True)

    # This makes the R package in the script parse faster
    print("Preprocessing trees file...", end = "")
    temp_file = output_dir/f"{basename}.trees"

    with open(filename) as in_file, open(temp_file, "w") as out_file:
        for line in in_file:
            if line.startswith("tree"):
                line = re.sub(r"(?:c_count|c_allTransitions)\[\d+\]=[0-9.]+,?", "", line)
                line = re.sub(r"count={[0-9.,]+},?", "", line)
                line = re.sub(r",\]", "]", line)
                line = re.sub(r"\[&\]", "", line)

            out_file.write(line)

    print("\tdone.")

    subprocess.run(["Rscript", "beast_to_ete3_helper.R", temp_file, output_dir]).check_returncode()

    trees = dict()

    tree_history = pd.read_csv(f"{output_dir}/{basename}-history.csv")
    tree_ancestry = pd.read_csv(f"{output_dir}/{basename}-ancestry.csv")
    tree_states = pd.read_csv(f"{output_dir}/{basename}-states.csv")

    SiteChange = namedtuple("SiteChange", ("when", "where", "from_base", "to_base"))

    with open(f"{output_dir}/{basename}-structures.nexus") as trees_file:
        for line in tqdm(trees_file, desc = "Converting trees to ete3"):

            # Find the tree definitions within the nexus file
            if not line.strip().startswith("TREE"):
                continue

            # Extract the tree_id and newick definition
            info, tree_definition = line.split(" = ")
            tree_id = int(re.search(r"TREE \* (\d+)", info).group(1))
            tree_definition = tree_definition.strip("[&R] ")

            tree = Tree(tree_definition)

            # Currently, only leaf nodes are labeled in the tree.
            # The internal node labels are saved in tree_data.
            # Let's populate them (we'll need them for the history)
            for node in tree.iter_leaves():
                while node.up and not node.up.name:
                    parent_label = tree_ancestry.parent[
                        (tree_ancestry.tree_id == tree_id) &
                        (tree_ancestry.node == int(node.name))
                    ].astype(int).values[0]

                    node.up.name = parent_label
                    node = node.up

            # Populate the sequences for each node
            for node in tree.traverse():
                sequence = tree_states.states[
                    (tree_states.tree_id == tree_id) &
                    (tree_states.node == int(node.name))
                ].values[0]

                node.add_features(
                    sequence = sequence,
                    x = phenotype.calculate_KD([sequence])[0],
                    event = "birth"
                )

            # Expand out the tree to reflect mutations along a branch.
            # Be sure to make a copy (list()) of the node sequence now,
            # because we are going to add nodes (that don't need traversing)
            for node in list(tree.traverse("levelorder")):
                history = tree_history[
                    (tree_history.tree_id == tree_id) &
                    (tree_history.node == int(node.name))
                ].sort_values(by = "when", ascending = False)

                site_changes = [
                    SiteChange(*row) for row in zip(
                        history.when,
                        history.site,
                        history.from_base,
                        history.to_base
                    )
                ]

                # Note that node.history contains the mutations that led *to* this node,
                # not that follow this node
                grow_branch(node, site_changes)

            trees[tree_id] = tree
        
    with open(f"{basename}-ete3_trees.pickle", "wb") as file:
        pickle.dump(trees, file)

    print("All done!")