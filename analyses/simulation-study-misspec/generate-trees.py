#!/usr/bin/env python3
# %%
import json
from pathlib import Path

from experiments import replay
from gcdyn.bdms import TreeError, TreeNode
from gcdyn.gpmap import AdditiveGPMap
from gcdyn.mutators import ContextMutator, SequencePhenotypeMutator
from gcdyn.poisson import ConstantResponse, SigmoidResponse

naive_sequence = "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA"
birth_response = SigmoidResponse(1, 0.2, 1.1, 0.9)
death_response = ConstantResponse(1.3)
mutation_response = ConstantResponse(1.1)
present_time = 4
survivor_sampling_prob = 0.8

dms = replay.dms(Path(__file__).parent / "bin" / "support" / "final_variant_scores.csv")

gp_map = AdditiveGPMap(dms["affinity"], nonsense_phenotype=dms["affinity"].min())
mutator = SequencePhenotypeMutator(
    ContextMutator(mutability=replay.mutability(), substitution=replay.substitution()),
    gp_map,
)

num_trees = 1500


def generate_tree():
    try:
        root = TreeNode()
        root.sequence = naive_sequence
        root.x = gp_map(naive_sequence)
        root.chain_2_start_idx = replay.CHAIN_2_START_IDX

        root.evolve(
            t=present_time,
            birth_response=birth_response,
            death_response=death_response,
            mutation_response=mutation_response,
            mutator=mutator,
            verbose=False,
            min_survivors=0,
        )

        root.sample_survivors(p=survivor_sampling_prob)
        root.prune()

        survivors = filter(
            lambda node: node.event == "sampling",
            root.get_leaves(),
        )

        if len(list(survivors)) == 0:
            return generate_tree()

        return root
    except ValueError:
        # TODO: why do we get this error
        return generate_tree()
    except TreeError:
        # Tree went extinct, try again
        return generate_tree()


def export_tree(tree):
    return {
        "affinity": tree.x,
        "time": tree.t,
        "event": tree.event if tree.event else "root",
        "children": [export_tree(child) for child in tree.children],
    }


# %%
trees = [generate_tree() for _ in range(num_trees)]
json_trees = [export_tree(tree) for tree in trees]

with open("trees.json", "w") as f:
    json.dump(json_trees, f)

# %%
