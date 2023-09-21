## Documentation: `beast-to-treejson`

BEAST .history.trees information extracter.

### Requirements

Tested on R 4.2.

```{r}
install.packages(c("tidyverse", "BiocManager"))
BiocManager::install("treeio")
```

### Usage

```{shell}
extract-tree-data.R [myfile.history.trees]
```

The input file should be the .history.trees file output by BEAST when `States > State Change Count Reconstruction > Reconstruct complete change history on tree` is enabled in BEAUti.
Be sure that the `compactHistory` option of the `markovJumpsTreeLikelihood` entry in your BEAST XML file is set to `false` (which is the default) or is unset.

The output (printed to stdout) will be a JSON list of objects representing each tree, obeying the TreeJSON spec (see below).

## TreeJSON spec

`beast-to-treejson` parses a `.history.trees` file from BEAST into a JSON list of objects that represent trees.
These objects obey the following schema:

```{json}
{
    /*
        Specifies the tree topology.
    */
    newick: [ /* a string, the Newick representation of the tree, with named tips */],

    /*
        Contains names and auxiliary information for each node.
    */
    "nodes": [
        {
            "name":   /* a number, assigned to each node in the tree for identification purposes */,
            "parent": /* a number, the name of the parent of this node */,
            "state":  /* a string, the sequence alignment at this node */,

            /*
                Specifies genetic mutations along the branch leading to this node.
                The list contains one object per mutation event.
            */
            "history": [
                {
                    site:      /* a number, the position in the sequence alignment `state` */,
                    when:      /* a number, the branch length between this node and the mutation */,
                    from_base: /* a string, the base that was present prior to mutation (for sanity check) */,
                    to_base:   /* a string, the base that is present post-mutation */
                }
            ]
        }
    ],

    /*
        Saves the original names of the tips in the tree, since the `newick` string uses the names defined in the `nodes` object.
    */
    original_tip_names: {
        "name": /* a number, matching the `nodes` object */,
        "original_name": /* a string, the original name of the tip */
    }
}
```

## Documentation: `get-affinity`

Reads a list of sequences (separated by newlines) from stdin or a file passed as an argument, and writes the KD values to stdout.

### Requirements

Tested on Python 3.9.

```{shell}
pip install git+https://github.com/matsengrp/gcdyn.git
```
