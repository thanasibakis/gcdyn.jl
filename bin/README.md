This folder gets added to the PATH for the environment that the nextflow script runs in.

## TreeJSON spec

Scripts in this folder pass around JSON lists of objects that represent trees.
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
