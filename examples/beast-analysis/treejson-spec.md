## TreeJSON spec

A `.history.trees` file from BEAST is represented as a JSON list of objects that represent trees.
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
            "state":  /* a string, the sequence at this node */,

            /*
                Specifies genetic mutations along the branch leading to this node.
                The list contains one object per mutation event.
            */
            "history": [
                {
                    site:      /* a number, the position in the sequence `state` */,
                    when:      /* a number, the branch length between the mutation and the most recent sampling time */,
                    from_base: /* a string, the base that was present prior to mutation (for sanity check) */,
                    to_base:   /* a string, the base that is present post-mutation */
                }
            ]
        }
    ]
}
```