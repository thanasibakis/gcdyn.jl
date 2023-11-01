## TreeJSON spec

A tree is represented as a JSON list of objects that represent nodes (ancestral and at the tips).
These objects obey the following schema:

```{json}
{
    "name":   /* a number, assigned to each node in the tree for identification purposes */,
    "parent": /* a number (or `null`), the name of the parent of this node */,
    "length": /* a number, the length of the branch leading to this node */,
    "state":  /* a string, the sequence at this node */,

    /*
        If applicable, specifies genetic mutations along the branch leading to this node.
        The list contains one object per mutation event.

        (It would be equivalent to not specify any history, and instead include unifurcating nodes
        in the JSON list of nodes.)
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
```

Consequently, a set of trees is represented as a JSON list of lists of these objects.
