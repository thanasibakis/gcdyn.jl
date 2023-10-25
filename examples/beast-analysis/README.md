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
beast-to-treejson [myfile.history.trees]
```

The input file should be the .history.trees file output by BEAST when `States > State Change Count Reconstruction > Reconstruct complete change history on tree` is enabled in BEAUti.
Be sure that the `compactHistory` option of the `markovJumpsTreeLikelihood` entry in your BEAST XML file is set to `false` (which is the default) or is unset.

The output (printed to stdout) will be a JSON list of objects representing each tree, obeying the TreeJSON spec (see below).

## Documentation: `get-affinity`

Reads a list of sequences (separated by newlines) from stdin or a file passed as an argument, and writes the KD values to stdout.

### Requirements

Tested on Python 3.9.

```{shell}
pip install git+https://github.com/matsengrp/gcdyn.git
```
