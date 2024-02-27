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

The output (printed to stdout) will be a JSON list containing representations of each tree accordin to the TreeJSON spec.
