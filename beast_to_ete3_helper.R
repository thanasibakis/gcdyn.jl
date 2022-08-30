# BiocManager::install(c("treeio", "ggtree"))
suppressPackageStartupMessages(library(ggtree))
suppressPackageStartupMessages(library(treeio))
suppressPackageStartupMessages(library(tidyverse))

extract_tree_data <- function(tree) {
    history <- get.data(tree) %>%
        select(node, starts_with("history")) %>%
        mutate(across(starts_with("history"),
            ~ map(.,
                ~ str_c(
                    .[seq(1, length(.), 3)],
                    .[seq(1, length(.), 3) + 1],
                    .[seq(1, length(.), 3) + 2],
                    sep = "|"
                )
            )
        )) %>%
        pivot_longer(starts_with("history"),
            names_to = "site",
            names_prefix = "history_",
            values_to = "history",
            values_drop_na = TRUE
        ) %>%
        unnest(history) %>%
        drop_na(history) %>%
        mutate(history = str_replace_all(history, "\\}|\\{", "")) %>%
        separate(history, c("when", "from_base", "to_base"), "\\|")

    states <- get.data(tree) %>%
        select(node, states)

    ancestry <- get.tree(tree)$edge %>%
        as_tibble(.name_repair = "minimal") %>%
        magrittr::set_colnames(c("parent", "node"))

    list(
        history = history,
        states = states,
        ancestry = ancestry
    )
}

get_output_filename <- function(suffix) {
    prefix <- str_split(basename(filename), "\\.")[[1]][1]
    str_glue("{output_dir}/{prefix}-{suffix}")
}

filename <- commandArgs(trailingOnly = TRUE)[1] # .history.trees file
output_dir <- commandArgs(trailingOnly = TRUE)[2]

# This step takes the longest
cat("Parsing trees...")
trees <- read.beast(filename)
cat("\t\tdone.\n")

cat("Preparing tree data...")
tree_data <- trees %>% map(extract_tree_data)
cat("\t\tdone.\n")

cat("Writing tree histories...")
tree_data %>%
    map_dfr(~ .$history, .id = "tree_id") %>%
    relocate(tree_id, node) %>%
    write_csv(get_output_filename("history.csv"))
cat("\tdone.\n")

cat("Writing tree ancestries...")
tree_data %>%
    map_dfr(~ .$ancestry, .id = "tree_id") %>%
    relocate(tree_id, node, parent) %>%
    write_csv(get_output_filename("ancestry.csv"))
cat("\tdone.\n")

cat("Writing tree states...")
tree_data %>%
    map_dfr(~ .$states, .id = "tree_id") %>%
    relocate(tree_id, node, states) %>%
    write_csv(get_output_filename("states.csv"))
cat("\t\tdone.\n")

cat("Writing tree structures...")
trees %>%
    map(get.tree) %>%
    magrittr::set_names(seq_along(trees)) %>%
    write.nexus(file = get_output_filename("structures.nexus"))
cat("\tdone.\n")
