#!/usr/bin/env Rscript

library(tidyverse)

posterior_medians <- read_csv("posterior-medians.csv") |>
	pivot_longer(
		-num_trees,
		names_to = c("Likelihood", "Parameter"),
		names_pattern = "(corrected|original)_(λ|μ)",
		values_to = "Median"
	)

truth <- tribble(
	~Parameter, ~Value,
	"λ", 1.8,
	"μ", 1
)

ggplot(posterior_medians) +
	geom_histogram(
		aes(Median, after_stat(density), fill = Likelihood),
		alpha = 0.5,
		position = "identity"
	) +
	geom_vline(data = truth, aes(xintercept = Value, color = "Truth")) +
	facet_grid(rows = vars(Parameter), cols = vars(num_trees), scales = "free_y")