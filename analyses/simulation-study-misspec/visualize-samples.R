#!/usr/bin/env Rscript

library(tidyverse)

# get path from cmd line
args <- commandArgs(trailingOnly = TRUE)
basename <- args[1] |> str_remove(".csv") |> str_remove("out/posterior-samples-")
posterior_samples <- read_csv(args[1])

prior_samples <- tibble(
	`φ[1]` = rlnorm(1000, 0.5, 0.75),
	`φ[2]` = rlnorm(1000, 0.5, 0.75),
	`φ[3]` = rnorm(1000, 0, sqrt(2)),
	`φ[4]` = rlnorm(1000, -0.5, 1.2),
	μ      = rlnorm(1000, 0, 0.5),
	δ      = rlnorm(1000, 0, 0.5)
) |>
	pivot_longer(
		c(starts_with("φ"), μ, δ),
		names_to = "Parameter",
		values_to = "Sample"
	) |>
	# Remove extreme samples to keep the histograms readable.
	# I should really start plotting prior density curves instead
	filter(Sample < quantile(Sample, 0.95), .by = Parameter)

truth <- tibble(
	`φ[1]` = 1.1,
	`φ[2]` = 1,
	`φ[3]` = 0.2,
	`φ[4]` = 0.9,
	μ      = 1.3,
	δ      = 1.1
) |>
	pivot_longer(
		c(starts_with("φ"), μ, δ),
		names_to = "Parameter",
		values_to = "Truth"
	)

posterior_summaries <- posterior_samples |>
	pivot_longer(
		c(starts_with("φ"), μ, δ),
		names_to = "Parameter",
		values_to = "Sample"
	) |>
	group_by(run, Parameter) |>
	summarise(
		Median = median(Sample),
		Q_025 = quantile(Sample, 0.025),
		Q_975 = quantile(Sample, 0.975),
		CI_Length = Q_975 - Q_025
	) |>
	ungroup()

ggplot() +
	geom_histogram(
		aes(Median, after_stat(density), fill = "Posterior medians"),
		data = posterior_summaries,
		alpha = 0.7
	) +
	geom_histogram(
		aes(Sample, after_stat(density), fill = "Prior"),
		data = prior_samples,
		alpha = 0.7
	) +
	geom_vline(
		aes(xintercept = Truth),
		data = truth,
		color = "black",
		size = 1.5
	) +
	scale_fill_manual(
		values = c("Posterior medians" = "dodgerblue4", "Prior" = "grey")
	) +
	facet_wrap(vars(Parameter), scales = "free") +
	labs(title = "Posterior median sampling distribution")

ggsave(paste0("out/posterior-medians-", basename, ".png"), width = 15, height = 8, dpi = 300)

# posterior_summaries |>
# 	full_join(truth, by = "Parameter") |>
# 	mutate(Relative_Error = (Median - Truth) / Truth) |>
# 	ggplot() +
# 	geom_histogram(aes(Relative_Error), fill = "dodgerblue4", alpha = 0.7) +
# 	xlim(-3, 3) +
# 	facet_wrap(vars(Parameter), scales = "free") +
# 	labs(title = "Relative error distribution for posterior medians")

# ggsave("relative-errors.png", width = 15, height = 8, dpi = 300)

# posterior_summaries |>
# 	full_join(truth, by = "Parameter") |>
# 	ggplot() +
# 	geom_histogram(aes(CI_Length / Truth), fill = "dodgerblue4", alpha = 0.7) +
# 	expand_limits(x = 0) +
# 	facet_wrap(vars(Parameter), scales = "free") +
# 	labs(title = "Length distribution of 95% CIs (normalized)")

# ggsave("ci-lengths.png", width = 15, height = 8, dpi = 300)

# posterior_summaries |>
# 	full_join(truth, by = "Parameter") |>
# 	group_by(Parameter) |>
# 	summarise(Coverage = mean((Truth >= Q_025) & (Truth <= Q_975))) |>
# 	write_tsv("ci-coverage-proportions.tsv")
