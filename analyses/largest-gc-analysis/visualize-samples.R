#!/usr/bin/env Rscript

cat("Loading packages...\n")
suppressPackageStartupMessages(library(tidyverse))
library(glue)

cat("Reading data...\n")
prior_samples <-
	read_csv("out/samples-prior.csv", show_col_types = FALSE) |>
	mutate(
		λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5),
		λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5),
		λ_xshift = λ_xshift_base * sqrt(2),
		λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5),
		μ        = exp(log_μ_base * 0.5),
		δ        = exp(log_δ_base * 0.5)
	) |>
	select(iteration, chain, λ_xscale, λ_yscale, λ_xshift, λ_yshift, μ, δ)

posterior_samples <-
	
	map_dfr(
		1:23,
		\(i) read_csv(glue("out/samples-posterior-{i}.csv"), show_col_types = FALSE),
		.id = "GC"
	) |>
	filter(iteration > 1000) |>
	mutate(
		λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5),
		λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5),
		λ_xshift = λ_xshift_base * sqrt(2),
		λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5),
		μ        = exp(log_μ_base * 0.5),
		δ        = exp(log_δ_base * 0.5)
	) |>
	select(GC, iteration, chain, λ_xscale, λ_yscale, λ_xshift, λ_yshift, μ, δ)

posterior_samples |>
	group_by(GC, chain) |>
	summarise(
		median = median(λ_xscale),
		q05 = quantile(λ_xscale, 0.05),
		q95 = quantile(λ_xscale, 0.95)
	) |>
	ggplot() +
	geom_pointrange(
		aes(y = factor(as.numeric(GC)), x = median, xmin = q05, xmax = q95)
	) +
	theme_bw() +
	theme(legend.position = "bottom") +
	labs(
		x = "λ_xscale",
		y = "Germinal center",
		title = "xscale medians and 95% CIs",
		subtitle = "across germinal centers"
	)

posterior_samples |>
	group_by(GC, chain) |>
	summarise(
		median = median(μ),
		q05 = quantile(μ, 0.05),
		q95 = quantile(μ, 0.95)
	) |>
	ggplot() +
	geom_pointrange(
		aes(y = factor(as.numeric(GC)), x = median, xmin = q05, xmax = q95)
	) +
	theme_bw() +
	theme(legend.position = "bottom") +
	labs(
		x = "μ",
		y = "Germinal center",
		title = "xscale medians and 95% CIs",
		subtitle = "across germinal centers"
	)
ggsave("out2.png", dpi = 600)
