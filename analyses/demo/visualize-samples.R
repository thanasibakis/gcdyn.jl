#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(tidyverse))

posterior_samples <- read_csv("posterior-samples.csv")

prior_samples <- tibble(
	λ_yscale = rlnorm(1000, 0.5, 0.75),
	λ_xscale = rlnorm(1000, 0.5, 0.75),
	λ_xshift = rnorm(1000, 5, 1),
	λ_yshift = rlnorm(1000, -0.5, 1.2),
)

truth <- tibble(
	λ_yscale = 1.5,
	λ_xscale = 1,
	λ_xshift = 5,
	λ_yshift = 1,
)

sigmoid <- function(x, λ_yscale, λ_xscale, λ_xshift, λ_yshift) {
	λ_yscale / (1 + exp(-λ_xscale * (x - λ_xshift))) + λ_yshift
}

prior_q <- prior_samples |>
	pmap(sigmoid, x = seq(0, 10, 0.05)) |>
	map(\(l) tibble(x = seq(0, 10, 0.05), sig = l)) |>
	bind_rows(.id = "Curve") |>
	group_by(x) |>
	summarise(
		q05 = quantile(sig, 0.05),
		q20 = quantile(sig, 0.2),
		q80 = quantile(sig, 0.8),
		q95 = quantile(sig, 0.95)
	) |>
	mutate(Dist = "Prior")

posterior_q <- posterior_samples |>
	select(λ_yscale, λ_xscale, λ_xshift, λ_yshift) |>
	pmap(sigmoid, x = seq(0, 10, 0.05)) |>
	map(\(l) tibble(x = seq(0, 10, 0.05), sig = l)) |>
	bind_rows(.id = "Curve") |>
	group_by(x) |>
	summarise(
		q05 = quantile(sig, 0.05),
		q20 = quantile(sig, 0.2),
		q80 = quantile(sig, 0.8),
		q95 = quantile(sig, 0.95)
	) |>
	mutate(Dist = "Posterior")

y_max <- 8

plt <- bind_rows(prior_q, posterior_q) |>
	mutate(Dist = factor(Dist, levels = c("Prior", "Posterior"))) |>
	ggplot(aes(x)) +
	geom_ribbon(
		aes(ymin = q05, ymax = pmin(q95, y_max), fill = "95% CI"),
		alpha = 0.5
	) +
	geom_ribbon(aes(ymin = q20, ymax = q80, fill = "80% CI"), alpha = 0.5) +
	geom_function(
		aes(fill = "Truth"),
		color = "dodgerblue4",
		fun = sigmoid,
		args = truth,
		linewidth = 1.5
	) +
	scale_fill_manual(
		values = c("95% CI" = "grey", "80% CI" = "#979696", "Truth" = "dodgerblue4"),
		name = NULL
	) +
	ylim(0, y_max) +
	ylab("$\\lambda_x$") +
	theme_bw(base_size = 32) +
	theme(legend.position = "bottom") +
	facet_wrap(vars(Dist))

ggsave("sigmoids.png", width = 15, height = 7)

plt <- posterior_samples |>
	select(iteration, starts_with("λ"), μ, δ) |>
	pivot_longer(-iteration, names_to = "Parameter", values_to = "Value") |>
	ggplot() +
	geom_line(aes(iteration, Value)) +
	facet_wrap(vars(Parameter), scales = "free_y")

ggsave("traceplots.png", width = 15, height = 7)