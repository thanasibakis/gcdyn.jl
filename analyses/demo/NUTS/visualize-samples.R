#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(tidyverse))

posterior_samples <- read_csv("posterior-samples.csv") |>
	rename(λ_xshift_ = "λ_xshift⁻") |>
	mutate(
		θ1 = exp(log_λ_yscale * 0.75 + 0.5),
		θ2 = exp(log_λ_xscale * 0.75 + 0.5),
		θ3 = λ_xshift_ + 5,
		θ4 = exp(log_λ_yshift * 1.2 - 0.5),
	)

prior_samples <- tibble(
	θ1 = rlnorm(1000, 0.5, 0.75),
	θ2 = rlnorm(1000, 0.5, 0.75),
	θ3 = rnorm(1000, 5, 1),
	θ4 = rlnorm(1000, -0.5, 1.2),
)

truth <- tibble(
	θ1 = 1.5,
	θ2 = 1,
	θ3 = 5,
	θ4 = 1,
)

sigmoid <- function(x, t1, t2, t3, t4) {
	t1 / (1 + exp(-t2 * (x - t3))) + t4
}

prior_q <- prior_samples |>
	pmap(\(θ1, θ2, θ3, θ4) tibble(x = seq(0, 10, 0.05), sig = sigmoid(x, θ1, θ2, θ3, θ4))) |>
	bind_rows(.id = "Curve") |>
	group_by(x) |>
	summarise(q05 = quantile(sig, 0.05), q20 = quantile(sig, 0.2), q80 = quantile(sig, 0.8), q95 = quantile(sig, 0.95)) |>
	mutate(Dist = "Prior")

posterior_q <- posterior_samples |>
	select(starts_with("θ")) |>
	pmap(\(θ1, θ2, θ3, θ4) tibble(x = seq(0, 10, 0.05), sig = sigmoid(x, θ1, θ2, θ3, θ4))) |>
	bind_rows(.id = "Curve") |>
	group_by(x) |>
	summarise(q05 = quantile(sig, 0.05), q20 = quantile(sig, 0.2), q80 = quantile(sig, 0.8), q95 = quantile(sig, 0.95)) |>
	mutate(Dist = "Posterior")

y_max <- 8

plt <- bind_rows(prior_q, posterior_q) |>
	mutate(Dist = factor(Dist, levels = c("Prior", "Posterior"))) |>
	ggplot(aes(x)) +
	geom_ribbon(aes(ymin = q05, ymax = pmin(q95, y_max), fill = "95% CI"), alpha = 0.5) +
	geom_ribbon(aes(ymin = q20, ymax = q80, fill = "80% CI"), alpha = 0.5) +
	geom_function(
		aes(fill = "Truth"),
		color = "dodgerblue4",
		fun = sigmoid,
		args = list(t1 = truth$θ1, t2 = truth$θ2, t3 = truth$θ3, t4 = truth$θ4),
		linewidth=1.5
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
	select(iteration, starts_with("log"), λ_xshift_) |>
	pivot_longer(-iteration, names_to = "Parameter", values_to = "Value") |>
	ggplot() +
	geom_line(aes(iteration, Value)) +
	facet_wrap(vars(Parameter), scales = "free_y")

ggsave("traceplots.png", width = 15, height = 7)