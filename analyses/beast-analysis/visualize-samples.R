#!/usr/bin/env Rscript

library(tidyverse)
library(glue)

sigmoid <- function(x, t1, t2, t3, t4) {
	t1 / (1 + exp(-t2 * (x - t3))) + t4
}

# Read data

samples_filename <- commandArgs(trailingOnly = TRUE)[1]
basename <- samples_filename |>
	str_replace(".csv", "") |>
	str_replace("out/samples-", "")

prior_samples <- read_csv("out/samples-prior.csv") |>
	mutate(
		λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5),
		λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5),
		λ_xshift = λ_xshift_base * sqrt(2),
		λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5),
		μ        = exp(log_μ_base * 0.5),
		δ        = exp(log_δ_base * 0.5)
	) |>
	select(iteration, chain, λ_xscale, λ_yscale, λ_xshift, λ_yshift, μ, δ)

posterior_samples <- read_csv(samples_filename) |>
	mutate(
		λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5),
		λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5),
		λ_xshift = λ_xshift_base * sqrt(2),
		λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5),
		μ        = exp(log_μ_base * 0.5),
		δ        = exp(log_δ_base * 0.5)
	) |>
	select(iteration, chain, λ_xscale, λ_yscale, λ_xshift, λ_yshift, μ, δ)

# Plot sigmoids

prior_q <- prior_samples |>
	select(starts_with("λ_")) |>
	pmap(\(λ_xscale, λ_yscale, λ_xshift, λ_yshift) tibble(x = seq(-5, 5, 0.05), sig = sigmoid(x, λ_xscale, λ_yscale, λ_xshift, λ_yshift))) |>
	bind_rows(.id = "Curve") |>
	group_by(x) |>
	summarise(median = quantile(sig, 0.5), q05 = quantile(sig, 0.05), q20 = quantile(sig, 0.2), q80 = quantile(sig, 0.8), q95 = quantile(sig, 0.95)) |>
	mutate(Dist = "Prior")

posterior_q <- posterior_samples |>
	select(starts_with("λ_")) |>
	pmap(\(λ_xscale, λ_yscale, λ_xshift, λ_yshift) tibble(x = seq(-5, 5, 0.05), sig = sigmoid(x, λ_xscale, λ_yscale, λ_xshift, λ_yshift))) |>
	bind_rows(.id = "Curve") |>
	group_by(x) |>
	summarise(median = quantile(sig, 0.5), q05 = quantile(sig, 0.05), q20 = quantile(sig, 0.2), q80 = quantile(sig, 0.8), q95 = quantile(sig, 0.95)) |>
	mutate(Dist = "Posterior")

y_max <- 8

bind_rows(prior_q, posterior_q) |>
	mutate(Dist = factor(Dist, levels = c("Prior", "Posterior"))) |>
	ggplot(aes(x)) +
	geom_ribbon(aes(ymin = q05, ymax = pmin(q95, y_max), fill="95% CI"), alpha = 0.5) +
	geom_ribbon(aes(ymin = q20, ymax = q80, fill="80% CI"), alpha = 0.5) +
	geom_line(aes(y = median, fill="Median"), color="dodgerblue4", linewidth=1.5) +
	scale_fill_manual(values = c("95% CI" = "grey", "80% CI" = "#979696", "Median" = "dodgerblue4"), name = NULL) +
	ylim(0, y_max) +
	ylab(expression(lambda[x])) +
	ggtitle("Birth rate sigmoid distribution") +
	theme_bw(base_size = 32) +
	theme(legend.position = "bottom") +
	facet_wrap(vars(Dist))

ggsave(glue("out/sigmoids-{basename}.png"), width = 15, height = 9, dpi=300)

# Plot histograms

prior_samples <- prior_samples |>
	pivot_longer(c(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ), names_to = "Parameter", values_to = "Sample")

posterior_samples <- posterior_samples |>
	pivot_longer(c(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ), names_to = "Parameter", values_to = "Sample")
	

ggplot() +
	geom_histogram(aes(Sample, after_stat(density), fill = "Posterior"), data = posterior_samples, alpha = 0.7) +
	geom_histogram(aes(Sample, after_stat(density), fill = "Prior"), data = prior_samples, alpha = 0.7) +
	scale_fill_manual(values = c("Posterior" = "dodgerblue4", "Prior" = "grey")) +
	facet_wrap(vars(Parameter), scales = "free") +
	theme_bw(base_size = 32) +
	theme(legend.position = "bottom") +
	labs(title = "Posterior histograms")

ggsave(glue("out/histograms-{basename}.png"), width = 18, height = 12, dpi = 300)

