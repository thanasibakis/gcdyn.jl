library(tidyverse)

posterior_samples <- read_csv("posterior-samples.csv")

prior_samples <- tibble(
	λ_xscale = rlnorm(1000, 0.5, 0.75),
	λ_xshift = rnorm(1000, 5, 1),
	λ_yscale = rlnorm(1000, 0.5, 0.75),
	λ_yshift = rlnorm(1000, -0.5, 1.2),
	μ        = rlnorm(1000, 0, 0.5),
	δ        = rlnorm(1000, 0, 0.5)
) |>
	pivot_longer(c(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ), names_to = "Parameter", values_to = "Sample")

truth <- tibble(
	λ_xscale = 1,
	λ_xshift = 5,
	λ_yscale = 1.5,
	λ_yshift = 1,
	μ        = 1.3,
	δ        = 1
) |>
	pivot_longer(c(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ), names_to = "Parameter", values_to = "Truth")

posterior_summaries <- posterior_samples |>
	rename(λ_xshift_ = "λ_xshift⁻") |>
	mutate(
		λ_xscale = exp(log_λ_xscale * 0.75 + 0.5),
		λ_xshift = λ_xshift_ + 5,
		λ_yscale = exp(log_λ_yscale * 0.75 + 0.5),
		λ_yshift = exp(log_λ_yshift * 1.2 - 0.5),
		μ        = exp(log_μ * 0.5),
		δ        = exp(log_δ * 0.5)
	) |>
	pivot_longer(c(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ), names_to = "Parameter", values_to = "Sample") |>
	group_by(run, Parameter) |>
	summarise(
		Median = median(Sample),
		Q_025 = quantile(Sample, 0.025),
		Q_975 = quantile(Sample, 0.975),
		CI_Length = Q_975 - Q_025
	) |>
	ungroup()

ggplot() +
	geom_histogram(aes(Median, after_stat(density), fill = "Posterior medians"), data = posterior_summaries, alpha = 0.7) +
	geom_histogram(aes(Sample, after_stat(density), fill = "Prior"), data = prior_samples, alpha = 0.7) +
	geom_vline(aes(xintercept = Truth), data = truth, color = "black", size = 1.5) +
	scale_fill_manual(values = c("Posterior medians" = "dodgerblue4", "Prior" = "grey")) +
	facet_wrap(vars(Parameter), scales = "free") +
	labs(title = "Posterior median sampling distribution")

ggsave("posterior-medians.png", width = 15, height = 8, dpi = 300)

posterior_summaries |>
	full_join(truth, by = "Parameter") |>
	mutate(Relative_Error = (Median - Truth) / Truth) |>
	ggplot() +
	geom_histogram(aes(Relative_Error), fill = "dodgerblue4", alpha = 0.7) +
	xlim(-3, 3) +
	facet_wrap(vars(Parameter), scales = "free") +
	labs(title = "Relative error distribution for posterior medians")

ggsave("relative-errors.png", width = 15, height = 8, dpi = 300)

posterior_summaries |>
	full_join(truth, by = "Parameter") |>
	ggplot() +
	geom_histogram(aes(CI_Length / Truth), fill = "dodgerblue4", alpha = 0.7) +
	expand_limits(x = 0) +
	facet_wrap(vars(Parameter), scales = "free") +
	labs(title = "Length distribution of 95% CIs (normalized)")

ggsave("ci-lengths.png", width = 15, height = 8, dpi = 300)

posterior_summaries |>
	full_join(truth, by = "Parameter") |>
	group_by(Parameter) |>
	summarise(Coverage = mean((Truth >= Q_025) & (Truth <= Q_975))) |>
	write_tsv("ci-coverage-proportions.tsv")