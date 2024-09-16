#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(tidyverse))

posterior_samples <- map(
	list.files("out", pattern = "posterior-samples-.+", full.names = TRUE),
	\(csv_path) {
		read_csv(csv_path, show_col_types = FALSE) |>
		mutate(
			`Type space` = basename(csv_path) |>
				str_remove("posterior-samples-") |>
				str_remove(".csv")
		) |>
		select(`Type space`, run, iteration, chain, starts_with("φ"), μ)
	}
) |>
	list_rbind()

prior_densities <- list(
	`φ[1]` = \(x) dlnorm(x, 0.5, 0.75),
	`φ[2]` = \(x) dlnorm(x, 0.5, 0.75),
	`φ[3]` = \(x) dnorm(x, 0, sqrt(2)),
	`φ[4]` = \(x) dlnorm(x, -0.5, 1.2),
	μ      = \(x) dlnorm(x, 0, 0.5)
)

prior_quantiles <- list(
	`φ[1]` = \(x) qlnorm(x, 0.5, 0.75),
	`φ[2]` = \(x) qlnorm(x, 0.5, 0.75),
	`φ[3]` = \(x) qnorm(x, 0, sqrt(2)),
	`φ[4]` = \(x) qlnorm(x, -0.5, 1.2),
	μ      = \(x) qlnorm(x, 0, 0.5)
)

truth <- tribble(
	~Parameter, ~Truth,
	"φ[1]",     1.1,
	"φ[2]",     1,
	"φ[3]",     0.2,
	"φ[4]",     0.9,
	"μ",        1.3
)

type_space <- tibble(
	`Type space` = c(rep("manual-6", 6), rep("quantile-6", 6), rep("manual-8", 8), rep("quantile-8", 8)),
	values = c(
		c(-2.4270176906430416, -1.4399117849363843, -0.2222600419251854, 0.28914500662573894, 1.3526378568771724, 2.1758707012574643),
		c(-0.6226600904206516, 0.0, 0.133576566862604, 0.5585342340982945, 1.013324787303958, 1.592738844284043),
		c(-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643),
		c(-0.9515369147366288, -0.07374813752966955, 0.0, 0.18401735874506064, 0.4942458884730887, 0.8592308603857539, 1.2483670286684694, 1.6914877080237338)
	)
)

# Plot posterior median sigmoid sampling distribution

X <- seq(-3, 3, 0.05)

sigmoid <- \(x, φ1, φ2, φ3, φ4) φ1 / (1 + exp(-φ2 * (x - φ3))) + φ4

posterior_median_λ_quantiles <-
	posterior_samples |>
	summarize(
		across(starts_with("φ"), median),
		.by = c(`Type space`, run)
	) |>
	mutate(Dist = paste("Posterior medians,", `Type space`)) |>
	select(-`Type space`) |>
	expand_grid(x = X) |>
	mutate(λ = sigmoid(x, `φ[1]`, `φ[2]`, `φ[3]`, `φ[4]`)) |>
	summarise(
		q05 = quantile(λ, 0.05),
		q10 = quantile(λ, 0.1),
		q20 = quantile(λ, 0.2),
		q30 = quantile(λ, 0.3),
		q40 = quantile(λ, 0.4),
		q60 = quantile(λ, 0.6),
		q70 = quantile(λ, 0.7),
		q80 = quantile(λ, 0.8),
		q90 = quantile(λ, 0.9),
		q95 = quantile(λ, 0.95),
		.by = c(Dist, x)
	)

prior_λ_quantiles <-
	map(
		truth$Parameter,
		\(param) tibble(
			Parameter = param,
			q05 = prior_quantiles[[param]](0.05),
			q10 = prior_quantiles[[param]](0.1),
			q20 = prior_quantiles[[param]](0.2),
			q30 = prior_quantiles[[param]](0.3),
			q40 = prior_quantiles[[param]](0.4),
			q60 = prior_quantiles[[param]](0.6),
			q70 = prior_quantiles[[param]](0.7),
			q80 = prior_quantiles[[param]](0.8),
			q90 = prior_quantiles[[param]](0.9),
			q95 = prior_quantiles[[param]](0.95),
		)
	) |>
	list_rbind() |>
	pivot_longer(
		c(starts_with("q")),
		names_to = "Quantile",
		values_to = "Value"
	) |>
	pivot_wider(
		names_from = Parameter,
		values_from = Value
	) |>
	expand_grid(x = X) |>
	mutate(λ = sigmoid(x, `φ[1]`, `φ[2]`, `φ[3]`, `φ[4]`)) |>
	select(x, λ, Quantile) |>
	pivot_wider(names_from = Quantile, values_from = λ) |>
	mutate(Dist = "Prior")

plot_sigmoid <- function(quantiles) {
	ggplot(quantiles, aes(x)) +
		facet_wrap(vars(Dist)) +
		geom_ribbon(
			aes(ymin = q05, ymax = q95), alpha = 0.15, fill = "dodgerblue4"
		) +
		geom_ribbon(
			aes(ymin = q10, ymax = q90), alpha = 0.15, fill = "dodgerblue4"
		) +
		geom_ribbon(
			aes(ymin = q20, ymax = q80), alpha = 0.15, fill = "dodgerblue4"
		) +
		geom_ribbon(
			aes(ymin = q30, ymax = q70), alpha = 0.15, fill = "dodgerblue4"
		) +
		geom_ribbon(
			aes(ymin = q40, ymax = q60), alpha = 0.15, fill = "dodgerblue4"
		) +
		geom_function(
			aes(color = "Truth"),
			fun = sigmoid,
			args = with(truth, list(
				φ1 = Truth[Parameter == "φ[1]"],
				φ2 = Truth[Parameter == "φ[2]"],
				φ3 = Truth[Parameter == "φ[3]"],
				φ4 = Truth[Parameter == "φ[4]"]
			)),
			linewidth = 1.5
		) +
		geom_vline(
			aes(xintercept = values),
			data = type_space |> mutate(Dist = paste("Posterior medians,", `Type space`)),
			linewidth = 1,
			linetype = "dashed"
		) +
		scale_color_manual(values = "black") +
		labs(
			title = "Posterior median sigmoid sampling distribution",
			y = expression(lambda(x))
		) +
		expand_limits(y = c(0, 2)) +
		theme_bw(base_size = 16) +
		theme(legend.position = "bottom", legend.title = element_blank())
}

ggsave(
	paste0("out/posterior-median-sigmoids.png"),
	bind_rows(prior_λ_quantiles, posterior_median_λ_quantiles) |> plot_sigmoid(),
	width = 15,
	height = 12,
	dpi = 300
)

ggsave(
	paste0("out/posterior-median-sigmoids-no-prior.png"),
	posterior_median_λ_quantiles |> plot_sigmoid(),
	width = 15,
	height = 12,
	dpi = 300
)

# Plot histograms

posterior_summaries <- posterior_samples |>
	pivot_longer(
		c(starts_with("φ"), μ),
		names_to = "Parameter",
		values_to = "Sample"
	) |>
	summarise(
		Median = median(Sample),
		Q_025 = quantile(Sample, 0.025),
		Q_975 = quantile(Sample, 0.975),
		CI_Length = Q_975 - Q_025,
		.by = c(`Type space`, run, Parameter)
	)

for (parameter in truth$Parameter) {
	p <- ggplot() +
		stat_function(
			aes(fill = "Prior"),
			fun = prior_densities[[parameter]],
			geom = "area",
		) +
		geom_histogram(
			aes(Median, after_stat(density), fill = "Posterior medians"),
			data = filter(posterior_summaries, Parameter == parameter),
			alpha = 0.5
		) +
		geom_vline(
			xintercept = with(truth, Truth[Parameter == parameter]),
			color = "black",
			linewidth = 1.5
		) +
		scale_fill_manual(
			values = c("Prior" = "grey", "Posterior medians" = "dodgerblue4")
		) +
		expand_limits(x = c(0, 3)) +
		theme_bw(base_size = 16) +
		theme(legend.position = "bottom", legend.title = element_blank()) +
		facet_wrap(vars(`Type space`)) +
		labs(title = "Posterior median sampling distribution", x = "Parameter")

	ggsave(
		paste0("out/posterior-medians-", parameter, ".png"),
		p,
		width = 15,
		height = 8,
		dpi = 300
	)
}

# Export coverage proportions

posterior_summaries |>
	full_join(truth, by = "Parameter") |>
	group_by(Parameter, `Type space`) |>
	summarise(Coverage = mean((Truth >= Q_025) & (Truth <= Q_975))) |>
	write_tsv("out/ci95-coverage-proportions.tsv")