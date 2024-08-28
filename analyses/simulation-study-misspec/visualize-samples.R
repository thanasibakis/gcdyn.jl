#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(tidyverse))

load_summaries <- function(csv_path) {
	read_csv(csv_path, show_col_types = F) |>
		select(-δ) |>
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
			.by = c(run, Parameter)
		) |>
		mutate(
			`Type space` = basename(csv_path) |>
				str_remove("posterior-samples-") |>
				str_remove(".csv")
		)
}

posterior_summaries <- map(
	list.files("out", pattern = "posterior-samples-.+", full.names = TRUE),
	load_summaries
) |>
	list_rbind()

priors <- list(
	`φ[1]` = \(x) dlnorm(x, 0.5, 0.75),
	`φ[2]` = \(x) dlnorm(x, 0.5, 0.75),
	`φ[3]` = \(x) dnorm(x, 0, sqrt(2)),
	`φ[4]` = \(x) dlnorm(x, -0.5, 1.2),
	μ      = \(x) dlnorm(x, 0, 0.5)
)

truth <- tribble(
	~Parameter, ~Truth,
	"φ[1]",     1.1,
	"φ[2]",     1,
	"φ[3]",     0.2,
	"φ[4]",     0.9,
	"μ",        1.3
)

for (parameter in truth$Parameter) {
	p <- ggplot() +
		stat_function(
			aes(fill = "Prior"),
			fun = priors[[parameter]],
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
		expand_limits(x = c(0, 3)) +
		theme_bw() +
		theme(base_size = 16) +
		facet_wrap(vars(`Type space`)) +
		labs(title = "Posterior median sampling distribution")

	ggsave(
		paste0("out/posterior-medians-", parameter, ".png"),
		p,
		width = 15,
		height = 8,
		dpi = 300
	)
}

posterior_summaries |>
	full_join(truth, by = "Parameter") |>
	group_by(Parameter, `Type space`) |>
	summarise(Coverage = mean((Truth >= Q_025) & (Truth <= Q_975))) |>
	write_tsv("out/ci95-coverage-proportions.tsv")