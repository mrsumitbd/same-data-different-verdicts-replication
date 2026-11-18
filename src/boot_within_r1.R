# boot_within_r1.R  (fast engine for src/boot_r1.py, BOOT_ENGINE=R)
#
# Repository cluster bootstrap of the repository-stratified Cox model, same scheme as
# src/analyze_rq1_master.py: repositories drawn with replacement from all repositories in the
# input (order of first appearance), each drawn copy its own stratum, Efron ties; the tier
# coefficient (column is_tier) is returned per draw. R's RNG (set.seed) differs from numpy's, so
# the draws are not the same resamples as the Python bootstrap; the scheme and B are.
#
# Usage: Rscript src/boot_within_r1.R input.csv output.csv n_boot seed n_cores [covariate ...]
suppressMessages(library(survival))
library(parallel)
a <- commandArgs(trailingOnly = TRUE)
x <- read.csv(a[1], stringsAsFactors = FALSE)
nb <- as.integer(a[3]); seed <- as.integer(a[4]); nc <- as.integer(a[5])
covs <- if (length(a) > 5) a[6:length(a)] else character(0)
repos <- unique(x$slug)
idx <- split(seq_len(nrow(x)), factor(x$slug, levels = repos))
set.seed(seed)
draws <- lapply(seq_len(nb), function(b) sample(length(repos), length(repos), replace = TRUE))
f <- as.formula(paste("Surv(duration_days, event) ~", paste(c("is_tier", covs), collapse = " + "), "+ strata(bstr)"))
one <- function(s) {
  rows <- idx[s]
  y <- x[unlist(rows), ]
  y$bstr <- rep(seq_along(s), lengths(rows))
  r <- tryCatch(coxph(f, data = y, ties = "efron"), error = function(e) NULL)
  if (is.null(r)) NA_real_ else exp(unname(coef(r)["is_tier"]))
}
hr <- unlist(mclapply(draws, one, mc.cores = nc, mc.preschedule = TRUE))
write.csv(data.frame(draw = seq_len(nb), hr = hr), a[2], row.names = FALSE)
