# cox_variance_r1.R 
#
# Refits every specification of the specification curve (src/specification_curve.py)
# with R survival::coxph (Efron ties, the same partial likelihood as lifelines), and
# reports for each fit the hazard ratio with four variance estimators:
#   model    : model-based (inverse information); what the paper uses for
#              unweighted across-repository fits
#   robust   : per-observation sandwich (Lin-Wei), what the paper uses for
#              PR-weighted across-repository fits (lifelines robust=True)
#   cl_repo  : sandwich clustered by repository (what the paper uses within repository)
#   cl_pr    : sandwich clustered by PR
# Sandwich = sum over clusters of (sum of weighted dfbeta residuals)^2, which is what
# coxph(..., cluster=) computes; no small-sample correction (as in lifelines and R).
#
# Comparisons: across (all repositories, unstratified), across_shared (unstratified,
# only repositories that contain both the tier and human code; new, C1c) and within
# (stratified by repository, repositories with both groups).
# Tiers: human_supervised, fully_autonomous, and pooled (all agents; RQ1 step 1).
# Extra definition 'structural_known' (v3 all lines): AST-structural with modified
# lines of unknown AST status dropped (RQ2 sensitivity).
#
# Usage: Rscript src/cox_variance_r1.R [n_cores] [window: full|c8|c8h|c10h]
# Input : results_v6/round1/work/spec_datasets.csv (+ the CSVs it lists)
# Output: results_v6/round1/c1_cox_variance.csv

suppressMessages(library(survival))
library(parallel)
args <- commandArgs(trailingOnly = TRUE)
ncores <- if (length(args) >= 1) as.integer(args[1]) else 8L
# C5: observation window as a specification choice (second argument)
#   full   all PRs, followed to the cutoff (the paper's design; >= 5 months)
#   c8     PRs with >= 8 months (240 days) of possible follow-up, followed to the cutoff
#   c8h    same cohort, every unit censored 240 days after merge (equal follow-up)
#   c10h   PRs with >= 10 months (300 days), censored 300 days after merge
window <- if (length(args) >= 2) args[2] else "full"
horizon <- c(full = NA, c8 = NA, c8h = 240, c10h = 300)[[window]]
min_fu <- c(full = 0, c8 = 240, c8h = 240, c10h = 300)[[window]]
wd <- "results_v6/round1"
ds <- read.csv(file.path(wd, "work", "spec_datasets.csv"), stringsAsFactors = FALSE)

fit_one <- function(x, weighted, strat) {
  f <- if (strat) Surv(d, e) ~ t + strata(slug) else Surv(d, e) ~ t
  fit <- if (weighted) coxph(f, data = x, weights = w, ties = "efron") else coxph(f, data = x, ties = "efron")
  b <- unname(coef(fit)[1])
  r <- residuals(fit, type = "dfbeta", weighted = TRUE)
  r <- as.numeric(r)
  v_model <- fit$var[1, 1]
  v_rob <- sum(r^2)
  v_repo <- sum(rowsum(r, x$slug)^2)
  v_pr <- sum(rowsum(r, x$pr)^2)
  out <- data.frame(coef = b, hr = exp(b), n = nrow(x), events = sum(x$e),
                    n_repo = length(unique(x$slug)), n_pr = length(unique(x$pr)),
                    n_pr_tier = length(unique(x$pr[x$t == 1])), n_repo_tier = length(unique(x$slug[x$t == 1])))
  for (nm in c("model", "robust", "cl_repo", "cl_pr")) {
    v <- switch(nm, model = v_model, robust = v_rob, cl_repo = v_repo, cl_pr = v_pr)
    se <- sqrt(v)
    out[[paste0("se_", nm)]] <- se
    out[[paste0("lo_", nm)]] <- exp(b - qnorm(0.975) * se)
    out[[paste0("hi_", nm)]] <- exp(b + qnorm(0.975) * se)
    out[[paste0("p_", nm)]] <- 2 * pnorm(-abs(b / se))
  }
  out
}

tasks <- list()
for (i in seq_len(nrow(ds))) {
  unit_kind <- ds$unit_kind[i]
  units <- if (unit_kind == "line") c("line", "PR-weighted") else c("hunk")
  defs <- "syntactic"
  if (unit_kind == "line" && ds$has_ast[i] %in% c(TRUE, "True")) defs <- c(defs, "structural")
  if (ds$key[i] == "v3all_line") defs <- c(defs, "structural_known")
  for (u in units) for (df in defs) for (cmp in c("across", "across_shared", "within"))
    for (tier in c("human_supervised", "fully_autonomous", "pooled"))
      tasks[[length(tasks) + 1]] <- list(i = i, unit = u, definition = df, comparison = cmp, tier = tier)
}
cat(length(tasks), "fits\n")

run_task <- function(tk, x) {
  if (tk$tier == "pooled") {
    x$t <- as.integer(x$grp != "human")
  } else {
    x <- x[x$grp %in% c(tk$tier, "human"), ]
    x$t <- as.integer(x$grp == tk$tier)
  }
  x$e <- switch(tk$definition, syntactic = x$e1, structural = x$e2, structural_known = x$e2)
  if (tk$definition == "structural_known") x <- x[x$ast_known == 1, ]
  if (tk$comparison != "across") {
    both <- tapply(x$t, x$slug, function(v) length(unique(v)) == 2)
    x <- x[x$slug %in% names(both)[both], ]
  }
  t0 <- Sys.time()
  res <- tryCatch(fit_one(x, tk$unit == "PR-weighted", tk$comparison == "within"),
                  error = function(e) { message(conditionMessage(e)); NULL })
  if (is.null(res)) return(NULL)
  cbind(data.frame(window = window, key = ds$key[tk$i], data = ds$data[tk$i], unit = tk$unit, definition = tk$definition,
                   comparison = tk$comparison, tier = tk$tier, secs = as.numeric(Sys.time() - t0, units = "secs")),
        res)
}

# one dataset at a time: load in the parent, fork workers over its fits
all_res <- list()
for (i in seq_len(nrow(ds))) {
  x <- read.csv(ds$path[i], stringsAsFactors = FALSE, colClasses = c(unit_id = "NULL"))
  if (window != "full") {
    fu <- read.csv(file.path(wd, "work", paste0("followup_", ds$key[i], ".csv")), stringsAsFactors = FALSE)
    stopifnot(nrow(fu) == nrow(x), all(fu$pr == x$pr))
    x <- x[fu$followup >= min_fu, ]
    if (!is.na(horizon)) {
      late <- x$d > horizon
      x$e1[late] <- 0L; x$e2[late] <- 0L; x$d[late] <- horizon
    }
  }
  tk_i <- Filter(function(tk) tk$i == i, tasks)
  t0 <- Sys.time()
  r <- mclapply(tk_i, run_task, x = x, mc.cores = ncores, mc.preschedule = FALSE)
  all_res[[i]] <- do.call(rbind, r)
  cat(ds$key[i], length(tk_i), "fits in", round(as.numeric(Sys.time() - t0, units = "mins"), 1), "min\n")
  rm(x); gc(verbose = FALSE)
  write.csv(do.call(rbind, all_res), file.path(wd, if (window == "full") "c1_cox_variance.csv" else paste0("c5_cox_window_", window, ".csv")), row.names = FALSE)
}
cat("done\n")
