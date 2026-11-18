# piecewise_r1.R 
# Piecewise Cox (0-30, 30-90, >90 days since merge; delayed entry), tier vs human, line level,
# identity tracking, all lines (spec-curve export of the primary data, Definition 1 and 2), as in
# src/check_ph_piecewise.py, with the variance estimators of C1:
#   across repositories: model-based (the paper's), clustered by repository, clustered by PR
#   within repository (strata = repository, repositories with both groups): clustered by repository
#   (the paper's), clustered by PR
suppressMessages(library(survival))
x <- read.csv("results_v6/round1/work/spec_v3all_line.csv", colClasses = c(unit_id = "NULL"))
cuts <- list(c(0, 30), c(30, 90), c(90, Inf))
out <- c()
for (dfn in c("e1", "e2")) for (tier in c("human_supervised", "fully_autonomous")) for (cmp in c("across", "within")) {
  y <- x[x$grp %in% c(tier, "human"), ]; y$t <- as.integer(y$grp == tier)
  if (cmp == "within") { both <- tapply(y$t, y$slug, function(v) length(unique(v)) == 2); y <- y[y$slug %in% names(both)[both], ] }
  for (ab in cuts) {
    a <- ab[1]; b <- ab[2]
    z <- y[y$d > a, ]; z$entry <- a; z$stop <- pmin(z$d, b); z$ev <- as.integer(z[[dfn]] == 1 & z$d <= b)
    f <- if (cmp == "within") coxph(Surv(entry, stop, ev) ~ t + strata(slug), data = z, ties = "efron")
         else coxph(Surv(entry, stop, ev) ~ t, data = z, ties = "efron")
    r <- as.numeric(residuals(f, type = "dfbeta"))
    b0 <- coef(f)[[1]]
    ses <- c(model = sqrt(f$var[1, 1]), cl_repo = sqrt(sum(rowsum(r, z$slug)^2)), cl_pr = sqrt(sum(rowsum(r, z$pr)^2)))
    row <- data.frame(definition = ifelse(dfn == "e1", "syntactic", "structural"), tier = tier, comparison = cmp,
                      interval = ifelse(is.finite(b), paste0(a, "-", b, " d"), paste0(">", a, " d")),
                      hr = exp(b0), events = sum(z$ev), n_repo = length(unique(z$slug)))
    for (k in names(ses)) { row[[paste0("lo_", k)]] <- exp(b0 - 1.959964 * ses[[k]]); row[[paste0("hi_", k)]] <- exp(b0 + 1.959964 * ses[[k]]); row[[paste0("p_", k)]] <- 2 * pnorm(-abs(b0 / ses[[k]])) }
    out <- rbind(out, row)
  }
}
write.csv(out, "results_v6/round1/c9e_piecewise.csv", row.names = FALSE)
print(out[, c("definition", "tier", "comparison", "interval", "hr", "lo_model", "hi_model", "lo_cl_repo", "hi_cl_repo", "p_cl_repo")], digits = 3)
