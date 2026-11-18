# km_logrank_r1.R 
# Log-rank tests per tier vs human on the primary data (identity tracking, all lines,
# vendored/generated removed; Definition 1), from the spec-curve export
# (results_v6/round1/work/spec_v3all_line.csv):
#   (1) unweighted log-rank (every line counts once; as in the paper)
#   (2) log-rank stratified by repository (unweighted)
#   (3) PR-weighted: the log-rank test is not defined for case weights; we report the robust
#       (cluster = PR) score test of the PR-weighted Cox model, its large-sample analogue,
#       and the same with cluster = repository.
# Also for Definition 2 (AST-structural).
suppressMessages(library(survival))
x <- read.csv("results_v6/round1/work/spec_v3all_line.csv", colClasses = c(unit_id = "NULL"))
out <- c()
for (dfn in c("e1", "e2")) for (tier in c("human_supervised", "fully_autonomous")) {
  y <- x[x$grp %in% c(tier, "human"), ]; y$t <- as.integer(y$grp == tier); y$e <- y[[dfn]]
  lr <- survdiff(Surv(d, e) ~ t, data = y)
  lrs <- survdiff(Surv(d, e) ~ t + strata(slug), data = y)
  f1 <- coxph(Surv(d, e) ~ t, data = y, weights = w, cluster = pr, ties = "efron")
  f2 <- coxph(Surv(d, e) ~ t, data = y, weights = w, cluster = slug, ties = "efron")
  s1 <- summary(f1)$robscore; s2 <- summary(f2)$robscore
  out <- rbind(out, data.frame(definition = ifelse(dfn == "e1", "syntactic", "structural"), tier = tier,
    logrank_chisq = lr$chisq, logrank_p = pchisq(lr$chisq, 1, lower.tail = FALSE),
    obs_tier = lr$obs[2], exp_tier = lr$exp[2],
    strat_logrank_chisq = lrs$chisq, strat_logrank_p = pchisq(lrs$chisq, 1, lower.tail = FALSE),
    prw_hr = exp(coef(f1)), prw_robust_score_pr = s1["test"], prw_robust_score_pr_p = s1["pvalue"],
    prw_robust_score_repo = s2["test"], prw_robust_score_repo_p = s2["pvalue"]))
}
write.csv(out, "results_v6/round1/c9d_logrank.csv", row.names = FALSE)
print(out)
