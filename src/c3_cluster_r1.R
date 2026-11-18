# c3_cluster_r1.R 
# Definition 2 current vs continued past leaf-only edits (results_v6/round1/work/c3_def2_continued.csv
# from src/c3_refit_r1.py): HR with model-based, repository-clustered and PR-clustered SEs,
# across repositories (all repositories and shared repositories) and within repository (strata).
suppressMessages(library(survival))
x <- read.csv("results_v6/round1/work/c3_def2_continued.csv", stringsAsFactors = FALSE)
out <- c()
for (v in c("cur", "con")) for (tier in c("human_supervised", "fully_autonomous")) for (cmp in c("across", "across_shared", "within")) {
  y <- x[x$grp %in% c(tier, "human"), ]; y$t <- as.integer(y$grp == tier)
  y$d <- y[[paste0("d_", v)]]; y$e <- y[[paste0("e_", v)]]
  if (cmp != "across") { both <- tapply(y$t, y$slug, function(z) length(unique(z)) == 2); y <- y[y$slug %in% names(both)[both], ] }
  f <- if (cmp == "within") coxph(Surv(d, e) ~ t + strata(slug), data = y, ties = "efron") else coxph(Surv(d, e) ~ t, data = y, ties = "efron")
  r <- as.numeric(residuals(f, type = "dfbeta")); b <- coef(f)[[1]]
  se <- c(model = sqrt(f$var[1, 1]), cl_repo = sqrt(sum(rowsum(r, y$slug)^2)), cl_pr = sqrt(sum(rowsum(r, y$pr)^2)))
  row <- data.frame(version = ifelse(v == "cur", "current (censored at leaf-only)", "continued past leaf-only"),
                    tier = tier, comparison = cmp, hr = exp(b), events_tier = sum(y$e[y$t == 1]), mr_tier = mean(y$e[y$t == 1]) * 100,
                    mr_human = mean(y$e[y$t == 0]) * 100)
  for (k in names(se)) { row[[paste0("lo_", k)]] <- exp(b - 1.959964 * se[[k]]); row[[paste0("hi_", k)]] <- exp(b + 1.959964 * se[[k]]); row[[paste0("p_", k)]] <- 2 * pnorm(-abs(b / se[[k]])) }
  out <- rbind(out, row)
}
write.csv(out, "results_v6/round1/c3_def2_clustered.csv", row.names = FALSE)
print(out[, c("version", "tier", "comparison", "hr", "mr_tier", "mr_human", "lo_model", "hi_model", "lo_cl_repo", "hi_cl_repo", "p_cl_repo")], digits = 3)
