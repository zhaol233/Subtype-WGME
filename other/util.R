
require("survival")
library(survminer)
require(parallel)
# library(optparse)


get_data <- function(samples, cluster_df) {

#   cluster_file <- paste(root_path, cancer,"/", cancer, ".", opt$method, sep = "") # nolint
#   # 临床信息 ./results/BRCA.clinic
#   clinic_file <- paste(root_path, cancer, "/", cancer,".clinic", sep = "") # nolint

#   # 读取文件
#   atac_data <- read.table(cluster_file, check.names = FALSE, row.names = 1, header = TRUE, sep = '\t',comment.char = "") # nolint
#   samples <- read.table(clinic_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "")  # nolint
  # 取交集

  ids <- intersect(rownames(cluster_df), rownames(samples))

  samples <- samples[ids,] 
  samples$label <- cluster_df[ids, 1]

  clustering <- samples[, "label"]


  names(clustering) <- rownames(samples)
  samples <- transform(samples, Survival = as.numeric(days), Death = as.numeric(status)) # nolint
  samples$Survival[is.na(samples$Survival)] <- 0
  samples$Death[is.na(samples$Death)] <- 0
  return(samples)
}

get_p <- function(samples) {
    surv_diff <- survdiff(Surv(Survival, Death) ~ label, data = samples)
    pvalue <- get.logrank.pvalue(surv_diff)
    print(pvalue)
    pvalue <- -log(pvalue,10)
    print(pvalue)
    return(round(pvalue,5))
}

get.logrank.pvalue <- function(survdiff.res) {
  1 - pchisq(survdiff.res$chisq, length(survdiff.res$n) - 1)  
}

check.survival <- function(groups, ori_surv_data) {
  patient.names = rownames(groups) # 
  # print(patient.names)
  # print(ori_surv_data[, 1])
  # patient.names.in.file = as.character(ori_surv_data[, 1]) # %in%: %in% is a more intuitive interface as a binary operator, which returns a logical vector indicating if there is a match or not for its left operand.
  patient.names.in.file = rownames(ori_surv_data)
  stopifnot(all(patient.names %in% patient.names.in.file)) # stopifnot: If any of the expressions (in ... or exprs) are not all TRUE, stop is called, producing an error message indicating the first expression which was not (all) true.
  indices = match(patient.names, patient.names.in.file) # match: match returns a vector of the positions of (first) matches of its first argument in its second.
  # print(indices)
  ordered.survival.data = ori_surv_data[indices,]
  ordered.survival.data <- transform(ordered.survival.data, Survival = as.numeric(days), Death = as.numeric(status))
  ordered.survival.data["cluster"] <- groups
  ordered.survival.data$Survival[is.na(ordered.survival.data$Survival)] = 0
  ordered.survival.data$Death[is.na(ordered.survival.data$Death)] = 0
  return(survdiff(Surv(Survival, Death) ~ cluster, data = ordered.survival.data)) # survdiff: Tests if there is a difference between two or more survival curves using the G-rho family of tests, or for a single curve against a known alternative.
}


get.empirical.surv <- function(clustering, ori_surv_data) {
  set.seed(42)
  surv.ret = check.survival(clustering, ori_surv_data)
  orig.chisq = surv.ret$chisq
  orig.pvalue = get.logrank.pvalue(surv.ret)
  print(orig.pvalue)
  # return(orig.pvalue)

  #The initial number of permutations to run
  num.perms = round(min(max(10 / orig.pvalue, 1000), 1e6))
  should.continue = T
  
  total.num.perms = 0
  total.num.extreme.chisq = 0
  
  while (should.continue) {
    print('Another iteration in empirical survival calculation')
    perm.chisq = as.numeric(mclapply(1:num.perms, function(i) {
      cur.clustering = sample(clustering)
      # print(rownames(cur.clustering))
      names(cur.clustering) = names(clustering)
      cur.chisq = check.survival(cur.clustering, ori_surv_data)$chisq
      return(cur.chisq)
    }, mc.cores = 1))
    
    total.num.perms = total.num.perms + num.perms
    total.num.extreme.chisq = total.num.extreme.chisq + sum(perm.chisq >= orig.chisq)
    
    binom.ret = binom.test(total.num.extreme.chisq, total.num.perms)
    cur.pvalue = binom.ret$estimate
    print(binom.ret$p.value)
    cur.conf.int = binom.ret$conf.int
    
    print(c(total.num.extreme.chisq, total.num.perms))
    print(cur.pvalue)
    print(cur.conf.int)
    
    print(cur.conf.int[2] - cur.pvalue)
    sig.threshold = 0.05
    is.conf.small = ((cur.conf.int[2] - cur.pvalue) < min(cur.pvalue / 10, 0.01)) & ((cur.pvalue - cur.conf.int[1]) < min(cur.pvalue / 10, 0.01))
    is.threshold.in.conf = cur.conf.int[1] < sig.threshold & cur.conf.int[2] > sig.threshold
    if ((is.conf.small & !is.threshold.in.conf) | (total.num.perms > 2e-7)) {
      should.continue = F
    } else {
      num.perms = 1e5
    }
  }
  print("final")
  print(cur.pvalue)
  return(cur.pvalue)
}