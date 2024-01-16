require("survival")
library(survminer)
library(optparse)


# 参数列表
option_list = list(
  optparse::make_option(c("-m", "--method"), action = "store", default = 'SubtypeWGME',
              type = 'character', help = "Path to genelist"),
  optparse::make_option(c("-t", "--type"), action = "store", default = 'OV',
              type = "character", help = "Path to genelist")
  # optparse::make_option(c("-e", "--experiment"), action = "store", default = 'results',
  #             type = "character", help = "Path to surv")
)


opt <- parse_args(optparse::OptionParser(option_list = option_list))

root_path <- "../results/"
# results_path <- paste(root_path, "pvalue", sep = "")
# if (!dir.exists(results_path)) {
#   dir.create(results_path)
# }

p_value <- c()
out_png <- c()

get.logrank.pvalue <- function(survdiff.res) {
  1 - pchisq(survdiff.res$chisq, length(survdiff.res$n) - 1)  
}

custom_theme <- function() {
  theme_survminer() %+replace%
    theme(
      plot.title=element_text(hjust=0.5)
    )
}

cancer_type <- c("CLLE","ESAD","MALY","OV","PACA","PAEN","RECA","BRCA") # nolint

data_operate <- function(cancer) {
  cluster_file <- paste(root_path, cancer,"/", cancer, ".", opt$method, sep = "") # nolint
  # 临床信息 ./results/BRCA.clinic
  clinic_file <- paste(root_path, cancer, "/clinic.csv", sep = "") # nolint

  # 读取文件
  atac_data <- read.table(cluster_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "") # nolint
  samples <- read.table(clinic_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "")  # nolint
  # 取交集

  ids <- intersect(rownames(atac_data), rownames(samples))
  samples <- samples[ids,] 
  samples$label <- atac_data[ids, 1]

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
  pvalue <- -log(pvalue,10)
  return(round(pvalue,3))
}

get_fig <- function(samples, cancer) {

  fit <- survfit(Surv(Survival, Death) ~ label, data = samples)

  gg <- ggsurvplot(fit, data = samples,
    title = cancer,
    # legend.title = "subtype",

    pval = TRUE,  # Add p-value and tervals
    # conf.int = TRUE,
    # surv.median.line = "hv", # Add medians survival
    font.main = c(16, "bold", "darkblue"),
    font.x = c(14, "bold.italic", "red"),
    font.y = c(14, "bold.italic", "darkred"),
    font.tickslab = c(12, "plain", "darkgreen"),
    # ggtheme = custom_theme()
    ggtheme = theme(
      panel.background = element_rect(fill='transparent'),
      plot.background = element_rect(fill='transparent', color=NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.background = element_rect(fill='transparent'),
      # legend.box.background = element_rect(fill='transparent'),
      

      axis.line.x = element_line(colour = "black",
                            size=1.0,
                            lineend = "butt"),
      axis.line.y = element_line(colour = "black",
                            size=1.0,
                            lineend = "butt"),
    )
  )

    tmp <- list(gg$plot)
    return(tmp)
}

if(opt$type != "all") {
    samples <- data_operate(opt$type)
    p <- get_p(samples)
    cat(p)
    tmp <- get_fig(samples,opt$type)
    out_file <- paste("C:/Users/xueyuAB/Desktop/文件/paper1/images/survival/",opt$type, ".png",sep="")
    png(out_file, width = 480, height = 480)
    print(tmp)
    dev.off()

    # d <- data.frame(opt$type, p, stringsAsFactors = FALSE)
    # p_path <- paste(results_path,"/survival_curs/",opt$type,"_p_value.txt", sep = "")
    # write.table(mat, 'C:/Users/xueyuAB/Desktop/文件/paper1/images/survival/test.txt', sep = ",", row.names = FALSE, quote = FALSE)
} else {
  for (cancer in cancer_type) {
      samples <- data_operate(cancer)

        p <- get_p(samples)
        tmp <- get_fig(samples,cancer)

    p_value <- c(p_value, p)
    out_png <- c(out_png, tmp)
  }
  print("successful")
  out_file <- paste("../../results/pvalue/survival_curs/surv.png", sep = "")
  png(out_file, width = 2000, height = 480)
  print(ggarrange(out_png[[1]], out_png[[2]], out_png[[3]], out_png[[4]], out_png[[5]], out_png[[6]], out_png[[7]], out_png[[8]],    # nolint
            ncol = 4, nrow = 2, align = "v"))
  dev.off()

#   d <- data.frame(cancer_type, p_value, stringsAsFactors = FALSE)
#   p_path <- paste(results_path, "/p_value.txt", sep = "")
#   write.table(d, p_path, sep = ",", row.names = FALSE, quote = FALSE)
}
