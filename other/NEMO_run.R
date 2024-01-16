require(optparse)
library(data.table)
require(SNFtool)
library(dplyr)
source("NEMO.R")
source("NEMO_RESULTS.R")
source("benchmark.R")
source("util.R")

option_list = list(
  make_option(c("-i", "--input"), action = "store", default = './input/fea.list',
              type = 'character', help = "all_data"),
    make_option(c("-t", "--type"), action = "store", default = 'all',
              type = 'character', help = "all_data"),
  make_option(c("-n", "--cluster_num"), action="store", default=28,
              type='integer', help="cluster_num")
)

cancer_type <- c("CLLE","ESAD","MALY","OV","PACA","PAEN","RECA","BRCA") # nolint
cluster_num <- c(6, 2, 3, 3, 6, 6, 4,4)

opt = parse_args(OptionParser(option_list = option_list))

omics_type <-c("CNA","Mut","rna","miRNA")
omics_num  <- length(omics_type)
cancer_num <- length(cancer_type)


if(opt$type == "all") {
    result <- data.frame(cancer_type)
    p_value = c()

    time_result <- data.frame(cancer_type)
    time <- c()
    for (c in 1:cancer_num){   
        cancer <- cancer_type[c]
        print(cancer)
        cluster_n = cluster_num[c]
        l <- list()

        # read feature file
        for (i in 1:omics_num) {
            omics_file_path <- paste("../../fea/",cancer,"/",omics_type[i],'.fea',sep='') # nolint
            if(!file.exists(omics_file_path))
                next
            a <- fread(omics_file_path, check.names = FALSE, sep = ",", header = TRUE,data.table = FALSE,stringsAsFactors = FALSE)  # nolint
            ids <- a[,1]
            a <- dplyr::select(a, -c(,1))
            a <- t(a)
            colnames(a) <- ids
            l[[i]] <- a
            print(omics_file_path)
          }

        start_time <- Sys.time()

        # clinic_file <- paste('../../results/', cancer, "/clinic.csv", sep = "") # nolint
        # samples <- read.table(clinic_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "")  # nolint
        subtype = nemo.clustering(l, num.clusters=cluster_n)

        end_time <- Sys.time()

        df <- as.data.frame(subtype)
        cluster_file <- paste('../../results/',cancer,'/',cancer,'.NEMO',sep='') # nolint
        write.table(df, file = cluster_file, quote = FALSE, sep = ",",row.names = TRUE) # nolint
        # data <- get_data(samples,df)
        # p <- get_p(data)
        # print(rownames(df))
        # print(rownames(samples))
        # p <- get.empirical.surv(df,samples)
        # p_value <- c(p_value,p)
        time <- c(time,end_time-start_time)

    }

    # result <- cbind(result,p_value)
    # fileout <-paste("../../results/","NEMO_pvalue", ".csv",sep="") # nolint
    # write.table(result, file = fileout, quote = FALSE, sep = ",",row.names = FALSE) # nolint

    time_result <- cbind(time_result,time)
    fileout <-paste("../../results/time/","NEMO_time", ".csv",sep="") # nolint
    write.table(time_result, file = fileout, quote = FALSE, sep = ",",row.names = FALSE) # nolint

} else {
    print("don't support")
  }
  





