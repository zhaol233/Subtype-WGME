require(optparse)
library(data.table)
library(dplyr)
require(PMA)
source("util.R")

get.elbow <- function(values, is.max) {
  second.derivatives = c()
  for (i in 2:(length(values) - 1)) {
    second.derivative = values[i + 1] + values[i - 1] - 2 * values[i]
    second.derivatives = c(second.derivatives, second.derivative)
  }
  print(second.derivatives)
  if (is.max) {
    return(which.max(second.derivatives) + 1)
  } else {
    return(which.min(second.derivatives) + 1)
  }
}

option_list = list(
  make_option(c("-i", "--input"), action = "store", default = './tmp/crc.list',
              type = 'character', help = "all_data"),
  make_option(c("-t", "--type"), action = "store", default = 'all',
              type = 'character', help = "all_data"),
  make_option(c("-n", "--cluster_num"), action = "store", default = 28,
              type = 'integer', help = "cluster_num")
)
opt = parse_args(OptionParser(option_list = option_list))


cancer_type <- c("CLLE","ESAD","MALY","OV","PACA","PAEN","RECA","BRCA") # nolint
cluster_num <- c(6,2,3,3,6,6,4,4)
omics_type <-c("CNA","Mut","rna","miRNA")
omics_num  <- length(omics_type)
cancer_num <- length(cancer_type)

penalty = NULL
rep.omic = 1
nb_fea = 1


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
            a <- fread(omics_file_path, check.names = FALSE, sep = ",", header = TRUE,data.table = FALSE,stringsAsFactors = FALSE)
            ids <- a[,1]
            a <- dplyr::select(a, -c(,1))
            a <- t(a)
            mat <- data.matrix(a)
            if (dim(mat)[1] == 0) { next }
            l[[i]] <- t(data.matrix(a))
            print(omics_file_path)

        }

#         clinic_file <- paste("../../results/", cancer, "/", cancer,".clinic", sep = "") # nolint
#         samples_df <- read.table(clinic_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "")  # nolint

        start_time <- Sys.time()

        cca.ret = PMA::MultiCCA(l, ncomponents = cluster_n, penalty = penalty)
        sample.rep = l[[rep.omic]] %*% cca.ret$ws[[rep.omic]]

        subtype = kmeans(sample.rep, cluster_n, iter.max = 100, nstart = 30)$cluster

        end_time <- Sys.time()

        df <- as.data.frame(subtype, col.names = c("MCCA"), row.names = ids)
        cluster_file <- paste("../../results/",cancer,'/',cancer,'.MCCA',sep='')
        write.table(df, file = cluster_file, quote = FALSE, sep = ",",row.names = TRUE)

#         data <- get_data(samples_df,df)
#         p <- get_p(data)
#         p_value <- c(p_value,p)
        time <- c(time,end_time-start_time)
      }

#     result <- cbind(result,p_value)
#     fileout <-paste("../../results/","MCCA_pvalue", ".csv",sep="")
#     write.table(result, file = fileout, quote = FALSE, sep = ",",row.names = FALSE)
#
#
    time_result <- cbind(time_result,time)
    fileout <-paste("../../results/time/","MCCA_time", ".csv",sep="")
    write.table(time_result, file = fileout, quote = FALSE, sep = ",",row.names = FALSE)


} else {
   print('dont support')
  }
  



