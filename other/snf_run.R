require(optparse)
require(SNFtool)
library(data.table)
library(dplyr)
source("util.R")

option_list = list(
  make_option(c("-i", "--input"), action = "store", default = './input/OV.list',
              type = 'character', help = "all_data"),
    make_option(c("-t", "--type"), action = "store", default = 'all',
              type = 'character', help = "all_data"),
  make_option(c("-n", "--cluster_num"), action="store", default=28,
              type='integer', help="cluster_num")
)

opt = parse_args(OptionParser(option_list = option_list))


cancer_type <- c("CLLE","ESAD","MALY","OV","PACA","PAEN","RECA","BRCA") # nolint
omics_type <-c("CNA","Mut","rna","miRNA")
omics_num  <- length(omics_type)
cluster_num <- c(6,2,3,3,6,6,4,4)

opt = parse_args(OptionParser(option_list = option_list))
alpha = 0.5
T = 20
K = 20

cancer_num <- length(cancer_type)

if(opt$type == "all") {

    result <- data.frame(cancer_type)
    time_result <- data.frame(cancer_type)
    time <- c()
    p_value = c()

    for (c in 1:cancer_num){
        cancer <- cancer_type[c]
        print(cancer)
        cluster_n = cluster_num[c]
        l <- list()

        for (j in 1:omics_num) {
            omics_file_path <- paste("../../fea/",cancer,"/",omics_type[j],'.fea',sep='') # nolint
             if(!file.exists(omics_file_path))
                next
            a <- fread(omics_file_path, check.names = FALSE, sep = ",", header = TRUE,data.table = FALSE,stringsAsFactors = FALSE)
            ids <- a[,1]
            a <- dplyr::select(a, -c(,1))
            a <- t(a)
            colnames(a) <- ids
            l[[j]] <- data.matrix(a)
            print(omics_file_path)
        }

        start_time <- Sys.time()
        
        alpha = 0.5
        T = 20
        K = 20
        similarity.data = lapply(l, function(x) { affinityMatrix(dist2(as.matrix(t(x)), as.matrix(t(x))),
                                                                K, alpha) })
        if (length(similarity.data) == 1) {
          W = similarity.data[[1]]
        } else {
          W = SNF(similarity.data, K, T)
        }


#         clinic_file <- paste('../../results/', cancer, "/", cancer,".clinic", sep = "") # nolint
#         samples <- read.table(clinic_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "")  # nolint

        subtype = spectralClustering(W, cluster_n)

        end_time <- Sys.time()
      
        df <- as.data.frame(subtype, col.names = c("SNF"), row.names = ids)
        cluster_file <- paste('../../results/',cancer,'/',cancer,'.SNF',sep='')
        write.table(df, file = cluster_file, quote = FALSE, sep = ",",row.names = TRUE)

        # data <- get_data(samples,df)
        # p <- get_p(data)
        # p_value <- c(p_value,p)
        time <- c(time,end_time-start_time)

      }

    # result <- cbind(result,p_value)
    # fileout <-paste("../../results/","SNF_pvalue", ".csv",sep="")
    # write.table(result, file = fileout, quote = FALSE, sep = ",",row.names = FALSE)

    time_result <- cbind(time_result,time)
    fileout <-paste("../../results/time/","SNF_time", ".csv",sep="")
    write.table(time_result, file = fileout, quote = FALSE, sep = ",",row.names = FALSE)
    }

else {
  print("not support yet")
  }
  

