
# library(corrplot) 
# library(vegan) 
# library(ggcor)
# library(ggplot2)




figure_latent_heatmap <- function(cancer){
    library(RColorBrewer)
    library(ComplexHeatmap, help, pos = 2, lib.loc = NULL)
    library(circlize)

    df<-read.csv(paste("H:/我的云端硬盘/ZL/paper1/fea/",cancer,"/",cancer,".fea",sep = ""),header=1,row.names = 1)
    df_c<-read.csv(paste("H:/我的云端硬盘/ZL/paper1/results/",cancer,"/",cancer,".SubtypeWGME",sep = ""),header=1,row.names = 1)
    df$subtype <- df_c[,'subtype']
    df <- df[order(df$subtype),]
    type <- df[257]
    type_n = type[nrow(type),]

    type_n = 3
    df<-df[,-257]

    df <- t(df)
    df <- as.matrix(df)

    col_fun <- colorRamp2(
    1:type_n,
    brewer.pal(n = type_n, name = 'Spectral')
    )

    ha = HeatmapAnnotation(
        subtype=type[,1],
        col = list(
            subtype = col_fun
        ),
        annotation_legend_param = list(
            subtype = list(
                title = "Subtype",
                at = 1:type_n,
                # labels = c("zero", "median", "one","zero", "median", "one"),
                # legend_gp = brewer.pal(n = type_n, name = 'Spectral')
                # gpar(fill=1:6)
                color_bar = "discrete"
                
            )
        )
    )
    t = Heatmap(df,
        name="feature",
        km=1,
        # col=colorRamp2(c(-10,0,10),c("green","white","red")),
        show_row_names = F,
        show_column_names=F,
        cluster_columns=F,
        # cluster_rows = F,
        top_annotation = ha,
        show_row_dend = F
        # heatmap_legend_param = list(direction = "horizontal")
    ) 
    png(file = paste("C:/Users/xueyuAB/Desktop/文件/paper1/images/latent_heatmap/",cancer,".png"),width = 400,height = 400,bg = "transparent")
    print(t)
    dev.off()
}



# figure nmi heat map
figure_nmi <- function(){
    library(corrplot)
    library(RColorBrewer)
    nmi <- read.csv("H:/我的云端硬盘/ZL/paper1/results/BRCA/BRCA.nmi",header=TRUE,
        row.names = 1,sep = ',')
    nmi <- as.matrix(nmi)
    m = par(no.readonly = TRUE) 
    par(family= "Times New Roman")
    png(file = "C:/Users/xueyuAB/Desktop/文件/paper1/images/BRCAnmi.png",width = 400,height = 400,bg = "transparent")
    corrplot(nmi, method = "circle",
        col = brewer.pal(n = 5, name = 'YlGn'),  #
        bg='#f0f0f0',   # 指定图的背景色
        title = 'BRCA',  # 标题
        mar=c(0, 0, 1, 0),
        # addgrid.col='black',  # 网格颜色
        # addCoef.col = '' ,        为相关系数添加颜色，
        # addCoefasPercent = TRUE,
        # addshade = 'all',
        addCoef.col = "black",
        # tl.cex = 1,BRCA
        # order = 'AOE',
        number.cex = 1.0,
        # order = 'hclust',    
        #  cl.pos = 'b',
        #  cl.cex = 1.5,

        #  tl.cex = 1.5,
        # insig = 'p-value',
        # pch.cex = 1.5,
        diag=TRUE,
        is.corr = FALSE,
        tl.col='black',
        tl.cex = 1.0,
        # tl.pos = 'lb'
        # type='upper'
    )
    dev.off()
    par(m) # 还原初始图形参数设置
}


figure_single_gene_survival<-function(cancer){
    # library(limma)
    library(survival)
    library(survminer) 
    root_path <- "C:/Users/xueyuAB/Desktop/文件/paper1/files/"
    clinic_file <- paste(root_path, "clinic/",cancer, "/clinic.csv", sep = "") # nolint
    fea_file <-  paste(root_path,"results3/biomarker_rawfea/" ,cancer, "/rawfea.csv", sep = "") # nolint

      # 读取文件
    fea <- read.table(fea_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "") # nolint
    samples <- read.table(clinic_file, check.names = FALSE, row.names = 1, header = TRUE, sep = ',',comment.char = "")  # nolint


    fea=cbind(as.data.frame(fea), samples)
    fea$days=fea$days/365

    # gene_list <- c("YPEL3","RP11-452L6.7","RP11-134G8.8","AC005682.5","RBPJL(ss)","GALNT7(ss)","CD200","ZNF217")

    gene_list <- c("ZNF217")
    gene_n  <- length(gene_list)
    for (c in 1:gene_n){
        gene <- gene_list[c]
        Type <- ifelse(fea[,gene]>median(fea[,gene]), "High", "Low")
        fea <- cbind(as.data.frame(fea), Type)

        diff <- survdiff(Surv(days, status) ~ Type, data=fea)
        pValue <- 1-pchisq(diff$chisq, df=1)
        if(pValue < 0.001){
            pValue="p<0.001"
        }else{
            pValue=paste0("p=", sprintf("%.03f",pValue))
        }
        fit <- survfit(Surv(days, status) ~ Type, data = fea) 

        surPlot=ggsurvplot(fit,
            data=fea,
            # conf.int=TRUE,
            pval=pValue,
            pval.size=2,
            
            # surv.median.line = "hv",
            legend.title=gene,
            legend.labs=c("High level", "Low level"),
            
            xlab="Time(years)",
            ylab="Survival probability",
            break.time.by = 1,
            palette=c("red", "blue"),
            
            # palette = "simpsons",
            risk.table=F,
            risk.table.title="",
            risk.table.col = "strata",
     
            font.y = c(5, "italic", "black"),
            font.x = c(5, "italic", "black"),
            font.tickslab = c(4,'bold','black'),
 
            # risk.table.height=.25,
            ggtheme = theme(
                panel.background = element_rect(fill='transparent'),
                plot.background = element_rect(fill='transparent', color=NA),
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                legend.background = element_rect(fill='transparent'),
                # legend.box.background = element_rect(fill='transparent'),
                
            
                axis.line.x = element_line(colour = "black",
                                     linewidth=0.03,
                                     lineend = "butt"),
                axis.line.y = element_line(colour = "black",
                                     linewidth=0.03,
                                     lineend = "butt"),
            )
        )
        png(file=paste('C:/Users/xueyuAB/Desktop/文件/paper1/images/results/biomarker_survival/',cancer,'/',gene, ".png",sep=""), res=300,width=800, height=800,bg = "transparent")
        print(surPlot)
        dev.off() 
    }

}

# figure_single_gene_survival('OV')

figure_latent_heatmap('ESAD')
