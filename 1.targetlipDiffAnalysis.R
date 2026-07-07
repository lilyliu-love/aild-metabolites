#本代码包含定量+过滤+删除内标Cer(d18:1/12:0)+多元统计分析/单维统计分析出具差异表单

library(dplyr)
library(stringr)
library(openxlsx)
library(tidyverse)
library(dplyr)
library(reshape2)
Arg <- commandArgs(T)
path <- as.character(Arg[1])
pairornot <- as.numeric(Arg[2])
status <- as.character(Arg[3])
target_pro <- c("s", "l", "c")
idx <- grep(status, target_pro)
Units <- c("ng/g", "ng/mL", "pg")
UNION = Units[idx]


#样本类型 1:固体solid；2:液体liquid；3:细胞cell；
#target_pro <- c("s", "l", "c")
#type <- grep(project, target_pro)
#Unit <- c("ug/g", "ug/mL", "ng")
#UNION = Unit[type]

setwd(path)
sty <- createStyle(fontName = "Arial", fontSize = 10)
stytitle <- createStyle(fontSize = 10, fontName = "Arial",textDecoration = "bold", halign = "left", fgFill = "#CCCCCC")
sty_color <- createStyle(fgFill="yellow")

stytitle1 <- createStyle(halign="center",valign="center",fontName = "Times New Roman",fontSize = 10, textDecoration = "bold", fgFill = "#CCCCCC")
sty_center1 <-createStyle(halign="center",valign="center",fontName = "Times New Roman", fontSize = 10, border = "TopBottom", borderColour = "black",borderStyle = "medium")
sty_center2 <-createStyle(halign="center",valign="center",fontName = "Times New Roman", fontSize = 10, border = "TopBottom", borderColour = "grey")
sty_center3 <-createStyle(halign="center",valign="center",fontName = "Times New Roman", fontSize = 10, border = "Bottom", borderColour = "black",borderStyle = "medium")
sty_center4 <-createStyle(halign="center",valign="center",fontName = "Times New Roman",fontSize = 10, border = "TopBottom", borderColour = "black",textDecoration = "bold", fgFill = "#CCCCCC",borderStyle = "medium")
all_exper <- read.table("final-out.xls",sep="\t",header=TRUE,quote = "",encoding='UTF-8',check.names=F) 

colnames(all_exper)=gsub("Rej.","polarity",colnames(all_exper))



if(file.exists("newname.xlsx")){
  newname <- read.xlsx(paste0(path,"/newname.xlsx"), rowNames = F, check.names = F)
  newname <- newname[,1:2]
  Name <- melt(newname, id = colnames(newname)[1])
  inputname <- dplyr::select(Name, colnames(Name)[1], value) %>% dplyr::filter(value != "")
  names(inputname) <- c("name", "group")
  inputname <- inputname[order(inputname$group),] #01.07  #保证同一组在表单排序在一起
}else{
  samplename <- colnames(all_exper)[grep("-\\d+$", colnames(all_exper))]
  inputname <- data.frame(name = samplename, group = str_remove(samplename, "-\\d+$"))
  inputname <- inputname[order(inputname$group),] 
}
if ("QC" %in% inputname$group){
  inputname <- rbind(dplyr::filter(inputname, group != "QC"), dplyr::filter(inputname, group == "QC")) 
}

folders <- read.table("groupvs.txt", sep = "\t", fill = TRUE, quote = "", header = FALSE, comment.char = "", check.names = F, stringsAsFactors = FALSE)
foldersall <- gsub("\\|", "_", folders[,1])
write.table(foldersall, file = "folders.txt", quote = F, sep = "\t", row.names = F, col.names = F) 
# 判断各比较组长度，字符数>31输出sheetname.txt, 用于VIP表sheet命名 #01.07
groupvs <- as.vector(folders[, 1])
sheetnames <- c()
sheetnames_txt <- c()
oneway_group <- c()
ttest_group  <- c()
for (i in 1:length(groupvs)) {
  if(grepl("\\|", groupvs[i], perl = T)){
    oneway_group <- c(oneway_group, groupvs[i])
  }else ttest_group <- c(ttest_group, groupvs[i])
  if(all(nchar(groupvs) < 31)){
    sheetnames[i] <- groupvs[i]
  }else{
    sheetnames[i] <- paste(i, str_sub(groupvs[i], end = (30-nchar(i))), sep = ".") #当不可写入sheet名时，sheetname[i]会带序号前缀
    sheetnames_txt[i] <- paste0(i, ".", groupvs[i])
    #print(paste0("[", groupvs[i], "]", " is too long to write in sheetname!"))
    write.table(sheetnames_txt, "Sheetnames.txt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE) # 给客户查看
    write.table(sheetnames, "tempname.txt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)  # 用于下游写xlsx工作簿中sheetname
  }
}

if(length(ttest_group) != 0) write.table(ttest_group, "folders1.txt", sep = "\t", quote = F, col.names = F, row.names = F)

if(length(oneway_group) != 0){
  write.table(gsub("\\|", "_", oneway_group, perl = T), "folders2.txt", sep = "\t", quote = F, col.names = F, row.names = F)
}

file_VIP<-paste(getwd(),"/VIP-SIMCA.xlsx",sep = "")
if(!file.exists("twoway.txt")&!file.exists("significanceA.txt")&!file.exists("significanceB.txt")){
  if (!file.exists(file_VIP)){
    Statpath <- paste0(path,"/统计分析")
    file2 <- paste(Statpath,"/VIP.xlsx",sep ="")
  }else{
    print("VIP-SIMCA.xlsx already exists")
    file2 <- "VIP-SIMCA.xlsx" 
  }
}

#temp <- all_exper[,-c(grep('IS',colnames(all_exper))[1]:ncol(all_exper))]
temp <- all_exper
#pure_exper <- temp[,grep("-\\d+$",toupper(colnames(temp)))]
#QC <- temp[,c(grep("QC-\\d+$",colnames(temp),fixed=FALSE))]	
pure_exper <- temp[,as.vector(inputname[which(inputname$group != "QC"),1])]
QC <- temp[,as.vector(inputname[which(inputname$group == "QC"),1])]

# QC[QC =="N/A"] <- NA
# QC[QC =="NA"] <- NA
# 计算RSD(RSD=STDEV/Average,stdev=sd,average=mean)
RSD <- function(x){
  m <- mean(x,na.rm=TRUE)
  s <- sd(x,na.rm=TRUE)                    
  RSD <- s/m
  return(RSD=RSD)
}
QC_RSD <- as.matrix(apply(QC,1,RSD))
temp$QCRSD <- QC_RSD
# QC_curve <- select(temp,polarity,QCRSD) %>% arrange(QCRSD)
# arrange(QC_curve,QCRSD)

QC_curve <- dplyr::select(temp,polarity,QCRSD)
QC_curve$QCRSD <- QC_curve$QCRSD*100
pdata<-table(QC_curve$QCRSD)
pframe<-data.frame(as.numeric(names(pdata)),as.numeric(pdata))
colnames(pframe)<-c("QCRSD","num")
rankData<-pframe[order(pframe[,1],decreasing=F),]
mulData<-data.frame()
sum=0
for(i in 1:nrow(rankData)){
  sum=sum+rankData[i,2]
  row=cbind(rankData$QCRSD[i],sum)
  mulData=rbind(mulData,row)
}
colnames(mulData)<-c("QCRSD","num")
curve_Data <-data.frame(RSD =mulData$QCRSD,PER =mulData$num/sum*100)
pc <- ggplot(curve_Data, aes(x = RSD, y = PER))+
  #geom_point() +
  geom_line(colour = "orange",size = 0.5) +
  scale_x_continuous(breaks = c(0,30,50,100,200))+
  theme_bw() + 
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line=element_line(size=0.4,colour="black"),
        axis.text.x=element_text(colour="black",size = 10), 
        axis.text.y=element_text(colour="black",size = 10), 
        axis.title.x=element_text(size = 12), 
        axis.title.y=element_text(size = 12)
        #plot.title = element_text(size = 11, hjust=0.5),
  )+
  geom_vline(xintercept = 30,lty =2, colour = "grey")+
  ylab("% of peaks") +
  xlab("RSD (%)") 
ggsave(file ="QCRSD_curve.png",pc,width = 10, height = 10, units = "cm",dpi=300)  #输出

tmprawdata <- temp
tmprawdata[tmprawdata == 0] <- NA

if(file.exists("twoway.txt")|file.exists("significanceA.txt")|file.exists("significanceB.txt")){
  tmprawdata <- subset(tmprawdata,tmprawdata$QCRSD<= 0.3) 
  tmprawdata$QCRSD <- as.numeric(tmprawdata$QCRSD)
  tmprawdata <- tmprawdata[,-1]
  #"polarity","hits.forward.x","hits.reverse.x","Class","FattyAcid","FA1","FA2","FA3","FA4","IonFormula","mzmed","rtmed","QCRSD",14:ncol(tmprawdata)
  tmprawdata <- tmprawdata[,c(12,1:8,10,9,11,13:ncol(tmprawdata))]
  colnames(tmprawdata)[1:3] <- c("name","LipidIon","LipidGroup")
  colnames(tmprawdata)[11:12] <-c("CalMz","RT-(min)") 
  
  class_numb <- length(unique(tmprawdata$Class))
  species_numb <- nrow(tmprawdata)
  table_class <- data.frame("Lipid class" = class_numb,"Lipid species" = species_numb, check.names = F)
  
  wb6 <- createWorkbook()
  addWorksheet(wb6, sheet = "class-species" ,gridLines = TRUE, zoom = 120)
  addStyle(wb6, sheet="class-species", sty_center1,cols =1:ncol(table_class),rows =(1:nrow(table_class)+1),gridExpand = TRUE)
  addStyle(wb6,sheet = "class-species",stytitle1,cols =1:ncol(table_class),rows =1,gridExpand = TRUE)
  writeData(wb6,sheet = "class-species",table_class ,colNames=TRUE,rowNames=FALSE)
  setColWidths(wb6, sheet = "class-species", cols= 1:ncol(table_class), widths = c(11,12))
  saveWorkbook(wb6,"Pic_class_species.xlsx",overwrite =TRUE)
  
  wb5 <- createWorkbook()
  for (s in (1:length(sheetnames))){
    print(sheetnames[s])
    tmprawdata_A <- tmprawdata
    tab <- str_replace_all(sheetnames[s], "\\s", "-")
    if (str_detect(tab, "_vs_")){
      sep <- "_vs_"
      tab <-  unlist(strsplit(as.character(tab), "_vs_", perl = TRUE))
      Foldchange <- c()
      for(mm in 1:nrow(tmprawdata_A)){
        #A1 <- as.numeric(tmprawdata_A[mm,grep(paste("^",tab[1],"-\\d+$",sep=""),names(tmprawdata_A),fixed=F)])
        #A2 <- as.numeric(tmprawdata_A[mm,grep(paste("^",tab[2],"-\\d+$",sep=""),names(tmprawdata_A),fixed=F)])
        A1 <- as.numeric(tmprawdata_A[mm,as.vector(inputname[which(inputname$group == tab[1]),1])])
        A2 <- as.numeric(tmprawdata_A[mm,as.vector(inputname[which(inputname$group == tab[2]),1])])
        if(any(!is.na(A1)) && any(!is.na(A2))){
          Foldchange[mm] <- c(mean(A1, na.rm = TRUE)/mean(A2, na.rm = TRUE))
        }else{  #通道值小于该组一半的不计算p-value与Fold change
          Foldchange[mm] <- NA
        }
      }
      tmprawdata_A$Foldchange <- Foldchange 
      tmprawdata_A <- tmprawdata_A[,c(1:13,ncol(tmprawdata_A),14:(ncol(tmprawdata_A)-1))]
      colnames(tmprawdata_A)[14] <- "Fold Change"
    }else{
      tmprawdata_A <- tmprawdata
      sep <- "\\|"
      tab <- unlist(strsplit(as.character(tab), "\\|", perl = TRUE)) 
    }
    
    addWorksheet(wb5, sheet = sheetnames[s],gridLines = TRUE)
    addStyle(wb5,sheet = sheetnames[s],sty,cols = 1:(ncol(tmprawdata_A)+1),rows = 1:(nrow(tmprawdata_A)+1),gridExpand = TRUE)
    addStyle(wb5,sheet = sheetnames[s],stytitle,cols =1:ncol(tmprawdata_A),rows =1,gridExpand = TRUE)
    writeData(wb5,sheet = sheetnames[s],tmprawdata_A,colNames=TRUE,rowNames=FALSE)
    setColWidths(wb5, sheet = sheetnames[s], cols =1:ncol(tmprawdata_A), widths = 10)
  }
  saveWorkbook(wb5,"附件1.xlsx",overwrite =TRUE)
  
}else {
  #循环计算p,FC(从VIP.xlsx中读取两两比较)
  wb2 <- createWorkbook()
  if (length(ttest_group) != 0){
    
    for (y in (1:length(ttest_group))){
      print(ttest_group[y])
      tab <- str_replace_all(ttest_group[y],"\\s","")
      tab <- str_split_fixed(tab,"_vs_",2) 
      pvalue <- c()
      Foldchange<- c()
      MEAN <- c()
      for (m in 1:nrow(tmprawdata)){
          # A1 <- as.numeric(tmprawdata[m,grep(paste("^",tab[1],"-\\d+$",sep=""),names(tmprawdata),fixed=F)])
          # A2 <- as.numeric(tmprawdata[m,grep(paste("^",tab[2],"-\\d+$",sep=""),names(tmprawdata),fixed=F)])
          A1 <- as.numeric(tmprawdata[m,as.vector(inputname[which(inputname$group == tab[1]),1])])
          A2 <- as.numeric(tmprawdata[m,as.vector(inputname[which(inputname$group == tab[2]),1])])
          A1 = round(A1, 6)
          A2 = round(A2, 6)
          if ((sum(is.na(A1)) < length(A1)/2)&(sum(is.na(A2)) < length(A2)/2)){ 
              Foldchange<- c(Foldchange,mean(A1,na.rm=TRUE)/mean(A2,na.rm=TRUE))
              if (!sd(A1,na.rm = T)==0&!sd(A2,na.rm = T)==0) { 
              if (pairornot == 0){  #t.test
                      #  pvalue<-c(pvalue,t.test(A1,A2,alternative ="two.sided",var.equal = TRUE)$p.value)
                      pvalue <- c(pvalue, wilcox.test(A1, A2)$p.value)  #------- 2025.6.13 zhangyu pvalue计算修改为wilcox
              }else{#paired t.test
                        pvalue <-c(pvalue, t.test(A1, A2, alternative = "two.sided", var.equal = TRUE,paired = TRUE)$p.value)
              }}else{pvalue=c(pvalue,1)}
          }else{
            pvalue<-c(pvalue,NA)
            Foldchange<- c(Foldchange,NA)
          }
      }   
      sigsig<- data.frame(Foldchange,pvalue)  
      #colnames(sigsig) <- c("Fold change","p-value","MEAN")
      
      data1 <- openxlsx::read.xlsx(xlsxFile = file2, sheet = y, colNames = T,check.names = F)
      VIPdata <- data.frame(polarity = data1[,1],VIP = data1[,4])
      tmprawdata2 <- cbind(sigsig,tmprawdata)
      
      rawdata <- merge(VIPdata,tmprawdata2,x.by="polarity",y.by="polarity")
      rawdata$DIFF <- ifelse(rawdata$`pvalue`<0.05 & (rawdata$Foldchange > 1.5 | rawdata$Foldchange < 1/1.5), "Y", NA)
      len <- nrow(rawdata[rawdata$`pvalue`<0.05 & (rawdata$Foldchange > 1.5 | rawdata$Foldchange < 1/1.5),])
      
      rawdata2 <- rawdata %>% group_by(polarity) %>% arrange(desc(DIFF))

      rawdata2 <- rawdata2[,c(1,ncol(rawdata2),6:19,21,20,22,(ncol(rawdata2)-1),3:4,2,23:(ncol(rawdata2)-2))]

      # 数据排列 准备输出。
      colnames(rawdata2)[1:4] <- c("name","DIFF","LipidIon","LipidGroup")
      colnames(rawdata2)[7] <- c("Class")
      colnames(rawdata2)[18:19] <-c("CalMz","RT-(min)") 
      colnames(rawdata2)[21:22] <- c("Fold Change","P-value")
      rawdata2$QCRSD <- as.numeric(rawdata2$QCRSD)
      rawdata2$unit = rep(UNION,nrow(rawdata2))  # 添加

      addWorksheet(wb2, sheet = ttest_group[y],gridLines = TRUE)
      addStyle(wb2,sheet = ttest_group[y],sty,cols = 1:(ncol(rawdata2)+1),rows = 1:(nrow(rawdata2)+1),gridExpand = TRUE)
      addStyle(wb2,sheet = ttest_group[y],stytitle,cols =1:ncol(rawdata2),rows =1,gridExpand = TRUE)
      if (len > 0){
        addStyle(wb2, sheet=ttest_group[y], sty_color, cols = 3:8,rows = 2:(len+1), gridExpand = TRUE,stack=TRUE)
      }
      
      writeData(wb2,sheet = ttest_group[y],rawdata2,colNames=TRUE,rowNames=FALSE)
      setColWidths(wb2, sheet = ttest_group[y], cols =1:ncol(rawdata2), widths = 10)
      
    }
  } 
  #file_oneway<-paste(getwd(),"/oneway.txt",sep = "")
  #oneway***********************************************************************************************
  #判断是否有两组至少两个通道有值
  pan <- function(x){
    num<-length(x)-sum(is.na(x))
    if(num>=2){
      return(na.omit(x))
    }else{
      return(na.omit(x))
    }
  }
  
  if (length(oneway_group) != 0){
    #oneway<-read.table("oneway.txt",header = FALSE,sep = '\t',fill = TRUE,quote = "",check.names = TRUE)  
    oneway <- oneway_group
    for (u in 1:length(oneway)){
      print(oneway[u])
      pvalue<-c()
      tab <- unlist(strsplit(as.character(oneway[u]),"\\|",perl = TRUE)) #tab************************************************
      #正态检验，统计true(p.value>0.05满足正态分布的个数)
      P.test<-c()
      n <- 0
      for(w in 1:nrow(pure_exper)) {
        testexpr <- pure_exper[w,]
        #testexpr <- as.numeric(testexpr)
        testexpr <- as.matrix(testexpr)
        sol <- shapiro.test(as.numeric(testexpr))
        P.test[w] <-sol$p.value
        if(P.test[w]>0.05){
          n <- n+1
        }else{
          n<- n
        }
      }
      outsummary <- summary(P.test>0.05)
      outsummary <- t(data.frame(a=names(outsummary),b=as.vector(outsummary)))
      write.table(outsummary,file = "summary.txt",quote = FALSE,sep = "\t",col.names = F,row.names = F)
      #**********************************************
      tmpexpr <- pure_exper
      tmpexpr[ tmpexpr == 0] <- NA
      
      #log_tmprawdata <- log2(tmpexpr) #待补充
      log_tmprawdata <- tmpexpr
      for (k in 1:nrow(tmprawdata)) {
        num<-0
        x<-c()
        A<-c()
        
        for (j in 1:length(tab)) {
          sample <- tab[j]
          #sample_data <- log_tmprawdata[,grep(paste("^",sample,"-\\d+$",sep=""),colnames(log_tmprawdata),perl=TRUE)][k,]
          sample_data <- log_tmprawdata[,as.vector(inputname[which(inputname$group ==sample),1])][k,]
          gp<-pan(as.numeric(sample_data))
          if(length(gp)>=2){
            num=num+1
            x<-c(x,gp)
            A<-c(A,rep(j,length(gp)))
          }
        }
        
        if(num>=2){
          A<-factor(A)
          lamp<-data.frame(x,A)
          lamp.acv<-aov(x~A,data = lamp)
          a<-summary(lamp.acv)
          pvalue[k]<-a[[1]]$`Pr(>F)`[1]
        }else{
          pvalue[k]<-1
        }
        
      }
      tmprawdata2 <- cbind(tmprawdata,pvalue)
      
      #所有信息rawdata
      rawdata <- tmprawdata2
      #rawdata <- subset(rawdata,rawdata$QCRSD<= 0.3)  #12.30正式上线QCRSD 30%（线下项目从12.04开始30%，之前为40%）
      rawdata$DIFF <- ifelse(rawdata$`pvalue`<0.05, "Y", NA)
      len <- nrow(rawdata[rawdata$`pvalue`<0.05,])
      rawdata2 <- rawdata %>% group_by(polarity) %>% arrange(desc(DIFF))
     
      rawdata2 <- rawdata2[,c(1,ncol(rawdata2),3:16,18,17,19,(ncol(rawdata2)-2),(ncol(rawdata2)-1),20:(ncol(rawdata2)-3))]
    
      colnames(rawdata2)[1:4] <- c("name","DIFF","LipidIon","LipidGroup")
      colnames(rawdata2)[7] <- c("Class")
      colnames(rawdata2)[18:19] <-c("CalMz","RT-(min)") 
      colnames(rawdata2)[21] <- "P-value"
      rawdata2$QCRSD <- as.numeric(rawdata2$QCRSD)
      rawdata2$unit = rep(UNION,nrow(rawdata2))  # 添加
      
      addWorksheet(wb2, sheet = oneway[u],gridLines = TRUE)
      addStyle(wb2,sheet = oneway[u],sty,cols = 1:(ncol(rawdata2)+1),rows = 1:(nrow(rawdata2)+1),gridExpand = TRUE)
      addStyle(wb2,sheet = oneway[u],stytitle,cols =1:ncol(rawdata2),rows =1,gridExpand = TRUE)
      if (len > 0){
        addStyle(wb2, sheet=oneway[u], sty_color, cols = 3:8,rows = 2:(len+1), gridExpand = TRUE,stack=TRUE)
      }
      
      writeData(wb2,sheet = oneway[u],rawdata2,colNames=TRUE,rowNames=FALSE)
      setColWidths(wb2, sheet = oneway[u], cols =1:ncol(rawdata2), widths = 10)
      
    } 
  }  
}

saveWorkbook(wb2,"附件1.Lipidomics表.xlsx",overwrite =TRUE)

file_Lipid <- "附件1.Lipidomics表.xlsx"
print(file_Lipid)
pic_table <- read.xlsx(xlsxFile = file_Lipid, sheet = 1, colNames = T,check.names = F)

class_numb <- length(unique(pic_table$Class))
species_numb <- nrow(pic_table)
table_class <- data.frame("Lipid class" = class_numb,"Lipid species" = species_numb, check.names = F)

wb3 <- createWorkbook()
addWorksheet(wb3, sheet = "class-species" ,gridLines = TRUE, zoom = 120)
addStyle(wb3, sheet="class-species", sty_center1,cols =1:ncol(table_class),rows =(1:nrow(table_class)+1),gridExpand = TRUE)
addStyle(wb3,sheet = "class-species",stytitle1,cols =1:ncol(table_class),rows =1,gridExpand = TRUE)
writeData(wb3,sheet = "class-species",table_class ,colNames=TRUE,rowNames=FALSE)
setColWidths(wb3, sheet = "class-species", cols= 1:ncol(table_class), widths = c(11,12))
saveWorkbook(wb3,"Pic_class_species.xlsx",overwrite =TRUE)

if ( "VIP"%in% colnames(pic_table)){
  # table_diff <- dplyr::filter(pic_table,VIP>1&`P-value` <0.05)
  table_diff <- dplyr::filter(pic_table, `P-value` < 0.05 & (`Fold.Change` > 1.5 | `Fold.Change` < 1/1.5))
  
  table_diff <- table_diff[order(table_diff$VIP, decreasing = T),]
  if (nrow(table_diff)>= 20) {
    len <- 20
  }else{
    len <-nrow(table_diff)
  }
  table_diff <- dplyr::select(table_diff,c("LipidIon",Class,"IonFormula","CalMz","RT-(min)","VIP","Fold.Change","P-value"))[1:len,]
  colnames(table_diff)[7] <- "Fold Change"
  wb4 <- createWorkbook()
  addWorksheet(wb4, sheet = "diff species" ,gridLines = TRUE, zoom = 120)
  addStyle(wb4, sheet="diff species", sty_center2,cols =1:ncol(table_diff),rows =1:nrow(table_diff),gridExpand = TRUE)
  addStyle(wb4, sheet="diff species", sty_center3,cols =1:ncol(table_diff),rows =nrow(table_diff)+1,gridExpand = TRUE)
  addStyle(wb4,sheet = "diff species",sty_center4,cols =1:ncol(table_diff),rows =1,gridExpand = TRUE)
  writeData(wb4,sheet = "diff species",table_diff ,colNames=TRUE,rowNames=FALSE)
  setColWidths(wb4, sheet = "diff species", cols= 1:ncol(table_diff), widths = c(22,6,20,11,11,11,11,11))
  saveWorkbook(wb4,"Pic_diff.xlsx",overwrite =TRUE)
}else{
  table_diff <- dplyr::filter(pic_table,`P-value` <0.05)
  table_diff <- table_diff[order(table_diff$`P-value`, decreasing =F),]
  if (nrow(table_diff)>= 20) {
    len <- 20
  }else{
    len <-nrow(table_diff)
  }
  table_diff <- dplyr::select(table_diff,c("LipidIon","Class","IonFormula","CalMz","RT-(min)","P-value"))[1:len,]
  wb4 <- createWorkbook()
  addWorksheet(wb4, sheet = "diff species" ,gridLines = TRUE, zoom = 120)
  addStyle(wb4, sheet="diff species", sty_center2,cols =1:ncol(table_diff),rows =1:nrow(table_diff),gridExpand = TRUE)
  addStyle(wb4, sheet="diff species", sty_center3,cols =1:ncol(table_diff),rows =nrow(table_diff)+1,gridExpand = TRUE)
  addStyle(wb4,sheet = "diff species",sty_center4,cols =1:ncol(table_diff),rows =1,gridExpand = TRUE)
  writeData(wb4,sheet = "diff species",table_diff ,colNames=TRUE,rowNames=FALSE)
  setColWidths(wb4, sheet = "diff species", cols= 1:ncol(table_diff), widths = c(22,6,20,11,11,11))
  saveWorkbook(wb4,"Pic_diff.xlsx",overwrite =TRUE)
}

