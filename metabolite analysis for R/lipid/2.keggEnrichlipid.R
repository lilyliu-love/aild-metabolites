library(dplyr)
library(stringr)
library(openxlsx)
library(tidyverse)
library(dplyr)
library(reshape2)
suppressMessages(library(openxlsx))
suppressMessages(library(dplyr))
library(ggplot2)
library(httr)
library(readr)
suppressMessages(library(rvest))
library(downloader)
library(magrittr)
library(RJSONIO)
suppressMessages(library(RCurl))
library(stringr)
suppressMessages(library(tidyr))
library(pheatmap)
suppressMessages(library(reshape2))
suppressMessages(library(ComplexHeatmap))
colpalette <- c("#1d953f", "#102b6a", "#c77eb5", "#ffce7b", "#2585a6", "purple", "#e0861a", "#d71345", "#6b473c", "#78a355", "#fdb933", "#5e7c85", "#411445", "#c37e00", "#bed742", "#009ad6", "#9d9087", "#aa2116", "#225a1f", "#f3715c", "#7bbfea", "#dbce8f", "#f8aba6", "#778899", "#2F4F4F", "#2E8B57", "#8B6969", "#EE2C2C", "#6495ED", "#87CEEB", "#0000CD", "#63B8FF", "#006400", "#00FF7F", "#9400D3", "#FFAEB9", "#8B5A2B", "#FFFF00", "#BDB76B", "#FF1493")


##提取KEGG所需要得文件diff.txt info-H2400.txt META_up/down.txt
data.Extract <- function(dat, Group,inputname){
  colnames(dat)[3] = "Name"
  tab <- unlist(strsplit(as.character(Group), "_vs_|\\|", perl = TRUE))
  folder <- gsub('\\|', '_', Group)
  
  csample<-inputname[inputname$group %in% tab,]$name
  if(length(tab) == 2){
    #------- 2025.6.13 zhangyu 修改差异筛选条件 start -------
    # data_sig <- dat[!is.na(dat$Name)  & dat$`P-value`< 0.05 & dat$VIP>1,]
    data_sig <- dat[!is.na(dat$Name)  & dat$`P-value`< 0.05 & (dat$`Fold.Change` > 1.5 | dat$`Fold.Change` < 1/1.5),]
    #------- 2025.6.13 zhangyu 修改差异筛选条件 end -------
  }else{
    data_sig <- dat[!is.na(dat$Name) &  dat$`P-value` < 0.05,]
  }
  data_sig<-data_sig[!is.na(data_sig$Name),]
  data_clu = data_sig[!duplicated(data_sig$Name),c("Name",csample)]
  
  # KEGG数据提取
  print("KEGG数据提取！！！！！！！！beging") 
  outpath_keg <- paste0(folder, "/KEGG")
  if(!dir.exists(outpath_keg)) dir.create(outpath_keg,recursive = T)
  
  if(length(tab) == 2){
    data_kegg <- select(data_sig,Name,KEGG,FC = 'Fold.Change', pvalue="P-value")
    data_kegg$MEAN<-apply(data_clu[,2:ncol(data_clu)],1,mean,na.rm=T)
    data_kegg<-data_kegg[,c("Name","MEAN","KEGG","FC","pvalue")]
    META_UP <- data_kegg[data_kegg$FC > 1,][,c("Name","KEGG")]
    META_DOWN <- data_kegg[data_kegg$FC < 1,][,c("Name","KEGG")]
    write.table(META_UP, paste(outpath_keg, "META_up.txt", sep = "/"), col.names = F, row.names = F, sep = "\t", quote = F)
    write.table(META_DOWN, paste(outpath_keg, "META_down.txt", sep = "/"), col.names = F, row.names = F, sep = "\t", quote = F)
  }else{
    data_kegg <- select(data_sig, Name, KEGG,'p-value' = 'P-value')
    data_kegg$MEAN<-apply(data_clu[,2:ncol(data_clu)],1,mean,na.rm=T)
    data_kegg<-data_kegg[,c("Name","MEAN","KEGG","p-value")]
  } 
  
  data_kegg<-data_kegg[!is.na(data_kegg$Name),]
  META_diff <- data_kegg[,c("Name","KEGG")]
  write.table(META_diff, paste(outpath_keg, "diff.txt", sep = "/"), col.names = F, row.names = F, sep = "\t", quote = F)
  write.table(data_kegg, file = paste(outpath_keg, paste0("info-", "H2400", ".txt"), sep = "/"), row.names = F, quote = F, sep = "\t")
  
  if(file.exists(paste0(outpath_keg, "/info-H2400.txt"))){
    
    input2 <- read.table(paste0(outpath_keg, "/info-H2400.txt"), sep = "\t", header = TRUE, encoding = 'UTF-8', check.names = F, stringsAsFactors = FALSE, quote = "")
    data <- input2
    data1 <- data %>% group_by(Name) %>% arrange(desc(MEAN))
    data1 <- data %>% group_by(Name)
    data_temp <- data1
    data_temp$Name <- tolower(data_temp$Name)
    index <- duplicated(data_temp$Name)
    data1$Name[index] <- NA
    data1 <- data1[order(data1$Name), -2]
    data_out <- data1[!is.na(data1$Name),]
    write.table(data_out, paste(outpath_keg, "info-diff.txt", sep = "/"), col.names = TRUE, row.names = FALSE, quote = FALSE, sep = "\t")
  }
}


#q2map()函数
#输出query2map的函数 参数为query.cpd和kegg_XX.compound
q2map <- function(aqko =query2cpd,pathway =pathway,background){
  rname <- c("Metabolite","cpdID","cpdName","Map_ID","Map_Name","URL","Pathway_Hierarchy1","Pathway_Hierarchy2")
  result <- as.data.frame(matrix(numeric(0),ncol=8))
  for(i in 1:nrow(aqko)){
    aline <- as.data.frame(matrix(numeric(0),ncol=8))
    names(aline) <- rname
    res_aline <- aline
    if(is.na(aqko[i,2])|aqko[i,2] == ""){
      res_aline <- data.frame(aqko[i,1],aqko[i,2],"","","","","","",stringsAsFactors=FALSE,check.names=F)
      names(res_aline) <- rname
      aline <- rbind(aline,res_aline)
    }else{
      if (aqko[i,2] %in% background){
        for (m in (1:length(pathway_list))){
          if (aqko[i,2] %in% pathway_list[[m]]$V3){
            mapurl <- paste0("https://www.kegg.jp/kegg-bin/show_pathway?",pathway[m,1],"+",aqko[i,2])
            res_aline <- data.frame(aqko[i,1],aqko[i,2],aqko[i,3],pathway[m,1],pathway[m,4],mapurl,pathway[m,6],pathway[m,5],stringsAsFactors=FALSE,check.names=F)
            names(res_aline) <- rname
            aline <- rbind(aline,res_aline) 
          }
        } 
      }else{
        res_aline <- data.frame(aqko[i,1],aqko[i,2],aqko[i,3],"","","","","",stringsAsFactors=FALSE,check.names=F)
        names(res_aline) <- rname
        aline <- rbind(aline,res_aline) 
      }
    }
    result <- rbind(result,aline)
  }
  return(result) 
}

#map2q()函数
#输出map2query2的函数 参数为query2map
map2q <- function(x = query2map){
  map_ids <- unique(x[which(x[,4]!=""),4])
  rname <- c("Pathway_Hierarchy1","Pathway_Hierarchy2","Map_ID","Map_Name","cpdName","cpd","Num_cpd","URL")
  result <- as.data.frame(matrix(numeric(0),ncol=8))
  colnames(result) <- rname
  for (i in 1:length(unique(map_ids))) {
    ap <- filter(x, x[,4]==as.character(unique(map_ids)[i]))
    mapid <- as.character(unique(map_ids)[i])
    myurl <- paste0("https://www.kegg.jp/kegg-bin/show_pathway?",mapid,"+",str_c(ap[,2],collapse = "+"))
    m2aq <- data.frame(ap[1,c(7,8)],mapid,ap[1,5],str_c(ap[,1],collapse = "|"),str_c(ap[,2],collapse = "+"),length(ap[,2]),myurl,stringsAsFactors=FALSE,check.names=F)
    result <- rbind(result,m2aq)
  }
  result <- as.data.frame(result,stringsAsFactors = FALSE)
  names(result) <- rname
  result$Num_cpd <- as.numeric(result$Num_cpd)
  result <- arrange(result,desc(Num_cpd))
  return(result)
}

####富集分析
Enrichment <- function(x= map2query,y = myback,difnum = TestAll,allnum = RefAll){
  myquery <- data.frame(x[,c(1:4,6:7)],rep(difnum,nrow(x)),stringsAsFactors = FALSE,check.names = FALSE)
  myquery$Num_cpd <- as.numeric(lapply(myquery$cpd,function(x){length(unique(unlist(str_split(x, pattern = "\\+"))))}))
  myquery <- myquery[,-5]
  myall <- data.frame(y,rep(allnum,nrow(y)),stringsAsFactors = FALSE,check.names = FALSE)
  myresult <- left_join(myquery,myall,by = c("Map_ID" = "V1"))
  myresult <- within(myresult,{
    testper <- round(myresult[,5]/myresult[,6]*100,6)
  })
  myresult <- within(myresult,{
    refper <- round(myresult[,7]/myresult[,8]*100,6)
  })
  myresult <- within(myresult,{
    ou <- apply(myresult,1,function(x){
      if(as.numeric(x[9]) >= as.numeric(x[10])){
        return("over")
      }else{
        return("under")}
    })
  })
  names(myresult) <- c("Pathway_Hierarchy1","Pathway_Hierarchy2","Map_ID","Map_Name","Test","TestAll","Ref","RefAll","Test_per","Ref_per","Over_Under")
  myresult$p.value <- phyper(myresult$Test-1,myresult$Ref, myresult$RefAll-myresult$Ref,myresult$TestAll,lower.tail = FALSE)
  myresult$FDR <-p.adjust(myresult$p.value,method="fdr",n=length(myresult$p.value))
  myresult$richFactor <- myresult$Test/myresult$Ref
  myresult <- arrange(myresult, p.value)
  myresult <- arrange(myresult, Over_Under)
  return(myresult)
}


###绘制KEGG气泡图
kegg_bubble <- function(data = em,pa = KeggPath){
  Enrich <- subset(data, data$p.value<0.05 & data$Over_Under == "over",select = names(data))#代谢绘制p<0.05的top20
  Enrich <- Enrich[order(Enrich$p.value),]
  Enrich <- head(Enrich,20)
  Enrich <- Enrich[order(Enrich$richFactor),]
  Enrich$richFactor <- round(Enrich$richFactor, digits = 2)
  Enrich$Map_Name <- factor(Enrich$Map_Name,levels=(unique(Enrich$Map_Name)))
  Enrich$type <- rep("KEGG Enrichment",nrow(Enrich))
  maxrol <- as.numeric(nrow(Enrich))
  
  xname <- "Enriched KEGG Pathways"
  txsize <- 14
  tysize <- 13
  pm <- c(4,2,1.5,1)
  lt <- 11
  vj <-0.4
  gtsize <- 3
  lkh <- 0.55
  txm1 <- 20
  ltsize <- 11
  if(maxrol >= 20){
    mywidth <- 0.5
    xname <- "Enriched KEGG Pathways (Top 20)"
    xsize <- 11.5
    ysize <- 9.5
    lkh <- 0.6
    pngh <- 16
    pdfh <- 8
  }else if(maxrol < 20 && maxrol >=15){
    mywidth <- 0.0197*maxrol+0.166
    xsize <- -0.25*maxrol+16
    ysize <- -0.25*maxrol+14
    txm1 <- 30
    pngh <- 14
    pdfh <- 7
  }else if(maxrol < 15 && maxrol >=10){
    mywidth <- 0.0197*maxrol+0.35
    xsize <- -0.25*maxrol+14.5
    ysize <- -0.25*maxrol+12.5
    vj < 0.45
    pngh <- 12
    pdfh <- 6
  }else if(maxrol < 10 && maxrol >=5){
    mywidth <- 0.0197*maxrol+0.286
    xsize <- -0.25*maxrol+14
    ysize <- -0.25*maxrol+12
    txm1 <- 25
    txsize <- 13
    tysize <- 12
    pm <- c(3.5,2,2,1)
    lkh <- 0.45
    vj <- 0.45
    gtsize <- 3.5
    pngh <- 11
    pdfh <- 5.5
  }else if(maxrol < 5 && maxrol >0){
    mywidth <- 0.0197*maxrol+0.35
    xsize <- -0.25*maxrol+13.5
    ysize <- -0.25*maxrol+11.5
    txsize <- 10.5
    tysize <- 8.5
    pm <- c(3,2,2,1)
    lkh <- 0.35
    ltsize <- 10
    gtsize <- 4
    pngh <- 8
    pdfh <- 4
  }
  if(maxrol > 0){
    p <- ggplot(data = Enrich,aes(x = richFactor,y = Map_Name)) +
      geom_point(aes(size=Test,color=-1*log10(p.value))) +
      scale_colour_gradient(low="green",high="red") + 
      theme_bw() +
      theme(
        axis.ticks.length = unit(-0.15, 'cm'),
        axis.text.x = element_text(colour = "black",vjust = 0.5,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
        axis.text.y = element_text(colour = "black",margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm'))
      )+
      labs(color=expression(-log[10](p.value)),size="Metabolite number",x="Rich factor",y=xname)+
      guides(color = guide_colorbar(order = 1),size= guide_legend(order = 2))
    ggsave(filename = paste0(pa,"EnrichmentBubble.png"), p, device = "png", width = 20, height = 20, units = "cm", dpi = 300)
    ggsave(filename = paste0(pa,"EnrichmentBubble.pdf"), p, device = "pdf", width = 20/2.54, height = 20/2.54)
    
    
    p1 <- ggplot(Enrich, aes(x= Map_Name, y = Test, fill = p.value)) +
      geom_bar(stat = "identity", width = mywidth) + 
      scale_fill_gradient(limits = c(0,0.05),low = "red", high = "gold2") + 
      labs(fill = "p.value", y = "Compound Number", x = xname) +
      scale_y_continuous(breaks=seq(1,ceiling(max(Enrich$Test)*1.1),by = 1),limits= c(0,ceiling(max(Enrich$Test)*1.1)))+
      coord_flip() +
      theme_bw() + 
      theme(
        panel.border = element_rect(size = 0.8),
        panel.grid.minor = element_blank(),
        axis.ticks.length = unit(-0.15, 'cm'),
        axis.text.x = element_text(colour = "black",size = ysize,vjust = 0.5,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
        axis.text.y = element_text(colour = "black",size = ysize,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
        axis.title.x = element_text(margin = margin(txm1,0,0,0),size = txsize),
        axis.title.y = element_text(margin = margin(0,30,0,20),size = tysize),
        plot.margin = unit(pm,"lines"),
        legend.margin = margin(c(0,0,0,15)),
        legend.key.height = unit(lkh,"cm"),
        legend.title = element_text(size = ltsize)) +
      guides(fill = guide_legend(label.position = "left", label.hjust = 1)) +
      geom_text(aes(label = Enrich$richFactor, vjust = vj, hjust = -0.3),size=gtsize)
    ggsave(filename = paste0(pa,"EnrichmentBar.png"), p1, device = "png", width = 20, height = 20, units = "cm", dpi = 300)
    ggsave(filename = paste0(pa,"EnrichmentBar.pdf"), p1, device = "pdf", width = 20/2.54, height = 20/2.54)
    
  }else{
    word <- "该比较组没有富集结果"
    write.table(word, file = paste0(pa,"/备注.txt"), row.names = F, col.names = F, quote = F)
  }
}

#画kegg TopMapStat
tms <- function(TopMapStat,pa){
  TopMapStat$Num_comp <- str_count(TopMapStat$cpd,"C")
  TopMapStat <- arrange(TopMapStat,desc(Num_comp))
  TopMapStat <- head(TopMapStat, 20)
  TopMapStat <- data.frame(TopMapStat, type = rep("KEGG TopMapStat",nrow(TopMapStat)),stringsAsFactors = F)
  TopMapStat$Map_Name <- factor(TopMapStat$Map_Name, levels = TopMapStat$Map_Name[nrow(TopMapStat):1])
  
  maxrol<- nrow(TopMapStat)
  p <- ggplot(TopMapStat, aes(x = Map_Name, y = Num_comp)) +
    geom_bar(stat = "identity", fill = "#473C8B", width = 0.7) +
    labs(x = "KEGG Pathway Name(Top 20)", y = "Compound Number") +
    theme_bw() +
    coord_flip() +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_blank(),
      axis.line.x = element_line(size = 0.5),
      axis.line.y = element_line(size = 0.5),
      axis.text.x = element_text(size = 10, vjust = 0,hjust = 1),
      axis.title.x=element_text(size = 13,vjust = -1), 
      strip.text = element_text(size = 12),
      axis.title.y=element_text(size = 13),
      plot.margin = unit(c(1.5,2,1.5,2),"lines"),
    ) +
    scale_y_continuous(expand = c(0, 0),limits = c(0,max(TopMapStat$Num_comp)*1.1)) +
    geom_text(aes(label = Num_comp, vjust = 0.4, hjust = -0.3),size=5)# +
    #facet_grid(cols = vars(type), scales = "free", space = "free")
  png(filename = paste0(pa,"/Top20Stat.png"),width = 25,height = 18,units = "cm",res=600)
  pdf(file=paste0(pa,"/Top20Stat.pdf"),width=12,height=9)
  print(p)
  dev.off()
  print(p)
  dev.off()
}

####绘制丰度差异图
DAScore_plot <- function(map2query=map2query,Enrichment=em,up_list = uplist,down_list =downlist,pa = pa) {
  da_fuc <- function(u,d,t){
    da <- (u-d)/t
    return(da)
  }
  data1 <- Enrichment[Enrichment$p.value <0.05&Enrichment$Over_Under == "over",3:5]
  if(nrow(data1) != 0){
    data2 <- left_join(data1,map2query,by = "Map_ID") 
    data3 <- data2[,c(1:5,7:8)]
    print("-------")
    print(up_list)
    print(down_list)
    print("---------")
    up_numb <- c()
    down_numb <-c() 
    for (i in 1:nrow(data3)){
      cpd <- data3$cpd[i]
      cpd <- unlist(strsplit(cpd,"\\+",perl = TRUE))
      up_numb <- c(up_numb,length(intersect(cpd,up_list[,2])))
      down_numb <- c(down_numb,length(intersect(cpd,down_list[,2])))
    }
    data3$up_numb <-  up_numb
    data3$down_numb <-  down_numb
    
    data3$da <- da_fuc(data3$up_numb,data3$down_numb,data3$Test)
    data3 <- data3[,c(4:5,1:3,7:10)]
    colnames(data3) <- c("Pathway_Hierarchy1","Pathway_Hierarchy2","Map_ID","Map_Name","Pathway.Size","cpdName","up_numb","down_numb","Diff.Abundance.Score")
    write.table(data3, file = paste0(pa,"/pathway_difference_abundance.xls"), sep = "\t", col.names = T, row.names = F, quote = F)
    
    p <- ggplot(data =data3,aes(x=Diff.Abundance.Score,y =Map_Name,colour =Pathway_Hierarchy1))+
      geom_segment(aes(yend =Map_Name),xend =0,colour = "grey50")+
      geom_point(aes(size = Pathway.Size,fill = Diff.Abundance.Score),shape = 21)+
      scale_size_area(max_size = 6)+
      scale_fill_gradient2(low= "blue",mid="grey",high ="red")+ 
      scale_colour_manual(values = c(rep("white",length(data3$Pathway_Hierarchy1))))+
      theme_bw()+
      theme(panel.grid.minor.y = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.border=element_rect(colour = "black", fill = NA),  #边框
            axis.line=element_line(size=0.2),
            axis.ticks.length = unit(-0.15, 'cm'),
            axis.text.x = element_text(colour = "black",vjust = 0.5,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
            axis.text.y = element_text(colour = "black",margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
            axis.title.x=element_text(size = 10,face="bold"),
            axis.title.y=element_text(size = 10,face="bold"),
            plot.title = element_text(size = 15, hjust=0.5,face="bold"))+
      geom_vline(xintercept = 0,lty = 3,colour = "grey")+
      guides(fill = guide_colourbar(order=1),size = guide_legend(order=2),colour = FALSE)+
      labs(y="Pathway",x = "Differential Abundance Score" ) 
    ggsave(filename = paste0(pa,"/DAScore_plot.png"),p, device = "png",width = 20, height = 21, units = "cm", dpi = 300)
    ggsave(filename = paste0(pa,"/DAScore_plot.pdf"),p, device = "pdf",width = 20/2.54, height = 21/2.54)
    #H2
    data_H2 <- data3[order(data3$Pathway_Hierarchy2),]
    data_H2$Pathway_Hierarchy2 <- factor(data_H2$Pathway_Hierarchy2,levels = unique(data_H2$Pathway_Hierarchy2))
    data_H2$Map_Name <- factor(data_H2$Map_Name,levels = data_H2$Map_Name)
    colorss <- colpalette[as.numeric(data_H2$Pathway_Hierarchy2)]
    uniq.cols <- unique(colorss)
    p_H2 <- ggplot(data =data_H2,aes(x=Diff.Abundance.Score,y =Map_Name,colour = Pathway_Hierarchy2))+
      geom_segment(aes(yend =Map_Name),xend =0,colour = "grey50")+
      geom_point(aes(size = Pathway.Size,fill = Diff.Abundance.Score),shape = 21)+
      scale_colour_manual(values = c(rep("white",length(uniq.cols))))+
      scale_size_area(max_size = 6)+
      scale_fill_gradient2(low= "blue",mid="grey",high ="red")+ 
      theme_bw()+
      theme(panel.grid.minor.y = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.border=element_rect(colour = "black", fill = NA),  #边框
            axis.line=element_line(size=0.2),
            axis.ticks.length = unit(-0.15, 'cm'),
            axis.text.x = element_text(colour = "black",vjust = 0.5,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
            axis.text.y = element_text(colour = colorss,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
            axis.title.x=element_text(size = 10,face="bold"),
            axis.title.y=element_text(size = 10,face="bold"),
            plot.title = element_text(size = 15, hjust=0.5,face="bold")
      )+
      geom_vline(xintercept = 0,lty = 3,colour = "grey")+
      guides(fill = guide_colourbar(order=1),size = guide_legend(order=2), colour = guide_legend(order=3,ncol = 2,override.aes=list(colour = uniq.cols,shape = 16)))+
      labs(y="Pathway",x = "Differential Abundance Score" ) 
    ggsave(filename = paste0(pa,"/DAScore_plot_H2.png"),p_H2, device = "png",width = 30, height = 21, units = "cm", dpi = 300)
    ggsave(filename = paste0(pa,"/DAScore_plot_H2.pdf"),p_H2, device = "pdf",width =30/2.54, height = 21/2.54)
    
    #H1
    data_H1 <- data3[order(data3$Pathway_Hierarchy1),]
    data_H1$Pathway_Hierarchy1 <- factor(data_H1$Pathway_Hierarchy1,levels = unique(data_H1$Pathway_Hierarchy1))
    data_H1$Map_Name <- factor(data_H1$Map_Name,levels = data_H1$Map_Name)
    colorss <- colpalette[as.numeric(data_H1$Pathway_Hierarchy1)]
    uniq.cols <- unique(colorss)
    p_H1 <- ggplot(data =data_H1,aes(x=Diff.Abundance.Score,y =Map_Name,colour = Pathway_Hierarchy1))+
      geom_segment(aes(yend =Map_Name),xend =0,colour = "grey50")+
      geom_point(aes(size = Pathway.Size,fill = Diff.Abundance.Score),shape = 21)+
      scale_colour_manual(values = c(rep("white",length(uniq.cols))))+
      scale_size_area(max_size = 6)+
      scale_fill_gradient2(low= "blue",mid="grey",high ="red")+ 
      theme_bw()+
      theme(panel.grid.minor.y = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.border=element_rect(colour = "black", fill = NA),  #边框
            axis.line=element_line(size=0.2),
            axis.ticks.length = unit(-0.15, 'cm'),
            axis.text.x = element_text(colour = "black",vjust = 0.5,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
            axis.text.y = element_text(colour = colorss,margin = unit(c(0.3, 0.3, 0.3, 0.3), 'cm')),
            axis.title.x=element_text(size = 10,face="bold"),
            axis.title.y=element_text(size = 10,face="bold"),
            plot.title = element_text(size = 15, hjust=0.5,face="bold"))+
      geom_vline(xintercept = 0,lty = 3,colour = "grey")+
      guides(fill = guide_colourbar(order=1),size = guide_legend(order=2), colour = guide_legend(order=3,ncol = 1,override.aes=list(colour = uniq.cols,shape = 16)))+
      labs(y="Pathway",x = "Differential Abundance Score" ) 
    ggsave(filename = paste0(pa,"/DAScore_plot_H1.png"),p_H1, device = "png",width = 22, height = 21, units = "cm", dpi = 300)
    ggsave(filename = paste0(pa,"/DAScore_plot_H1.pdf"),p_H1, device = "pdf",width =22/2.54, height = 21/2.54)
  }else cat('', file = paste0(pa, '/无Differential Abundance Score结果.txt'))
}



getInputname<-function(samplelist){
  if(file.exists(samplelist)){
    newname <- read.xlsx(samplelist, rowNames = F, check.names = F)
    newname <- newname[,1:2]
    Name <- melt(newname, id = colnames(newname)[1])
    inputname <- dplyr::select(Name, colnames(Name)[1], value) %>% dplyr::filter(value != "")
    names(inputname) <- c("name", "group")
    inputname <- inputname[order(inputname$group),] #01.07  #保证同一组在表单排序在一起
  }else{
    stop("please specify the absoulte path of newname.xlsx")
  }
  if ("QC" %in% inputname$group){
    inputname <- rbind(dplyr::filter(inputname, group != "QC"), dplyr::filter(inputname, group == "QC")) 
  }
  return(inputname)
}



extractMain<-function(file,vs,inputname){
  ##for 循环提取需要得文件
  groupvs<-as.vector(read.table(vs)$V1)
  for(i in seq(length(groupvs))){
    print(paste0(i, ".", groupvs[i]))
    folder <- gsub('\\|', '_', groupvs[i])
    data<-read.xlsx(xlsxFile = file, sheet = groupvs[i], colNames = T,check.names = F)
    data.Extract(data,groupvs[i],inputname)
  }
}


####KEGG主函数
keggMain<-function(path,species,n){
  ####读取KEGG数据库
  ####读取比较组文件夹
  resultdir <- list.dirs(getwd(),full.names = TRUE,recursive = FALSE) 
  resultdir <- resultdir[grepl("_vs_|_",resultdir)]
  
  for (difdir in resultdir){
    KEGG_folder <- paste0(difdir, "/KEGG/")
    path_pack <- strsplit(KEGG_folder,"/")
    folder <- path_pack[[1]][length(path_pack[[1]])-1]
    KeggPath <- paste0(path,"/报告及附件/附件2 Result/07. Lipid KEGG Analysis/",folder,"/")
    dir.create(KeggPath,recursive = T,showWarnings = F)
    
    print(folder)
    if(file.exists(paste0(KEGG_folder,"/diff.txt"))){
      fit0 <- try(read.table(paste0(KEGG_folder,"/diff.txt"),header = FALSE,sep = '\t',fill = TRUE,quote = "",
                             stringsAsFactors = FALSE,check.names = TRUE),silent = TRUE)
      if (!'try-error' %in% class(fit0)) {
        diflist <- read.table(paste0(KEGG_folder,"/diff.txt"),header = FALSE,sep = '\t',fill = TRUE,quote = "",
                              stringsAsFactors = FALSE,check.names = TRUE)
        query2cpd <- diflist
        
        names(query2cpd) <- c("Metabolite", "cpdID") 
        query2cpd_OUT <- query2cpd
        colnames(query2cpd_OUT) <- c("Metabolite","KEGG.ID")
        
        write.table(query2cpd,paste0(KEGG_folder,"/query.cpd"),row.names = FALSE,col.names = FALSE,sep = "\t",quote=F,na = "")
        
        query2cpd <- separate_rows(query2cpd,cpdID,sep ="/")
        
        query2cpd <- left_join(query2cpd,cpdid2cpd ,by="cpdID")
        
        #输出query2map.txt文件
        query2map <- q2map(query2cpd,pathway,background)
        write.table(query2map[,1:6],paste0(KEGG_folder,"/query2map.txt"),row.names = FALSE,col.names = TRUE,sep = "\t",quote=F,na= "")
        #生成map2query文件
        fit <- try(map2q(query2map),silent = TRUE)
        if(!'try-error' %in% class(fit)){
          map2query <- map2q(query2map)
          write.table(map2query,paste0(KEGG_folder,"/map2query.txt"),row.names = FALSE,col.names = TRUE,sep = "\t",quote=F,na= "")
          if(n==1){
            TestAll <- length(unique(query2map[!is.na(query2map$cpdID),]$cpdID))
            myback <- pathway[pathway$V1 %in% map2query$Map_ID,1:2]
            em <- Enrichment(x= map2query,y = myback,difnum = TestAll,allnum = RefAll)
            write.table(em,paste0(KEGG_folder,"/Enrichment.txt"),row.names = FALSE,col.names = TRUE,sep = "\t",quote=F,na= "")
            Kegg_em_path <- paste0(KeggPath,"KEGG Enrichment Analysis/")
            dir.create(Kegg_em_path,recursive = T,showWarnings = F)   
            tms(map2query,Kegg_em_path)
            kegg_bubble(em,Kegg_em_path)
            row_num <- as.numeric(nrow(filter(em, p.value < 0.05 & Over_Under == "over")))
          }
          
          ###生成KEGG.xlsx
          wb <- createWorkbook()
          modifyBaseFont(wb, fontName = "Arial", fontSize = 10.5)
          addWorksheet(wb, sheetName = "query2map")
          addWorksheet(wb, sheetName = "map2query")
          addWorksheet(wb, sheetName = "IDmapping")
          header_style <- createStyle(textDecoration = "bold", halign = "left")
          addStyle(wb, sheet = "query2map", rows = 1, cols = 1:ncol(query2map), style = header_style)
          addStyle(wb, sheet = "map2query", rows = 1, cols = 1:ncol(map2query), style = header_style)
          addStyle(wb, sheet = "IDmapping", rows = 1, cols = 1:ncol(query2cpd), style = header_style)
          setColWidths(wb, sheet = "query2map", cols = 1, widths = 30)
          setColWidths(wb, sheet = "query2map", cols = 4, widths = 11)
          setColWidths(wb, sheet = "query2map", cols = 5, widths = 42)
          setColWidths(wb, sheet = "map2query", cols = 1, widths = 30)
          setColWidths(wb, sheet = "map2query", cols = 2, widths = 42)
          setColWidths(wb, sheet = "map2query", cols = 4, widths = 50)
          setColWidths(wb, sheet = "IDmapping", cols = 1, widths = 30)
          query2map_out <- query2map[,1:6]
          colnames(query2map_out)[2] <- "KEGG.ID"
          map2query_out <- map2query
          colnames(map2query_out)[6] <- "KEGG.ID"
          writeData(wb, sheet = "query2map", x = query2map_out)
          writeData(wb, sheet = "map2query", x = map2query_out)
          writeData(wb, sheet = "IDmapping", x = query2cpd_OUT)
          #print("page 789")
          if (file.exists(paste0(difdir,"/KEGG/Enrichment.txt"))){
            addWorksheet(wb, sheetName = "Enrichment")
            addStyle(wb, sheet = "Enrichment", rows = 1, cols = 1:ncol(em), style = header_style)
            setColWidths(wb, sheet = "Enrichment", cols = 1, widths = 39)
            setColWidths(wb, sheet = "Enrichment", cols = 2, widths = 42)
            setColWidths(wb, sheet = "Enrichment", cols = 4, widths = 55)
            setColWidths(wb, sheet = "Enrichment", cols = 13, widths = 10)
            if (row_num > 0){
              addStyle(wb, sheet = "Enrichment", style = createStyle(fgFill="yellow"), rows = 2:(row_num + 1), cols = c(4,11,12), gridExpand = T)
            }
            writeData(wb, sheet = "Enrichment", x = em)
          }
          saveWorkbook(wb, paste0(KeggPath,"Kegg.xlsx"), overwrite = T)
          DEP_file <- read.table(file = paste0(KEGG_folder,"/info-H2400.txt"), sep = "\t", header = T, stringsAsFactors = F, quote = "",fileEncoding = "UTF-8",check.names=F)
          #========================== map INPUT FILE ==========================#
          files <- list.files(path =KEGG_folder)
          
          if ("META_up.txt" %in% files & "META_down.txt" %in% files){
            DEP_file<-DEP_file[,c("Name","KEGG","FC","pvalue")]
            names(DEP_file) <- c("metaid","keggid","fc","pvalue")
            fit_up<- try(uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = ""),silent = TRUE)
            if('try-error' %in% class(fit_up)){
              upko <- query2cpd_OUT[1,][-1,]
              if(file.info(paste0(KEGG_folder,"/META_up.txt"))$size != 0){
                uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else{
                uplist <- data.frame(V1 = 1,V2=1)
              }
            }else{
              ########################################
              if(file.info(paste0(KEGG_folder,"/META_up.txt"))$size != 0){
                uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else{
                uplist <- data.frame(V1 = 1,V2=1)
              }
              upko <- uplist
              names(upko) <- c("metaid", "keggid")
            }
            fit_down<- try(downlist <- read.table(file = paste0(KEGG_folder,"/META_down.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = ""),silent = TRUE) 
            if('try-error' %in% class(fit_down)){
              downko <- query2cpd_OUT[1,][-1,]
              if(file.info(paste0(KEGG_folder,"/META_down.txt"))$size != 0){
                downlist <- read.table(file = paste0(KEGG_folder,"/META_down.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else downlist <- data.frame(V1 = 1,V2 = 1)
              
            }else{
              if(file.info(paste0(KEGG_folder,"/META_down.txt"))$size != 0){
                downlist <- read.table(file = paste0(KEGG_folder,"/META_down.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else downlist <- data.frame(V1 = 1,V2 = 1)
            }
            pathway_folder <- paste0(KeggPath, "KEGG Map/")
            if (dir.exists(pathway_folder)){
              setwd(pathway_folder)
            }else{
              dir.create(pathway_folder)
            }
            #========================== ad==========================#
            DEP_info <- mutate(DEP_file, information = paste0(metaid, " ( FC=", round(fc,3), "; P.value=", signif(pvalue, 3), " )"))
            DEP_info <- DEP_info[,c("metaid", "information")]
            Kegg_DA_path <- paste0(KeggPath,"Differential Abundance Score/")
            dir.create(Kegg_DA_path,recursive = T,showWarnings = F)
            DAScore_plot(map2query=map2query,Enrichment=em,up_list = uplist,down_list =downlist,pa = Kegg_DA_path) 
          }else{
            print(" Oneway FC not exits,No DA plot")
          }
          setwd(path)
        }else{
          cat("", file= paste0(KeggPath,"差异物无map信息,无kegg分析.txt"), append = TRUE)
        }
      }else{
        cat("", file= paste0(KeggPath,"无差异物，无kegg分析.txt"), append = TRUE)
      }
    }
  }
}


args <- commandArgs(T)
path <- as.character(args[1])
vs <- "groupvs.txt"
file<- "附件1.Lipidomics表.xlsx"
samplelist<- "newname.xlsx"
species <- as.character(args[2])
n <- as.numeric(args[3])

####读取KEGG数据库
Hierarchy <- read.table("/database/metabolome/code_Version1/KEGG/species_specific/br08901.list",
					  header=F,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)
cpdid2cpd <- read.table("/database/metabolome/database/cpdid_20201021/cpdid2cpd.txt",
					  header=F,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)
## hsaxxxx num CXXX1;CXXX2;...
pathway <- read.table(paste0("/database/metabolome/code_Version1/KEGG/species_specific/","kegg_",species,".Compound"),
					header=F,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)
##  name  Cxxx
all_ko <- read.table("/database/metabolome/code_Version1/KEGG/metabo.txt",
				   header=T,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)
colnames(Hierarchy) <- c("Map_ID","Map_Name","Pathway_Hierarchy2","Pathway_Hierarchy1")
Hierarchy$Map_ID <- sub(pattern = "map", replacement = species, Hierarchy$Map_ID)
colnames(cpdid2cpd) <- c("cpdID","cpdName")
cpdid2cpd$cpdName <- str_replace_all(cpdid2cpd$cpdName,"; ",";")
pathway <- left_join(pathway,Hierarchy,by=c("V1"="Map_ID"))
pathway_list <- apply(pathway, 1, function(x){strsplit(x[3], ',')}) #物种特异性
background <- unique(unlist(pathway_list))
RefAll <- length(background)


inputname<-getInputname(samplelist)
extractMain(file,vs,inputname)
keggMain(path,species,n)






