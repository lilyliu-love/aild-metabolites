rm(list=ls())
print("kegg analysis beging!")

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
stringsAsFactors = FALSE


setwd(path)

args <- commandArgs(T)
path <- as.character(args[1])
species <- as.character(args[2])
n <- as.numeric(args[3])  #n = 1  #富集分析 n = 0 不富集
setwd(path)
colpalette <- c("#1d953f", "#102b6a", "#c77eb5", "#ffce7b", "#2585a6", "purple", "#e0861a", "#d71345", "#6b473c", "#78a355", "#fdb933", "#5e7c85", "#411445", "#c37e00", "#bed742", "#009ad6", "#9d9087", "#aa2116", "#225a1f", "#f3715c", "#7bbfea", "#dbce8f", "#f8aba6", "#778899", "#2F4F4F", "#2E8B57", "#8B6969", "#EE2C2C", "#6495ED", "#87CEEB", "#0000CD", "#63B8FF", "#006400", "#00FF7F", "#9400D3", "#FFAEB9", "#8B5A2B", "#FFFF00", "#BDB76B", "#FF1493")
command <- paste("Rscript kegg_batch_meta.R",path,species,n,sep = " ")
write.table(command, file = "command_KEGG.txt", quote = F, sep = "\t", row.names = F, col.names = F) 

# regroup or not
if(file.exists("newname.xlsx")){
  newname <- read.xlsx("newname.xlsx", rowNames = F, check.names = F)
  Name <- melt(newname, id = colnames(newname)[1])
  inputname <- dplyr::select(Name, colnames(Name)[1], value) %>% filter(value != "")
  names(inputname) <- c("name", "group")
  inputname <- rbind(filter(inputname, group != "QC"), filter(inputname, group == "QC"))
}else{
  DATA <- read.csv("pos-dele-iso.csv", sep = ",", check.names = F)
  samplename <- colnames(DATA)[grep("-\\d+$", colnames(DATA))]
  inputname <- data.frame(name = samplename, group = str_remove(samplename, "-\\d+$"))
}

if ("QC" %in% inputname$group){
  inputname <- rbind(filter(inputname, group != "QC"), filter(inputname, group == "QC")) 
}


Hierarchy <- read.table("/database/metabolome/code_Version1/KEGG/species_specific/br08901.list",
                        header=F,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)
cpdid2cpd <- read.table("/database/metabolome/code_Version1/KEGG/species_specific/cpdid2cpd.txt",
			header=F,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)
## hsaxxxx num CXXX1;CXXX2;...
pathway <- read.table(paste0("/database/metabolome/code_Version1/KEGG/species_specific/","kegg_",species,".Compound"),
                      header=F,sep="\t",fill=TRUE,quote = "",encoding='UTF-8',check.names=F,stringsAsFactors = FALSE)

colnames(Hierarchy) <- c("Map_ID","Map_Name","Pathway_Hierarchy2","Pathway_Hierarchy1")

Hierarchy$Map_ID <- sub(pattern = "map", replacement = species, Hierarchy$Map_ID)

colnames(cpdid2cpd) <- c("cpdID","cpdName")
cpdid2cpd$cpdName <- str_replace_all(cpdid2cpd$cpdName,"; ",";")
pathway <- left_join(pathway,Hierarchy,by=c("V1"="Map_ID"))
pathway_list <- apply(pathway, 1, function(x){strsplit(x[3], ',')}) #物种特异性
background <- unique(unlist(pathway_list))
RefAll <- length(background)

#names(all_ko) <- c("Metabolite", "cpdID")  #all_ko 全部的metabo.txt
#all_ko[all_ko == ""] <- NA
#allko <- all_ko
#allko <- allko[complete.cases(allko),] #allko 非空的metabo.txt

print("服务器工作结束！！！！！！！")
# *******************************************************************************************************************
# querycpd()函数-------输出query.cpd的函数
# *******************************************************************************************************************
querycpd <- function(x,al,cpdid2cpd){
  query <- left_join(x,al,by=c("V1"="Metabolite"))
  query <-  query[order(query[,2]),]
  return(query)
}

# *******************************************************************************************************************
# q2map()函数---------输出query2map的函数 参数为query.cpd和kegg_XX.compound
# *******************************************************************************************************************
#q2map(query2cpd,pathway)
q2map <- function(aqko =query2cpd,pathway =pathway){
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


# *******************************************************************************************************************
# map2q()函数---------输出map2query2的函数 参数为query2map
# *******************************************************************************************************************
map2q <- function(x = query2map){
  map_ids <- unique(x[which(x[,4]!=""),4])
  rname <- c("Pathway_Hierarchy1","Pathway_Hierarchy2","Map_ID","Map_Name","cpdName","cpd","Num_cpd","URL")
  result <- as.data.frame(matrix(numeric(0),ncol=8))
  colnames(result) <- rname
  for (i in 1:length(unique(map_ids))) {
    ap <- filter(x, x[,4]==as.character(unique(map_ids)[i]))
    mapid <- as.character(unique(map_ids)[i])
    myurl <- paste0("https://www.kegg.jp/kegg-bin/show_pathway?",mapid,"+",str_c(ap[,2],collapse = "+"))
    m2aq <- data.frame(ap[1,c(7,8)],mapid,ap[1,5],str_c(ap[,1],collapse = "|"),str_c(ap[,2],collapse = "+"),length(unique(ap[,2])),myurl,stringsAsFactors=FALSE,check.names=F)
    result <- rbind(result,m2aq)
  }
  result <- as.data.frame(result,stringsAsFactors = FALSE)
  names(result) <- rname
  result$Num_cpd <- as.numeric(result$Num_cpd)
  result <- arrange(result,desc(Num_cpd))
  return(result)
}

# *******************************************************************************************************************
# Enrichment()函数---------
# *******************************************************************************************************************
Enrichment <- function(x= map2query,y = myback,difnum = TestAll,allnum = RefAll){
  myquery <- data.frame(x[,c(1:4,7)],rep(difnum,nrow(x)),stringsAsFactors = FALSE,check.names = FALSE)
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

# *******************************************************************************************************************
# kegg_bubble()函数---------kegg富集气泡图,同时输出kegg富集柱状图备用
# *******************************************************************************************************************
kegg_bubble <- function(data = em,pa = KeggPath){
  #Enrich <- subset(data, data$Over_Under == "over",select = names(data)) #同蛋白一致，绘制p 从小到大top20
  Enrich <- subset(data, data$p.value<0.05 & data$Over_Under == "over",select = names(data))#代谢绘制p<0.05的top20
  Enrich <- Enrich[order(Enrich$p.value),]
  Enrich <- head(Enrich,20)
  Enrich <- Enrich[order(Enrich$richFactor),]
  Enrich$richFactor <- round(Enrich$richFactor, digits = 2)
  Enrich$Map_Name <- factor(Enrich$Map_Name,levels=(unique(Enrich$Map_Name)))
  Enrich$type <- rep("KEGG Enrichment",nrow(Enrich))
  maxrol <- as.numeric(nrow(Enrich))

  xname <- "Enriched KEGG Pathways"
  #xname <- "KEGG Pathways(Top 20)"
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
    ggsave(filename = paste0(pa,"EnrichmentBubble.png"), p, device = "png", width = 20, height = 20, units = "cm", dpi = 300,limitsize = FALSE)
    ggsave(filename = paste0(pa,"EnrichmentBubble.pdf"), p, device = "pdf", width = 20/2.54, height = 20/2.54,limitsize = FALSE)
    
    
    p1 <- ggplot(Enrich, aes(x= Map_Name, y = Test, fill = p.value)) +
      geom_bar(stat = "identity", width = mywidth) + 
      scale_fill_gradient(limits = c(0,0.05),low = "red", high = "gold2") + 
      labs(fill = "p.value", y = "Compound Number", x = xname) +
      #ylim(0, ceiling(max(Enrich$Test)*1.1)) +
      scale_y_continuous(breaks=seq(1,ceiling(max(Enrich$Test)*1.1),by = 1),limits= c(0,ceiling(max(Enrich$Test)*1.1)))+
      #scale_y_continuous(breaks = "auto")+
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

      if (max(Enrich$Test) < 7){a = 4}else if (max(Enrich$Test) < 10){ a = 2}else if (max(Enrich$Test) < 40){a = 1.5}else if (max(Enrich$Test) >= 40){a = 1.4}
    print(Enrich$Test)

    ggsave(filename = paste0(pa,"EnrichmentBar.png"), p1, device = "png", width = (max(Enrich$Test)+2)*a , height = 20, units = "cm", dpi = 300,limitsize = FALSE)
    ggsave(filename = paste0(pa,"EnrichmentBar.pdf"), p1, device = "pdf", width = (max(Enrich$Test)+2)*a/2.54 , height = 20/2.54,limitsize = FALSE)
    
  }else{
    word <- "该比较组没有富集结果"
    write.table(word, file = paste0(pa,"/备注.txt"), row.names = F, col.names = F, quote = F)
  }
}

# *******************************************************************************************************************
# get_pic_ud()函数---------分上下调的获取map图的函数
# *******************************************************************************************************************
get_pic_ud <- function(url, pathway_id){
  pic <- try(read_html(url) %>% html_nodes("img") %>% extract2(2) %>% html_attr("src"), silent = T)
  error <- 1
  while (class(pic) == "try-error"){
    if (error > 20){
      stop(paste0(url, " read_html error !!!"))
    }
    Sys.sleep(2)
    pic <- try(read_html(url) %>% html_nodes("img") %>% extract2(2) %>% html_attr("src"), silent = T)
    error <- error + 1
  }
  
  # info <- try(download(paste0("https://www.kegg.jp", pic), paste0(pathway_id, ".png"), mode = "wget"), silent = T)
  info <- try(download(paste0("https://www.kegg.jp", pic), paste0(pathway_id, ".png"), mode = "wb"), silent = T)
  error <- 1
  while (class(info) == "try-error"){
    if (error > 20){
      stop(paste0(url, " download error !!!"))
    }
    Sys.sleep(2)
    info <- try(download(paste0("https://www.kegg.jp", pic), paste0(pathway_id, ".png"), mode = "wget"), silent = T)
    error <- error + 1
  }
}
# *******************************************************************************************************************
# get_pic()函数---------正常获取map图的函数(不去分上下调)
# *******************************************************************************************************************
get_pic <- function(x){
  pathway_id <- map2query[x,3]
  url <- map2query[x,ncol(map2query)]
  pic <- try(read_html(url) %>% html_nodes("img") %>% extract2(2) %>% html_attr("src"), silent = T)
  error <- 1
  while (class(pic) == "try-error"){
    if (error > 20){
      stop(paste0(url, " read_html error !!!"))
    }
    Sys.sleep(2)
    pic <- try(read_html(url) %>% html_nodes("img") %>% extract2(2) %>% html_attr("src"), silent = T)
    error <- error + 1
  }
  info <- try(download(paste0("https://www.kegg.jp", pic), paste0(pathway_id, ".png"), mode = "wb"), silent = T)
  error <- 1
  while (class(info) == "try-error"){
    if (error > 20){
      stop(paste0(url, " download error !!!"))
    }
    Sys.sleep(2)
    info <- try(download(paste0("https://www.kegg.jp", pic), paste0(pathway_id, ".png"), mode = "wget"), silent = T)
    error <- error + 1
  }
}
# *******************************************************************************************************************
# get_map2query()函数----------
# *******************************************************************************************************************
get_map2query <- function(map2query,DEP_info,upko,downko,targetname){
  for(i in 1:nrow(map2query)){
    if (paste0(map2query$Map_ID[i], ".png") %in% list.files(pattern = ".png") & paste0(map2query$Map_ID[i], ".html") %in% list.files(pattern = ".html")){
      next
    }
    url <- paste0("https://www.genome.jp/kegg-bin/show_pathway?", map2query$Map_ID[i])
    result <- try(read_html(url),silent=T)
    num <- 1
    while (!class(result)[1] == "xml_document") {
      result <- try(read_xml(url), silent = T)
      if(num == 3){
        print(paste0("please check map ",map2query$Map_ID[i]))
        break()
      }
    }
    res_list <- result %>% html_nodes("area")
    res <- plyr::llply(res_list, function(x){
      shape <- html_attr(x, "shape")
      coords <- html_attr(x, "coords")
      title <- html_attr(x, "title")
      href <- paste0("http://www.kegg.jp", html_attr(x, "href"))
      keggid <- strsplit(strsplit(href, "\\?")[[1]][2], "\\+")[[1]]
      info <- list()
      if (any(upko$keggid %in% keggid)){
        pro_up_ids <- upko$metaid[upko$keggid %in% keggid]
        for(j in 1:length(pro_up_ids)){
          metaid2keggid <- upko$keggid[pro_up_ids[j] == upko$metaid]
          metaid2information <- DEP_info$information[pro_up_ids[j] == DEP_info$metaid]
          info <- dplyr::combine(info, list(c(list(title = metaid2keggid), list(subTitle = targetname), list(text = metaid2information))))
        }
      }
      if (any(downko$keggid %in% keggid)){
        pro_down_ids <- downko$metaid[downko$keggid %in% keggid]
        for(j in 1:length(pro_down_ids)){
          metaid2keggid <- downko$keggid[pro_down_ids[j] == downko$metaid]
          metaid2information <- DEP_info$information[pro_down_ids[j] == DEP_info$metaid]
          info <- dplyr::combine(info, list(c(list(title = metaid2keggid), list(subTitle = "Down regulated metabolite"), list(text = metaid2information))))
        }
      }
      if (length(info) == 0){
        list(shape = shape, coords = coords, href = href, title = title)
      }else{
        list(shape = shape, coords = coords, href = href, title = title, info = info)
      }
    })
    # col_url <- lapply(res, function(x){
    #   if (!is.null(x$info)){
    #     subtitle <- unlist(lapply(x$info, function(x)x["subTitle"]))
    # 
    #     if (any(grepl("up", subtitle, ignore.case = T)) & any(grepl("down", subtitle, ignore.case = T))){
    #       col <- paste0(unlist(lapply(x$info, function(x){x["title"]})), "%09", "yellow")
    #     }else{
    #       col <- ifelse(grepl("up|target", subtitle, ignore.case = T),
    #                     paste0(unlist(lapply(x$info, function(x){x["title"]})), "%09", "red"),
    #                     paste0(unlist(lapply(x$info, function(x){x["title"]})), "%09", "green"))
    #     }
    #   }
    # })
    #col_url <- unique(na.omit(unlist(col_url)))
    # if (is.null(col_url)){
    #   res_url <- paste0("https://www.kegg.jp/kegg-bin/show_pathway?", map2query$Map_ID[i])
    # }else{
    #   res_url <- paste0("https://www.kegg.jp/kegg-bin/show_pathway?", map2query$Map_ID[i], "/", paste(col_url, collapse = "/"))
    # }
    get_pic(i)
    dd <- toJSON(res)
    #服务器测试
    content <- read_lines(file = "/database/metabolome/code_Version1/KEGG/template.html")
    #本地测试
    #content <- read_lines(file = "E:/commoncodes/Non-targeted/KEGG/template.html")
    json <- paste0("var imageList = ", dd)
    content <- sub("pathway_id", map2query$Map_ID[i], content)
    content <- sub("js_local", json, content) 
    write_lines(content, path = paste0(map2query$Map_ID[i], ".html"))
  }
}
# *******************************************************************************************************************
# DAScore_plot()函数----------绘制KEGG DA score气泡图
# *******************************************************************************************************************
DAScore_plot <- function(map2query=map2query,Enrichment=em,up_list = uplist,down_list =downlist,pa = pa) {
  da_fuc <- function(u,d,t){
    da <- (u-d)/t
    return(da)
  }
  # map2query <- map2query
  # Enrichment <- em
  # up_list <- uplist
  # down_list <- downlist
  data1 <- Enrichment[Enrichment$p.value <0.05&Enrichment$Over_Under == "over",3:5]
  if(nrow(data1) != 0){
	data2 <- left_join(data1,map2query,by = "Map_ID") 
  data3 <- data2[,c(1:5,7)]
  print("-------")
  print(up_list)
  print(down_list)
  print("---------")
  up_numb <- c()
  down_numb <-c() 
  for (i in 1:nrow(data3)){
    cpd <- data3$cpdName[i]
    cpd <- unlist(strsplit(cpd,"\\|",perl = TRUE))
    up_numb <- c(up_numb,length(intersect(cpd,up_list[,1])))
    down_numb <- c(down_numb,length(intersect(cpd,down_list[,1])))
  }
  data3$up_numb <-  up_numb
  data3$down_numb <-  down_numb
  
  data3$da <- da_fuc(data3$up_numb,data3$down_numb,data3$Test)
  data3 <- data3[,c(4:5,1:3,6:9)]
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
    #geom_text(label = data_H2$Map_Name,colour = colorss, hjust = 0.5)+
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
    #guides(fill=guide_legend(title =NULL,override.aes = list(size = 5)))+
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
    #geom_text(label = data_H1$Map_Name,colour = colorss, hjust = 0.5)+
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
    #guides(fill=guide_legend(title =NULL,override.aes = list(size = 5)))+
    guides(fill = guide_colourbar(order=1),size = guide_legend(order=2), colour = guide_legend(order=3,ncol = 1,override.aes=list(colour = uniq.cols,shape = 16)))+
    labs(y="Pathway",x = "Differential Abundance Score" ) 
  ggsave(filename = paste0(pa,"/DAScore_plot_H1.png"),p_H1, device = "png",width = 22, height = 21, units = "cm", dpi = 300)
  ggsave(filename = paste0(pa,"/DAScore_plot_H1.pdf"),p_H1, device = "pdf",width =22/2.54, height = 21/2.54)
  }else cat('', file = paste0(pa, '/无Differential Abundance Score结果.txt'))
  
}
# *******************************************************************************************************************
# Pathway_heatmap()函数----------绘制KEGG通路热图
# *******************************************************************************************************************
# Pathway_heatmap <- function(query2map= query2map,Metabolites =Metabolites,Enrichment =em,pa = pa){
#   print("pathway heatmap")
#   kegg2cpd <- left_join(Metabolites,query2map[,c(1,4,5)],by=c("Name"="Metabolite")) 
#   kegg2cpd <- left_join(Enrichment[,c(2,4,12)],kegg2cpd,by= "Map_Name")
#   
#   kegg2cpd <- filter(kegg2cpd,p.value <0.05)  #Enriched pathway
#   kegg2samples<- kegg2cpd[,-c(1,3,4,ncol(kegg2cpd))]
#   
#   kegg2samples_sum <- data.frame(pathway=kegg2samples$Map_Name,stringsAsFactors = F)
#   for(cols in colnames(kegg2samples[2:ncol(kegg2samples)])){
#     #print(cols)
#     kegg2samples_sum[,cols] <- ave(kegg2samples[,cols],kegg2samples$Map_Name,FUN=sum)
#   }
#   kegg2samples_sum <- filter(kegg2samples_sum,pathway!="") %>% unique()
#   kegg2samples_sum$sum <- apply(kegg2samples_sum[,2:ncol(kegg2samples_sum)],1,sum,na.rm=T)
#   kegg2samples_sum <- filter(kegg2samples_sum,sum>0)
#   kegg2samples_sum <- kegg2samples_sum[order(kegg2samples_sum$sum,decreasing = T),]
#   write.table(kegg2samples_sum,paste0(pa,"pathway_per_samples.xls"),sep = "\t",col.names = T,row.names = F,quote = F)
#   
#   #normalize
#   kegg2samples_sum[kegg2samples_sum==0] <- NA
#   clu <- apply(kegg2samples_sum[,2:(ncol(kegg2samples_sum)-1)],1,function(x){(x-mean(x,na.rm = T))/sd(x,na.rm = T)}) %>% t() %>% as.data.frame()
#   rownames(clu) <- kegg2samples_sum$pathway
#   annotation_col <- data.frame(group= str_replace_all(colnames(clu),"-\\d+$",""))
#   rownames(annotation_col) <- colnames(clu)
#   
#   max_row_width <- max(nchar(rownames(clu)))
#   max_col_height <- max(nchar(colnames(clu)))
#   pic_width <- (ncol(clu) + max_row_width) * 0.3
#   pic_height <- (nrow(clu) + max_col_height) / 6.5
#   
#   if(nrow(clu) < 12){
#     pic_height = 10
#   }else if(nrow(clu) < 22){
#     pic_height = 10 + nrow(clu)/6.5
#   }else if(nrow(clu) < 42){
#     pic_height = 12 + nrow(clu)/6.5
#   }else if(nrow(clu) < 102){
#     pic_height = 15 + nrow(clu)/6.5
#   }else pic_height = nrow(clu) * 0.3
#   while (!is.null(dev.list()))  dev.off() 
#   png(filename = paste0(pa,"/Pathway_heatmap.png"),width = 1.2*pic_width,height = 1.2*pic_height,units = "cm",res = 300)
#   pp <- pheatmap(clu,annotation_col = annotation_col,annotation_legend = T,annotation_names_col=F,
#                  show_colnames = T,show_rownames=T,fontsize=10,
#                  #labels_row=labels_row,
#                  cellwidth= 12,cellheight = 9,
#                  clustering_distance_cols = "euclidean",
#                  clustering_distance_rows = "euclidean",
#                  color = colorRampPalette(c("navy", "white", "firebrick3"))(100)
#   )
#   dev.off()
#   
#   pdf(file = paste0(pa,"/Pathway_heatmap.pdf"),width = 0.5*pic_width,height = 0.5*pic_height)
#   pp <- pheatmap(clu,annotation_col = annotation_col,annotation_legend = T,annotation_names_col=F,
#                  show_colnames = T,show_rownames=T,fontsize=10,
#                  #labels_row=labels_row,
#                  cellwidth= 12,cellheight = 9,
#                  clustering_distance_cols = "euclidean",
#                  clustering_distance_rows = "euclidean",
#                  color = colorRampPalette(c("navy", "white", "firebrick3"))(100)
#   )
#   dev.off()
#   
# }
# *******************************************************************************************************************
# heatmap_per_Path())函数----------绘制注释到的通路中，代谢物数量大于5的热图
# *******************************************************************************************************************
heatmap_per_Path <- function(query2map = query2map,Metabolites = Metabolites,Enrichment = em,pa = pa){
  print("heatmap_per_Path")
  print(Enrichment)
  kegg2cpd1 <- left_join(Metabolites,query2map[,c(1,4,5)],by=c("Name"="Metabolite")) 
  heatmapframe <- filter(Enrichment,Test >5)
  heatmaplist <- heatmapframe$Map_ID
  kegg2cpd1 <- kegg2cpd1[,c((ncol(kegg2cpd1)-1),ncol(kegg2cpd1),1:(ncol(kegg2cpd1)-2))]
  kegg2cpd1 <- kegg2cpd1[order(kegg2cpd1$Map_ID),]
  kegg2cpd1 <- kegg2cpd1[kegg2cpd1$Map_ID != "",]
  colnames(kegg2cpd1)[3] <- "Name"
  kegg2cpd1 <- unique(kegg2cpd1)
  write.table(kegg2cpd1,paste0(pa,"meta_allpathway.xls"),sep = "\t",col.names = T,row.names = F,quote = F)
  print("heatmapframe")
  print(heatmapframe)
  if (nrow(heatmapframe) != 0 ){
    kegg2cpd1 <- kegg2cpd1[kegg2cpd1$Map_ID %in% heatmaplist, ]
    wb1 <- createWorkbook()
    for (i in 1:length(heatmaplist)){
      map_id_temp <-  heatmaplist[i]
      map_meta <- filter(kegg2cpd1,kegg2cpd1$Map_ID == map_id_temp)
      colnames(map_meta)[3] <- "Name"
      addWorksheet(wb1, sheet = map_id_temp,gridLines = TRUE)
      writeData(wb1,sheet = map_id_temp,map_meta,colNames=TRUE,rowNames=FALSE)
      setColWidths(wb1, sheet = map_id_temp, cols= 1:ncol(map_meta), widths = "auto")
      
      #normalize
      map_meta[map_meta==0] <- NA
      map_meta <- map_meta[,-c(1,2)]
      clu <- apply(map_meta[,2:ncol(map_meta)],1,function(x){(x-mean(x,na.rm = T))/sd(x,na.rm = T)}) %>% t() %>% as.data.frame()
      rownames(clu) <- map_meta$Name
      #clu_anno <- inputname[match(colnames(clu),inputname$name[which(inputname$group!= "QC")]),][2]
      annotation_col <- data.frame(group = inputname[match(colnames(clu),inputname$name[which(inputname$group!= "QC")]),][2])
      rownames(annotation_col) <- colnames(clu)
      
      max_row_width <- max(nchar(rownames(clu)))
      max_col_height <- max(nchar(colnames(clu)))
      pic_width <- (ncol(clu) + max_row_width) * 0.5
      pic_height <- (nrow(clu) + max_col_height) / 6.5
      
      if(nrow(clu) < 12){
        pic_height = 10
      }else if(nrow(clu) < 22){
        pic_height = 10 + nrow(clu)/6.5
      }else if(nrow(clu) < 42){
        pic_height = 12 + nrow(clu)/6.5
      }else if(nrow(clu) < 102){
        pic_height = 15 + nrow(clu)/6.5
      }else pic_height = nrow(clu) * 0.3
      while (!is.null(dev.list()))  dev.off() 
      png(filename = paste0(pa,map_id_temp,"_heatmap.png"),width = 1.2*pic_width,height = 1.2*pic_height,units = "cm",res = 300)
      pp <- pheatmap(clu,annotation_col = annotation_col,annotation_legend = T,annotation_names_col=F,
                     show_colnames = T,show_rownames=T,fontsize=10,
                     #labels_row=labels_row,
                     cellwidth= 12,cellheight = 9,
                     clustering_distance_cols = "euclidean",
                     clustering_distance_rows = "euclidean",
                     color = colorRampPalette(c("navy", "white", "firebrick3"))(100)
      )
      dev.off()
      
      pdf(file = paste0(pa,map_id_temp,"_heatmap.pdf"),width = 0.5*pic_width,height = 0.5*pic_height)
      pp <- pheatmap(clu,annotation_col = annotation_col,annotation_legend = T,annotation_names_col=F,
                     show_colnames = T,show_rownames=T,fontsize=10,
                     #labels_row=labels_row,
                     cellwidth= 12,cellheight = 9,
                     clustering_distance_cols = "euclidean",
                     clustering_distance_rows = "euclidean",
                     color = colorRampPalette(c("navy", "white", "firebrick3"))(100)
      )
      dev.off()
# ####################################
      png(filename = paste0(pa,map_id_temp,"_heatmap2.png"),width = 1.2*pic_width,height = 1.2*pic_height,units = "cm",res = 300)
      pp2 <- pheatmap(clu,annotation_col = annotation_col,annotation_legend = T,annotation_names_col=F,
                     show_colnames = T,show_rownames=T,fontsize=10,
                     #labels_row=labels_row,
                     cluster_col=FALSE,
                     cellwidth= 12,cellheight = 9,
                     clustering_distance_cols = "euclidean",
                     clustering_distance_rows = "euclidean",
                     color = colorRampPalette(c("navy", "white", "firebrick3"))(100)
      )
      dev.off()
      
      pdf(file = paste0(pa,map_id_temp,"_heatmap2.pdf"),width = 0.5*pic_width,height = 0.5*pic_height)
      pp2 <- pheatmap(clu,annotation_col = annotation_col,annotation_legend = T,annotation_names_col=F,
                     show_colnames = T,show_rownames=T,fontsize=10,
                     #labels_row=labels_row,
                     cluster_col=FALSE,
                     cellwidth= 12,cellheight = 9,
                     clustering_distance_cols = "euclidean",
                     clustering_distance_rows = "euclidean",
                     color = colorRampPalette(c("navy", "white", "firebrick3"))(100)
      )
      dev.off()
    }
    saveWorkbook(wb1, paste0(pa,"meta_per_map.xlsx"), overwrite = T)
    
  }else{
    cat("无包含5个以上差异代谢物的通路，heatmap无法绘制\n", file= paste0(pa,"备注.txt",sep = ""),sep = "")
  } 
  
}

#遍历差异文件夹，生成query.ko和query2map.txt文件等等文件
resultdir <- list.dirs(getwd(),full.names = TRUE,recursive = FALSE) 
resultdir <- resultdir[grepl("_vs_|_",resultdir)] 
for (difdir in resultdir){
  print(difdir)
  KEGG_folder <- paste0(difdir, "/KEGG/")
  path_pack <- strsplit(KEGG_folder,"/")
  #print(path_pack)
  folder <- path_pack[[1]][length(path_pack[[1]])-1]
  #print(folder)
  KeggPath <- paste0(path,"/报告及附件/附件2 Result/05. Bioinformatics Analysis/KEGG Analysis/",folder,"/")
  dir.create(KeggPath,recursive = T,showWarnings = F)
  
  if(file.exists(paste0(KEGG_folder,"/diff.txt"))){
    fit0 <- try(read.table(paste0(KEGG_folder,"/diff.txt"),header = FALSE,sep = '\t',fill = TRUE,quote = "",
                          stringsAsFactors = FALSE,check.names = TRUE),silent = TRUE)
    if (!'try-error' %in% class(fit0)) {
      diflist <- read.table(paste0(KEGG_folder,"/diff.txt"),header = FALSE,sep = '\t',fill = TRUE,quote = "",na.strings = "",
                          stringsAsFactors = FALSE,check.names = TRUE)
      #print("kaishi querycpd function!")
      #query2cpd <- querycpd(diflist,all_ko)#部分代谢物未被metabo.txt收录，要用all_ko
	  query2cpd <- diflist
      #print("ending querycpd function!")

      names(query2cpd) <- c("Metabolite", "cpdID") 
      query2cpd_OUT <- query2cpd
      colnames(query2cpd_OUT) <- c("Metabolite","KEGG.ID")
      write.table(query2cpd,paste0(KEGG_folder,"/query.cpd"),row.names = FALSE,col.names = FALSE,sep = "\t",quote=F,na = "")
      query2cpd <- separate_rows(query2cpd,cpdID,sep ="/")
      #print(head(query2cpd))
      #print(head(cpdid2cpd))
      query2cpd <- left_join(query2cpd,cpdid2cpd ,by="cpdID")
      
      #输出query2map.txt文件
      query2map <- q2map(query2cpd,pathway)
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
          kegg_bubble(em,Kegg_em_path)
          row_num <- as.numeric(nrow(filter(em, p.value < 0.05 & Over_Under == "over")))
        }
        #print("page 765")
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
          #print("001")
          addWorksheet(wb, sheetName = "Enrichment")
          addStyle(wb, sheet = "Enrichment", rows = 1, cols = 1:ncol(em), style = header_style)
          setColWidths(wb, sheet = "Enrichment", cols = 1, widths = 39)
          setColWidths(wb, sheet = "Enrichment", cols = 2, widths = 42)
          setColWidths(wb, sheet = "Enrichment", cols = 4, widths = 55)
          setColWidths(wb, sheet = "Enrichment", cols = 13, widths = 10)
          if (row_num > 0){
            #print("002")
            addStyle(wb, sheet = "Enrichment", style = createStyle(fgFill="yellow"), rows = 2:(row_num + 1), cols = c(4,11,12), gridExpand = T)
          }
          writeData(wb, sheet = "Enrichment", x = em)
        }
        saveWorkbook(wb, paste0(KeggPath,"Kegg.xlsx"), overwrite = T)
        #========================== map INPUT FILE ==========================#
        #names(allko) <- c("metaid", "keggid")  #非空的metabo.txt
        DEP_file <- read.table(file = paste0(KEGG_folder,"/info-Q650.txt"), sep = "\t", header = T, stringsAsFactors = F, quote = "",fileEncoding = "UTF-8",check.names=F)
        #DEP_file<-DEP_file[,c("Name","KEGG","pvalue")]
        pathway_folder <- paste0(KeggPath, "KEGG Map/")
        if (dir.exists(pathway_folder)){
              setwd(pathway_folder)
        }else{
              dir.create(pathway_folder)
              setwd(pathway_folder)
        }
        #========================== pathway map ==========================#
        map2query$URL <- str_replace_all(map2query$URL,"http://","https://") #旧版为"http://"
        files <- list.files(path =KEGG_folder)
        #print(files)
        if ("META_up.txt" %in% files & "META_down.txt" %in% files){
           DEP_file<-DEP_file[,c("Name","KEGG","FC","pvalue")]
           names(DEP_file) <- c("metaid","keggid","fc","pvalue")
           fit_up<- try(uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = ""),silent = TRUE)
           if('try-error' %in% class(fit_up)){
               upko <- query2cpd_OUT[1,][-1,]
                     if(file.info(paste0(KEGG_folder,"/META_up.txt"))$size != 0){
                uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else{
                uplist <- data.frame(V1 = 1)
              }
               
           }else{
               ########################################
              if(file.info(paste0(KEGG_folder,"/META_up.txt"))$size != 0){
                uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else{
                uplist <- data.frame(V1 = 1)
              }
			  upko <- uplist
              names(upko) <- c("metaid", "keggid")
              #upko <- filter(allko, metaid %in% uplist$V1)
              #upko <-  separate_rows(upko,keggid,sep ="/")
              ##########################################
              # # upko <- allko[1,][-1,]
              # if(file.info(paste0(KEGG_folder,"/META_up.txt"))$size != 0){
              #     uplist <- read.table(file = paste0(KEGG_folder,"/META_up.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              # }else uplist <- data.frame(V1 = 1)
              # upko <- filter(allko, metaid %in% uplist$V1)
              # upko <-  separate_rows(upko,keggid,sep ="/")
           }
          fit_down<- try(downlist <- read.table(file = paste0(KEGG_folder,"/META_down.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = ""),silent = TRUE) 
          if('try-error' %in% class(fit_down)){
              downko <- query2cpd_OUT[1,][-1,]
                if(file.info(paste0(KEGG_folder,"/META_down.txt"))$size != 0){
                  downlist <- read.table(file = paste0(KEGG_folder,"/META_down.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else downlist <- data.frame(V1 = 1)

          }else{
              if(file.info(paste0(KEGG_folder,"/META_down.txt"))$size != 0){
                  downlist <- read.table(file = paste0(KEGG_folder,"/META_down.txt"), sep = "\t", header = F, stringsAsFactors = F,quote = "")
              }else downlist <- data.frame(V1 = 1)
              #downko <- filter(allko, metaid %in% downlist$V1)
              #downko <-  separate_rows(downko,keggid,sep ="/")
          }
          DEP_info <- mutate(DEP_file, information = paste0(metaid, " ( FC=", round(fc,3), "; P.value=", signif(pvalue, 3), " )"))#[,c(1,5)]
          DEP_info <- DEP_info[,c("metaid", "information")]
          #get_map2query(map2query = map2query,DEP_info,upko,downko,"Up regulated metabolite")
          #tryCatch({ get_map2query(map2query = map2query,DEP_info,upko,downko,"Up regulated metabolite")},error=function(e){print(e)})
          Kegg_DA_path <- paste0(KeggPath,"Differential Abundance Score/")
          dir.create(Kegg_DA_path,recursive = T,showWarnings = F)
          DAScore_plot(map2query=map2query,Enrichment=em,up_list = uplist,down_list =downlist,pa = Kegg_DA_path) 
        }else{        #oneway
          print(" not exits")
          #names(DEP_file) <- c("metaid", "pvalue")
          DEP_file<-DEP_file[,c("Name","KEGG","p-value")]
          names(DEP_file) <- c("metaid", "keggid", "pvalue")
          print("---------------mutate begin")
          print(head(DEP_file))
          DEP_info <- mutate(DEP_file, information = paste0(metaid, "(P.value=", signif(as.numeric(pvalue), 3), " )"))
          print("--------------mutate over !")
          DEP_info <- DEP_info[,c("metaid", "information")]
		  uplist <- DEP_file[, grep("metaid", colnames(DEP_file)):grep("keggid", colnames(DEP_file))]
          upko <- as.data.frame(uplist)
          names(upko) <- c("metaid", "keggid")
		  downko <- as.data.frame(query2cpd_OUT[1,][-1,])
          #uplist <- DEP_file$metaid
          #upko <- filter(allko, metaid %in% uplist)
          #upko <-  separate_rows(upko,keggid,sep ="/")
          #downko <- allko[1,][-1,]
          #get_map2query(map2query = map2query,DEP_info,upko,downko,"Target metabolite")
        }
        #print("page 857!!!!!!")
        mapnmb <- nrow(map2query)
        pngfile <- dir(pattern = "png")
        if("TRUE" %in% grepl("\\~\\$",pngfile)){
          pngfile <- pngfile[-grep("^\\~\\$",pngfile)]
        }
        htmlfile <- dir(pattern = "html")
        if("TRUE" %in% grepl("\\~\\$",htmlfile)){
          htmlfile <- htmlfile[-grep("^\\~\\$",htmlfile)]
        }
        if (mapnmb != length(pngfile)){
          cat("kegg map不全", file= paste(path,"/missing-map.txt",sep = ""),sep = "") 
        }
        if (mapnmb != length(htmlfile)){
          cat("kegg html不全", file= paste(path,"/missing-html.txt",sep = ""),sep = "")  
        }
        setwd(KEGG_folder)
        Metabolites <- read.table("../Metabolites.txt",header = T,sep = "\t",quote="",stringsAsFactors = F,check.names = F)
        #Kegg_ph_path <- paste0(KeggPath,"KEGG Pathway_Metabolites Heatmap/")
        #dir.create(Kegg_ph_path,recursive = T,showWarnings = F) 
        #heatmap_per_Path(query2map= query2map,Metabolites =Metabolites,Enrichment =em,pa =Kegg_ph_path)
        setwd(path)
      }else{
        cat("", file= paste0(KeggPath,"差异物无map信息,无kegg分析.txt"), append = TRUE)
      }
    }else{
      cat("", file= paste0(KeggPath,"无差异物，无kegg分析.txt"), append = TRUE)
    }
  }
}
print("kegg over!!!!!!!!!!!!!")
