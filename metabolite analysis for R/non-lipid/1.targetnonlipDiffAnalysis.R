args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript G3_Non_target_auto_v3.R <path> <pairornot> <project>")
}

path <- as.character(args[1])
pairornot <- as.numeric(args[2])
project <- tolower(as.character(args[3]))
options(stringsAsFactors = FALSE)

analysis_tag <- "Q650"
pvalue_cutoff <- 0.05
fold_change_cutoff <- 1.5

target_pro <- c("s", "l", "c")
unit_map <- c("ng/g", "ng/mL", "pg")
UNION <- unit_map[grep(project, target_pro)]

if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages(
    "pacman",
    dependencies = TRUE,
    repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"
  )
}

if (!requireNamespace("pacman", quietly = TRUE)) {
  stop("pacman installation failed")
}

pacman::p_load(openxlsx, stringr, dplyr, reshape2, ggplot2)
setwd(path)

sty <- openxlsx::createStyle(fontName = "Arial", fontSize = 10)
stytitle <- openxlsx::createStyle(
  fontSize = 10,
  fontName = "Arial",
  textDecoration = "bold",
  halign = "left",
  fgFill = "#CCCCCC"
)
sty_color1 <- openxlsx::createStyle(fgFill = "yellow")

get_valid_values <- function(x) {
  na.omit(x)
}

get_group_tabs <- function(group) {
  unlist(strsplit(as.character(group), "_vs_|\\|", perl = TRUE))
}

get_group_folder <- function(group) {
  gsub("\\|", "_", group)
}

filter_diff_data <- function(dat) {
  is_diff <- !is.na(dat$Name) & dat$pvalue < pvalue_cutoff

  if ("Foldchange" %in% colnames(dat)) {
    is_diff <- is_diff &
      (dat$Foldchange > fold_change_cutoff | dat$Foldchange < 1 / fold_change_cutoff)
  }

  dat[is_diff, ]
}

data.Extract <- function(dat, Group) {
  colnames(dat)[1] <- "Name"

  tab <- get_group_tabs(Group)
  folder <- get_group_folder(Group)
  data_sig <- filter_diff_data(dat)

  outpath_clu <- paste(folder, paste("Cluster", analysis_tag, sep = "-"), sep = "/")
  if (!dir.exists(outpath_clu)) {
    dir.create(outpath_clu)
  }

  data_clu <- data.frame(Name = data_sig$Name)
  for (j in seq_along(tab)) {
    group_samples <- inputname[which(inputname$group == tab[j]), 1]
    data_clu <- data.frame(
      data_clu,
      data_sig[, colnames(data_sig) %in% group_samples],
      check.names = FALSE
    )
  }
  data_clu <- data_clu[!duplicated(data_clu$Name), ]
  write.table(
    data_clu,
    file = paste(outpath_clu, "1.txt", sep = "/"),
    row.names = FALSE,
    quote = FALSE,
    sep = "\t"
  )

  outpath_keg <- paste0(folder, "/KEGG")
  if (!dir.exists(outpath_keg)) {
    dir.create(outpath_keg)
  }

  if (length(tab) == 2) {
    data_kegg <- dplyr::select(data_sig, Name, MEAN, KEGG, FC = Foldchange, pvalue)
    meta_up <- data_kegg[data_kegg$FC > 1, c("Name", "KEGG")]
    meta_down <- data_kegg[data_kegg$FC < 1, c("Name", "KEGG")]

    write.table(meta_up, paste(outpath_keg, "META_up.txt", sep = "/"),
      col.names = FALSE, row.names = FALSE, sep = "\t", quote = FALSE
    )
    write.table(meta_down, paste(outpath_keg, "META_down.txt", sep = "/"),
      col.names = FALSE, row.names = FALSE, sep = "\t", quote = FALSE
    )
  } else {
    data_kegg <- dplyr::select(data_sig, Name, MEAN, KEGG, `p-value` = pvalue)
  }

  meta_diff <- data_kegg[, c("Name", "KEGG")]
  write.table(meta_diff, paste(outpath_keg, "diff.txt", sep = "/"),
    col.names = FALSE, row.names = FALSE, sep = "\t", quote = FALSE
  )
  write.table(
    data_kegg,
    file = paste(outpath_keg, paste0("info-", analysis_tag, ".txt"), sep = "/"),
    row.names = FALSE,
    quote = FALSE,
    sep = "\t"
  )

  info_file <- paste0(outpath_keg, "/info-", analysis_tag, ".txt")
  if (file.exists(info_file)) {
    input2 <- read.table(
      info_file,
      sep = "\t",
      header = TRUE,
      encoding = "UTF-8",
      check.names = FALSE,
      stringsAsFactors = FALSE,
      quote = ""
    )

    data1 <- input2 %>% dplyr::group_by(Name) %>% dplyr::arrange(dplyr::desc(MEAN))
    data_temp <- data1
    data_temp$Name <- tolower(data_temp$Name)
    index <- duplicated(data_temp$Name)
    data1$Name[index] <- NA
    data1 <- data1[order(data1$Name), -2]
    data_out <- data1[!is.na(data1$Name), ]

    write.table(data_out, paste(outpath_keg, "info-diff.txt", sep = "/"),
      col.names = TRUE, row.names = FALSE, quote = FALSE, sep = "\t"
    )
  }

  if (length(tab) == 2) {
    outpath_vol <- paste0(folder, "/Volcano-", analysis_tag)
    if (!dir.exists(outpath_vol)) {
      dir.create(outpath_vol)
    }

    data_vol <- dplyr::select(dat, FC = Foldchange, Pvalue = pvalue)
    data_vol <- na.omit(data_vol)
    write.csv(data_vol, file = paste(outpath_vol, "diff.csv", sep = "/"), row.names = FALSE)
    write.table(
      paste(tab, collapse = "_vs_"),
      file = paste(outpath_vol, "groupname.txt", sep = "/"),
      row.names = FALSE,
      col.names = FALSE,
      quote = FALSE,
      sep = "\t"
    )
  }
}

groupvs <- read.table("groupvs.txt", header = FALSE, sep = "\t", quote = "")
groupvs <- as.vector(groupvs[, 1])

if (all(nchar(groupvs) < 31)) {
  sheetname <- groupvs
} else {
  sheetname <- as.vector(read.table("tempname.txt", header = FALSE, sep = "\t", quote = "")[, 1])
}

if (file.exists("newname.xlsx")) {
  newname <- openxlsx::read.xlsx("newname.xlsx", rowNames = FALSE, check.names = FALSE)
  Name <- reshape2::melt(newname, id = colnames(newname)[1])
  inputname <- dplyr::select(Name, colnames(Name)[1], value) %>%
    dplyr::filter(value != "")
  names(inputname) <- c("name", "group")
  inputname <- rbind(
    dplyr::filter(inputname, group != "QC"),
    dplyr::filter(inputname, group == "QC")
  )
} else {
  DATA <- read.csv("pos-dele-iso.csv", sep = ",", check.names = FALSE)
  samplename <- colnames(DATA)[grep("-\\d+$", colnames(DATA))]
  inputname <- data.frame(name = samplename, group = stringr::str_remove(samplename, "-\\d+$"))
}

dir.create("报告及附件/", showWarnings = FALSE)
QCPath <- paste0(path, "/报告及附件/附件2 Result/01. QC/")
dir.create(QCPath, recursive = TRUE, showWarnings = FALSE)

plot_qcrsd_curve <- function(input_file, output_path) {
  data <- read.csv(input_file, header = TRUE, sep = "\t", check.names = FALSE)
  colnames(data)[1] <- "name"
  data <- subset(data, data$name != "")

  qc_curve <- dplyr::select(data, name, dplyr::contains("RSD", ignore.case = TRUE))
  colnames(qc_curve)[2] <- "QCRSD"
  qc_curve$QCRSD <- qc_curve$QCRSD * 100

  pdata <- table(qc_curve$QCRSD)
  rank_data <- data.frame(
    QCRSD = as.numeric(names(pdata)),
    num = as.numeric(pdata)
  )
  rank_data <- rank_data[order(rank_data$QCRSD, decreasing = FALSE), ]

  curve_data <- data.frame(
    RSD = rank_data$QCRSD,
    PER = cumsum(rank_data$num) / sum(rank_data$num) * 100
  )

  pc <- ggplot2::ggplot(curve_data) +
    ggplot2::geom_line(ggplot2::aes(x = RSD, y = PER), colour = "orange", size = 0.5) +
    ggplot2::geom_smooth(
      ggplot2::aes(x = RSD, y = PER),
      colour = "orange",
      size = 0.5,
      method = "lm",
      formula = y ~ I(poly(x, 20))
    ) +
    ggplot2::scale_x_continuous(breaks = c(0, 30, 50, 100, 200)) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(size = 0.4, colour = "black"),
      axis.text.x = ggplot2::element_text(colour = "black", size = 10),
      axis.text.y = ggplot2::element_text(colour = "black", size = 10),
      axis.title.x = ggplot2::element_text(size = 12),
      axis.title.y = ggplot2::element_text(size = 12)
    ) +
    ggplot2::geom_vline(xintercept = 30, lty = 2, colour = "grey") +
    ggplot2::ylab("% of peaks") +
    ggplot2::xlab("RSD (%)")

  ggplot2::ggsave(
    file = paste0(output_path, "QCRSD_curve", ".png"),
    pc,
    width = 10,
    height = 10,
    units = "cm",
    dpi = 300
  )
}

plot_qcrsd_curve("file11.xls", QCPath)

Non_tar_step1 <- function() {
  MVDApath <- paste0(path, "/统计分析/")

  if (!file.exists("significanceA.txt") && !file.exists("significanceB.txt")) {
    file.copy(paste0(MVDApath, "QC.png"), QCPath)
    file.copy(paste0(MVDApath, "MCC.png"), QCPath)
    file.copy(paste0(MVDApath, "MultiScatter.png"), QCPath)
    file.copy(paste0(MVDApath, "Hotelling-s T2Range Line Plot.png"), QCPath)

    MVDApath2 <- paste0(path, "/报告及附件/附件2 Result/03. Multivariate Statistical Analysis/")
    dir.create(MVDApath2, recursive = TRUE, showWarnings = FALSE)

    file.copy(paste0(MVDApath, list.files(MVDApath, include.dirs = TRUE)), MVDApath2)
    file.remove(file.path(MVDApath2, c("pic_OPLSDA.xlsx", "pic_PCA.xlsx", "pic_PLSDA.xlsx")))
    file.copy(paste0(MVDApath, list.files(MVDApath, include.dirs = TRUE)), MVDApath2)
    file.remove(file.path(MVDApath2, c("预警信息.txt", "Hotelling-s T2Range Line Plot.png", "MCC.png")))
  }

  data <- openxlsx::read.xlsx(
    xlsxFile = "normeddata_for_simca.xlsx",
    sheet = 1,
    colNames = TRUE,
    check.names = FALSE
  )
  wb2 <- openxlsx::createWorkbook()

  for (i in seq_along(groupvs)) {
    message("Processing ", i, ". ", groupvs[i])

    folder <- get_group_folder(groupvs[i])
    dir.create(folder, showWarnings = FALSE)
    tab <- get_group_tabs(groupvs[i])
    file2 <- paste0(MVDApath, "VIP.xlsx")

    if (!file.exists("significanceA.txt") && !file.exists("significanceB.txt")) {
      MVDApath_group <- paste0(MVDApath, folder, "/")
      MVDApath2_group <- paste0(MVDApath2, folder, "/")
      dir.create(MVDApath2_group, recursive = TRUE, showWarnings = FALSE)
      file.copy(
        paste(MVDApath_group, list.files(MVDApath_group, include.dirs = TRUE), sep = ""),
        MVDApath2_group
      )
    }

    data_tab <- seq_len(nrow(data))
    Replicated <- c()
    for (j in seq_along(tab)) {
      group_samples <- inputname[which(inputname$group == tab[j]), 1]
      Replicated <- c(Replicated, sum(colnames(data) %in% group_samples))
      data_tab <- data.frame(
        data_tab,
        data[, colnames(data) %in% group_samples],
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
    }

    data_tab <- subset(data_tab, select = -data_tab)
    MEAN <- rowMeans(data_tab, na.rm = TRUE)

    outfiles <- list.files(path = folder, pattern = "xlsx", full.names = TRUE)
    if (any(grepl("\\~\\$", basename(outfiles)))) {
      outfiles <- outfiles[!grepl("\\~\\$", basename(outfiles))]
    }

    if (length(tab) == 2) {
      outpfile <- outfiles[grep(paste0(analysis_tag, "/significance"), outfiles)]
    } else {
      outpfile <- outfiles[grep(paste0(analysis_tag, "/twoway"), outfiles)]
    }

    OUTP <- FALSE
    if (length(outpfile) != 0) {
      outp_data <- openxlsx::read.xlsx(outpfile, check.names = FALSE)
      index <- match(data$ID, outp_data$ID)
      outp_data <- outp_data[index, ]
      out_pvalue <- outp_data[, ncol(outp_data)]
      OUTP <- TRUE
    }

    pvalue <- c()
    if (length(tab) == 2) {
      Foldchange <- c()
      for (m in seq_len(nrow(data))) {
        group1_samples <- inputname[which(inputname$group == tab[1]), 1]
        group2_samples <- inputname[which(inputname$group == tab[2]), 1]
        A11 <- as.numeric(data[m, colnames(data) %in% group1_samples]) %>% round(3)
        A22 <- as.numeric(data[m, colnames(data) %in% group2_samples]) %>% round(3)
        A1 <- as.numeric(data[m, colnames(data) %in% group1_samples])
        A2 <- as.numeric(data[m, colnames(data) %in% group2_samples])

        if (sum(is.na(A1)) < ceiling(Replicated[1] / 2) &&
          sum(is.na(A2)) < ceiling(Replicated[2] / 2)) {
          Foldchange[m] <- mean(A1, na.rm = TRUE) / mean(A2, na.rm = TRUE)

          if (sd(A11, na.rm = TRUE) != 0 && sd(A22, na.rm = TRUE) != 0) {
            if (OUTP) {
              pvalue[m] <- out_pvalue[m]
            } else if (pairornot == 0) {
              pvalue[m] <- wilcox.test(A1, A2)$p.value
            } else {
              pvalue[m] <- t.test(A1, A2, alternative = "two.side", var.equal = TRUE, paired = TRUE)$p.value
            }
          } else {
            pvalue[m] <- 1
          }
        } else {
          pvalue[m] <- 1
          Foldchange[m] <- NA
        }
      }

      sigsig <- data.frame(Foldchange, pvalue, MEAN)
      rawdata1 <- cbind(data, sigsig)
      data1 <- openxlsx::read.xlsx(xlsxFile = file2, sheet = sheetname[i], colNames = TRUE, check.names = FALSE)
      dataVIP <- data.frame(Name = data1[, 1], VIP = data1$VIP_oplsda)

      colnames(rawdata1)[1] <- "Name"
      rawdata1 <- merge(rawdata1, dataVIP, by = "Name")
      difflist <- rep("-", nrow(rawdata1))
      difflist[
        which(rawdata1$pvalue < pvalue_cutoff &
          (rawdata1$Foldchange > fold_change_cutoff | rawdata1$Foldchange < 1 / fold_change_cutoff))
      ] <- "diff"
      rawdata1$diff <- difflist
    } else {
      post_hoc <- c()
      log_data_tab <- data_tab

      for (m in seq_len(nrow(log_data_tab))) {
        x <- c()
        A <- c()
        num <- 0
        result <- ""

        for (j in seq_along(tab)) {
          sample_data <- log_data_tab[, colnames(log_data_tab) %in% inputname[which(inputname$group == tab[j]), 1]][m, ]
          gp <- get_valid_values(as.numeric(sample_data))

          if (length(gp) >= 2) {
            num <- num + 1
            x <- c(x, gp)
            A <- c(A, rep(tab[j], length(gp)))
          }
        }

        if (num >= 2) {
          A <- factor(A)
          lamp <- data.frame(x, A)
          lamp.acv <- aov(x ~ A, data = lamp)
          a <- summary(lamp.acv)
          pvalue[m] <- a[[1]]$`Pr(>F)`[1]
          post <- TukeyHSD(lamp.acv)

          if (OUTP) {
            pvalue[m] <- out_pvalue[m]
          }

          for (n in seq_len(nrow(post$A))) {
            if (!is.na(post$A[n, 4]) && length(post$A[n, 4]) != 0 && post$A[n, 4] < pvalue_cutoff) {
              result <- paste(result, row.names(post$A)[n], sep = ";")
            }
          }
        } else {
          pvalue[m] <- 1
        }

        if (result == "" || pvalue[m] > pvalue_cutoff) {
          post_hoc[m] <- "/"
        } else {
          post_hoc[m] <- substring(result, 2)
        }
      }

      sigsig <- data.frame(pvalue, MEAN, post_hoc)
      rawdata1 <- cbind(data, sigsig)
      colnames(rawdata1)[1] <- "Name"
      rawdata1$diff <- as.factor(ifelse(!is.na(rawdata1$Name) & rawdata1$pvalue < pvalue_cutoff, "diff", NA))
    }

    colnames(rawdata1)[1] <- "Name"
    rawdata1 <- rawdata1[order(rawdata1$Name), ]

    openxlsx::addWorksheet(wb2, sheet = sheetname[i], gridLines = TRUE)
    openxlsx::addStyle(
      wb2,
      sheet = sheetname[i],
      sty,
      cols = 1:(ncol(rawdata1) + 1),
      rows = 1:(nrow(rawdata1) + 1),
      gridExpand = TRUE
    )
    openxlsx::writeData(wb2, sheet = sheetname[i], rawdata1, colNames = TRUE, rowNames = FALSE)
    openxlsx::setColWidths(wb2, sheet = sheetname[i], cols = 1:ncol(rawdata1), widths = "auto")
  }

  openxlsx::saveWorkbook(wb2, paste(analysis_tag, ".xlsx", sep = ""), overwrite = TRUE)
}

Non_tar_step2 <- function(condition = FALSE) {
  if (!condition) {
    return(invisible(NULL))
  }

  wb4 <- openxlsx::createWorkbook()

  dataoriginal <- read.csv("fileori.xls", header = TRUE, sep = "\t", check.names = FALSE)
  xoriginal <- dataoriginal[, colnames(dataoriginal) %in% inputname$name]
  dataoriginal <- dataoriginal[apply(xoriginal, 1, function(y) any(!is.na(y))), ]
  dataoriginal$unit <- rep(UNION, nrow(dataoriginal))
  openxlsx::addWorksheet(wb4, sheet = "all-ori", gridLines = TRUE)
  openxlsx::writeData(wb4, sheet = "all-ori", dataoriginal, colNames = TRUE, rowNames = FALSE)

  datadele <- read.csv("file11.xls", header = TRUE, sep = "\t", check.names = FALSE)
  datadele$unit <- rep(UNION, nrow(datadele))
  openxlsx::addWorksheet(wb4, sheet = "all", gridLines = TRUE)
  openxlsx::writeData(wb4, sheet = "all", datadele, colNames = TRUE, rowNames = FALSE)

  dataraw <- openxlsx::read.xlsx(xlsxFile = "normeddata_for_simca.xlsx", sheet = 1, colNames = TRUE, check.names = FALSE)
  openxlsx::addWorksheet(wb4, sheet = "norm", gridLines = TRUE)
  openxlsx::writeData(wb4, sheet = "norm", dataraw, colNames = TRUE, rowNames = FALSE)

  for (i in seq_along(groupvs)) {
    message("Exporting ", i, ". ", groupvs[i])

    folder <- get_group_folder(groupvs[i])
    data_neg <- openxlsx::read.xlsx(xlsxFile = paste0(analysis_tag, ".xlsx"), sheet = i, colNames = TRUE, check.names = FALSE)
    colnames(data_neg)[1] <- "Name"

    data.Extract(dat = data_neg, Group = groupvs[i])

    tab <- get_group_tabs(groupvs[i])
    sig_data <- data_neg[data_neg[, grep("p-?\\s?value", colnames(data_neg))] < pvalue_cutoff, ]

    check_File <- sig_data %>% dplyr::group_by(Name) %>% dplyr::arrange(dplyr::desc(pvalue))
    check_File2 <- data_neg %>% dplyr::group_by(Name) %>% dplyr::arrange(dplyr::desc(diff))
    len3 <- nrow(check_File[check_File$diff %in% "diff", ])

    index <- duplicated(tolower(check_File$Name))
    check_File$Name[index] <- NA
    check_File <- check_File[order(check_File$Name), ]

    Metabolites <- data.frame(Name = check_File$Name)
    for (j in seq_along(tab)) {
      group_samples <- inputname[which(inputname$group == tab[j]), 1]
      Metabolites <- data.frame(
        Metabolites,
        check_File[, colnames(check_File) %in% group_samples],
        check.names = FALSE
      )
    }
    Metabolites <- Metabolites %>% dplyr::filter(Name != "")

    write.table(Metabolites, paste0(folder, "/Metabolites.txt"), sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(check_File, paste0(folder, "/checkFile.txt"), sep = "\t", quote = FALSE, row.names = FALSE, na = "")
    write.table(data_neg, paste0(folder, "/checkFile_all.txt"), sep = "\t", quote = FALSE, row.names = FALSE, na = "")

    openxlsx::addWorksheet(wb4, sheet = sheetname[i], gridLines = TRUE)
    openxlsx::addStyle(
      wb4,
      sheet = sheetname[i],
      sty,
      cols = 1:(ncol(check_File) + 1),
      rows = 1:(nrow(check_File) + 1),
      gridExpand = TRUE
    )
    openxlsx::addStyle(
      wb4,
      sheet = sheetname[i],
      stytitle,
      cols = 1:ncol(check_File),
      rows = 1,
      gridExpand = TRUE
    )

    if (len3 > 0) {
      openxlsx::addStyle(
        wb4,
        sheet = sheetname[i],
        sty_color1,
        cols = 1,
        rows = 2:(len3 + 1),
        gridExpand = TRUE,
        stack = TRUE
      )
    }

    openxlsx::writeData(wb4, sheet = sheetname[i], check_File2, colNames = TRUE, rowNames = FALSE)
  }

  openxlsx::saveWorkbook(wb4, "附件1_样本定性定量和差异分析列表.xlsx", overwrite = TRUE)
}

Non_tar_step1()
Non_tar_step2(condition = file.exists(paste0(analysis_tag, ".xlsx")))
