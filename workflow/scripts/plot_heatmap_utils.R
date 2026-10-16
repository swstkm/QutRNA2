library(dplyr)
library(ggplot2)


################################################################################
# command line tools
################################################################################

# built arguments for processing score and coverage data frame
get_process_args <- function(opts) {
  sort_by_callback <- NULL
  if (!is.null(opts$options$sort_by)) {
    if (opts$options$sort_by == "coverage") {
      sort_by_callback <- order_by_coverage
    } else if (opts$options$sort_by == "trna") {
      sort_by_callback <- order_by_trna
    } else {
      stop("Unknown sort_by: ", opts$options$sort_by)
    }
  }

  return(
    list(
      "sort_by_callback" = sort_by_callback
    ))
}

get_plot_args <- function(df, opts) {
  fill_in <- function(template) {
    if (is.null(template)) {
      return(NULL)
    }
    template <- gsub("\\{condition1\\}", opts$options$condition1, template)
    template <- gsub("\\{condition2\\}", opts$options$condition2, template)
    template <- gsub("\\{score\\}", opts$options$score_column, template)
    template <- gsub("\\{bam_type\\}", opts$options$bam_type, template)
    gsub("\\{position_column\\}", opts$options$position_column, template)
  }

  title <- fill_in(opts$options$title)
  figure_title <- fill_in(opts$options$figure_title)

  position_xlab <- "position"
  if (opts$options$position_column == "sprinzl") {
    position_xlab <- "Sprinzl pos."
  } else if (opts$options$position_column == "seq_position") {
    position_xlab <- "pos. in tRNA seq."
  } else {
    stop("Unknown position column: ", opts$options$position_column)
  }

  coverage_xlab <- NULL
  if (!is.null(opts$options$estimate_coverage)) {
    coverage_xlab <- "median coverage"
  } else if (opts$options$coverage_info) {
    coverage_xlab <- "reads"
  }

  harmonize_scaling <- opts$options$harmonize_scaling
  if (!is.null(harmonize_scaling)) {
    if (harmonize_scaling == "NONE") {
      harmonize_scaling <- NULL
    } else if (harmonize_scaling == "MAX_SCORE") {
      harmonize_scaling <- max(df$score, na.rm = TRUE)
    } else if (harmonize_scaling == "CONTRASTS") {
      max_scores <- read.table(opts$options$max_scores, header = TRUE, sep = "\t")
      harmonize_scaling <- max_scores |>
        filter(cond1 == opts$options$condition1,
               cond2 == opts$options$condition2,
               bam_type == opts$options$bam_type) |>
        pull(opts$options$score_column) |>
        max()
    } else if (harmonize_scaling == "OVERALL") {
      max_scores <- read.table(opts$options$max_scores, header = TRUE, sep = "\t")
      harmonize_scaling <- max(max_scores[[opts$options$score]])
    } else {
      harmonize_scaling <- as.numeric(harmonize_scaling)
    } 
  }

  # quick fix
  if (is.element("mods", colnames(df))) {
    i <- is.na(df$mods)
    df$mods[i] <- ""
  }
  show_mods <- is.element("mods", colnames(df)) &&
    any(df$mods != "") &&
    !opts$options$hide_mods

  return(list(
    "title" = title,
    "figure_title" = figure_title,
    "position_xlab" = position_xlab,
    "coverage_xlab" = coverage_xlab,
    "harmonize_scaling" = harmonize_scaling,
    "show_mods" = show_mods,
    "show_ref" = FALSE #opts$options$show_ref
  ))
}

################################################################################
# estimate coverage by median from observed base calls per tRNA
################################################################################

estimate_coverage_info <- function(df, condition1, condition2) {
  # a read spanning a position gives either a base call or a deletion. JACUSA2's
  # own reads column is base_calls + insertions + deletions, which counts a read
  # carrying an insertion twice
  spanning <- df["trna"]
  for (col in grep("^base_calls_\\d+_\\d+$", names(df), value = TRUE)) {
    deletions <- sub("^base_calls_", "deletions_", col)
    if (!deletions %in% names(df)) {
      stop("estimate_coverage_info: ", col, " has no matching ", deletions,
           ". Pass -D to JACUSA2 so deletions are reported.")
    }
    spanning[[sub("^base_calls_", "reads_", col)]] <- df[[col]] + df[[deletions]]
  }

  coverage_summary <- spanning |>
    group_by(trna) |>
    summarise_all(median) |>
    ungroup() |>
    tidyr::pivot_longer(starts_with("reads_"),
                        names_to = "condition",
                        values_to = "reads") |>
    mutate(replicate = gsub("^reads_\\d+_", "", condition),
           condition = gsub("^reads_|_\\d+$", "", condition),
           condition_i = condition,
           condition = case_match(
             condition,
             "1" ~ condition1,
             "2" ~ condition2)) |>
    as.data.frame()

  # reorder condition
  df_condition <- coverage_summary |>
    dplyr::select(all_of(c("condition_i", "condition"))) |>
    distinct() |>
    arrange(condition_i)
  coverage_summary$condition <- factor(coverage_summary$condition,
                                       levels = df_condition$condition,
                                       ordered = TRUE)

  return(coverage_summary |> dplyr::select(-all_of("condition_i")))
}


coverage_groups <- function(coverage_summary, whisker = 1.5) {
  expression <- coverage_summary |>
    group_by(trna) |>
    summarise(reads = median(reads), .groups = "drop")

  # the groups are the parts of a boxplot of log coverage: below the box, the
  # box, above the box, and whatever lies past the whisker
  reads <- log10(expression$reads)
  quartiles <- quantile(reads, c(.25, .75))
  edges <- c(quartiles, quartiles[2] + whisker * diff(quartiles))
  expression$bin <- cut(reads, c(-Inf, edges, Inf), labels = FALSE)

  # each bin is named by its coverage range and by its part of the boxplot
  parts <- c("lowest quartile", "middle half", "upper quartile",
             "beyond 1.5 IQR")
  labels <- expression |>
    group_by(bin) |>
    summarise(label = sprintf("%.0f-%.0f reads, %s", min(reads), max(reads),
                              parts[first(bin)]),
              .groups = "drop")

  expression |>
    inner_join(labels, by = join_by(bin)) |>
    transmute(trna, reads,
              coverage_group = factor(label, levels = labels$label))
}


plot_coverage_distribution <- function(expression, base_size = 9,
                                       xlab = "median coverage",
                                       remove_prefix = "") {
  beyond <- expression |>
    filter(coverage_group == levels(coverage_group)[nlevels(coverage_group)]) |>
    mutate(label = sub(remove_prefix, "", trna))

  ggplot(expression, aes(x = reads, y = "")) +
    stat_boxplot(geom = "errorbar", width = .25, colour = "grey45",
                 linewidth = .3) +
    geom_boxplot(outlier.shape = NA, width = .55, colour = "grey45",
                 linewidth = .3) +
    geom_jitter(aes(colour = coverage_group), height = .16, size = rel(.9),
                alpha = .75) +
    geom_text(data = beyond, aes(label = label), size = rel(2.2),
              angle = 90, hjust = 0, nudge_y = .24, colour = "grey25") +
    scale_x_log10(labels = scales::label_number(
      scale_cut = scales::cut_short_scale())) +
    scale_colour_manual(values = c("#004488", "#DDAA33", "#BB5566", "#000000")) +
    expand_limits(y = c(0.6, 1.9)) +
    labs(x = xlab, y = "") +
    theme_classic(base_size = base_size) +
    theme(legend.position = "top",
          legend.title = element_blank(),
          legend.margin = margin(0, 0, 0, 0),
          axis.line.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}

################################################################################
# plots
################################################################################

plot_coverage <- function(coverage_summary, opts, xlab = "reads") {
  p <- coverage_summary |>
    ggplot(aes(x = reads, y = trna_label,
               colour = condition, fill = condition)) +
    scale_colour_manual(values = c("#bf568b", "#bcd357")) +
    scale_fill_manual(values = c("#bf568b", "#bcd357")) +
    scale_x_continuous(breaks = scales::breaks_pretty(3),
                       labels = scales::label_number(
                         scale_cut = scales::cut_short_scale())) +
    ylab("") +
    xlab(xlab) +
    theme_bw(base_size = opts$options$base_size) +
    theme(plot.margin = margin(0, 0, 0, 0),
          legend.position = "top",
          legend.title = element_blank(),
          legend.direction = "vertical",
          axis.title.y = element_blank())

  if (opts$options$debug) {
    p <- p + theme(plot.background = element_rect(fill = "yellow"))
  }

  if (unique(coverage_summary$replicate) |> length() == 1) {
    p <- p + geom_col(position = position_dodge2())
  } else {
    p <- p + geom_point(position = position_dodge2(width = 0.5), size = rel(.75))
  }

  p
}


read_mod_abbrevs <- function(mod_abbrevs_fname) {
  mod_abbrevs <- read.table(mod_abbrevs_fname,
                            header = TRUE,
                            sep = "\t",
                            quote = "",
                            comment.char = "")

  return(mod_abbrevs)
}


plot_mod_abbrevs_table <- function(mod_abbrevs) {
  mod_abbrevs <- mod_abbrevs |>
    dplyr::select(all_of(c("short_name", "abbrev"))) |>
    filter(!is.na(abbrev)) |>
    distinct() |>
    dplyr::select(all_of(c("abbrev", "short_name"))) |>
    dplyr::rename(`RNAMods\ncode` = "abbrev", `Mod.` = "short_name")

  return(gridExtra::tableGrob(mod_abbrevs, rows = NULL))
}


Q_GATE <- 0.05


plot_heatmap <- function(df, base_size = 9, xlab = "position",
                         harmonize_scaling = NULL,
                         title = NULL,
                         show_ref = FALSE, show_mods = FALSE) {

  show_ref <- FALSE
  if (show_ref && show_mods) {
    df <- df |>
      mutate(combined_label = paste0(ref, mod_label))
  }

  df$tile <- ifelse(df$gate, .95, .3)
  keys <- paste(c("FDR >=", "FDR <"), opts$options$gate_q)
  df$gate_label <- factor(keys[df$gate + 1], levels = keys)

  p <- df |>
    ggplot(aes(x = position, y = trna_label, fill = score)) +
    # a warm hairline edge keeps a low-score tile apart from the page
    geom_tile(aes(width = tile, height = tile), colour = "#D8C180",
              linewidth = .05) +
    ylab("") +
    xlab(xlab) +
    theme_bw(base_size = base_size) +
    theme(panel.background = element_rect(fill = "white", color = "white"),
          legend.position = "top",
          panel.border = element_blank(),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
          #axis.title.x = element_text(size = 10),
          #axis.text.y = element_text(size = 10),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          axis.title.y = element_blank(),
          plot.margin = margin(0, .5, 0, 0))

  # geom_tile sizes carry no key, so an unseen layer supplies one
  if (any(!df$gate)) {
    p <- p +
      geom_point(aes(size = gate_label), alpha = 0, shape = 22,
                 fill = "grey30", colour = NA) +
      scale_size_manual(name = NULL, values = c(1.5, 4)) +
      # without an explicit order the two guides swap between blocks
      guides(fill = guide_colourbar(order = 1),
             size = guide_legend(order = 2,
                                 override.aes = list(alpha = 1)))
  }

  scale_f <- scale_fill_gradient
  if (opts$options$keep_negative_score) {
    scale_f <- scale_fill_gradient2
  }
  # ensure same scaling for all plots
  if (!is.null(harmonize_scaling)) {
    p <- p +
      scale_f(low = "yellow", high = "blue",
              na.value = "grey",
              limits = c(0, harmonize_scaling), oob = scales::squish)
  } else {
    p <- p +
      scale_f(low = "yellow", high = "blue",
              na.value = "grey")
  }

  # an overlay drawing its own key would paint over the score key
  # mark flagged positions
  if (any(df$flag_position)) {
    p <- p + geom_point(data = df |>
                        filter(flag_position == TRUE),
                        shape = 4, color = "darkred", show.legend = FALSE)
  }
  # mark score exceeding max_score
  if (any(df$exceed_max_score)) {
    p <- p + geom_point(data = df |>
                        filter(exceed_max_score == TRUE),
                        shape = 4, color = "red", show.legend = FALSE)
  }

  # mark special positions
  if (any(df$mark_position)) {
    p <- p + geom_tile(data = df |>
                         filter(mark_position == TRUE),
                       aes(width = tile, height = tile),
                       color = "red", show.legend = FALSE)
  }

  rel_size <- 1.7
  if (show_ref && show_mods) {
    p <- p + geom_text(aes(label = combined_label), size = rel(rel_size)) +
      coord_equal(ratio = 1 / 2)
  } else if (show_ref) {
    p <- p + geom_text(aes(label = ref), size = rel(rel_size)) +
      coord_equal()
  } else if (show_mods) {
    p <- p + geom_text(aes(label = mod_label), size = rel(rel_size)) +
      coord_equal()
  } else {
    p <- p + coord_equal()
  }

  if (!is.null(title)) {
    title <- gsub("\\{anti_codon\\}",
                  paste0(unique(na.omit(df$anti_codon)), collapse = ", "), title)
    title <- gsub("\\{amino_acid\\}",
                  paste0(unique(na.omit(df$amino_acid)), collapse = ", "), title)
    title <- gsub("\\{group\\}", paste0(unique(df$group), collapse = ", "), title)
    p <- p + ggtitle(title)
  }

  p
}


################################################################################
## infer dimensions


infer_height <- function(df) {
  trna_height <- df$trna |>
    unique() |>
    length()

  height <- trna_height + 15

  return(height / 10)
}


infer_width <- function(df, coverage_summary = NULL, abbrevs = NULL) {
  position_width <- df$position |>
    unique() |>
    length()

  trna_width <- df$trna |>
    unique() |>
    as.character() |>
    nchar() |>
    max()

  width <- position_width + trna_width
  if (!is.null(coverage_summary)) {
    width <- width + trna_width
  }
  if (!is.null(abbrevs)) {
    width <- width +
      nchar(abbrevs$short_name) |> max()
      nchar(abbrevs$abbrev) |> max()
  }

  return(width / 10)
}


infer_ggsave_opts <- function(df, coverage_summary = NULL,
                              mod_abbrevs = NULL,
                              ggsave_opts) {

  if (is.null(ggsave_opts)) {
    ggsave_opts <- list(
      "width" = infer_width(df, coverage_summary, mod_abbrevs),
      "height" = infer_height(df)
    )
  } else {
    if (is.null(ggsave_opts$width)) {
      ggsave_opts$width <- infer_width(df, coverage_summary)
    }
    if (is.null(ggsave_opts$height)) {
      ggsave_opts$height <- infer_height(df)
    }
  }

  return(ggsave_opts)
}

#################################################################################
## group tRNA and output
#################################################################################

add_groups <- function(group_opts, df, coverage_summary = NULL) {
  if (!is.null(group_opts)) {
    group_opts <- strsplit(group_opts, ",")[[1]]
    df <- df |>
      tidyr::unite(group, all_of(group_opts), sep = "-", remove = FALSE)
    if (!is.null(coverage_summary)) {
      coverage_summary <- coverage_summary |>
        tidyr::unite(group, all_of(group_opts), sep = "-", remove = FALSE)
    }
    # uniting gives a character column, which would sort the plots by name
    if (length(group_opts) == 1 && is.factor(df[[group_opts]])) {
      order <- levels(df[[group_opts]])
      df$group <- factor(df$group, levels = order)
      if (!is.null(coverage_summary)) {
        coverage_summary$group <- factor(coverage_summary$group, levels = order)
      }
    }
  } else {
    df$group <- "_"
    if (!is.null(coverage_summary)) {
      coverage_summary$group <- "_"
    }
  }

  return(
    list(
      df = df,
      coverage_summary = coverage_summary))
}

split_groups <- function(df, coverage_summary = NULL) {
  # split based on group
  df <- split(df, df$group)

  # container for split coverage_summary
  if (!is.null(coverage_summary)) {
    coverage_summary <- split(coverage_summary, coverage_summary$group)
  } else {
    coverage_summary <- list()
  }

  return(
    list(
      df = df,
      coverage_summary = coverage_summary))
}

process_group <- function(df, coverage_summary, process_args) {
  df <- remove_missing_positions(df)

  # optional sort df and coverage
  if (!is.null(process_args$sort_by_callback)) {
    sorted <- process_args$sort_by_callback(df, coverage_summary)
    df <- sorted$df
    coverage_summary <- sorted$coverage_summary
  }

  return(
    list(
      df = df,
      coverage_summary = coverage_summary))
}

group_trna <- function(group_opts,
                       df, coverage_summary = NULL,
                       output_dir,
                       opts) {
  plot_args <- get_plot_args(df, opts)
  process_args <- get_process_args(opts)
  
  plots <- data.frame(
    fname = c(),
    type = c(),
    group = c(),
    object = c()
  )
  
  if (!is.null(group_opts)) {
    i <- is.na(df$amino_acid)
    if (any(i)) {
      warning("Missing 'amino_acid'")
    }
    i <- is.na(df$anti_codon)
    if (any(i)) {
      warning("Missing 'anti_codon'")
    }
  }

  l <- add_groups(group_opts, df, coverage_summary)
  groups <- split_groups(l$df,
                         l$coverage_summary)
  # TODO catch NA

  stacked <- list(heatmap = list(), heatmap_cov = list(), heatmap_mods = list())
  counts <- c()
  mod_abbrevs <- NULL

  for (group in names(groups$df)) {
    l <- process_group(groups$df[[group]],
                       groups$coverage_summary[[group]],
                       process_args)

    plots <- dplyr::bind_rows(
      plots,
      tibble::tribble(
        ~name,                ~type, ~group, ~object,
          "df",               "tsv", group,  l$df,
          "coverage_summary", "tsv", group,  l$df
      )
    )
    # TODO remove
    #group_plots <- list(
    #  "df" = l$df,
    #  "coverage_summary" = l$coverage_summary
    #)
        
    counts[group] <- length(unique(l$df$trna))

    ncol <- 1
    heatmap <- plot_heatmap(l$df,
                            base_size = opts$options$base_size,
                            xlab = plot_args[["position_xlab"]],
                            harmonize_scaling = plot_args$harmonize_scaling,
                            title = plot_args$title,
                            show_ref = plot_args$show_ref,
                            show_mods = plot_args$show_mods)
    stacked$heatmap[[group]] <- heatmap

    # add coverage summary
    if (!is.null(l$coverage_summary)) {
      ncol <- ncol + 1
      widths <- unit(c(-1, .10), c("null", "npc"))
      heatmap <- heatmap +
        guides(y.sec = guide_axis_label_trans(identity)) +
        plot_coverage(l$coverage_summary, opts, plot_args[["coverage_xlab"]]) +
        theme(axis.text.y = element_blank())
      
      stacked$heatmap_cov[[group]] <- heatmap +
        plot_layout(ncol = ncol, widths = widths)
    }

    # quick fix
    if (is.element("mod_abbrevs", colnames(l$df))) {
      i <- is.na(l$df$mod_label)
      l$df$mod_label <- ""
    }
    # add mod abbrevs table
    if (is.element("mod_abbrevs", colnames(l$df)) && any(l$df$mod_label != "")) {
      ncol <- ncol + 1
      widths <- unit(c(-1, .10, .10), c("null", "npc", "npc"))
      
      mod_abbrevs <- read_mod_abbrevs(opts$options$abbrevs)
      mods <- paste0(l$df$mods, " ") |>
        unique() |>
        strsplit(" ")

      mod_abbrevs <- mod_abbrevs |>
        dplyr::filter(is.element(short_name, mods))

      heatmap <- heatmap + plot_mod_abbrevs_table(mod_abbrevs)
      stacked$heatmap_mods[[group]] <- heatmap +
        plot_layout(ncol = ncol, widths = widths)
    }
    #TODO remove
    # plots[[group]] <- group_plots
  }

  # the groups are stacked into one figure, each with its own scales
  for (variant in names(stacked)) {
    if (length(stacked[[variant]]) == 0) {
      next
    }
    # each group takes the height its tRNAs need
    p <- patchwork::wrap_plots(stacked[[variant]], ncol = 1,
                               heights = counts[names(stacked[[variant]])])
    # one title above the stack, beside the per-block coverage ranges
    if (!is.null(plot_args$figure_title)) {
      p <- p + patchwork::plot_annotation(title = plot_args$figure_title)
    }

    ggsave_opts <- infer_ggsave_opts(
      df,
      if (variant == "heatmap") NULL else coverage_summary,
      if (variant == "heatmap_mods") mod_abbrevs else NULL,
      ggsave_opts = opts$options$ggsave_opts)
    if (variant != "heatmap") {
      ggsave_opts$width <- ggsave_opts$width + 3
    }
    ggsave_opts$height <- ggsave_opts$height +
      1.5 * (length(stacked[[variant]]) - 1)

    output_fname <- file.path(output_dir, paste0(variant, ".pdf"))
    custom_ggsave(output_fname,
                  p,
                  ggsave_opts = ggsave_opts,
                  opts$options$no_crop)
    plots <- dplyr::bind_rows(
      plots,
      tibble::tribble(
        ~name,                            ~type, ~group, ~object,
        output_fname,                     "pdf", "_",    p,
        gsub("pdf$", "png", output_fname), "png", "_",   p,
        gsub("pdf$", "rds", output_fname), "rds", "_",   p
      )
    )
  }

  return(plots)
}

flag_positions <- function(df, flag_positions) {
  i <- is.element(df$trna_coords, flag_positions)
  if (any(i)) {
    df[i, "score"] <- NA
    df[i, "flag_position"] <- TRUE
  }

  return(df)
}


mark_positions <- function(df, pattern) {
  i <- grepl(pattern, df$trna_coords)
  if (!any(i)) {
    # a target position dropped by min_cov leaves nothing to mark
    warning("Pattern '", pattern, "' marked no position")
    return(df)
  }
  df[i, "mark_position"] <- TRUE

  return(df)
}


remove_3adapter <- function(df, trna_annotation, five_adapter, three_adapter) {
  five_adapter <- ifelse(is.null(five_adapter), 0, five_adapter)
  df_trna_annotation <- trna_annotation |>
    mutate(trna_length = nchar(seq)) |>
    dplyr::select(trna, trna_length)

  df <- left_join(df, df_trna_annotation, by = join_by(trna)) |>
    filter(seq_position <= (trna_length + five_adapter)) |>
    dplyr::select(-trna_length)
  stopifnot(!any(is.na(df$trna_length)))

  return(df)
}


remove_missing_positions <- function(df) {
  tbl <- table(df$position, is.na(df$ref))
  if (is.element("FALSE", colnames(tbl))) {
    tbl[, "FALSE"] == 0
    i <- tbl[, "FALSE"] == 0
    if (any(i)) {
      missing_positions <- rownames(tbl[i, ])
      df <- df |>
        filter(!is.element(position, missing_positions))
    }
  }

  return(df)
}


add_missing_values <- function(df, cols = c("trna_label", "position")) {
  fill <- list("exceed_max_score" = FALSE,
               "mark_position" = FALSE,
               "flag_position" = FALSE,
               # a position the tRNA lacks was never tested. Its tile stays small
               "gate" = FALSE)

  df <- tidyr::complete(df, !!! rlang::syms(cols), fill = fill)

  # fixing trna and strand
  fixes <- df |>
    dplyr::select(trna_label, trna, strand) |>
    filter(!if_any(everything(), is.na)) |>
    distinct()
  df <- df |>
    dplyr::select(-c(trna, strand)) |>
    inner_join(
      fixes,
      by = join_by(trna_label))

  if (any(!is.na(df$amino_acid))) {
    fixes <- df |>
      dplyr::select(trna_label, amino_acid) |>
      filter(!if_any(everything(), is.na)) |>
      distinct()
    if (nrow(fixes) > 0) {
      df <- df |>
        dplyr::select(-amino_acid) |>
        left_join(
          fixes,
          by = join_by(trna_label))
    }
  }
  if (any(!is.na(df$anti_codon))) {
    fixes <- df |>
      dplyr::select(trna_label, anti_codon) |>
      filter(!if_any(everything(), is.na)) |>
      distinct()
    if (nrow(fixes) > 0) {
      df <- df |>
        dplyr::select(-anti_codon) |>
        left_join(
          fixes,
          by = join_by(trna_label))
    }
  }

  # fixing reference
  i <- is.na(df$ref)
  if (any(i)) {
    df[i, "ref"] <- "*"
  }
  if (!is.null(df$mods)) {
    i <- is.na(df$mods)
    if (any(i)) {
      df[i, "mods"] <- ""
    }
  }
  if (!is.null(df$mod_label)) {
    i <- is.na(df$mod_label)
    if (any(i)) {
      df[i, "mod_label"] <- ""
    }
  }
  
  # fixing coords
  df$coords <- trna_coords(df)

  df
}


process_sprinzl_coords <- function(df, sprinzl_fname, hide_varm, show_introns, intron_start = "37") {
  # remove positions with no sprinzl mapping
  df <- df[!is.element(df$sprinzl, c("-", ".")), ] # - -> gap, . unmatched
  sprinzl_coords <- read.table(sprinzl_fname, header = FALSE)$V1
  i <- sprinzl_coords %in% c("-")
  sprinzl_coords <- sprinzl_coords[!i]

  if (hide_varm) {
    # remove from sequenced trnas
    df <- df[!grepl("e[0-9]+", df$sprinzl), ]
    # remove from sprinzl model
    sprinzl_coords <- sprinzl_coords[!grepl("e[0-9]+", sprinzl_coords)]
  }
  if (!show_introns) {
    # remove from sequenced trnas
    df <- df[!grepl("i[0-9]+", df$sprinzl), ]
    # remove from sprinzl model
    sprinzl_coords <- sprinzl_coords[!grepl("i[0-9]+", sprinzl_coords)]
  } else {
    # add intron coordinates to sprinzl coordinates
    i <- grepl("i[0-9]+", df$sprinzl)
    if (any(i)) {
      intron_start_i <- which(sprinzl_coords == intron_start)
      sprinzl_coords <- c(sprinzl_coords[1:intron_start_i],
                          df[i, "sprinzl"] |>
                            unique() |>
                            stringr::str_sort(numeric = TRUE),
                          sprinzl_coords[(intron_start_i + 1):length(sprinzl_coords)])
    }
  }

  df$position <- factor(df$sprinzl, levels = sprinzl_coords, ordered = TRUE)

  return(df)
}


process_seq_coords <- function(df) {
  df$position <- factor(df$position,
                        levels = stringr::str_sort(
                          unique(df$position),
                          numeric = TRUE), ordered = TRUE)

  return(df)
}

save_plot <- function(p, filename, opts) {
  output <- file.path(opts$options$output_dir, filename)
  
  save(p, file = output)
  
  return(output)
}

##
# Order data frame by different keys
##

order_by_helper <- function(df, coverage_summary, trna_labels) {
  df$trna_label <- factor(df$trna_label, levels = trna_labels, ordered = TRUE)
  if (!is.null(coverage_summary)) {
    coverage_summary$trna_label <- factor(coverage_summary$trna_label,
                                          levels = trna_labels, ordered = TRUE)
  }

  return(list("df" = df, coverage_summary = coverage_summary))
}
order_by_coverage <- function(df, coverage_summary) {
  total_reads_info <- coverage_summary |>
    summarise(.by = c("trna_label"), total_reads = sum(reads)) |>
    dplyr::select(all_of(c("trna_label", "total_reads"))) |>
    distinct()
  trna_labels <- total_reads_info$trna_label[order(total_reads_info$total_reads)]

  return(order_by_helper(df, coverage_summary, trna_labels))
}
order_by_trna <- function(df, coverage_summary) {
  trna_labels <- unique(df$trna_label) |>
    rev()

  df$trna <- factor(df$trna_label, levels = trna_labels, ordered = TRUE)
  if (!is.null(coverage_summary)) {
    coverage_summary$trna <- factor(coverage_summary$trna, levels = trna_labels, ordered = TRUE)
  }

  return(order_by_helper(df, coverage_summary, trna_labels))
}
