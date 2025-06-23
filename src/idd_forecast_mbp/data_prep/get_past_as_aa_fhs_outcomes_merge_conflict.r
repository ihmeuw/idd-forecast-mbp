rm(list = ls())
#

require(sf)
require(glue)
require(arrow)
require(data.table)
require(ncdf4)

source("/ihme/cc_resources/libraries/current/r/get_outputs.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_age_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_population.R")

outdir <- "/mnt/team/idd/pub/forecast-mbp/02-processed_data/age_specific_fhs"
# make directory if it doesn't exist
if (!dir.exists(outdir)) {
  dir.create(outdir, recursive = TRUE)
}


# Age distribution globally and for dengue for the 4 countries
age_metadata <- as.data.frame(get_age_metadata(age_group_set_id = 24, release_id=9))
write_parquet(age_metadata, glue("{outdir}/age_metadata.parquet"))
age_metadata <- age_metadata[,c("age_group_id", "age_group_name")]

# FHS Hierarchy
fhs_hierarchy <- as.data.frame(get_location_metadata(location_set_id = 39, release_id=9))
fhs_hierarchy <- fhs_hierarchy[,c("location_id", "parent_id", "path_to_top_parent", "level", "most_detailed", "location_name", "super_region_id", "super_region_name",
                                  "region_id", "region_name")]
write_parquet(fhs_hierarchy, glue("{outdir}/fhs_hierarchy.parquet"))

# For GBD 2023 pulls
release_id <- 16
compare_v <- 8234 

causes <- c("malaria", "dengue")
cause_ids <- c(345, 357)
measures <- c("death", "incidence")
measure_ids <- c(1, 6)
metric_ids <- c(1,3)
metrics <- c("count", "rate")

years = 2000:2023

for (cause_num in seq_along(causes)){
  #
  message(glue("Processing cause: {causes[cause_num]}"))
  #
  cause <- causes[cause_num]
  cause_id <- cause_ids[cause_num]
  #
  for (measure_num in seq_along(measures)){
    #
    message(glue("Processing measure: {measures[measure_num]}"))
    #
    measure <- measures[measure_num]
    measure_id <- measure_ids[measure_num]
    #
    for (metric_num in seq_along(metrics)){
      #
      message(glue("Processing metric: {metrics[metric_num]}"))
      #
      message(glue("Processing age-specific data for cause: {cause}, measure: {measure}, metric: {metrics[metric_num]}"))
      #
      metric <- metrics[metric_num]
      metric_id <- metric_ids[metric_num]
      #
      sex_ids <- 1:2
      df <- as.data.frame(get_outputs("cause", 
                                      year_id = years, location_id = fhs_hierarchy$location_id,
                                      age_group_id = age_metadata$age_group_id,
                                      compare_version_id = compare_v,
                                      cause_id = cause_id,
                                      measure_id = measure_id,
                                      release_id = release_id,
                                      metric_id = metric_id,
                                      sex_id = sex_ids))
      df <- df[,c("age_group_id", "cause_id", "location_id", "measure_id", "metric_id", "sex_id", "year_id", "val", "upper", "lower")]
      df_with_hierarchy <- merge(df, fhs_hierarchy, by = c("location_id"), all.x = TRUE)
      df_full <- merge(df_with_hierarchy, age_metadata, by = "age_group_id", all.x = TRUE)
      
      metadata <- list(
        cause = cause, cause_id = cause_id,
        measure = measure, measure_id = measure_id,
        metric = metric, metric_id = metric_id,
        sex_ids = sex_ids,
        version_id = compare_v,
        release_id = release_id,
        hierarchy = "fhs",
        extraction_date = as.character(Sys.Date()),
        min_year = min(years),
        max_year = max(years)
      )
      
      df_name <- glue("as_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_fhs.parquet")
      write_parquet(df_full, glue("{outdir}/{df_name}"))
      #
      message(glue("Processing all-age data for cause: {cause}, measure: {measure}, metric: {metrics[metric_num]}"))
      #
      df <- as.data.frame(get_outputs("cause", 
                                      year_id = years, location_id = fhs_hierarchy$location_id,
                                      age_group_id = 22,
                                      compare_version_id = compare_v,
                                      cause_id = cause_id,
                                      measure_id = measure_id,
                                      release_id = release_id,
                                      metric_id = metric_id,
                                      sex_id = 3))
      df <- df[,c("age_group_id", "cause_id", "location_id", "measure_id", "metric_id", "sex_id", "year_id", "val", "upper", "lower")]
      df_with_hierarchy <- merge(df, fhs_hierarchy, by = c("location_id"), all.x = TRUE)
      df_with_hierarchy$age_group_name <- "All Age"
      df_full <- df_with_hierarchy

      df_name <- glue("aa_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_fhs.parquet")
      write_parquet(df_full, glue("{outdir}/{df_name}"))
    }
  }
}



# For GBD 2023 pulls
release_id <- 16
compare_v <- 8234 

as.data.frame(get_outputs("cause", 
                          year_id = 2011, location_id = 198,
                          age_group_id = 22,
                          compare_version_id = compare_v,
                          cause_id = 345,
                          measure_id = 1,
                          release_id = release_id,
                          metric_id = 1,
                          sex_id = 3))


as.data.frame(get_outputs("cause", 
                          year_id = 2011, location_id = 198,
                          age_group_id = 22,
                          cause_id = 345,
                          measure_id = 1,
                          release_id = 9,
                          metric_id = 1,
                          sex_id = 3))
