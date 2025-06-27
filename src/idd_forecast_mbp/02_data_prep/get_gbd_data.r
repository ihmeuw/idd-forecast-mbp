require(glue)
require(arrow)

source("/ihme/cc_resources/libraries/current/r/get_cause_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_draws.R")
source("/ihme/cc_resources/libraries/current/r/get_outputs.R")
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_demographics.R")
source("/ihme/cc_resources/libraries/current/r/get_ids.R")
source("/ihme/cc_resources/libraries/current/r/get_sequela_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_age_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_covariate_estimates.R")
source("/ihme/cc_resources/libraries/current/r/get_population.R")



# Output
output_dir <- "/mnt/team/idd/pub/forecast-mbp/01-raw_data/gbd"
# Make sure the directory exists and create it if it doesn't
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

release_id <- 16
como_v <- 1591
codcorrect_v <- 461
dalynator_v <- 96
burdenator_v <- 360
compare_v <- 8234

# Constants
past_years = 2000:2023

# GBD Constants
release_id_2021 = 9
gbd_2023_release_id = 16

gbd_location_set_id = 35
fhs_location_set_id = 39



como_v_2023 = 1591
codcorrect_v_2023 = 461
dalynator_v_2023 = 96
burdenator_v_2023 = 360
compare_v_2023 = 8234
dengue_id = 357
malaria_id = 345

# Get age-meta-data
gbd_2023_age_metadata <- as.data.frame(get_age_metadata(release_id = gbd_2023_release_id))
gbd_2023_age_metadata <- gbd_2023_age_metadata[, c("age_group_id", "age_group_years_start", "age_group_years_end", "age_group_name")]
new_rows <- data.frame(age_group_id = c(1,22),
                       age_group_years_start = c(0, 0),
                       age_group_years_end = c(5, 125),
                       age_group_name = c("Under 5", "All age"))
gbd_2023_age_metadata <- rbind(gbd_2023_age_metadata, new_rows)
write.csv(gbd_2023_age_metadata, glue("{output_dir}/gbd_2023_age_metadata.csv"), row.names = FALSE)

# Get hierarchy
gbd_2023_modeling_hierarchy <- as.data.frame(get_location_metadata(location_set_id = gbd_location_set_id, release_id=gbd_2023_release_id))
fhs_2023_modeling_hierarchy <- as.data.frame(get_location_metadata(location_set_id = fhs_location_set_id, release_id=gbd_2023_release_id))


fhs_2023_modeling_hierarchy[which(fhs_2023_modeling_hierarchy$location_id == 44858),]

44858

gbd_2023_modeling_hierarchy[which(gbd_2023_modeling_hierarchy$parent_id == "135"),]


col_names_to_delete <- c("start_date", "end_date", "date_inserted", "last_updated", "last_updated_by", "last_updated_action")
toss_col_locs <- which(names(gbd_2023_modeling_hierarchy) %in% col_names_to_delete)

gbd_2023_modeling_hierarchy <- gbd_2023_modeling_hierarchy[,-toss_col_locs]
fhs_2023_modeling_hierarchy <- fhs_2023_modeling_hierarchy[,-toss_col_locs]


gbd_2023_modeling_hierarchy_path = glue("{output_dir}/gbd_2023_modeling_hierarchy.parquet")
fhs_2023_modeling_hierarchy_path = glue("{output_dir}/fhs_2023_modeling_hierarchy.parquet")
arrow::write_parquet(gbd_2023_modeling_hierarchy, gbd_2023_modeling_hierarchy_path)
arrow::write_parquet(fhs_2023_modeling_hierarchy, fhs_2023_modeling_hierarchy_path)


# Get population
gbd_2023_population = as.data.frame(get_population(age_group_id = gbd_2023_age_metadata$age_group_id,
                                                   release_id = gbd_2023_release_id,
                                                   year_id = past_years,
                                                   location_id = gbd_2023_modeling_hierarchy$location_id,
                                                   sex_id = 1:3))

gbd_2023_population <- gbd_2023_population[ ,c("age_group_id", "location_id", "year_id", "sex_id", "population")]

# Write fhs_2023_population as a parquet file
gbd_2023_population_path <- glue("{output_dir}/gbd_2023_population.parquet")
arrow::write_parquet(gbd_2023_population, gbd_2023_population_path)

# Get population
fhs_2023_population = as.data.frame(get_population(age_group_id = gbd_2023_age_metadata$age_group_id,
                                                   release_id = gbd_2023_release_id,
                                                   year_id = past_years,
                                                   location_id = fhs_2023_modeling_hierarchy$location_id,
                                                   sex_id = 1:3))

fhs_2023_population <- fhs_2023_population[ ,c("age_group_id", "location_id", "year_id", "sex_id", "population")]

# Write fhs_2023_population as a parquet file
fhs_2023_population_path <- glue("{output_dir}/fhs_2023_population.parquet")
arrow::write_parquet(fhs_2023_population, fhs_2023_population_path)

####
## Dengue
####

4750       


require(data.table)
require(glue)
require(sf)
source("/ihme/cc_resources/libraries/current/r/get_location_metadata.R")
source("/ihme/cc_resources/libraries/current/r/get_life_table.R")
source("/ihme/cc_resources/libraries/current/r/get_population.R")

"%ni%" <- Negate("%in%")

wd <- "/mnt/share/homes/bcreiner/DENV/TAK"
data_dir <- "/mnt/team/sae/pub/takeda/data"
rra_dir <- "/mnt/team/rapidresponse/pub/malaria-denv"

config_template <- read.csv("/mnt/team/rapidresponse/pub/takeda-dengue/thailand_sample/calibr_inputs.csv")


###
# Grab all life table data for Brazil A0 & A1
###

RELEASE_ID <- 16
dengue_id <- 357

sexes <- 3
release <- 16
Years <- 2000:2024

release_id <- 16
como_v <- 1591
codcorrect_v <- 461
dalynator_v <- 96
burdenator_v <- 360
compare_v <- 8234

modeling_hierarchy_2023 <- get_location_metadata(location_set_id = 35, release_id = RELEASE_ID)

Brazil_A1_modeling_hierarchy <- modeling_hierarchy_2023[which(modeling_hierarchy_2023$parent_id == 135),]
Brazil_A1_location_ids <- Brazil_A1_modeling_hierarchy$location_id

Brazil_A1_population <- get_population(age_group_id=22, location_id=Brazil_A1_location_ids, year_id=2022, sex_id=3, release_id=RELEASE_ID)

df <- as.data.frame(get_life_table(location_id = 4750, 
                                                year_id = 2022, sex_id = 3,
                                                release_id = gbd_2023_release_id, with_hiv = 1, with_shock = 1))

df[which(df$life_table_parameter_id == 3),]
Brazil_lt_with_hiv_with_shock








cause_df <- as.data.frame(get_outputs("cause", cause_id = dengue_id,
                                      measure_id = 6, #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                      year_id = past_years,
                                      location_id = 4750, 
                                      age_group_id = 1, 
                                      release_id = gbd_2023_release_id,
                                      metric_id = 1,  #rate =3, counts =1
                                      sex_id = 3, # males =1, females =2, both =3
                                      compare_version_id = compare_v_2023))








# All age results
cause_df <- as.data.frame(get_outputs("cause", cause_id = dengue_id,
                                                    measure_id = 1:6, #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                                    year_id = past_years,
                                                    location_id = gbd_2023_modeling_hierarchy$location_id, 
                                                    age_group_id = 22, 
                                                    release_id = gbd_2023_release_id,
                                                    metric_id = c(1,3),  #rate =3, counts =1
                                                    sex_id = 3, # males =1, females =2, both =3
                                                    compare_version_id = compare_v_2023))

# Merge location hierarchy
cause_df <- merge(cause_df, gbd_2023_modeling_hierarchy, all.x = TRUE, sort = FALSE)
cause_df <- merge(cause_df, gbd_2023_population, all.x = TRUE, sort = FALSE)

cause_df_path <- glue("{output_dir}/gbd_2023_dengue_aa.parquet")
arrow::write_parquet(cause_df, cause_df_path)


# age-specific
cause_df <- as.data.frame(get_outputs("cause", cause_id = dengue_id,
                                      measure_id = c(1,6), #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                      year_id = past_years,
                                      location_id = gbd_2023_modeling_hierarchy$location_id, 
                                      age_group_id = gbd_2023_age_metadata$age_group_id, 
                                      release_id = gbd_2023_release_id,
                                      metric_id = c(1,3),  #rate =3, counts =1
                                      sex_id = 1:3, # males =1, females =2, both =3
                                      compare_version_id = compare_v_2023))

# Merge location hierarchy
cause_df <- merge(cause_df, gbd_2023_modeling_hierarchy, all.x = TRUE, sort = FALSE)
cause_df <- merge(cause_df, gbd_2023_population, all.x = TRUE, sort = FALSE)

cause_df_path <- glue("{output_dir}/gbd_2023_dengue_as.parquet")
arrow::write_parquet(cause_df, cause_df_path)

####
## Malaria
####

# All age results
cause_df <- as.data.frame(get_outputs("cause", cause_id = malaria_id,
                                      measure_id = 1:6, #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                      year_id = past_years,
                                      location_id = gbd_2023_modeling_hierarchy$location_id, 
                                      age_group_id = 22, 
                                      release_id = gbd_2023_release_id,
                                      metric_id = c(1,3),  #rate =3, counts =1
                                      sex_id = 3, # males =1, females =2, both =3
                                      compare_version_id = compare_v_2023))

# Merge location hierarchy
cause_df <- merge(cause_df, gbd_2023_modeling_hierarchy, all.x = TRUE, sort = FALSE)
cause_df <- merge(cause_df, gbd_2023_population, all.x = TRUE, sort = FALSE)
cause_df_path <- glue("{output_dir}/gbd_2023_malaria_aa.parquet")
arrow::write_parquet(cause_df, cause_df_path)

# Age-specific results
cause_df <- as.data.frame(get_outputs("cause", cause_id = malaria_id,
                                      measure_id = c(1,6) , #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                      year_id = past_years,
                                      location_id = gbd_2023_modeling_hierarchy$location_id, 
                                      age_group_id = gbd_2023_age_metadata$age_group_id, 
                                      release_id = gbd_2023_release_id,
                                      metric_id = c(1,3),  #rate =3, counts =1
                                      sex_id = 1:3, # males =1, females =2, both =3
                                      compare_version_id = compare_v_2023))

# Merge location hierarchy
cause_df <- merge(cause_df, gbd_2023_modeling_hierarchy, all.x = TRUE, sort = FALSE)
cause_df <- merge(cause_df, gbd_2023_population, all.x = TRUE, sort = FALSE)
cause_df_path <- glue("{output_dir}/gbd_2023_malaria_as.parquet")
arrow::write_parquet(cause_df, cause_df_path)


