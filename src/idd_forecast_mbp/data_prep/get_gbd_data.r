require(glue)

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
output_dir <- "/mnt/team/idd/pub/forecast-mbp/01-raw_data"

release_id <- 16
como_v <- 1591
codcorrect_v <- 461
dalynator_v <- 96
burdenator_v <- 360
compare_v <- 8234

# Constants
past_years = 1970:2023

# GBD Constants
release_id_2021 = 9
release_id_2023 = 16

gbd_location_set_id = 35
fhs_location_set_id = 39



como_v_2023 = 1591
codcorrect_v_2023 = 461
dalynator_v_2023 = 96
burdenator_v_2023 = 360
compare_v_2023 = 8234
ages = 22
sexes = 3
dengue_id = 357
malaria_id = 345


# Get hierarchy
gbd_modeling_hierarchy_2023 <- as.data.frame(get_location_metadata(location_set_id = gbd_location_set_id, release_id=release_id_2023))
fhs_modeling_hierarchy_2023 <- as.data.frame(get_location_metadata(location_set_id = fhs_location_set_id, release_id=release_id_2023))

col_names_to_delete <- c("start_date", "end_date", "date_inserted", "last_updated", "last_updated_by", "last_updated_action")
toss_col_locs <- which(names(gbd_modeling_hierarchy_2023) %in% col_names_to_delete)

gbd_modeling_hierarchy_2023 <- gbd_modeling_hierarchy_2023[,-toss_col_locs]
fhs_modeling_hierarchy_2023 <- fhs_modeling_hierarchy_2023[,-toss_col_locs]


write.csv(gbd_modeling_hierarchy_2023, glue("{output_dir}/gbd_modeling_hierarchy_2023.csv"), row.names = FALSE)
write.csv(fhs_modeling_hierarchy_2023, glue("{output_dir}/fhs_modeling_hierarchy_2023.csv"), row.names = FALSE)


# Get population
gbd_population_2023 = as.data.frame(get_population(age_group_id = ages,
                                                   release_id = release_id_2023,
                                                   year_id = past_years,
                                                   location_id = gbd_modeling_hierarchy_2023$location_id,
                                                   sex_id = sexes))

gbd_population_2023 <- gbd_population_2023[ ,c("age_group_id", "location_id", "year_id", "sex_id", "population")]

write.csv(gbd_population_2023, glue("{output_dir}/gbd_population_2023.csv"), row.names = FALSE)

####
## Dengue
####

# All age results
cause_df <- as.data.frame(get_outputs("cause", cause_id = dengue_id,
                                                    measure_id = 1:6, #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                                    year_id = past_years,
                                                    location_id = gbd_modeling_hierarchy_2023$location_id, 
                                                    age_group_id = ages, 
                                                    release_id = release_id_2023,
                                                    metric_id = c(1,3),  #rate =3, counts =1
                                                    sex_id = sexes, # males =1, females =2, both =3
                                                    compare_version_id = compare_v_2023))

# Merge location hierarchy
cause_df <- merge(cause_df, gbd_modeling_hierarchy_2023, all.x = TRUE, sort = FALSE)
cause_df <- merge(cause_df, gbd_population_2023, all.x = TRUE, sort = FALSE)

write.csv(cause_df, glue("{output_dir}/gbd_dengue_aa_2023.csv"), row.names = FALSE)

####
## Malaria
####

# All age results
cause_df <- as.data.frame(get_outputs("cause", cause_id = malaria_id,
                                      measure_id = 1:6, #prev =5 , inc =6 , deaths =1 , dalys =2 , ylds = 3, ylls = 4
                                      year_id = past_years,
                                      location_id = gbd_modeling_hierarchy_2023$location_id, 
                                      age_group_id = ages, 
                                      release_id = release_id_2023,
                                      metric_id = c(1,3),  #rate =3, counts =1
                                      sex_id = sexes, # males =1, females =2, both =3
                                      compare_version_id = compare_v_2023))

# Merge location hierarchy
cause_df <- merge(cause_df, gbd_modeling_hierarchy_2023, all.x = TRUE, sort = FALSE)
cause_df <- merge(cause_df, gbd_population_2023, all.x = TRUE, sort = FALSE)

write.csv(cause_df, glue("{output_dir}/gbd_malaria_aa_2023.csv"), row.names = FALSE)



