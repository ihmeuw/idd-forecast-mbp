require(glue)
require(data.table)

USER <- Sys.getenv('USER')

repo_dir <- glue("/ihme/homes/{USER}/repos/idd-forecast-mbp")
message_dir <- "/mnt/team/idd/pub"

draws <- sprintf("%03d", 0:99)
ssp_scenarios <- c("ssp126", "ssp245", "ssp585")

param_map_filepath <- glue("/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue_param_map.csv")
param_map <- data.table(expand.grid(draw_num = 0:99,
                                    ssp_scenario = ssp_scenarios))
write.csv(param_map,  param_map_filepath, row.names = FALSE)


## QSUB Command
job_name <- glue("forecast_dengue")   # name of the job
thread_flag <- "-c 1" 
mem_flag <- "--mem=50G" 
runtime_flag <- "-t 50"
#jdrive_flag <- "-l archive" # archive nodes can access the J drive. They're a little harder to get though. If you need J drive access, uncomment this and add it to the qsub_command
queue_flag <- "-p all.q" # long or all

throttle_flag <- "1000" # how many tasks are allowed to run at once. 800 is a good limit for smaller jobs. Bigger jobs will need smaller throttles

# n_jobs <- paste0("1-", nrow(param_map), "%", throttle_flag) # this means you're running one task for every row of the param map you made. 
n_jobs <- paste0("1-", nrow(param_map)) # this means you're running one task for every row of the param map you made.

# n_jobs <- "1-1" # sometimes this is helpful if you just want to test the first row without running the entire thing
# filepath to the script you want to launch. Make sure you saved it first.
script <- glue("{repo_dir}/src/idd_forecast_mbp/forecasting/forecast_dengue_admin_2s_rocket.r") 
error_filepath <- glue("-e {message_dir}/stderr/%x.e%j") # where are errors going to be saved
output_filepath <- glue("-o {message_dir}/stdout/%x.o%j") # where are outputs going to be saved
project_flag<- "-A proj_rapidresponse"  # make sure this is one you have permissions for
# add jdrive_flag if needed
qsub_command <- glue( "sbatch -J {job_name} {mem_flag} {thread_flag} {project_flag}  {mem_flag} {runtime_flag} {queue_flag} -a '{n_jobs}' {error_filepath} {output_filepath} /ihme/singularity-images/rstudio/shells/execRscript.sh -i /ihme/singularity-images/rstudio/ihme_rstudio_4222.img -s {script}")
system(qsub_command) # this is the go button