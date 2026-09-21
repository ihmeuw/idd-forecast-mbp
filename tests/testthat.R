# R test runner. From the repo root, inside the pipeline's R image:
#   /ihme/singularity-images/rstudio/shells/execRscript.sh -i <image> -s tests/testthat.R
# Slow tests that read the real past inputs run only when MBP_PAST_INPUTS names the parquet.
library(testthat)
test_dir("tests/testthat", stop_on_failure = TRUE)
