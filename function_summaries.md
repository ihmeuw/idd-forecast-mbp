# Function Documentation Summary

## YAML and Configuration Functions

### `load_yaml_dictionary(yaml_path: str) -> dict`
**Purpose**: Loads a YAML file and extracts the COVARIATE_DICT section.

**Inputs**:
- `yaml_path`: String path to the YAML file

**Outputs**:
- Dictionary containing the COVARIATE_DICT from the YAML file

**Functions it uses**: None (uses built-in `yaml.safe_load`)

**Functions that use it**: `parse_yaml_dictionary()`

---

### `parse_yaml_dictionary(covariate: str) -> dict`
**Purpose**: Parses covariate-specific configuration from the YAML dictionary and calculates derived values.

**Inputs**:
- `covariate`: String name of the covariate to extract configuration for

**Outputs**:
- Dictionary with parsed covariate configuration including:
  - `covariate_name`: Name of the covariate
  - `covariate_resolution`: Calculated resolution (numerator/denominator)
  - `years`: List of years from start to end
  - `synoptic`: Synoptic flag
  - `cc_sensitive`: Climate change sensitivity flag
  - `summary_statistic`: Summary statistic method
  - `path`: File path

**Functions it uses**: `load_yaml_dictionary()`

**Functions that use it**: Not directly called by other functions in this module

---

## Data Merging and Reading Functions

### `merge_dataframes(model_df, dfs)`
**Purpose**: Merges multiple DataFrames with a base model DataFrame on location_id and year_id.

**Inputs**:
- `model_df`: Base pandas DataFrame
- `dfs`: Dictionary of DataFrames to merge

**Outputs**:
- Merged pandas DataFrame with suffixes added for duplicate columns

**Functions it uses**: None (uses pandas merge)

**Functions that use it**: Not directly called by other functions in this module

---

### `read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)`
**Purpose**: Reads multiple income data files, filters by RCP scenario, and processes them.

**Inputs**:
- `income_paths`: Dictionary of file paths
- `rcp_scenario`: RCP scenario to filter by
- `VARIABLE_DATA_PATH`: Base path for variable data

**Outputs**:
- Dictionary of filtered pandas DataFrames (scenario column dropped)

**Functions it uses**: `read_parquet_with_integer_ids()`

**Functions that use it**: Not directly called by other functions in this module

---

### `read_urban_paths(urban_paths, VARIABLE_DATA_PATH)`
**Purpose**: Reads multiple urban data files and standardizes column names.

**Inputs**:
- `urban_paths`: Dictionary of file paths
- `VARIABLE_DATA_PATH`: Base path for variable data

**Outputs**:
- Dictionary of processed pandas DataFrames with standardized column names

**Functions it uses**: None (uses pandas read_parquet)

**Functions that use it**: Not directly called by other functions in this module

---

## Data Type and I/O Utility Functions

### `ensure_id_columns_are_integers(df)`
**Purpose**: Converts columns ending with '_id' to integer type.

**Inputs**:
- `df`: pandas DataFrame

**Outputs**:
- DataFrame with ID columns converted to integers

**Functions it uses**: None (uses pandas type operations)

**Functions that use it**: `read_parquet_with_integer_ids()`

---

### `read_parquet_with_integer_ids(path, **kwargs)`
**Purpose**: Reads a parquet file and ensures ID columns are integers.

**Inputs**:
- `path`: File path to parquet file
- `**kwargs`: Additional arguments for pd.read_parquet

**Outputs**:
- pandas DataFrame with integer ID columns

**Functions it uses**: `ensure_id_columns_are_integers()`

**Functions that use it**: `read_income_paths()`

---

### `write_parquet(df, filepath, max_retries=3, compression='snappy', index=False, **kwargs)`
**Purpose**: Writes parquet files with validation and retry logic for robustness.

**Inputs**:
- `df`: pandas DataFrame to write
- `filepath`: Destination file path
- `max_retries`: Number of retry attempts (default: 3)
- `compression`: Compression method (default: 'snappy')
- `index`: Whether to include index (default: False)
- `**kwargs`: Additional arguments for to_parquet

**Outputs**:
- Boolean indicating success/failure

**Functions it uses**: None (uses pandas and os operations)

**Functions that use it**: 
- `rake_aa_count_lsae_to_gbd()`
- `make_aa_rate_variable()`
- `aggregate_aa_count_lsae_to_gbd()`
- `make_full_aa_rate_df_from_aa_count_df()`

---

## Raking Functions

### `prep_df(df, hierarchy_df)`
**Purpose**: Prepares DataFrame by adding level column and removing parent_id if present.

**Inputs**:
- `df`: pandas DataFrame to prepare
- `hierarchy_df`: Hierarchy DataFrame containing location_id and level mappings

**Outputs**:
- Prepared DataFrame with level column added and parent_id removed

**Functions it uses**: None (uses pandas merge and drop)

**Functions that use it**: 
- `rake_aa_count_lsae_to_gbd()`
- `aggregate_aa_count_lsae_to_gbd()`
- `aggregate_aa_rate_lsae_to_gbd()`

---

### `rake_level(count_variable, level_df, level_m1_df, hierarchy_df)`
**Purpose**: Rakes (adjusts) data at one level to match aggregated totals from the next higher level.

**Inputs**:
- `count_variable`: Name of the count variable to rake
- `level_df`: DataFrame for current level
- `level_m1_df`: DataFrame for the level above (level minus 1)
- `hierarchy_df`: Hierarchy DataFrame

**Outputs**:
- DataFrame with raked values that sum to the higher level totals

**Functions it uses**: None (uses pandas operations)

**Functions that use it**: `rake_aa_count_lsae_to_gbd()`

---

### `rake_aa_count_lsae_to_gbd(count_variable, hierarchy_df, gbd_aa_count_df, lsae_aa_count_df, full_aa_count_df_path, return_full_df=False)`
**Purpose**: Rakes LSAE age-aggregated count data to match GBD totals across hierarchy levels.

**Inputs**:
- `count_variable`: Name of count variable
- `hierarchy_df`: Hierarchy DataFrame
- `gbd_aa_count_df`: GBD age-aggregated count data
- `lsae_aa_count_df`: LSAE age-aggregated count data
- `full_aa_count_df_path`: Output file path
- `return_full_df`: Whether to return the DataFrame (default: False)

**Outputs**:
- Optionally returns full raked DataFrame if return_full_df=True

**Functions it uses**: 
- `prep_df()`
- `rake_level()`
- `write_parquet()`

**Functions that use it**: Not directly called by other functions in this module

---

### `make_aa_rate_variable(count_variable, full_aa_count_df, aa_population_df, full_lsae_aa_rate_df_path, return_full_df=False)`
**Purpose**: Converts age-aggregated count data to rate data using population denominators.

**Inputs**:
- `count_variable`: Name of count variable
- `full_aa_count_df`: Full age-aggregated count DataFrame
- `aa_population_df`: Age-aggregated population DataFrame
- `full_lsae_aa_rate_df_path`: Output file path
- `return_full_df`: Whether to return DataFrame (default: False)

**Outputs**:
- Optionally returns rate DataFrame if return_full_df=True

**Functions it uses**: `write_parquet()`

**Functions that use it**: Not directly called by other functions in this module

---

## Aggregation Functions

### `aggregate_level(count_variable, level_df, hierarchy_df)`
**Purpose**: Aggregates count data from one hierarchy level to the next higher level.

**Inputs**:
- `count_variable`: Name of count variable to aggregate
- `level_df`: DataFrame for current level
- `hierarchy_df`: Hierarchy DataFrame

**Outputs**:
- DataFrame with aggregated counts at the parent level

**Functions it uses**: None (uses pandas operations)

**Functions that use it**: `aggregate_aa_count_lsae_to_gbd()`

---

### `aggregate_aa_count_lsae_to_gbd(count_variable, hierarchy_df, lsae_aa_count_df, full_aa_count_df_path, return_full_df=False)`
**Purpose**: Aggregates LSAE age-aggregated count data up through all hierarchy levels (5 to 0).

**Inputs**:
- `count_variable`: Name of count variable
- `hierarchy_df`: Hierarchy DataFrame
- `lsae_aa_count_df`: LSAE age-aggregated count data
- `full_aa_count_df_path`: Output file path
- `return_full_df`: Whether to return DataFrame (default: False)

**Outputs**:
- Optionally returns full aggregated DataFrame if return_full_df=True

**Functions it uses**: 
- `prep_df()`
- `aggregate_level()`
- `write_parquet()`

**Functions that use it**: `aggregate_aa_rate_lsae_to_gbd()`

---

### `make_full_aa_rate_df_from_aa_count_df(rate_variable, count_variable, full_aa_count_df, aa_population_df, full_aa_rate_df_path=None, return_full_df=False)`
**Purpose**: Converts aggregated count data to rate data using population.

**Inputs**:
- `rate_variable`: Name of rate variable to create
- `count_variable`: Name of count variable
- `full_aa_count_df`: Full age-aggregated count DataFrame
- `aa_population_df`: Age-aggregated population DataFrame
- `full_aa_rate_df_path`: Optional output file path
- `return_full_df`: Whether to return DataFrame (default: False)

**Outputs**:
- Optionally returns rate DataFrame if return_full_df=True

**Functions it uses**: `write_parquet()`

**Functions that use it**: `aggregate_aa_rate_lsae_to_gbd()`

---

### `aggregate_aa_rate_lsae_to_gbd(rate_variable, hierarchy_df, lsae_aa_rate_df, aa_population_df, full_aa_rate_df_path=None, return_full_df=False)`
**Purpose**: Aggregates LSAE age-aggregated rate data by first converting to counts, aggregating, then converting back to rates.

**Inputs**:
- `rate_variable`: Name of rate variable
- `hierarchy_df`: Hierarchy DataFrame
- `lsae_aa_rate_df`: LSAE age-aggregated rate data
- `aa_population_df`: Age-aggregated population DataFrame
- `full_aa_rate_df_path`: Optional output file path
- `return_full_df`: Whether to return DataFrame (default: False)

**Outputs**:
- Optionally returns full aggregated rate DataFrame if return_full_df=True

**Functions it uses**: 
- `prep_df()`
- `aggregate_aa_count_lsae_to_gbd()`
- `make_full_aa_rate_df_from_aa_count_df()`

**Functions that use it**: Not directly called by other functions in this module

---

## Function Dependency Tree

```
Configuration Functions:
├── load_yaml_dictionary()
└── parse_yaml_dictionary() → uses load_yaml_dictionary()

Data I/O Functions:
├── ensure_id_columns_are_integers()
├── read_parquet_with_integer_ids() → uses ensure_id_columns_are_integers()
├── read_income_paths() → uses read_parquet_with_integer_ids()
├── read_urban_paths()
├── merge_dataframes()
└── write_parquet() → used by multiple functions

Raking Functions:
├── prep_df() → used by multiple raking/aggregation functions
├── rake_level() → used by rake_aa_count_lsae_to_gbd()
├── rake_aa_count_lsae_to_gbd() → uses prep_df(), rake_level(), write_parquet()
└── make_aa_rate_variable() → uses write_parquet()

Aggregation Functions:
├── aggregate_level() → used by aggregate_aa_count_lsae_to_gbd()
├── aggregate_aa_count_lsae_to_gbd() → uses prep_df(), aggregate_level(), write_parquet()
├── make_full_aa_rate_df_from_aa_count_df() → uses write_parquet()
└── aggregate_aa_rate_lsae_to_gbd() → uses prep_df(), aggregate_aa_count_lsae_to_gbd(), make_full_aa_rate_df_from_aa_count_df()
```
