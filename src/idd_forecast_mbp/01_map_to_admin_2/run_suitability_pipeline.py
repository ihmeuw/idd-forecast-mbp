"""
Wrapper to run climate-data suitability pipeline with custom curves.
"""

import pandas as pd
from pathlib import Path
from typing import Callable
from collections import defaultdict
import numpy as np

# These imports come from the climate-data repo
from climate_data.generate import scenario_annual, utils
from climate_data import constants as cdc

from idd_forecast_mbp import constants as mbpc
PROCESSED_DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
malaria_temp_suitabilities_df_path = PROCESSED_DATA_PATH / 'malaria_temp_suitabilities_df.parquet'


def run_custom_suitability_curves(
    dataframe_path: str | Path,
    pathogen: str,
    suitability_column: str,
    combination_columns: list[str],
    climate_data_repo: str | Path,
    output_dir: str | Path | None = None,
    scenarios: list[str] | None = None,
    years: list[str] | None = None,
    gcm_members: list[str] | None = None,
    progress_bar: bool = True,
    dry_run: bool = False,
    run_aggregation: bool = True,
    agg_version: str = "custom_suitability",
    agg_hierarchy: str = "lsae_1209",
) -> list[str]:
    """
    Run the suitability pipeline for each unique combination of curve parameters.

    Parameters
    ----------
    dataframe_path
        Path to parquet with all curves. Must have 'temperature' column and
        columns specified in suitability_column and combination_columns.
    pathogen
        Disease name (e.g., 'malaria', 'dengue')
    suitability_column
        Column name containing suitability values (e.g., 'rel_suit')
    combination_columns
        Columns whose unique combinations define separate curves (e.g., ['method', 'shift'])
    climate_data_repo
        Path to the climate-data repo root
    output_dir
        Where to write gridded results (default: cdc.MODEL_ROOT)
    scenarios
        SSP scenarios to run (default: ssp126, ssp245, ssp585)
    years
        Years to process - will be automatically split into historical (1950-2023) 
        and forecast (2024-2100). Default: all years.
    gcm_members
        GCM members to use (default: all available)
    progress_bar
        Show progress bar during computation
    dry_run
        If True, only generate curve files and print what would run
    run_aggregation
        If True, run population-weighted aggregation to admin units (default: True)
    agg_version
        Version name for aggregation outputs
    agg_hierarchy
        Location hierarchy for aggregation (default: gbd_2021)

    Returns
    -------
    list[str]
        Names of target variables that were processed
    """
    df = pd.read_parquet(dataframe_path)
    climate_data_repo = Path(climate_data_repo)
    output_dir = Path(output_dir) if output_dir else cdc.MODEL_ROOT
    
    supplementary_data_dir = (
        climate_data_repo / "src" / "climate_data" / "generate" / "supplementary_data"
    )
    supplementary_data_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    scenarios = scenarios or ["ssp126", "ssp245", "ssp585"]
    
    # Parse years into historical and forecast
    if years is None:
        history_years = cdc.HISTORY_YEARS
        forecast_years = cdc.FORECAST_YEARS
    else:
        history_years = [y for y in years if y in cdc.HISTORY_YEARS]
        forecast_years = [y for y in years if y in cdc.FORECAST_YEARS]
    
    # Get unique combinations
    unique_combos = df[combination_columns].drop_duplicates()
    print(f"Found {len(unique_combos)} unique curve combinations")

    target_variables = []

    for idx, row in unique_combos.iterrows():
        # Build target variable name
        suffix_parts = []
        for col in combination_columns:
            val = row[col]
            if isinstance(val, float):
                if val < 0:
                    val_str = f"m{abs(val)}".replace(".", "_")
                elif val > 0:
                    val_str = f"p{val}".replace(".", "_")
                else:
                    val_str = "0_0"
            else:
                val_str = str(val).replace(" ", "_").replace("-", "_")
            suffix_parts.append(f"{col}_{val_str}")

        suffix = "_".join(suffix_parts)
        target_var = f"{pathogen}_suitability_{suffix}"
        target_variables.append(target_var)

        # Filter dataframe to this combination
        mask = pd.Series(True, index=df.index)
        for col in combination_columns:
            mask &= df[col] == row[col]

        subset = df.loc[mask, ["temperature", suitability_column]].copy()
        subset = subset.rename(columns={suitability_column: "suitability"})
        subset = subset.sort_values("temperature").reset_index(drop=True)

        # Write parquet file
        parquet_path = supplementary_data_dir / f"{target_var}.parquet"
        subset.to_parquet(parquet_path, index=False)
        print(f"Created: {parquet_path.name} ({len(subset)} rows)")

        # Inject transform into TRANSFORM_MAP
        scenario_annual.TRANSFORM_MAP[target_var] = utils.Transform(
            source_variables=["mean_temperature"],
            transform_funcs=[
                _make_custom_suitability_mapper(target_var),
                utils.annual_sum,
            ],
        )

    if dry_run:
        print(f"\nDry run complete. Generated {len(target_variables)} curve files.")
        print(f"Target variables: {target_variables}")
        print(f"Historical years to run: {len(history_years)} ({history_years[0] if history_years else 'none'} - {history_years[-1] if history_years else 'none'})")
        print(f"Forecast years to run: {len(forecast_years)} ({forecast_years[0] if forecast_years else 'none'} - {forecast_years[-1] if forecast_years else 'none'})")
        print(f"Scenarios: {scenarios}")
        return target_variables

    # Import additional modules
    from climate_data.data import ClimateData, PopulationModelData, ClimateAggregateData
    from climate_data.generate.draws import compile_gcm_main
    from climate_data.aggregate.pixel import pixel_main
    from climate_data.aggregate.hierarchy import hierarchy_main
    
    cdata = ClimateData(output_dir, read_only=True)
    
    # Get all available GCMs
    all_gcms = gcm_members or cdata.get_gcms(["tas"])
    
    # ============================================
    # STEP 1: Generate scenario annual results
    # ============================================
    print("\n" + "="*60)
    print("STEP 1: Generating scenario annual results")
    print("  Output: Annual suitability rasters per GCM/year/scenario")
    print("="*60)
    
    for target_var in target_variables:
        print(f"\nProcessing: {target_var}")
        
        # Historical
        if history_years:
            print(f"  Running historical ({len(history_years)} years)...")
            for year in history_years:
                scenario_annual.generate_scenario_annual_main(
                    target_variable=target_var,
                    scenario="historical",
                    year=year,
                    gcm_member="era5",
                    output_dir=output_dir,
                    progress_bar=progress_bar,
                )
        
        # Scenarios
        if forecast_years:
            for scenario in scenarios:
                print(f"  Running {scenario} ({len(forecast_years)} years × {len(all_gcms)} GCMs)...")
                for year in forecast_years:
                    for gcm in all_gcms:
                        scenario_annual.generate_scenario_annual_main(
                            target_variable=target_var,
                            scenario=scenario,
                            year=year,
                            gcm_member=gcm,
                            output_dir=output_dir,
                            progress_bar=progress_bar,
                        )

    # ============================================
    # STEP 2: Compile GCM results
    # ============================================
    print("\n" + "="*60)
    print("STEP 2: Compiling GCM results")
    print("  Output: Combined historical+scenario file per GCM")
    print("="*60)
    
    for target_var in target_variables:
        print(f"\nCompiling: {target_var}")
        for scenario in scenarios:
            for gcm in all_gcms:
                print(f"  {scenario} / {gcm}")
                compile_gcm_main(
                    target_variable=target_var,
                    cmip6_experiment=scenario,
                    gcm_member=gcm,
                    ouptut_dir=str(output_dir),
                )

    # ============================================
    # STEP 3: Create draws with mapping
    # ============================================
    print("\n" + "="*60)
    print("STEP 3: Creating 100 draws")
    print("  Output: Draw symlinks + draw_mapping_{variable}.parquet")
    print("="*60)
    
    draw_mappings = {}
    for target_var in target_variables:
        print(f"\nCreating draws for: {target_var}")
        mapping_df = _draws_main_with_mapping(target_var, str(output_dir), scenarios)
        draw_mappings[target_var] = mapping_df
        
        # Save mapping
        mapping_path = output_dir / f"draw_mapping_{target_var}.parquet"
        mapping_df.to_parquet(mapping_path, index=False)
        print(f"  Saved draw mapping to: {mapping_path}")

    if not run_aggregation:
        print("\n" + "="*60)
        print("COMPLETE (aggregation skipped)")
        print("="*60)
        return target_variables

    # ============================================
    # STEP 4: Pixel aggregation
    # ============================================
    print("\n" + "="*60)
    print("STEP 4: Pixel aggregation (population-weighted)")
    print("  Output: Admin-level values per draw/block")
    print("="*60)
    
    # Monkey-patch AGGREGATION_MEASURES to include our custom variables
    original_measures = cdc.AGGREGATION_MEASURES.copy()
    cdc.AGGREGATION_MEASURES = target_variables
    
    # Also need to patch in the aggregate.pixel module
    from climate_data.aggregate import pixel as pixel_module
    pixel_module.cdc.AGGREGATION_MEASURES = target_variables
    
    pm_data = PopulationModelData(cdc.POPULATION_MODEL_ROOT)
    ca_data = ClimateAggregateData(cdc.AGGREGATE_ROOT)
    
    modeling_frame = pm_data.load_modeling_frame()
    block_keys = modeling_frame["block_key"].unique().tolist()
    
    draws = [f"{d:03d}" for d in range(100)]
    
    print(f"  Processing {len(draws)} draws × {len(block_keys)} blocks...")
    for draw in draws:
        print(f"\n  Draw {draw}")
        for block_key in block_keys:
            pixel_main(
                agg_version=agg_version,
                block_key=block_key,
                draw=draw,
                hierarchy=agg_hierarchy,
                population_model_root=str(cdc.POPULATION_MODEL_ROOT),
                climate_data_root=str(output_dir),
                output_dir=str(cdc.AGGREGATE_ROOT),
                progress_bar=False,
            )

    # ============================================
    # STEP 5: Hierarchy aggregation
    # ============================================
    print("\n" + "="*60)
    print("STEP 5: Hierarchy aggregation (roll up to all admin levels)")
    print("  Output: Final parquet files per variable")
    print("="*60)
    
    for target_var in target_variables:
        for scenario in scenarios:
            print(f"\n  Aggregating: {target_var} / {scenario}")
            hierarchy_main(
                agg_version=agg_version,
                hierarchy=agg_hierarchy,
                measure=target_var,
                scenario=scenario,
                population_model_dir=str(cdc.POPULATION_MODEL_ROOT),
                output_dir=str(cdc.AGGREGATE_ROOT),
                progress_bar=progress_bar,
            )
    
    # Restore original measures
    cdc.AGGREGATION_MEASURES = original_measures
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Gridded rasters: {output_dir}/results/annual/")
    print(f"  Draw mappings: {output_dir}/draw_mapping_*.parquet")
    print(f"  Aggregated data: {cdc.AGGREGATE_ROOT}/{agg_version}/")
    
    return target_variables


def _make_custom_suitability_mapper(curve_name: str) -> Callable:
    """Create a suitability mapping function for a custom curve."""
    import numpy as np
    import xarray as xr
    
    def smap(ds: xr.Dataset) -> xr.Dataset:
        df = pd.read_parquet(
            Path(utils.__file__).parent / "supplementary_data" / f"{curve_name}.parquet"
        )
        t = df["temperature"].to_numpy()
        s = df["suitability"].to_numpy()
        
        ds["value"] = (("date", "latitude", "longitude"), np.interp(ds["value"], t, s))
        return ds

    return smap


def _draws_main_with_mapping(
    target_variable: str, 
    output_dir: str, 
    scenarios: list[str],
) -> pd.DataFrame:
    """Run draws and return the draw-to-GCM mapping."""
    from climate_data.data import ClimateData
    
    cdata = ClimateData(output_dir)

    # Only look at scenarios we actually ran
    scenario_gcm_members = {}
    for scenario in scenarios:
        paths = (cdata.compiled_annual_results / scenario / target_variable).glob("*.nc")
        scenario_gcm_members[scenario] = [p.stem for p in paths]

    # Get members present in all scenarios
    all_members = set.intersection(*[set(m) for m in scenario_gcm_members.values()])
    all_members = sorted(all_members)
    
    source_member_map = defaultdict(list)
    for gcm_member in all_members:
        source, member = gcm_member.split("_")
        source_member_map[source].append(member)

    num_draws = 100
    rs = np.random.RandomState(42)  # Same seed as original - ensures matching draws
    
    draw_mapping = []
    for draw in range(num_draws):
        gcm = rs.choice(list(source_member_map))
        member = rs.choice(source_member_map[gcm])
        gcm_member = f"{gcm}_{member}"
        draw_mapping.append({
            "draw": draw,
            "gcm": gcm,
            "member": member, 
            "gcm_member": gcm_member
        })
        
        for scenario in scenarios:
            cdata.link_annual_draw(
                draw=draw,
                variable=target_variable,
                scenario=scenario,
                gcm_member=gcm_member,
            )
    
    return pd.DataFrame(draw_mapping)


# ============================================
# Example usage
# ============================================
if __name__ == "__main__":
    
    # ------------------------------------------
    # USE CASE 1: Run the whole thing
    # ------------------------------------------
    # run_custom_suitability_curves(
    #     dataframe_path=malaria_temp_suitabilities_df_path,
    #     pathogen="malaria",
    #     suitability_column="rel_suit",
    #     combination_columns=["method", "shift"],
    #     climate_data_repo="/ihme/homes/bcreiner/repos/climate-data",
    #     # All defaults: all years, all scenarios, aggregation enabled
    # )
    
    # ------------------------------------------
    # USE CASE 2: Test run - one year, one scenario, no aggregation
    # ------------------------------------------
    run_custom_suitability_curves(
        dataframe_path=malaria_temp_suitabilities_df_path,
        pathogen="malaria",
        suitability_column="rel_suit",
        combination_columns=["method", "shift"],
        climate_data_repo="/ihme/homes/bcreiner/repos/climate-data",
        scenarios=["ssp245"],
        years=["2010", "2050"],  # One historical, one forecast
        run_aggregation=True,
        dry_run=False,  # Set to False to actually run
    )