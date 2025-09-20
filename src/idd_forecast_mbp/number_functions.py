__all__ = [
    "get_multiplier",
    "get_summary_ds_from_ds",
    "get_value_w_UI",
    "format_mean_lower_upper",
    "smart_UI_format"
]

import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore

def get_multiplier(number, scale=2, allow_nonstandard_units=False, override_multiplier=None):
    # For counts, adapt based on magnitude
    if (number < scale * 100 and override_multiplier is None) or override_multiplier == 1:
        multiplier = 1
        multiplier_text = ""
    elif (number < scale * 1000 and override_multiplier is None) or override_multiplier == 100:
        multiplier = 0.01
        multiplier_text = " (in 100s)"
    elif (number < scale * 10000 and override_multiplier is None) or override_multiplier == 1000:
        multiplier = 0.001
        multiplier_text = " (in 1,000s)"
    elif (number < scale * 100000 and override_multiplier is None) or override_multiplier == 10000:
        multiplier = 0.0001
        multiplier_text = " (in 10,000s)"
    elif (number < scale * 1_000_000 and override_multiplier is None) or override_multiplier == 100000:
        multiplier = 0.00001
        multiplier_text = " (in 100,000s)"
    elif (number < scale * 10_000_000 and override_multiplier is None and allow_nonstandard_units) or (number < scale * 1_000_000_000 and not allow_nonstandard_units and override_multiplier is None) or override_multiplier == 1000000:
        multiplier = 0.000001
        multiplier_text = " (in Millions)"
    elif (number < scale * 100_000_000 and override_multiplier is None and allow_nonstandard_units) or override_multiplier == 1000000:
        multiplier = 0.0000001
        multiplier_text = " (in 10 Millions)"
    elif (number < scale * 1_000_000_000 and override_multiplier is None and allow_nonstandard_units) or override_multiplier == 10000000:
        multiplier = 0.00000001
        multiplier_text = " (in 100 Millions)"
    else:
        multiplier = 0.000000001
        multiplier_text = " (in Billions)"
    return multiplier, multiplier_text

def get_mean_UI_matrix(df):
    row_mean = df.mean(axis=1)
    row_lower = df.quantile(0.025, axis=1)
    row_upper = df.quantile(0.975, axis=1)
    summary_df = pd.DataFrame({
        'mean': row_mean,
        'lower': row_lower,
        'upper': row_upper
    })
    return summary_df

def get_summary_ds_from_ds(ds, var_name='val', dim='draw_id', ui=[0.025, 0.975], suffix=None):
    if isinstance(ds, xr.Dataset):
        data_var = ds[var_name]
    else:
        data_var = ds
    stat_names = ["val", "lower", "upper"]
    if suffix is not None:
        stat_names = [f"{x}.{suffix}" for x in stat_names]
    summary_ds = xr.Dataset({
        stat_names[0]: data_var.mean(dim=dim),
        stat_names[1]: data_var.quantile(ui[0], dim=dim).drop_vars('quantile', errors='ignore'),
        stat_names[2]: data_var.quantile(ui[1], dim=dim).drop_vars('quantile', errors='ignore')
    })
    return summary_ds

def smart_UI_format(val, units=False, reference_val=None, percentage=False, rate=False, small_number = None, multiplier_adjustment=True):
    """Format number with 3 sig figs or percentage-specific rounding."""
    val = float(val)
    original_val = val  # Keep original for unit decisions

    if percentage:
        val *= 100
        if reference_val is not None:
            reference_val *= 100
    elif rate:
        val *= 100000
        if reference_val is not None:
            reference_val *= 100000


    # Determine scale based on reference_val (or val if no reference)
    if multiplier_adjustment:
        use_millions = False
        use_billions = False
        if not percentage and not rate:
            check_val = reference_val if reference_val is not None else original_val
            if abs(check_val) >= 1_000_000_000:
                use_billions = True
                val /= 1_000_000_000
            elif abs(check_val) >= 1_000_000:
                use_millions = True
                val /= 1_000_000

    # Format based on type
    if percentage or rate:
        # Always 1 decimal place for percentages
        rounded = round(val, 1)
        decimals = 1
        formatted = f"{rounded:.{decimals}f}" if decimals > 0 else str(int(rounded))
    else:
        if val == 0:
            formatted = "0·00"
        else:
            # Always use 3 significant figures
            power = int(np.floor(np.log10(abs(val))))
            scale = 10 ** (power - 2)
            rounded = np.round(val / scale) * scale

            # Recalculate power after rounding (in case rounding changed the magnitude)
            if rounded != 0:
                power = int(np.floor(np.log10(abs(rounded))))

            # Calculate decimal places for exactly 3 sig figs
            if power >= 2:  # >= 100
                decimals = 0
            elif power >= 1:  # >= 10
                decimals = 1
            elif power >= 0:  # >= 1
                decimals = 2
            else:  # < 1
                decimals = 2 - power

            formatted = f"{rounded:.{decimals}f}" if decimals > 0 else str(int(rounded))

    # Add separators and units
    formatted = formatted.replace('.', '·')
    # Only count digits for separator logic, ignore minus sign
    integer_part = formatted.split('·')[0]
    num_digits = len(integer_part.lstrip('-'))
    if num_digits > 4:  # DONT ADD THE GAP FOR NUMBERS IN THE THOUSANDS
        parts = formatted.split('·')
        integer = parts[0]
        # Handle minus sign separately
        sign = ''
        if integer.startswith('-'):
            sign = '-'
            integer = integer[1:]
        formatted_int = ""
        for i, digit in enumerate(reversed(integer)):
            if i > 0 and i % 3 == 0:
                formatted_int = '\u2009' + formatted_int
            formatted_int = digit + formatted_int
        formatted = sign + formatted_int + ('·' + parts[1] if len(parts) > 1 else '')

    # Add units
    if percentage and units:
        formatted += '%'
    elif not percentage and units:
        if use_billions:
            formatted += ' billion'
        elif use_millions:
            formatted += ' million'

    if small_number is not None: 
        if percentage or rate:
            if val < 0 and val > -0.05:
                formatted = '>-0.05'
            elif val > 0 and val < 0.05:
                formatted = '<0.05'
            else:
                formatted = formatted
        elif small_number > 0:
            if val < 0 and val > -1 * small_number:
                formatted = f'>-{small_number}'
            elif val > 0 and val < small_number:
                formatted = f'<{small_number}'
            else:
                formatted = formatted
        else:
                formatted = formatted
    else:
        formatted = formatted

    return formatted

def get_value_w_UI(vec, nested=False, percentage=False, rate=False, first=False, 
                   use_precalculated=False, two_lines=False, units = True, small_number=None,
                   separator='\u2013', multiplier_adjustment=True):
    """Main function to format values with uncertainty intervals."""
    if use_precalculated:
        if 'val' in vec.columns:
            mean_val = vec['val'].iloc[0]
        elif 'mean' in vec.columns:
            mean_val = vec['mean'].iloc[0]
        else:
            raise ValueError("DataFrame must have 'val' or 'mean' column when use_precalculated=True")
        lower = vec['lower'].iloc[0] if 'lower' in vec.columns else vec['lower'].iloc[0]
        upper = vec['upper'].iloc[0] if 'upper' in vec.columns else vec['upper'].iloc[0]
    else:
        if hasattr(vec, 'to_series'):
            vec = vec.to_series()
        mean_val = round(vec.mean(), 6)
        lower, upper = vec.quantile([0.025, 0.975]).values

    # Format all values
    mean_fmt = smart_UI_format(mean_val, units = units, percentage=percentage, rate=rate, small_number=small_number, multiplier_adjustment=multiplier_adjustment)
    lower_fmt = smart_UI_format(lower, units=False, reference_val=mean_val, percentage=percentage, rate=rate, small_number=small_number, multiplier_adjustment=multiplier_adjustment)
    upper_fmt = smart_UI_format(upper, units=False, reference_val=mean_val, percentage=percentage, rate=rate, small_number=small_number, multiplier_adjustment=multiplier_adjustment)
    if lower < 0 and upper > 0:
        separator = ' to '

    # Build output strings
    raw_text = f"{mean_val} (95% UI: {lower}, {upper})"
    open_bracket = '[' if nested else '('
    close_bracket = ']' if nested else ')'
    if first:
        formatted_text = f"{mean_fmt} {'[95% UI ' if nested else '(95% UI '}{lower_fmt}{separator}{upper_fmt}{close_bracket}"
    else:
        formatted_text = f"{mean_fmt} {open_bracket}{lower_fmt}{separator}{upper_fmt}{close_bracket}"
    if two_lines:
        return f'{mean_fmt}\n{open_bracket}{lower_fmt}{separator}{upper_fmt}{close_bracket}'
    else:
        return f"{raw_text}\n{formatted_text}"

# Convenience functions for backward compatibility
def get_UI_text(vec, first=False, nested=False, percentage=False, small_number=None,separator='\u2013', multiplier_adjustment=True):
    return get_value_w_UI(vec, nested=nested, percentage=percentage, first=first, 
                          small_number=small_number, separator=separator, multiplier_adjustment=multiplier_adjustment)

def format_mean_lower_upper(vec, percentage=False, rate=False, first=False, two_lines=False, units = True, 
                            small_number=None, separator='\u2013', multiplier_adjustment=True):
    return get_value_w_UI(vec, percentage=percentage, rate=rate, first=first, use_precalculated=True, two_lines=two_lines, 
                          units = units, small_number=small_number, separator=separator, multiplier_adjustment=multiplier_adjustment)