from __future__ import annotations
from typing import Dict, List, Optional
import sys
import functools
# Global unbuffered print for immediate output
print = functools.partial(print, flush=True)

import pandas as pd
import numpy as np


def prepare_time_index(df: pd.DataFrame, dt_col: str, freq: str = '15min',
                       station_col: Optional[str] = 'Station_ID') -> pd.DataFrame:
    """
    Reindex DataFrame to strict frequency to handle temporal gaps correctly.

    CRITICAL FIX: This prevents the "shift bug" where shift(1) would pull data
    from the previous row instead of the previous time step during data gaps.

    Example of the bug:
        - Data has a 2-hour gap at 14:00 (sensor maintenance)
        - Without reindexing: At 14:15, shift(1) pulls from 12:00 (2 hours ago!)
        - With reindexing: Gap rows become NaN, shift(1) correctly pulls from 14:00

    Args:
        df: Input DataFrame
        dt_col: Datetime column name
        freq: Target frequency (e.g., '15min', '1H', '1D')
        station_col: Station ID column for multi-station data (None for single station)

    Returns:
        DataFrame reindexed to strict frequency with gaps filled as NaN rows

    Example:
        Before: 12:00, 12:15, [gap], 14:15, 14:30
        After:  12:00, 12:15, 12:30, ..., 14:00, 14:15, 14:30
                (missing rows have NaN values)
    """
    print(f"[TIME INDEX] Reindexing to strict {freq} frequency to handle temporal gaps...")

    out = df.copy()

    # Ensure datetime column is actually datetime type
    if not pd.api.types.is_datetime64_any_dtype(out[dt_col]):
        out[dt_col] = pd.to_datetime(out[dt_col])

    if station_col is not None and station_col in out.columns:
        # Multi-station: reindex each station separately
        stations = out[station_col].unique()
        reindexed_parts = []
        deduped_rows_total = 0

        for station in stations:
            station_data = out[out[station_col] == station].copy()

            # Get time range for this station
            station_times = station_data[dt_col]
            if len(station_times) == 0:
                reindexed_parts.append(station_data)
                continue

            start_time = station_times.min()
            end_time = station_times.max()

            # Create complete time index
            full_idx = pd.date_range(start=start_time, end=end_time, freq=freq, name=dt_col)

            # Handle duplicates: drop exact duplicates, keep first occurrence
            dup_count = station_data.duplicated(subset=[dt_col]).sum()
            if dup_count > 0:
                print(f"[TIME INDEX] Station {station}: Dropping {dup_count} duplicate timestamps")
                station_data = station_data.drop_duplicates(subset=[dt_col], keep='first')

            deduped_rows_total += len(station_data)

            # Reindex this station's data
            station_data = station_data.set_index(dt_col).reindex(full_idx).reset_index()

            # Restore Station_ID (may be lost during reindex)
            station_data[station_col] = station

            reindexed_parts.append(station_data)

        # Combine all stations
        result = pd.concat(reindexed_parts, axis=0, ignore_index=True)

        # Count added rows
        original_rows = deduped_rows_total
        new_rows = len(result)
        gaps_filled = new_rows - original_rows
        print(f"[TIME INDEX] Filled {gaps_filled} temporal gap rows across {len(stations)} station(s)")
        if gaps_filled > 0:
            gap_pct = 100 * gaps_filled / new_rows
            print(f"[TIME INDEX] Gap percentage: {gap_pct:.1f}% of data")

    else:
        # Single station: simple reindex
        start_time = out[dt_col].min()
        end_time = out[dt_col].max()

        # Create complete time index
        full_idx = pd.date_range(start=start_time, end=end_time, freq=freq, name=dt_col)

        # Handle duplicates: drop exact duplicates, keep first occurrence
        dup_count = out.duplicated(subset=[dt_col]).sum()
        if dup_count > 0:
            print(f"[TIME INDEX] Dropping {dup_count} duplicate timestamps")
            out = out.drop_duplicates(subset=[dt_col], keep='first')

        # Reindex
        original_rows = len(out)
        result = out.set_index(dt_col).reindex(full_idx).reset_index()

        # Count added rows
        new_rows = len(result)
        gaps_filled = new_rows - original_rows
        print(f"[TIME INDEX] Filled {gaps_filled} temporal gap rows")
        if gaps_filled > 0:
            gap_pct = 100 * gaps_filled / new_rows
            print(f"[TIME INDEX] Gap percentage: {gap_pct:.1f}% of data")

    return result


def add_time_features(df: pd.DataFrame, dt_col: str, hour: bool = True, dow: bool = True) -> pd.DataFrame:
    """
    Add cyclical time features (global, station-independent).

    Args:
        df: Input DataFrame
        dt_col: Datetime column name
        hour: Whether to add hour sin/cos
        dow: Whether to add day-of-week sin/cos

    Returns:
        DataFrame with time features added
    """
    out = df.copy()
    if hour:
        h = out[dt_col].dt.hour
        out["hour_sin"] = np.sin(2 * np.pi * h / 24)
        out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    if dow:
        d = out[dt_col].dt.dayofweek
        out["dow_sin"] = np.sin(2 * np.pi * d / 7)
        out["dow_cos"] = np.cos(2 * np.pi * d / 7)
    return out


def add_fourier_seasonality(df: pd.DataFrame, dt_col: str, period: float = 365.25,
                              harmonics: int = 4) -> pd.DataFrame:
    """
    Adds sophisticated seasonal features using Fourier harmonics.

    PHYSICS RATIONALE: H2S dispersion changes drastically by season
    - Winter: Temperature inversions trap pollutants near ground
    - Summer: Convection disperses pollutants upward
    - The model needs to know "Time of Year" not just "Time of Day"

    Fourier series captures periodic patterns efficiently:
    - harmonics=1: Annual cycle (Winter vs Summer)
    - harmonics=2: Semi-annual (Equinox vs Solstice transitions)
    - harmonics=3: Quarterly patterns
    - harmonics=4: Seasonal fine-tuning

    Using sine/cosine pairs avoids discontinuity at year boundary.

    Args:
        df: Input DataFrame
        dt_col: Datetime column name
        period: Period of the cycle in days (default: 365.25 for annual cycle)
        harmonics: Number of Fourier harmonic pairs to add (default: 4)

    Returns:
        DataFrame with seasonal features added

    Example:
        With harmonics=4, adds 8 features:
        - season_sin_1, season_cos_1 (annual cycle)
        - season_sin_2, season_cos_2 (semi-annual)
        - season_sin_3, season_cos_3 (third harmonic)
        - season_sin_4, season_cos_4 (fourth harmonic)
    """
    print(f"[FEATURES] Adding Fourier seasonality features (harmonics={harmonics})...")
    print(f"[FEATURES]   This captures seasonal dispersion physics:")
    print(f"[FEATURES]   - Winter: Inversions trap H2S near ground")
    print(f"[FEATURES]   - Summer: Convection disperses H2S upward")

    out = df.copy()

    # Convert datetime to day of year (float for sub-daily precision)
    # dayofyear is 1-366, adding hour/24 gives fractional day for time-of-day alignment
    doy = out[dt_col].dt.dayofyear + (out[dt_col].dt.hour / 24.0)

    for k in range(1, harmonics + 1):
        # Fourier harmonic: sin(2*pi*k*t/T) and cos(2*pi*k*t/T)
        theta = 2 * np.pi * k * doy / period

        out[f'season_sin_{k}'] = np.sin(theta)
        out[f'season_cos_{k}'] = np.cos(theta)

    print(f"[FEATURES]   Added {2 * harmonics} seasonal features (sin/cos pairs)")

    return out


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes Wind Direction (WD) cyclically to solve 359°≈0° discontinuity.
    Converts 0-360 degrees into Sine and Cosine components.
    Keeps raw WD for tree-based sector splits.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with WD_sin and WD_cos added (raw WD preserved)
    """
    out = df.copy()
    # Find WD column (case insensitive search)
    wd_col = next((c for c in out.columns if c in ['WD', 'wind_dir', 'wd', 'WindDir']), None)

    if wd_col:
        print(f"[FEATURES] Encoding circular wind direction for '{wd_col}'")
        # Convert to radians
        wd_rad = np.deg2rad(out[wd_col])
        out[f"{wd_col}_sin"] = np.sin(wd_rad)
        out[f"{wd_col}_cos"] = np.cos(wd_rad)
        print(f"[FEATURES] Added {wd_col}_sin and {wd_col}_cos (preserving raw {wd_col})")
    else:
        print("[WARN] No wind direction column found for circular encoding")

    return out


def add_lags(df: pd.DataFrame, col: str, ks: List[int], dt_col: str) -> pd.DataFrame:
    """
    Add lagged features with station-aware grouping to prevent cross-station leakage.

    Args:
        df: Input DataFrame
        col: Column to lag
        ks: List of lag values
        dt_col: Datetime column for sorting

    Returns:
        DataFrame with lag features added
    """
    out = df.copy()
    # Sort for shift safety
    if 'Station_ID' in out.columns:
        out = out.sort_values(['Station_ID', dt_col])
        # Critical: GroupBy ensures we don't shift data from Station A to Station B
        for k in ks:
            out[f"{col}_lag{k}"] = out.groupby('Station_ID')[col].shift(k)
    else:
        out = out.sort_values(dt_col)
        for k in ks:
            out[f"{col}_lag{k}"] = out[col].shift(k)
    return out


def add_rolling(df: pd.DataFrame, col: str, window: int, stats: List[str], dt_col: str) -> pd.DataFrame:
    """
    Add rolling window features with station-aware grouping.
    Special handling for circular wind direction using vector averaging.

    Args:
        df: Input DataFrame
        col: Column to compute rolling stats on
        window: Window size
        stats: List of statistics (e.g., ['mean', 'std'])
        dt_col: Datetime column for sorting

    Returns:
        DataFrame with rolling features added

    Note:
        For wind direction (WD, WindDir, wd), uses circular statistics:
        - Mean calculated via vector averaging (avoids 350°+10° = 180° bug)
        - Std calculated using circular standard deviation
        - Min/max skipped (ambiguous for circular data)
    """
    out = df.copy()

    # SPECIAL HANDLING FOR WIND DIRECTION (circular variable)
    is_wind_direction = col.upper() in ['WD', 'WINDDIR', 'WIND_DIR', 'WIND_DIRECTION']

    if is_wind_direction:
        print(f"[FEATURES] Using circular statistics for {col} rolling window (fixes 350°+10°=180° bug)")

        # Convert to radians and then to vector components
        wd_rad = np.deg2rad(out[col])
        wd_sin_temp = np.sin(wd_rad)
        wd_cos_temp = np.cos(wd_rad)

        # Temporarily add sin/cos columns for rolling
        out[f'_{col}_sin_temp'] = wd_sin_temp
        out[f'_{col}_cos_temp'] = wd_cos_temp

        if 'Station_ID' in out.columns:
            out = out.sort_values(['Station_ID', dt_col])
            grouped_sin = out.groupby('Station_ID')[f'_{col}_sin_temp'].rolling(window=window, min_periods=1)
            grouped_cos = out.groupby('Station_ID')[f'_{col}_cos_temp'].rolling(window=window, min_periods=1)

            # Calculate rolling mean of sin and cos components
            roll_sin_mean = grouped_sin.mean().reset_index(level=0, drop=True)
            roll_cos_mean = grouped_cos.mean().reset_index(level=0, drop=True)
        else:
            out = out.sort_values(dt_col)
            roll_sin_mean = out[f'_{col}_sin_temp'].rolling(window=window, min_periods=1).mean()
            roll_cos_mean = out[f'_{col}_cos_temp'].rolling(window=window, min_periods=1).mean()

        # Convert back to degrees using atan2 (circular mean)
        if 'mean' in stats:
            circular_mean = np.rad2deg(np.arctan2(roll_sin_mean, roll_cos_mean))
            circular_mean = (circular_mean + 360) % 360  # Normalize to [0, 360)
            out[f"{col}_roll{window}_mean"] = circular_mean

        # Circular standard deviation (magnitude of mean resultant vector)
        if 'std' in stats:
            # R = sqrt(mean(sin)^2 + mean(cos)^2)
            # Circular std = sqrt(-2 * log(R)) in radians, convert to degrees
            R = np.sqrt(roll_sin_mean**2 + roll_cos_mean**2)
            # Avoid log(0) by clipping R to small positive value
            R_clipped = np.clip(R, 1e-10, 1.0)
            circular_std_rad = np.sqrt(-2 * np.log(R_clipped))
            circular_std_deg = np.rad2deg(circular_std_rad)
            out[f"{col}_roll{window}_std"] = circular_std_deg

        # Note: min/max are ambiguous for circular data, skip them
        if 'min' in stats or 'max' in stats:
            print(f"[WARN] Min/max stats not meaningful for circular variable {col}, skipping")

        # Clean up temporary columns
        out.drop([f'_{col}_sin_temp', f'_{col}_cos_temp'], axis=1, inplace=True)

    else:
        # STANDARD ROLLING FOR NON-CIRCULAR VARIABLES
        if 'Station_ID' in out.columns:
            out = out.sort_values(['Station_ID', dt_col])
            # GroupBy Rolling
            grouped = out.groupby('Station_ID')[col].rolling(window=window, min_periods=1)
            for st in stats:
                # Result has multi-index (Station_ID, original_index), drop Station_ID level to assign back
                res = getattr(grouped, st)().reset_index(level=0, drop=True)
                # Assign by index alignment
                out[f"{col}_roll{window}_{st}"] = res
        else:
            out = out.sort_values(dt_col)
            roll = out[col].rolling(window=window, min_periods=1)
            for st in stats:
                out[f"{col}_roll{window}_{st}"] = getattr(roll, st)()

    return out


def calculate_filtered_derivative(series: pd.Series, target_index: pd.DatetimeIndex, column_name: str) -> pd.Series:
    """
    Calculates time derivative using 6h low-pass filter and 6h windowed weighted slope.

    Args:
        series (pd.Series): Raw data series (index must be datetime)
        target_index (pd.DatetimeIndex): Timestamps to calculate derivatives at
        column_name (str): Name for logging

    Returns:
        pd.Series: Derivative values aligned to target_index
    """
    try:
        from scipy import signal

        # 1. Prepare Data
        valid_data = series.dropna()
        if len(valid_data) < 10:
            print(f"   [WARN] Insufficient data for {column_name} derivative")
            return pd.Series(index=target_index, dtype=float)

        # 2. Low-Pass Filtering (4th order Butterworth, 6h cutoff)
        # Estimate sampling freq
        time_diffs = np.diff(valid_data.index).astype('timedelta64[s]').astype(float)
        med_dt = np.median(time_diffs)

        # Safety check for sampling rate
        if med_dt <= 0:
            med_dt = 900.0  # Default to 15 min if calc fails

        fs = 1.0 / med_dt
        cutoff = 1.0 / (6 * 3600)  # 6 hour period
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist

        # Apply filter if parameters are valid
        if 0 < normalized_cutoff < 1:
            b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
            # CRITICAL FIX: Use causal filter (lfilter) instead of filtfilt
            # filtfilt is non-causal (looks at future data) causing temporal leakage
            # lfilter is causal (only looks at past) - no leakage
            filtered_values = signal.lfilter(b, a, valid_data.values)
            filtered_series = pd.Series(filtered_values, index=valid_data.index)
        else:
            filtered_series = valid_data  # Fallback to raw if data is too sparse

        # 3. Windowed Weighted Slope Calculation
        window_hours = 6.0
        derivatives = pd.Series(index=target_index, dtype=float)

        # Convert timestamps to seconds for numeric calculation
        target_secs = target_index.astype(np.int64) // 10**9
        source_secs = filtered_series.index.astype(np.int64) // 10**9
        source_vals = filtered_series.values

        window_half_width = window_hours * 1800  # seconds

        # Calculate slope for each target timestamp
        for i, t_center in enumerate(target_secs):
            # Find indices within window
            mask = (source_secs >= t_center - window_half_width) & \
                   (source_secs <= t_center + window_half_width)

            times_window = source_secs[mask]
            vals_window = source_vals[mask]

            if len(times_window) >= 3:
                # Time delta from center
                dt = times_window - t_center

                # Weighting: 1 - sqrt(|dt|/max_dt) - gives higher weight to center
                max_dist = np.max(np.abs(dt)) if len(dt) > 1 else 1
                weights = 1.0 - (np.abs(dt) / max_dist) ** 0.5
                weights = np.maximum(weights, 0.1)

                # Weighted Linear Regression
                w_sum = np.sum(weights)
                mean_t = np.sum(weights * dt) / w_sum
                mean_y = np.sum(weights * vals_window) / w_sum

                numerator = np.sum(weights * (dt - mean_t) * (vals_window - mean_y))
                denominator = np.sum(weights * (dt - mean_t) ** 2)

                if denominator > 0:
                    slope_per_sec = numerator / denominator
                    derivatives.iloc[i] = slope_per_sec * 3600.0  # Convert to Unit/hr

        return derivatives

    except Exception as e:
        print(f"   [ERROR] Failed to calc derivative for {column_name}: {e}")
        return pd.Series(index=target_index, dtype=float)


def add_met_derivatives(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    Generates station-aware filtered time derivatives for meteorological variables.
    Adds columns: d_WS_dt, d_WD_dt, d_Temp_dt, d_Pressure_dt.
    Handles Wind Direction unwrapping (360->0 crossing) per station.

    Args:
        df: DataFrame with meteorological data
        dt_col: Name of datetime column

    Returns:
        DataFrame with derivative columns added
    """
    print("[FEATURES] Generating meteorological time derivatives (per station)...")

    out = df.copy()

    # Define variable mappings
    met_vars = {
        'Pressure': ['Pressure', 'pressure', 'press', 'barometric', 'bp'],
        'Temp': ['TEMP', 'Temp', 'temp', 't_air', 'temperature'],
        'WS': ['WS', 'wind_speed', 'ws', 'wind_vel'],
        'WD': ['WD', 'wind_dir', 'wd', 'wind_direction']
    }

    # Check if multi-station or single-station
    if 'Station_ID' not in out.columns:
        # Fallback to single-station behavior (backward compatibility)
        out['Station_ID'] = 'Unknown'
        was_single_site = True
    else:
        was_single_site = False

    stations = out['Station_ID'].unique()
    results = []

    for station in stations:
        st_data = out[out['Station_ID'] == station].copy()

        # Set datetime index
        if dt_col in st_data.columns:
            st_data = st_data.set_index(dt_col).sort_index()
        elif isinstance(st_data.index, pd.DatetimeIndex):
            st_data = st_data.sort_index()
        else:
            print(f"   [WARN] No datetime index found for {station}, skipping derivatives")
            results.append(st_data.reset_index())
            continue

        # Process each meteorological variable
        for metric, patterns in met_vars.items():
            # Find column
            col_name = None
            for col in st_data.columns:
                if col.startswith('n_') or col == 'Station_ID':
                    continue
                if col in patterns or col.lower() in [p.lower() for p in patterns]:
                    col_name = col
                    break

            if col_name:
                if len(stations) > 1:
                    print(f"   [{station}] Calculating d_{metric}_dt from {col_name}...")
                else:
                    print(f"   Calculating d_{metric}_dt from {col_name}...")

                series_to_process = st_data[col_name].copy()

                # Wind Direction unwrapping
                if metric == 'WD':
                    valid_mask = series_to_process.notna()
                    if valid_mask.sum() > 0:
                        wd_rad = np.deg2rad(series_to_process[valid_mask])
                        wd_unwrapped_vals = np.rad2deg(np.unwrap(wd_rad))
                        series_to_process = pd.Series(index=series_to_process.index, dtype=float)
                        series_to_process[valid_mask] = wd_unwrapped_vals

                # Calculate derivative
                deriv = calculate_filtered_derivative(series_to_process, st_data.index, metric)
                st_data[f"d_{metric}_dt"] = deriv

        results.append(st_data.reset_index())

    # Re-stack all stations
    combined = pd.concat(results, axis=0, ignore_index=True)

    # Remove temporary Station_ID if this was single-site
    if was_single_site:
        combined = combined.drop(columns=['Station_ID'])

    print(f"   [OK] Generated derivative features for {len(stations)} station(s)")
    return combined


def add_met_accumulation(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    Calculates accumulation of meteorological conditions (Integrals)
    to complement the derivatives (Rates of Change).

    Physics rationale:
    - Derivatives (d_WS_dt) detect CHANGES in weather (transitions)
    - Accumulations detect PERSISTENCE (stagnation, inversions)
    - H2S spikes occur when dispersion is blocked for DURATION, not just at moment of change

    Features added:
    - stagnation_2h, stagnation_6h: Inverse wind speed sum (measures gas buildup)
    - recirculation_index: Wind vector consistency (straight flow vs circular trapping)

    Args:
        df: DataFrame with meteorological data
        dt_col: Name of datetime column

    Returns:
        DataFrame with accumulation features added
    """
    print("[FEATURES] Adding Meteorological Accumulation indices...")
    out = df.copy()

    # Check if wind features exist
    if 'WS' not in out.columns:
        print("[WARN] Wind Speed (WS) not found, skipping accumulation features")
        return out

    # 1. STAGNATION ACCUMULATION (Inverse Wind Speed Sum)
    # Lower wind = Higher stagnation = More gas buildup
    # Avoid division by zero
    inv_ws = 1.0 / np.maximum(out['WS'], 0.1)

    # Station-aware grouping
    if 'Station_ID' in out.columns:
        # Ensure sorted by datetime within each station
        out = out.sort_values(['Station_ID', dt_col])
        grouper = out.groupby('Station_ID', group_keys=False)

        # 2-hour accumulation (8 steps @ 15min = 120min)
        # 6-hour accumulation (24 steps @ 15min = 360min)
        print("   [ACCUMULATION] Calculating stagnation indices (2h, 6h windows)...")
        out['stagnation_2h'] = grouper[['WS']].transform(
            lambda x: (1.0 / np.maximum(x, 0.1)).rolling(window=8, min_periods=1).sum()
        )['WS']
        out['stagnation_6h'] = grouper[['WS']].transform(
            lambda x: (1.0 / np.maximum(x, 0.1)).rolling(window=24, min_periods=1).sum()
        )['WS']

        # 2. RECIRCULATION INDEX (Vector Flow Consistency)
        # Requires WD_sin/WD_cos from add_wind_features
        if 'WD_sin' in out.columns and 'WD_cos' in out.columns:
            print("   [ACCUMULATION] Calculating recirculation index (3h window)...")

            # Vector components of flow
            out['_flow_u'] = out['WS'] * out['WD_sin']
            out['_flow_v'] = out['WS'] * out['WD_cos']

            # Sum vectors over 3 hours (12 steps @ 15min)
            sum_u = grouper['_flow_u'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
            sum_v = grouper['_flow_v'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
            sum_speed = grouper['WS'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())

            # Ratio of Vector Sum to Scalar Sum
            # 1.0 = Straight line flow (good dispersion)
            # 0.0 = Perfect recirculation (trapping)
            net_displacement = np.sqrt(sum_u**2 + sum_v**2)
            out['recirculation_index'] = 1.0 - (net_displacement / np.maximum(sum_speed, 0.1))

            # Clean up temporary columns
            out.drop(['_flow_u', '_flow_v'], axis=1, inplace=True)
        else:
            print("   [WARN] WD_sin/WD_cos not found, skipping recirculation index")
            print("   [HINT] Enable features.engineering.wd_sin_cos: true in config")

    else:
        # Single-station mode (backward compatibility)
        out = out.sort_values(dt_col)
        print("   [ACCUMULATION] Calculating stagnation indices (single-site mode)...")

        out['stagnation_2h'] = (1.0 / np.maximum(out['WS'], 0.1)).rolling(window=8, min_periods=1).sum()
        out['stagnation_6h'] = (1.0 / np.maximum(out['WS'], 0.1)).rolling(window=24, min_periods=1).sum()

        if 'WD_sin' in out.columns and 'WD_cos' in out.columns:
            print("   [ACCUMULATION] Calculating recirculation index (single-site mode)...")

            out['_flow_u'] = out['WS'] * out['WD_sin']
            out['_flow_v'] = out['WS'] * out['WD_cos']

            sum_u = out['_flow_u'].rolling(window=12, min_periods=1).sum()
            sum_v = out['_flow_v'].rolling(window=12, min_periods=1).sum()
            sum_speed = out['WS'].rolling(window=12, min_periods=1).sum()

            net_displacement = np.sqrt(sum_u**2 + sum_v**2)
            out['recirculation_index'] = 1.0 - (net_displacement / np.maximum(sum_speed, 0.1))

            out.drop(['_flow_u', '_flow_v'], axis=1, inplace=True)

    print(f"   [OK] Added accumulation features: stagnation_2h, stagnation_6h" +
          (", recirculation_index" if 'recirculation_index' in out.columns else ""))

    return out


def build_features(df: pd.DataFrame, cfg: Dict, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    eng = cfg.get("engineering", {})

    # ========================================================================
    # CRITICAL FIX: Reindex to strict frequency BEFORE feature engineering
    # ========================================================================
    # This prevents the "shift bug" where shift(k) would pull data from k rows
    # ago instead of k time steps ago during temporal gaps (sensor maintenance,
    # data outages, etc.)
    #
    # Example bug:
    #   - Data has 2-hour gap at 14:00
    #   - Without fix: At 14:15, shift(1) pulls from 12:00 (wrong!)
    #   - With fix: Gap rows become NaN, shift(1) pulls from 14:00 (correct!)
    # ========================================================================
    timebase = cfg.get("timebase", "15min")
    out = prepare_time_index(out, dt_col, freq=timebase)

    # 1. Circular Wind Encoding (NEW - fixes 359°≈0° discontinuity)
    if eng.get("wd_sin_cos", False):
        out = add_wind_features(out)

    # 2. Add meteorological derivatives if enabled (station-aware)
    if eng.get("met_derivatives", False):
        out = add_met_derivatives(out, dt_col)

    # 2b. Add meteorological accumulation indices (stagnation, recirculation)
    if eng.get("met_accumulation", False):
        out = add_met_accumulation(out, dt_col)

    # 3. Add Fourier seasonality features (Time of Year)
    # CRITICAL: H2S dispersion physics changes by season
    # - Winter: Inversions trap pollutants near ground
    # - Summer: Convection disperses pollutants upward
    if eng.get("fourier_seasonality", False):
        season_harmonics = eng.get("season_harmonics", 4)
        out = add_fourier_seasonality(out, dt_col, harmonics=season_harmonics)

    # 4. Add cyclical time features (global, not station-specific)
    if eng.get("hour_sin_cos", False) or eng.get("dow_sin_cos", False):
        out = add_time_features(out, dt_col, eng.get("hour_sin_cos", False), eng.get("dow_sin_cos", False))

    # 5. Add lagged features (station-aware to prevent cross-site leakage)
    # Now shift(1) correctly shifts by 1 time step, not 1 row
    #
    # CRITICAL: Do NOT add lags of the target variable (H2S)!
    # The S4 model has its own internal memory (state space) that captures
    # temporal dependencies. Adding explicit H2S lags would cause:
    # 1) Data leakage if not masked properly
    # 2) Overfitting to recent values
    # 3) Redundancy with the SSM's memory
    #
    # Note: The caller (cli.py) validates that target_col is not in lag config.
    for lag_item in eng.get("lags", []):
        out = add_lags(out, lag_item["col"], lag_item["k"], dt_col)

    # 6. Add rolling window features (station-aware)
    # Now rolling windows correctly account for time gaps
    for roll_item in eng.get("rollings", []):
        out = add_rolling(out, roll_item["col"], int(roll_item["window"]), list(roll_item["stats"]), dt_col)

    return out
