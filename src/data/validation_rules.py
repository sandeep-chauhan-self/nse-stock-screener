"""
Validation rules implementation - specific validation logic for data quality checks.
This module contains concrete implementations of validation rules:
- FreshnessCheck: Validates data timeliness (T+1 compliance)
- EnhancedConsistencyCheck: Validates OHLCV relationships and data integrity
- CrossProviderDiscrepancyCheck: Detects discrepancies between data sources
"""
import logging
from datetime import datetime
from typing import Optional, Any
import numpy as np
import pandas as pd
from .validation_core import ValidationRule, ValidationIssue, ValidationLevel

# Configure logger
logger = logging.getLogger(__name__)
class FreshnessCheck(ValidationRule):
    """Check if data is fresh (updated recently) - FS.2 T+1 compliance."""
    def __init__(self, max_age_hours: int = 25, market_hours_only: bool = True, **kwargs) -> None:
        super().__init__("freshness_check", **kwargs)

        # T+1 + buffer
        self.max_age_hours = max_age_hours
        self.market_hours_only = market_hours_only
    def validate(self, symbol: str, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Check data freshness against T+1 requirement."""
        issues = []
        metadata = metadata or {}
        if data.empty:
            return self._create_empty_data_issue(symbol)
        current_time = datetime.now()

        # Check last data point date
        issues.extend(self._validate_data_freshness(symbol, data, current_time))

        # Check metadata freshness if available
        if metadata:
            issues.extend(self._validate_metadata_freshness(symbol, metadata, current_time))
        return issues
    def _create_empty_data_issue(self, symbol: str) -> List[ValidationIssue]:
        """Create issue for empty data."""
        return [ValidationIssue(
            rule_name=self.name,
            level=ValidationLevel.CRITICAL,
            message=f"No data available for symbol {symbol}",
            symbol=symbol,
            metadata={"issue_type": "empty_data"}
        )]
    def _validate_data_freshness(self, symbol: str, data: pd.DataFrame,
                                 current_time: datetime) -> List[ValidationIssue]:
        """Validate freshness of data based on Date column."""
        issues = []
        try:

            # Check for missing Date column
            if 'Date' not in data.columns:
                issues.extend(self._create_missing_date_issue(symbol))
            else:

                # Validate date column format
                date_validation = self._validate_date_column(symbol, data)
                if date_validation:
                    issues.extend(date_validation)
                else:

                    # Check for valid dates
                    latest_date = self._get_latest_date(data)
                    if latest_date is None:
                        issues.extend(self._create_no_valid_dates_issue(symbol))
                    else:

                        # Calculate age and check threshold
                        issues.extend(self._check_age_threshold(symbol, latest_date, current_time))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error checking freshness for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"issue_type": "freshness_check_error", "error": str(e)}
            ))
        return issues
    def _create_missing_date_issue(self, symbol: str) -> List[ValidationIssue]:
        """Create issue for missing Date column."""
        return [ValidationIssue(
            rule_name=self.name,
            level=ValidationLevel.ERROR,
            message=f"Date column missing in data for {symbol}",
            symbol=symbol,
            metadata={"issue_type": "missing_date_column"}
        )]
    def _validate_date_column(self, symbol: str, data: pd.DataFrame) -> Optional[List[ValidationIssue]]:
        """Validate and convert date column."""
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            try:
                pd.to_datetime(data['Date'])
                return None
            except Exception as e:
                return [ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.ERROR,
                    message=f"Cannot parse Date column for {symbol}: {str(e)}",
                    symbol=symbol,
                    metadata={"issue_type": "date_parse_error", "error": str(e)}
                )]
        return None
    def _get_latest_date(self, data: pd.DataFrame) -> Optional[datetime]:
        """Get the latest date from the data."""
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            dates = data['Date']
        else:
            dates = pd.to_datetime(data['Date'])
        latest_date = dates.max()
        return latest_date.to_pydatetime() if not pd.isna(latest_date) else None
    def _create_no_valid_dates_issue(self, symbol: str) -> List[ValidationIssue]:
        """Create issue for no valid dates."""
        return [ValidationIssue(
            rule_name=self.name,
            level=ValidationLevel.ERROR,
            message=f"No valid dates found for {symbol}",
            symbol=symbol,
            metadata={"issue_type": "no_valid_dates"}
        )]
    def _check_age_threshold(self, symbol: str, latest_date: datetime,
                             current_time: datetime) -> List[ValidationIssue]:
        """Check if data age exceeds threshold."""
        issues = []
        age_hours = (current_time - latest_date).total_seconds() / 3600
        if age_hours > self.max_age_hours:
            level = ValidationLevel.CRITICAL if age_hours > 48 else ValidationLevel.WARNING
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=level,
                message=f"Data for {symbol} is stale: {age_hours:.1f} hours old (max: {self.max_age_hours})",
                symbol=symbol,
                metadata={
                    "age_hours": age_hours,
                    "latest_date": latest_date.isoformat(),
                    "max_age_hours": self.max_age_hours
                }
            ))
        return issues
    def _validate_metadata_freshness(self, symbol: str, metadata: Dict[str, Any],
                                     current_time: datetime) -> List[ValidationIssue]:
        """Validate freshness based on metadata timestamp."""
        issues = []
        try:
            if 'last_updated' in metadata:
                if isinstance(metadata['last_updated'], str):
                    last_updated = datetime.fromisoformat(metadata['last_updated'].replace('Z', '+00:00'))
                elif isinstance(metadata['last_updated'], datetime):
                    last_updated = metadata['last_updated']
                else:
                    return []
                age_hours = (current_time - last_updated).total_seconds() / 3600
                if age_hours > self.max_age_hours:
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        level=ValidationLevel.WARNING,
                        message=f"Metadata for {symbol} indicates stale data: {age_hours:.1f} hours old",
                        symbol=symbol,
                        metadata={
                            "metadata_age_hours": age_hours,
                            "last_updated": last_updated.isoformat()
                        }
                    ))
        except Exception:

            # Ignore metadata parsing errors
            pass
        return issues
class EnhancedConsistencyCheck(ValidationRule):
    """Enhanced data consistency validation with comprehensive OHLCV checks."""
    def __init__(self, **kwargs) -> None:
        super().__init__("enhanced_consistency_check", **kwargs)
    def validate(self, symbol: str, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Perform comprehensive consistency validation."""
        issues = []
        metadata = metadata or {}
        if data.empty:
            return [ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.CRITICAL,
                message=f"No data to validate for {symbol}",
                symbol=symbol
            )]

        # Validate OHLC relationships
        issues.extend(self._validate_ohlc_relationships(symbol, data))

        # Validate volume consistency
        issues.extend(self._validate_volume_consistency(symbol, data))

        # Validate completeness
        issues.extend(self._validate_completeness(symbol, data))

        # Detect price anomalies
        issues.extend(self._detect_price_anomalies(symbol, data))

        # Validate data sequence
        issues.extend(self._validate_data_sequence(symbol, data))
        return issues
    def _validate_ohlc_relationships(self, symbol: str, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate OHLC price relationships."""
        issues = []
        try:
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.ERROR,
                    message=f"Missing OHLC columns for {symbol}: {missing_cols}",
                    symbol=symbol,
                    metadata={"missing_columns": missing_cols}
                ))
                return issues

            # Check High >= max(Open, Close)
            invalid_high = data['High'] < data[['Open', 'Close']].max(axis=1)
            if invalid_high.any():
                count = invalid_high.sum()
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid High prices for {symbol}: {count} rows where High < max(Open, Close)",
                    symbol=symbol,
                    metadata={"invalid_high_count": int(count)}
                ))

            # Check Low <= min(Open, Close)
            invalid_low = data['Low'] > data[['Open', 'Close']].min(axis=1)
            if invalid_low.any():
                count = invalid_low.sum()
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid Low prices for {symbol}: {count} rows where Low > min(Open, Close)",
                    symbol=symbol,
                    metadata={"invalid_low_count": int(count)}
                ))

            # Check for zero or negative prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in data.columns:
                    invalid_prices = (data[col] <= 0) | data[col].isna()
                    if invalid_prices.any():
                        count = invalid_prices.sum()
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=ValidationLevel.ERROR,
                            message=f"Invalid {col} prices for {symbol}: {count} zero/negative/NaN values",
                            symbol=symbol,
                            metadata={f"invalid_{col.lower()}_count": int(count)}
                        ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error validating OHLC relationships for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
    def _validate_volume_consistency(self, symbol: str, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate volume data consistency."""
        issues = []
        try:
            if 'Volume' not in data.columns:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.WARNING,
                    message=f"Volume column missing for {symbol}",
                    symbol=symbol
                ))
                return issues

            # Check for negative volume
            negative_volume = data['Volume'] < 0
            if negative_volume.any():
                count = negative_volume.sum()
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.ERROR,
                    message=f"Negative volume detected for {symbol}: {count} rows",
                    symbol=symbol,
                    metadata={"negative_volume_count": int(count)}
                ))

            # Check for excessive zero volume
            zero_volume = data['Volume'] == 0
            zero_volume_pct = (zero_volume.sum() / len(data)) * 100

            # >15% zero volume is concerning
            if zero_volume_pct > 15:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.WARNING,
                    message=f"High zero volume percentage for {symbol}: {zero_volume_pct:.1f}%",
                    symbol=symbol,
                    metadata={"zero_volume_percentage": zero_volume_pct}
                ))

            # Check for volume spikes (potential data errors)
            if len(data) > 5:
                volume_ma = data['Volume'].rolling(5).mean()

                # 10x average
                volume_spikes = (data['Volume'] > volume_ma * 10).sum()
                if volume_spikes > 0:
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        level=ValidationLevel.WARNING,
                        message=f"Volume spikes detected for {symbol}: {volume_spikes} occurrences",
                        symbol=symbol,
                        metadata={"volume_spikes": int(volume_spikes)}
                    ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error validating volume for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
    def _validate_completeness(self, symbol: str, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data completeness."""
        issues = []
        try:
            critical_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            important_cols = ['Volume']
            issues.extend(self._validate_critical_columns(symbol, data, critical_cols))
            issues.extend(self._validate_important_columns(symbol, data, important_cols))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error validating completeness for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
    def _validate_critical_columns(self, symbol: str, data: pd.DataFrame,
                                   critical_cols: List[str]) -> List[ValidationIssue]:
        """Validate critical columns for completeness."""
        issues = []
        for col in critical_cols:
            if col not in data.columns:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.CRITICAL,
                    message=f"Critical column missing for {symbol}: {col}",
                    symbol=symbol,
                    metadata={"missing_critical_column": col}
                ))
            else:
                missing_stats = self._get_missing_stats(data[col])
                if missing_stats['count'] > 0:
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        level=ValidationLevel.ERROR,
                        message=f"Missing critical data for {symbol}: {col} has {missing_stats['count']} NaN values",
                        symbol=symbol,
                        metadata={"missing_critical_data": missing_stats}
                    ))
        return issues
    def _validate_important_columns(self, symbol: str, data: pd.DataFrame,
                                    important_cols: List[str]) -> List[ValidationIssue]:
        """Validate important columns for completeness."""
        issues = []
        for col in important_cols:
            if col in data.columns:
                missing_stats = self._get_missing_stats(data[col])
                data_type = "volume" if col == "Volume" else col.lower()

                # >5% missing volume is concerning
                if missing_stats['pct'] > 5:
                    level = ValidationLevel.WARNING if missing_stats['pct'] <= 20 else ValidationLevel.ERROR
                    message_text = (f"Missing {data_type} data: {col} has {missing_stats['count']} "
                                   f"NaN values ({missing_stats['pct']:.1f}%)")
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        level=level,
                        message=message_text,
                        symbol=symbol,
                        metadata={"missing_important_data": missing_stats}
                    ))
        return issues
    def _get_missing_stats(self, series: pd.Series) -> Dict[str, float]:
        """Get statistics about missing values in a series."""
        missing_count = series.isna().sum()
        total_count = len(series)
        missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
        return {
            "count": int(missing_count),
            "total": int(total_count),
            "pct": missing_pct
        }
    def _detect_price_anomalies(self, symbol: str, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detect price anomalies and extreme movements."""
        issues = []
        try:
            if 'Close' not in data.columns or len(data) < 2:
                return issues

            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            if len(returns) == 0:
                return issues

            # 50% change
            extreme_threshold = 0.5
            extreme_moves = (abs(returns) > extreme_threshold).sum()
            if extreme_moves > 0:
                max_return = returns.abs().max()
                extreme_msg = (f"Extreme price movements: {extreme_moves} days with "
                              f">{extreme_threshold*100}% change (max: {max_return:.1%})")
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.WARNING,
                    message=extreme_msg,
                    symbol=symbol,
                    metadata={
                        "extreme_moves": int(extreme_moves),
                        "max_return": float(max_return),
                        "threshold": extreme_threshold
                    }
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error detecting price anomalies for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
    def _validate_data_sequence(self, symbol: str, data: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data sequence and ordering."""
        issues = []
        try:
            if 'Date' not in data.columns or len(data) < 2:
                return issues

            # Convert dates if needed
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                dates = pd.to_datetime(data['Date'], errors='coerce')
            else:
                dates = data['Date']

            # Check for date duplicates
            duplicates = dates.duplicated()
            if duplicates.any():
                dup_count = duplicates.sum()
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.ERROR,
                    message=f"Duplicate dates found for {symbol}: {dup_count} duplicates",
                    symbol=symbol,
                    metadata={"duplicate_dates": int(dup_count)}
                ))

            # Check if data is properly sorted
            if not dates.is_monotonic_increasing:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.WARNING,
                    message=f"Data not sorted by date for {symbol}",
                    symbol=symbol,
                    metadata={"data_ordering": "not_sorted"}
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error validating data sequence for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
class CrossProviderDiscrepancyCheck(ValidationRule):
    """Detect discrepancies between multiple data providers."""
    def __init__(self, tolerance_pct: float = 2.0, **kwargs) -> None:
        super().__init__("cross_provider_discrepancy", **kwargs)
        self.tolerance_pct = tolerance_pct
    def validate_multiple_sources(self, symbol: str, data_sources: Dict[str, pd.DataFrame]) -> List[ValidationIssue]:
        """Validate data across multiple sources."""
        issues = []
        if len(data_sources) < 2:
            return issues

        # Compare all pairs of sources
        sources = List[str](data_sources.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                data1, data2 = data_sources[source1], data_sources[source2]
                pair_issues = self._compare_data_sources(symbol, source1, data1,
                                                        source2, data2)
                issues.extend(pair_issues)
        return issues
    def validate(self, symbol: str, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[ValidationIssue]:
        """Standard validation interface - requires multiple sources in metadata."""
        if not metadata or 'additional_sources' not in metadata:
            return []
        data_sources = {'primary': data}
        data_sources.update(metadata['additional_sources'])
        return self.validate_multiple_sources(symbol, data_sources)
    def _compare_data_sources(self, symbol: str, source1: str, data1: pd.DataFrame,
                              source2: str, data2: pd.DataFrame) -> List[ValidationIssue]:
        """Compare two data sources for discrepancies."""
        issues = []
        try:

            # Find common dates
            common_dates = self._find_common_dates(data1, data2)
            if not common_dates:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    level=ValidationLevel.WARNING,
                    message=f"No common dates between {source1} and {source2} for {symbol}",
                    symbol=symbol,
                    metadata={"source1": source1, "source2": source2}
                ))
                return issues

            # Filter to common dates
            filtered_data1 = self._filter_to_common_dates(data1, common_dates)
            filtered_data2 = self._filter_to_common_dates(data2, common_dates)

            # Check price discrepancies
            price_issues = self._check_price_discrepancies(symbol, source1, filtered_data1,
                                                          source2, filtered_data2)
            issues.extend(price_issues)

            # Check volume discrepancies
            volume_issues = self._check_volume_discrepancies(symbol, source1, filtered_data1,
                                                            source2, filtered_data2)
            issues.extend(volume_issues)
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error comparing {source1} and {source2} for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
    def _extract_dates(self, data: pd.DataFrame) -> Optional[List[datetime]]:
        """Extract dates from dataframe."""
        if 'Date' not in data.columns:
            return None
        try:
            dates = self._convert_to_datetime(data)
            return dates.dropna().dt.to_pydatetime().tolist() if dates is not None else None
        except Exception:
            return None
    def _convert_to_datetime(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Convert Date column to datetime."""
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            return data['Date']
        else:
            return pd.to_datetime(data['Date'], errors='coerce')
    def _filter_to_common_dates(self, data: pd.DataFrame, common_dates: List[datetime]) -> pd.DataFrame:
        """Filter dataframe to common dates."""
        if 'Date' not in data.columns:
            return data.copy()
        try:
            if pd.api.types.is_datetime64_any_dtype(data['Date']):
                dates = data['Date']
            else:
                dates = pd.to_datetime(data['Date'], errors='coerce')
            common_dates_set = Set[str](common_dates)
            mask = dates.dt.to_pydatetime().isin(common_dates_set)
            return data[mask].copy()
        except Exception:
            return data.copy()
    def _find_common_dates(self, data1: pd.DataFrame, data2: pd.DataFrame) -> List[datetime]:
        """Find common dates between two dataframes."""
        dates1 = self._extract_dates(data1)
        dates2 = self._extract_dates(data2)
        if not dates1 or not dates2:
            return []
        common = Set[str](dates1) & Set[str](dates2)
        return sorted(common)
    def _check_price_discrepancies(self, symbol: str, source1: str, data1: pd.DataFrame,
                                   source2: str, data2: pd.DataFrame) -> List[ValidationIssue]:
        """Check for price discrepancies between sources."""
        issues = []
        try:
            price_cols = ['Open', 'High', 'Low', 'Close']
            price_cols = [col for col in price_cols if col in data1.columns and col in data2.columns]
            if not price_cols:
                return issues

            # Merge on Date for comparison
            merged = pd.merge(data1, data2, on='Date', suffixes=('_1', '_2'), how='inner', validate='one_to_one')
            if merged.empty:
                return issues
            for col in price_cols:
                col1, col2 = f"{col}_1", f"{col}_2"
                if col1 in merged.columns and col2 in merged.columns:

                    # Calculate percentage difference
                    diff_pct = abs((merged[col1] - merged[col2]) / merged[col1] * 100)
                    discrepant = diff_pct > self.tolerance_pct
                    discrepant_count = discrepant.sum()
                    if discrepant_count > 0:
                        discrepancy_pct = (discrepant_count / len(merged)) * 100
                        max_diff = diff_pct.max()
                        avg_diff = diff_pct[discrepant].mean()
                        level = ValidationLevel.CRITICAL if discrepancy_pct > 10 else ValidationLevel.WARNING
                        message_text = (f"Price discrepancies between {source1} and {source2}: "
                                       f"{discrepant_count}/{len(merged)} days ({discrepancy_pct:.1f}%) "
                                       f"exceed {self.tolerance_pct}% tolerance "
                                       f"(max: {max_diff:.1f}%, avg: {avg_diff:.1f}%)")
                        issues.append(ValidationIssue(
                            rule_name=self.name,
                            level=level,
                            message=message_text,
                            symbol=symbol,
                            metadata={
                                "price_column": col,
                                "source1": source1,
                                "source2": source2,
                                "discrepant_days": int(discrepant_count),
                                "total_days": int(len(merged)),
                                "discrepancy_percentage": discrepancy_pct,
                                "max_difference": float(max_diff),
                                "avg_difference": float(avg_diff),
                                "tolerance": self.tolerance_pct
                            }
                        ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error checking price discrepancies for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
    def _check_volume_discrepancies(self, symbol: str, source1: str, data1: pd.DataFrame,
                                    source2: str, data2: pd.DataFrame) -> List[ValidationIssue]:
        """Check for volume discrepancies between sources."""
        issues = []
        try:
            if 'Volume' not in data1.columns or 'Volume' not in data2.columns:
                return issues

            # Volume often differs significantly
            volume_tolerance = self.tolerance_pct * 10

            # Merge on Date for comparison
            volume_data = pd.merge(data1[['Date', 'Volume']], data2[['Date', 'Volume']],
                                  on='Date', suffixes=('_1', '_2'), how='inner', validate='one_to_one')
            if volume_data.empty:
                return issues

            # Calculate percentage difference (avoid division by zero)
            vol1_nonzero = volume_data['Volume_1'] != 0
            vol2_nonzero = volume_data['Volume_2'] != 0
            both_nonzero = vol1_nonzero & vol2_nonzero
            if both_nonzero.any():
                valid_data = volume_data[both_nonzero]
                diff_pct = abs((valid_data['Volume_1'] - valid_data['Volume_2']) / valid_data['Volume_1'] * 100)
                discrepant = diff_pct > volume_tolerance
                discrepant_count = discrepant.sum()
                if discrepant_count > 0:
                    discrepancy_pct = (discrepant_count / len(valid_data)) * 100
                    max_diff = diff_pct.max()
                    level = ValidationLevel.WARNING if discrepancy_pct <= 25 else ValidationLevel.ERROR
                    volume_message = (f"Volume discrepancies between {source1} and {source2}: "
                                     f"{discrepant_count}/{len(volume_data)} days ({discrepancy_pct:.1f}%) "
                                     f"exceed {volume_tolerance}% tolerance (max: {max_diff:.1f}%)")
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        level=level,
                        message=volume_message,
                        symbol=symbol,
                        metadata={
                            "volume_discrepancies": {
                                "source1": source1,
                                "source2": source2,
                                "discrepant_days": int(discrepant_count),
                                "total_days": int(len(volume_data)),
                                "discrepancy_percentage": discrepancy_pct,
                                "max_difference": float(max_diff),
                                "tolerance": volume_tolerance
                            }
                        }
                    ))
        except Exception as e:
            issues.append(ValidationIssue(
                rule_name=self.name,
                level=ValidationLevel.ERROR,
                message=f"Error checking volume discrepancies for {symbol}: {str(e)}",
                symbol=symbol,
                metadata={"error": str(e)}
            ))
        return issues
