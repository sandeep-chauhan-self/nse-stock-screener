"""
Cross-platform path utilities for NSE Stock Screener

This module provides centralized path management using pathlib for consistent,
cross-platform behavior. All file operations should use these utilities to
ensure compatibility across Windows, Linux, and macOS.

Key Features:
- Repository-root-relative path resolution
- Cross-platform path handling with pathlib
- Command-line output argument support
- Consistent directory structure management
- Working directory independence

Usage:
    from common.paths import PathManager

    pm = PathManager()
    output_file = pm.get_data_path('nse_symbols.txt')
    pm.ensure_data_dir()
    output_file.write_text(data)
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict
import argparse


class PathManager:
    """
    Centralized path management for cross-platform compatibility.

    Automatically determines repository root and provides methods for
    accessing standard directories (data, output, etc.) in a portable way.
    """

    def __init__(self, custom_base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize path manager.

        Args:
            custom_base_dir: Optional custom base directory. If None, auto-detects
                           repository root based on current file location.
        """
        if custom_base_dir:
            self._base_dir = Path(custom_base_dir).resolve()
        else:
            # Auto-detect repository root by finding the directory containing 'src'
            self._base_dir = self._find_repo_root()

        # Standard directory structure
        self._standard_dirs = {
            'src': self._base_dir / 'src',
            'data': self._base_dir / 'data',
            'output': self._base_dir / 'output',
            'reports': self._base_dir / 'output' / 'reports',
            'charts': self._base_dir / 'output' / 'charts',
            'backtests': self._base_dir / 'output' / 'backtests',
            'docs': self._base_dir / 'docs',
            'scripts': self._base_dir / 'scripts',
            'temp': self._base_dir / 'data' / 'temp'
        }

    def _find_repo_root(self) -> Path:
        """
        Find repository root by locating directory containing 'src' folder.

        Returns:
            Path to repository root

        Raises:
            FileNotFoundError: If repository root cannot be determined
        """
        # Start from current file location and walk up
        current = Path(__file__).resolve().parent

        # If we're in src/common, go up two levels
        if current.name == 'common' and current.parent.name == 'src':
            return current.parent.parent

        # Otherwise, walk up looking for 'src' directory
        while current != current.parent:  # Stop at filesystem root
            if (current / 'src').is_dir():
                return current
            current = current.parent

        # Fallback: use the parent of the directory containing this file
        fallback = Path(__file__).resolve().parent.parent.parent
        if (fallback / 'src').is_dir():
            return fallback

        raise FileNotFoundError(
            f"Could not determine repository root. "
            f"Looked for 'src' directory starting from {Path(__file__).resolve()}"
        )

    @property
    def base_dir(self) -> Path:
        """Repository base directory."""
        return self._base_dir

    def get_data_path(self, filename: str) -> Path:
        """
        Get path to file in data directory.

        Args:
            filename: Name of file in data directory

        Returns:
            Full path to data file
        """
        return self._standard_dirs['data'] / filename

    def get_output_path(self, subdir: str, filename: str) -> Path:
        """
        Get path to file in output subdirectory.

        Args:
            subdir: Subdirectory name ('reports', 'charts', 'backtests')
            filename: Name of output file

        Returns:
            Full path to output file
        """
        if subdir not in ['reports', 'charts', 'backtests']:
            raise ValueError(f"Invalid output subdir: {subdir}. Must be one of: reports, charts, backtests")

        return self._standard_dirs[subdir] / filename

    def get_temp_path(self, filename: str) -> Path:
        """
        Get path to file in temp directory.

        Args:
            filename: Name of temp file

        Returns:
            Full path to temp file
        """
        return self._standard_dirs['temp'] / filename

    def get_standard_dir(self, dir_name: str) -> Path:
        """
        Get path to standard directory.

        Args:
            dir_name: Directory name ('data', 'output', 'reports', etc.)

        Returns:
            Full path to directory
        """
        if dir_name not in self._standard_dirs:
            available = ', '.join(self._standard_dirs.keys())
            raise ValueError(f"Unknown directory: {dir_name}. Available: {available}")

        return self._standard_dirs[dir_name]

    def ensure_dir(self, dir_path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, creating it if necessary.

        Args:
            dir_path: Path to directory

        Returns:
            Path object for the directory
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_data_dir(self) -> Path:
        """Ensure data directory exists."""
        return self.ensure_dir(self._standard_dirs['data'])

    def ensure_output_dirs(self) -> Dict[str, Path]:
        """
        Ensure all output directories exist.

        Returns:
            Dictionary mapping directory names to Path objects
        """
        output_dirs = {}
        for name in ['output', 'reports', 'charts', 'backtests']:
            output_dirs[name] = self.ensure_dir(self._standard_dirs[name])
        return output_dirs

    def ensure_temp_dir(self) -> Path:
        """Ensure temp directory exists."""
        return self.ensure_dir(self._standard_dirs['temp'])

    def resolve_path(self, path_str: str) -> Path:
        """
        Resolve a path string that may contain relative components.

        Handles legacy paths like "../data/file.txt" by converting them
        to proper repository-relative paths.

        Args:
            path_str: Path string (may be relative or absolute)

        Returns:
            Resolved absolute Path object
        """
        path = Path(path_str)

        # If it's already absolute, return as-is
        if path.is_absolute():
            return path

        # Handle common relative patterns
        if str(path).startswith("../data/"):
            # Convert "../data/file.txt" to proper data path
            filename = str(path)[8:]  # Remove "../data/"
            return self.get_data_path(filename)
        elif str(path).startswith("../output/"):
            # Convert "../output/subdir/file.txt" to proper output path
            parts = Path(str(path)[10:]).parts  # Remove "../output/"
            if len(parts) >= 2:
                return self.get_output_path(parts[0], '/'.join(parts[1:]))
            else:
                return self._standard_dirs['output'] / str(path)[10:]
        else:
            # Resolve relative to repository root
            return (self._base_dir / path).resolve()

    def get_working_dir_independent_path(self, target_path: Union[str, Path]) -> Path:
        """
        Get a path that works regardless of current working directory.

        This is crucial for scripts that may be called from different directories.

        Args:
            target_path: Target path (relative or absolute)

        Returns:
            Absolute path that works from any working directory
        """
        return self.resolve_path(str(target_path))


def add_output_argument(parser: argparse.ArgumentParser,
                       default_filename: str,
                       help_text: str = "Output file path") -> None:
    """
    Add standardized --output argument to argument parser.

    Args:
        parser: ArgumentParser instance
        default_filename: Default filename (will be placed in appropriate directory)
        help_text: Help text for the argument
    """
    parser.add_argument(
        '--output', '-o',
        type=str,
        help=f"{help_text}. Can be absolute path or relative to repository root. "
             f"Default: {default_filename}"
    )


def resolve_output_path(args_output: Optional[str],
                       default_filename: str,
                       output_type: str = 'data') -> Path:
    """
    Resolve output path from command line arguments.

    Args:
        args_output: Output path from command line (or None)
        default_filename: Default filename to use
        output_type: Type of output ('data', 'reports', 'charts', 'backtests')

    Returns:
        Resolved output path
    """
    pm = PathManager()

    if args_output:
        # User specified custom output path
        return pm.resolve_path(args_output)
    else:
        # Use default location
        if output_type == 'data':
            return pm.get_data_path(default_filename)
        elif output_type in ['reports', 'charts', 'backtests']:
            return pm.get_output_path(output_type, default_filename)
        else:
            raise ValueError(f"Invalid output_type: {output_type}")


# Global instance for convenience
DEFAULT_PATH_MANAGER = PathManager()

# Convenience functions that use the global instance
def get_data_path(filename: str) -> Path:
    """Get path to file in data directory."""
    return DEFAULT_PATH_MANAGER.get_data_path(filename)

def get_output_path(subdir: str, filename: str) -> Path:
    """Get path to file in output subdirectory."""
    return DEFAULT_PATH_MANAGER.get_output_path(subdir, filename)

def get_temp_path(filename: str) -> Path:
    """Get path to file in temp directory."""
    return DEFAULT_PATH_MANAGER.get_temp_path(filename)

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    return DEFAULT_PATH_MANAGER.ensure_dir(dir_path)

def resolve_path(path_str: str) -> Path:
    """Resolve legacy relative path to absolute path."""
    return DEFAULT_PATH_MANAGER.resolve_path(path_str)


# Export key components
__all__ = [
    'PathManager',
    'DEFAULT_PATH_MANAGER',
    'add_output_argument',
    'resolve_output_path',
    'get_data_path',
    'get_output_path',
    'get_temp_path',
    'ensure_dir',
    'resolve_path'
]


if __name__ == "__main__":
    # Test the path manager
    pm = PathManager()
    print(f"Repository root: {pm.base_dir}")
    print(f"Data directory: {pm.get_standard_dir('data')}")
    print(f"Output directory: {pm.get_standard_dir('output')}")
    print(f"Sample data path: {pm.get_data_path('nse_symbols.txt')}")
    print(f"Sample output path: {pm.get_output_path('reports', 'analysis.csv')}")

    # Test legacy path resolution
    print(f"Legacy path '../data/test.txt' resolves to: {pm.resolve_path('../data/test.txt')}")