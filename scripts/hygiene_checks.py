#!/usr/bin/env python3
"""
Custom Hygiene Checks for Pre-commit Hooks

This script provides specific hygiene checks that can be run as pre-commit hooks
to prevent code hygiene issues from being committed.

Features:
- Check for print statements (should use logging)
- Check for placeholder implementations (...)
- Check for duplicate imports
- Check for incomplete docstrings
- Integrates with pre-commit framework
"""

from pathlib import Path
import argparse
import logging
import re
import sys

from typing import List, Set, Dict, Tuple
import ast

# Suppress logging by default for pre-commit
logging.basicConfig(level=logging.ERROR)


class HygieneChecker:
    """Performs specific hygiene checks on Python files"""

    def __init__(self):
        self.errors_found = 0

    def check_print_statements(self, file_paths: List[str]) -> int:
        """Check for print statements that should be logging calls"""
        errors = 0

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    # Skip comments and strings
                    if line.strip().startswith('#'):
                        continue

                    # Look for print statements (but allow some exceptions)
                    if re.search(r'\bprint\s*\(', line):
                        # Allow prints in specific contexts
                        if any(context in line.lower() for context in [
                            'debug', 'test', 'main()', '__main__',
                            'example', 'demo', 'cli'
                        ]):
                            continue

                        # Allow prints with logger context
                        if 'logger' in line or 'log' in line.lower():
                            continue

                        print(f"{file_path}:{i}: Found print statement - use logging instead")
                        print(f"  {line.strip()}")
                        errors += 1

            except Exception as e:
                logging.error(f"Error checking {file_path}: {e}")
                errors += 1

        return errors

    def check_placeholder_implementations(self, file_paths: List[str]) -> int:
        """Check for placeholder implementations with ellipsis"""
        errors = 0

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse AST to find functions with placeholders
                try:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if function has placeholder implementation
                            if self._has_placeholder(node):
                                print(f"{file_path}:{node.lineno}: Function '{node.name}' has placeholder implementation")
                                errors += 1

                except SyntaxError:
                    # File has syntax errors, skip AST check
                    lines = content.splitlines()
                    for i, line in enumerate(lines, 1):
                        if re.search(r'\.\.\.|pass\s*#.*TODO', line):
                            print(f"{file_path}:{i}: Found placeholder implementation")
                            errors += 1

            except Exception as e:
                logging.error(f"Error checking {file_path}: {e}")
                errors += 1

        return errors

    def check_duplicate_imports(self, file_paths: List[str]) -> int:
        """Check for duplicate import statements"""
        errors = 0

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.splitlines()
                seen_imports = set()

                for i, line in enumerate(lines, 1):
                    stripped = line.strip()

                    # Check import statements
                    if stripped.startswith(('import ', 'from ')) and ' import ' in stripped:
                        # Normalize the import
                        normalized = self._normalize_import(stripped)

                        if normalized in seen_imports:
                            print(f"{file_path}:{i}: Duplicate import - {stripped}")
                            errors += 1
                        else:
                            seen_imports.add(normalized)

            except Exception as e:
                logging.error(f"Error checking {file_path}: {e}")
                errors += 1

        return errors

    def check_docstrings(self, file_paths: List[str]) -> int:
        """Check for missing or incomplete docstrings"""
        errors = 0

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                try:
                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            # Skip private functions/classes
                            if node.name.startswith('_'):
                                continue

                            # Check for docstring
                            docstring = ast.get_docstring(node)

                            if not docstring:
                                node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                                print(f"{file_path}:{node.lineno}: {node_type} '{node.name}' missing docstring")
                                errors += 1
                            elif len(docstring.strip()) < 10:
                                node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                                print(f"{file_path}:{node.lineno}: {node_type} '{node.name}' has minimal docstring")
                                errors += 1

                except SyntaxError:
                    # Skip files with syntax errors
                    pass

            except Exception as e:
                logging.error(f"Error checking {file_path}: {e}")
                errors += 1

        return errors

    def _has_placeholder(self, node: ast.FunctionDef) -> bool:
        """Check if function has placeholder implementation"""
        for child in ast.walk(node):
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Ellipsis):
                return True
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                if child.value.value in ("...", Ellipsis):
                    return True
        return False

    def _normalize_import(self, import_stmt: str) -> str:
        """Normalize import statement for comparison"""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', import_stmt.strip())

        # Remove comments
        if '#' in normalized:
            normalized = normalized.split('#')[0].strip()

        return normalized


def main():
    """Main entry point for hygiene checks"""
    parser = argparse.ArgumentParser(description="Run hygiene checks on Python files")
    parser.add_argument("--check-prints", action="store_true",
                       help="Check for print statements")
    parser.add_argument("--check-placeholders", action="store_true",
                       help="Check for placeholder implementations")
    parser.add_argument("--check-duplicates", action="store_true",
                       help="Check for duplicate imports")
    parser.add_argument("--check-docstrings", action="store_true",
                       help="Check for missing docstrings")
    parser.add_argument("files", nargs="*", help="Files to check")

    args = parser.parse_args()

    if not args.files:
        return 0

    # Filter for Python files only
    python_files = [f for f in args.files if f.endswith('.py')]

    if not python_files:
        return 0

    checker = HygieneChecker()
    total_errors = 0

    if args.check_prints:
        errors = checker.check_print_statements(python_files)
        total_errors += errors
        if errors > 0:
            logging.error(f"\n❌ Found {errors} print statement(s) that should use logging")

    if args.check_placeholders:
        errors = checker.check_placeholder_implementations(python_files)
        total_errors += errors
        if errors > 0:
            logging.error(f"\n❌ Found {errors} placeholder implementation(s)")

    if args.check_duplicates:
        errors = checker.check_duplicate_imports(python_files)
        total_errors += errors
        if errors > 0:
            logging.error(f"\n❌ Found {errors} duplicate import(s)")

    if args.check_docstrings:
        errors = checker.check_docstrings(python_files)
        total_errors += errors
        if errors > 0:
            logging.error(f"\n❌ Found {errors} missing/incomplete docstring(s)")

    if total_errors == 0:
        print("✅ All hygiene checks passed")

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())