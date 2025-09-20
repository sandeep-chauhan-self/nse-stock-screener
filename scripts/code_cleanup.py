#!/usr/bin/env python3
"""
Automated Code Cleanup Tool

This script provides automated fixes for common code hygiene issues identified by the analyzer.
Safely applies fixes with backup and validation.

Features:
- Remove duplicate imports
- Fix placeholder implementations
- Standardize import organization 
- Clean up style issues
- Generate safe backup before changes
- Validate changes don't break syntax
"""

from datetime import datetime
from pathlib import Path
import logging
import re

from typing import Dict, List, Set, Tuple, Optional
import ast
import shutil

from .code_hygiene_analyzer import CodeHygieneAnalyzer, HygieneIssue, FileAnalysis

logger = logging.getLogger(__name__)


class CodeCleaner:
    """Automated code cleanup with safety validations"""
    
    def __init__(self, root_dir: Path, dry_run: bool = True):
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.backup_dir = self.root_dir / "backup" / f"hygiene_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.changes_made = []
        
    def apply_fixes(self, analyzer: CodeHygieneAnalyzer) -> Dict[str, int]:
        """Apply automated fixes for hygiene issues"""
        if not self.dry_run:
            self._create_backup()
            
        fixes_applied = {
            "duplicate_imports": 0,
            "print_statements": 0,
            "placeholder_functions": 0,
            "import_organization": 0,
            "total_files_modified": 0
        }
        
        for file_path, analysis in analyzer.file_analyses.items():
            file_modified = False
            
            # Apply file-level fixes
            if self._fix_file_issues(file_path, analysis):
                file_modified = True
                
            if file_modified:
                fixes_applied["total_files_modified"] += 1
                
        logger.info(f"Applied fixes: {fixes_applied}")
        return fixes_applied
        
    def _create_backup(self):
        """Create backup of all files before modification"""
        logger.info(f"Creating backup in: {self.backup_dir}")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy entire src directory
        src_backup = self.backup_dir / "src"
        shutil.copytree(self.root_dir / "src", src_backup)
        
        logger.info("Backup created successfully")
        
    def _fix_file_issues(self, file_path: str, analysis: FileAnalysis) -> bool:
        """Fix issues in a single file"""
        path = Path(file_path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                lines = original_content.splitlines()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return False
            
        modified_lines = lines.copy()
        changes_made = False
        
        # Fix duplicate imports
        if self._fix_duplicate_imports(modified_lines, analysis):
            changes_made = True
            
        # Fix print statements
        if self._fix_print_statements(modified_lines):
            changes_made = True
            
        # Fix placeholder implementations
        if self._fix_placeholder_implementations(modified_lines, analysis):
            changes_made = True
            
        # Organize imports
        if self._organize_imports(modified_lines):
            changes_made = True
            
        if changes_made and not self.dry_run:
            # Validate syntax before writing
            new_content = '\n'.join(modified_lines)
            if self._validate_syntax(new_content, file_path):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Fixed issues in: {path.relative_to(self.root_dir)}")
                self.changes_made.append(str(path))
            else:
                logger.error(f"Syntax validation failed for {file_path}, skipping")
                return False
        elif changes_made:
            logger.info(f"[DRY RUN] Would fix issues in: {path.relative_to(self.root_dir)}")
            
        return changes_made
        
    def _fix_duplicate_imports(self, lines: List[str], analysis: FileAnalysis) -> bool:
        """Remove duplicate import statements"""
        duplicate_issues = [
            issue for issue in analysis.issues 
            if issue.issue_type == "duplicate_import"
        ]
        
        if not duplicate_issues:
            return False
            
        lines_to_remove = set()
        for issue in duplicate_issues:
            # Mark line for removal (1-based to 0-based index)
            lines_to_remove.add(issue.line_number - 1)
            
        # Remove lines in reverse order to maintain indices
        for line_idx in sorted(lines_to_remove, reverse=True):
            if 0 <= line_idx < len(lines):
                logger.debug(f"Removing duplicate import: {lines[line_idx].strip()}")
                lines.pop(line_idx)
                
        return len(lines_to_remove) > 0
        
    def _fix_print_statements(self, lines: List[str]) -> bool:
        """Replace print statements with logging calls"""
        changes_made = False
        logging_imported = False
        
        # Check if logging is already imported
        for line in lines:
            if re.search(r'import logging|from.*logging', line):
                logging_imported = True
                break
                
        for i, line in enumerate(lines):
            # Look for print statements
            print_match = re.search(r'(\s*)print\s*\((.*)\)', line)
            if print_match and 'logger' not in line:
                indent = print_match.group(1)
                content = print_match.group(2)
                
                # Add logging import if not present
                if not logging_imported:
                    # Find good place to add import (after other imports)
                    import_line = 0
                    for j, import_line_content in enumerate(lines):
                        if import_line_content.strip().startswith(('import ', 'from ')):
                            import_line = j + 1
                    lines.insert(import_line, "import logging")
                    lines.insert(import_line + 1, "logger = logging.getLogger(__name__)")
                    logging_imported = True
                    i += 2  # Adjust index for inserted lines
                    
                # Replace print with logger.info
                new_line = f"{indent}logger.info({content})"
                lines[i] = new_line
                changes_made = True
                logger.debug(f"Replaced print statement: {line.strip()} -> {new_line.strip()}")
                
        return changes_made
        
    def _fix_placeholder_implementations(self, lines: List[str], analysis: FileAnalysis) -> bool:
        """Fix placeholder implementations with proper TODO markers"""
        placeholder_issues = [
            issue for issue in analysis.issues 
            if issue.issue_type in ("placeholder_function", "empty_function")
        ]
        
        if not placeholder_issues:
            return False
            
        changes_made = False
        
        for issue in placeholder_issues:
            line_idx = issue.line_number - 1
            if line_idx < len(lines):
                # Find the function and replace placeholder
                func_line = lines[line_idx]
                
                # Look for the placeholder in subsequent lines
                for j in range(line_idx + 1, min(line_idx + 10, len(lines))):
                    if '...' in lines[j] or lines[j].strip() == 'pass':
                        # Get indentation
                        indent = len(lines[j]) - len(lines[j].lstrip())
                        indent_str = ' ' * indent
                        
                        # Replace with proper TODO
                        lines[j] = f'{indent_str}# TODO: Implement {issue.description.split("'")[1] if "'" in issue.description else "this function"}'
                        lines.insert(j + 1, f'{indent_str}raise NotImplementedError("Function not yet implemented")')
                        changes_made = True
                        logger.debug(f"Fixed placeholder in function at line {issue.line_number}")
                        break
                        
        return changes_made
        
    def _organize_imports(self, lines: List[str]) -> bool:
        """Organize imports according to PEP8 style"""
        import_lines = []
        from_import_lines = []
        other_lines = []
        
        in_imports_section = True
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('import ') and not stripped.startswith('import.'):
                if in_imports_section:
                    import_lines.append(line)
                else:
                    other_lines.append(line)
            elif stripped.startswith('from ') and ' import ' in stripped:
                if in_imports_section:
                    from_import_lines.append(line)
                else:
                    other_lines.append(line)
            else:
                if stripped and not stripped.startswith('#') and stripped != '"""' and not stripped.startswith('"""'):
                    in_imports_section = False
                other_lines.append(line)
                
        # Check if reorganization is needed
        original_import_section = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and ' import ' in stripped:
                original_import_section.append(line)
            elif stripped and not stripped.startswith('#'):
                break
                
        # Sort imports
        import_lines.sort(key=lambda x: x.strip().lower())
        from_import_lines.sort(key=lambda x: x.strip().lower())
        
        # Reconstruct organized imports
        organized_imports = []
        if import_lines:
            organized_imports.extend(import_lines)
        if from_import_lines:
            if import_lines:
                organized_imports.append('')  # Blank line between import groups
            organized_imports.extend(from_import_lines)
            
        # Check if changes were made
        new_import_section = [line for line in organized_imports if line.strip()]
        original_import_section = [line for line in original_import_section if line.strip()]
        
        if new_import_section != original_import_section:
            # Rebuild the file with organized imports
            lines.clear()
            
            # Add file header comments
            for line in other_lines:
                if line.strip().startswith('#') or line.strip().startswith('"""') or not line.strip():
                    lines.append(line)
                else:
                    break
                    
            # Add organized imports
            lines.extend(organized_imports)
            if organized_imports:
                lines.append('')  # Blank line after imports
                
            # Add rest of the file
            found_content = False
            for line in other_lines:
                if not found_content and (line.strip().startswith('#') or line.strip().startswith('"""') or not line.strip()):
                    continue
                found_content = True
                lines.append(line)
                
            return True
            
        return False
        
    def _validate_syntax(self, content: str, file_path: str) -> bool:
        """Validate that the modified content has valid Python syntax"""
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in modified {file_path}: {e}")
            return False


def create_docstring_templates() -> Dict[str, str]:
    """Create templates for standardized docstrings"""
    return {
        "function": '''"""
    {description}
    
    Args:
        {args}
    
    Returns:
        {returns}
        
    Raises:
        {raises}
    """''',
        
        "class": '''"""
    {description}
    
    Attributes:
        {attributes}
    """''',
        
        "method": '''"""
    {description}
    
    Args:
        {args}
    
    Returns:
        {returns}
    """'''
    }


def generate_docstring_for_function(func_info, template: str) -> str:
    """Generate a proper docstring for a function"""
    # Extract function information
    description = f"TODO: Add description for {func_info.name}"
    
    # Generate args documentation
    args_doc = []
    for param in func_info.parameters:
        if param != 'self':
            args_doc.append(f"{param}: TODO: Describe parameter")
    args_text = "\n        ".join(args_doc) if args_doc else "None"
    
    # Generate return documentation
    returns_text = func_info.return_annotation or "TODO: Describe return value"
    
    # Generate raises documentation  
    raises_text = "TODO: Document exceptions that may be raised"
    
    return template.format(
        description=description,
        args=args_text,
        returns=returns_text,
        raises=raises_text
    )


def main():
    """Main entry point for code cleanup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated code cleanup tool")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                       help="Root directory of the project")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be changed without making changes")
    parser.add_argument("--apply", action="store_true",
                       help="Actually apply the changes (overrides --dry-run)")
    parser.add_argument("--backup", action="store_true", default=True,
                       help="Create backup before applying changes")
    
    args = parser.parse_args()
    
    # Determine if this is a dry run
    dry_run = args.dry_run and not args.apply
    
    if not dry_run:
        logger.warning("This will modify your source code. Make sure you have a backup!")
        confirmation = input("Continue? (y/N): ")
        if confirmation.lower() != 'y':
            logger.info("Operation cancelled")
            return 0
            
    logger.info(f"Running code cleanup {'(DRY RUN)' if dry_run else '(APPLYING CHANGES)'}")
    
    # First, analyze the code
    analyzer = CodeHygieneAnalyzer(args.root)
    analyzer.analyze_project()
    
    # Apply fixes
    cleaner = CodeCleaner(args.root, dry_run=dry_run)
    fixes = cleaner.apply_fixes(analyzer)
    
    # Report results
    print(f"\n🧹 Cleanup Results:")
    print(f"📁 Files modified: {fixes['total_files_modified']}")
    print(f"🔄 Duplicate imports removed: {fixes['duplicate_imports']}")
    print(f"📝 Print statements converted: {fixes['print_statements']}")
    print(f"⚡ Placeholder functions fixed: {fixes['placeholder_functions']}")
    print(f"📚 Import sections organized: {fixes['import_organization']}")
    
    if not dry_run and fixes['total_files_modified'] > 0:
        print(f"💾 Backup created in: {cleaner.backup_dir}")
        print("🔍 Please review changes and run tests to ensure everything still works!")
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())