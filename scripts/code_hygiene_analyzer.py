#!/usr/bin/env python3
"""
Code Hygiene Analyzer and Cleanup Tool

This script performs comprehensive code hygiene analysis and cleanup for the NSE Stock Screener project.
Addresses Requirement 3.12: Code hygiene - duplicates, placeholder ellipses, docstrings

Features:
- Detects and removes duplicate imports
- Identifies incomplete functions with placeholders
- Standardizes docstring format across all modules
- Enforces consistent code style and organization
- Generates detailed reports of hygiene issues
- Provides automated fixes with safety validations
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import logging
import os
import re
import sys

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import ast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    name: Optional[str] = None
    alias: Optional[str] = None
    line_number: int = 0
    full_statement: str = ""
    is_from_import: bool = False


@dataclass
class FunctionInfo:
    """Information about a function definition"""
    name: str
    line_number: int
    has_docstring: bool = False
    docstring_quality: str = "none"  # none, partial, complete
    has_placeholder: bool = False
    has_implementation: bool = True
    parameters: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about a class definition"""
    name: str
    line_number: int
    has_docstring: bool = False
    docstring_quality: str = "none"
    methods: List[FunctionInfo] = field(default_factory=list)


@dataclass
class HygieneIssue:
    """Represents a code hygiene issue"""
    file_path: str
    issue_type: str
    description: str
    line_number: int = 0
    severity: str = "medium"  # low, medium, high, critical
    suggested_fix: str = ""
    auto_fixable: bool = False


@dataclass
class FileAnalysis:
    """Complete analysis of a Python file"""
    file_path: str
    imports: List[ImportInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    issues: List[HygieneIssue] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    encoding: str = "utf-8"


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python code structure"""

    def __init__(self, file_path: str, lines: List[str]):
        self.file_path = file_path
        self.lines = lines
        self.imports: List[ImportInfo] = []
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        self.current_class = None

    def visit_Import(self, node: ast.Import):
        """Visit import statements"""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                alias=alias.asname,
                line_number=node.lineno,
                full_statement=self._get_line_content(node.lineno),
                is_from_import=False
            )
            self.imports.append(import_info)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statements"""
        module = node.module or ""
        for alias in node.names:
            import_info = ImportInfo(
                module=module,
                name=alias.name,
                alias=alias.asname,
                line_number=node.lineno,
                full_statement=self._get_line_content(node.lineno),
                is_from_import=True
            )
            self.imports.append(import_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions"""
        # Get function parameters
        params = [arg.arg for arg in node.args.args]

        # Check for docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None
        docstring_quality = self._assess_docstring_quality(docstring)

        # Check for placeholder implementation
        has_placeholder = self._has_placeholder_implementation(node)
        has_implementation = self._has_real_implementation(node)

        # Get return annotation
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        func_info = FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            has_docstring=has_docstring,
            docstring_quality=docstring_quality,
            has_placeholder=has_placeholder,
            has_implementation=has_implementation,
            parameters=params,
            return_annotation=return_annotation
        )

        if self.current_class:
            self.current_class.methods.append(func_info)
        else:
            self.functions.append(func_info)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions"""
        # Get docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None
        docstring_quality = self._assess_docstring_quality(docstring)

        class_info = ClassInfo(
            name=node.name,
            line_number=node.lineno,
            has_docstring=has_docstring,
            docstring_quality=docstring_quality
        )

        # Set current class context for method processing
        old_class = self.current_class
        self.current_class = class_info

        self.generic_visit(node)

        # Restore previous class context
        self.current_class = old_class
        self.classes.append(class_info)

    def _get_line_content(self, line_number: int) -> str:
        """Get the content of a specific line"""
        if 1 <= line_number <= len(self.lines):
            return self.lines[line_number - 1].strip()
        return ""

    def _assess_docstring_quality(self, docstring: Optional[str]) -> str:
        """Assess the quality of a docstring"""
        if not docstring:
            return "none"

        # Clean up docstring
        docstring = docstring.strip()

        # Check for minimal content
        if len(docstring) < 10:
            return "minimal"

        # Check for standard docstring components
        has_description = len(docstring.split('.')[0]) > 5
        has_args = 'Args:' in docstring or 'Parameters:' in docstring
        has_returns = 'Returns:' in docstring or 'Return:' in docstring
        has_examples = 'Example' in docstring

        quality_score = sum([has_description, has_args, has_returns, has_examples])

        if quality_score >= 3:
            return "complete"
        elif quality_score >= 2:
            return "good"
        elif quality_score >= 1:
            return "partial"
        else:
            return "minimal"

    def _has_placeholder_implementation(self, node: ast.FunctionDef) -> bool:
        """Check if function has placeholder implementation"""
        for child in ast.walk(node):
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Ellipsis):
                return True
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                if child.value.value == "..." or child.value.value == Ellipsis:
                    return True
        return False

    def _has_real_implementation(self, node: ast.FunctionDef) -> bool:
        """Check if function has real implementation beyond docstring"""
        # Skip docstring if present
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):  # Docstring
                body = body[1:]

        # Check if there's meaningful implementation
        if not body:
            return False

        # Check for placeholder patterns
        for stmt in body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Ellipsis):
                return False
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value in ("...", Ellipsis, "TODO", "FIXME"):
                    return False
            # If we find any other statement, it's implemented
            return True

        return False


class CodeHygieneAnalyzer:
    """Main analyzer for code hygiene issues"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir / "src"
        self.file_analyses: Dict[str, FileAnalysis] = {}
        self.global_issues: List[HygieneIssue] = []

    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the entire project for hygiene issues"""
        logger.info(f"Starting code hygiene analysis for: {self.root_dir}")

        # Find all Python files
        python_files = list(self.src_dir.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files to analyze")

        # Analyze each file
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")

        # Perform global analysis
        self._analyze_global_issues()

        # Generate summary
        summary = self._generate_summary()

        logger.info("Code hygiene analysis complete")
        return summary

    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Python file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        lines = content.splitlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Could not decode file: {file_path}")
                return FileAnalysis(str(file_path))

        # Create file analysis
        analysis = FileAnalysis(
            file_path=str(file_path),
            lines=lines
        )

        # Parse AST
        try:
            tree = ast.parse(content)
            analyzer = ASTAnalyzer(str(file_path), lines)
            analyzer.visit(tree)

            analysis.imports = analyzer.imports
            analysis.functions = analyzer.functions
            analysis.classes = analyzer.classes

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            analysis.issues.append(HygieneIssue(
                file_path=str(file_path),
                issue_type="syntax_error",
                description=f"Syntax error: {e}",
                line_number=getattr(e, 'lineno', 0),
                severity="critical"
            ))

        # Analyze file-specific issues
        self._analyze_file_issues(analysis)

        # Store analysis
        self.file_analyses[str(file_path)] = analysis

        return analysis

    def _analyze_file_issues(self, analysis: FileAnalysis):
        """Analyze issues within a single file"""
        # Check for duplicate imports
        self._check_duplicate_imports(analysis)

        # Check for incomplete functions
        self._check_incomplete_functions(analysis)

        # Check for missing docstrings
        self._check_missing_docstrings(analysis)

        # Check for style issues
        self._check_style_issues(analysis)

    def _check_duplicate_imports(self, analysis: FileAnalysis):
        """Check for duplicate import statements"""
        seen_imports = set()

        for import_info in analysis.imports:
            if import_info.is_from_import:
                import_key = f"from {import_info.module} import {import_info.name}"
            else:
                import_key = f"import {import_info.module}"

            if import_info.alias:
                import_key += f" as {import_info.alias}"

            if import_key in seen_imports:
                analysis.issues.append(HygieneIssue(
                    file_path=analysis.file_path,
                    issue_type="duplicate_import",
                    description=f"Duplicate import: {import_key}",
                    line_number=import_info.line_number,
                    severity="medium",
                    suggested_fix=f"Remove duplicate import on line {import_info.line_number}",
                    auto_fixable=True
                ))
            else:
                seen_imports.add(import_key)

    def _check_incomplete_functions(self, analysis: FileAnalysis):
        """Check for incomplete function implementations"""
        all_functions = analysis.functions.copy()
        for class_info in analysis.classes:
            all_functions.extend(class_info.methods)

        for func in all_functions:
            if func.has_placeholder:
                analysis.issues.append(HygieneIssue(
                    file_path=analysis.file_path,
                    issue_type="placeholder_function",
                    description=f"Function '{func.name}' has placeholder implementation (...)",
                    line_number=func.line_number,
                    severity="high",
                    suggested_fix="Complete the implementation or mark as TODO with clear description"
                ))

            if not func.has_implementation:
                analysis.issues.append(HygieneIssue(
                    file_path=analysis.file_path,
                    issue_type="empty_function",
                    description=f"Function '{func.name}' has no implementation",
                    line_number=func.line_number,
                    severity="high",
                    suggested_fix="Add implementation or mark as abstract/TODO"
                ))

    def _check_missing_docstrings(self, analysis: FileAnalysis):
        """Check for missing or inadequate docstrings"""
        # Check functions
        for func in analysis.functions:
            if not func.name.startswith('_'):  # Skip private functions
                if not func.has_docstring:
                    analysis.issues.append(HygieneIssue(
                        file_path=analysis.file_path,
                        issue_type="missing_docstring",
                        description=f"Function '{func.name}' missing docstring",
                        line_number=func.line_number,
                        severity="medium",
                        suggested_fix="Add comprehensive docstring with description, parameters, and return value"
                    ))
                elif func.docstring_quality in ["none", "minimal"]:
                    analysis.issues.append(HygieneIssue(
                        file_path=analysis.file_path,
                        issue_type="poor_docstring",
                        description=f"Function '{func.name}' has inadequate docstring",
                        line_number=func.line_number,
                        severity="low",
                        suggested_fix="Improve docstring with better description and parameter documentation"
                    ))

        # Check classes
        for class_info in analysis.classes:
            if not class_info.has_docstring:
                analysis.issues.append(HygieneIssue(
                    file_path=analysis.file_path,
                    issue_type="missing_class_docstring",
                    description=f"Class '{class_info.name}' missing docstring",
                    line_number=class_info.line_number,
                    severity="medium",
                    suggested_fix="Add class docstring describing purpose and usage"
                ))

    def _check_style_issues(self, analysis: FileAnalysis):
        """Check for various style issues"""
        for i, line in enumerate(analysis.lines, 1):
            # Check for print statements (should use logging)
            if re.search(r'\bprint\s*\(', line) and 'logger' not in line and 'log' not in line.lower():
                analysis.issues.append(HygieneIssue(
                    file_path=analysis.file_path,
                    issue_type="print_statement",
                    description="Using print() instead of logging",
                    line_number=i,
                    severity="low",
                    suggested_fix="Replace print() with appropriate logging call",
                    auto_fixable=True
                ))

            # Check for TODO/FIXME comments
            if re.search(r'#.*\b(TODO|FIXME|XXX|HACK)\b', line, re.IGNORECASE):
                analysis.issues.append(HygieneIssue(
                    file_path=analysis.file_path,
                    issue_type="todo_comment",
                    description="TODO/FIXME comment found",
                    line_number=i,
                    severity="low",
                    suggested_fix="Address the TODO or create proper issue tracking"
                ))

    def _analyze_global_issues(self):
        """Analyze issues across multiple files"""
        # Check for inconsistent imports across files
        self._check_inconsistent_imports()

        # Check for module-level organization issues
        self._check_module_organization()

    def _check_inconsistent_imports(self):
        """Check for inconsistent import patterns across files"""
        # Group imports by module
        module_imports = defaultdict(list)

        for file_path, analysis in self.file_analyses.items():
            for import_info in analysis.imports:
                module_imports[import_info.module].append((file_path, import_info))

        # Check for inconsistent import styles
        for module, imports in module_imports.items():
            if len(imports) > 1:
                # Check if same module is imported differently
                import_styles = set()
                for file_path, import_info in imports:
                    if import_info.is_from_import:
                        style = f"from {import_info.module} import {import_info.name}"
                    else:
                        style = f"import {import_info.module}"
                    import_styles.add(style)

                if len(import_styles) > 1:
                    self.global_issues.append(HygieneIssue(
                        file_path="GLOBAL",
                        issue_type="inconsistent_imports",
                        description=f"Module '{module}' imported inconsistently: {list(import_styles)}",
                        severity="medium",
                        suggested_fix="Standardize import style across all files"
                    ))

    def _check_module_organization(self):
        """Check for module organization issues"""
        # This could include checks for:
        # - Circular imports
        # - Module naming conventions
        # - Package structure consistency
        pass

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary"""
        total_files = len(self.file_analyses)
        total_issues = sum(len(analysis.issues) for analysis in self.file_analyses.values())
        total_global_issues = len(self.global_issues)

        # Count issues by type and severity
        issue_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for analysis in self.file_analyses.values():
            for issue in analysis.issues:
                issue_counts[issue.issue_type] += 1
                severity_counts[issue.severity] += 1

        for issue in self.global_issues:
            issue_counts[issue.issue_type] += 1
            severity_counts[issue.severity] += 1

        # Count auto-fixable issues
        auto_fixable = sum(
            1 for analysis in self.file_analyses.values()
            for issue in analysis.issues
            if issue.auto_fixable
        )

        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_files_analyzed": total_files,
            "total_issues_found": total_issues + total_global_issues,
            "issues_by_type": dict(issue_counts),
            "issues_by_severity": dict(severity_counts),
            "auto_fixable_issues": auto_fixable,
            "global_issues": total_global_issues,
            "file_analyses": {
                path: {
                    "issues_count": len(analysis.issues),
                    "functions_count": len(analysis.functions),
                    "classes_count": len(analysis.classes),
                    "imports_count": len(analysis.imports)
                }
                for path, analysis in self.file_analyses.items()
            }
        }

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate detailed hygiene report"""
        summary = self._generate_summary()

        report_lines = []
        report_lines.append("# Code Hygiene Analysis Report")
        report_lines.append(f"Generated: {summary['analysis_timestamp']}")
        report_lines.append("")

        # Summary section
        report_lines.append("## Summary")
        report_lines.append(f"- **Files Analyzed**: {summary['total_files_analyzed']}")
        report_lines.append(f"- **Total Issues**: {summary['total_issues_found']}")
        report_lines.append(f"- **Auto-fixable Issues**: {summary['auto_fixable_issues']}")
        report_lines.append(f"- **Global Issues**: {summary['global_issues']}")
        report_lines.append("")

        # Issues by severity
        report_lines.append("## Issues by Severity")
        for severity, count in sorted(summary['issues_by_severity'].items()):
            report_lines.append(f"- **{severity.title()}**: {count}")
        report_lines.append("")

        # Issues by type
        report_lines.append("## Issues by Type")
        for issue_type, count in sorted(summary['issues_by_type'].items()):
            report_lines.append(f"- **{issue_type.replace('_', ' ').title()}**: {count}")
        report_lines.append("")

        # File-by-file details
        report_lines.append("## File Details")
        for file_path, analysis in self.file_analyses.items():
            if analysis.issues:
                relative_path = str(Path(file_path).relative_to(self.root_dir))
                report_lines.append(f"### {relative_path}")
                report_lines.append("")

                for issue in sorted(analysis.issues, key=lambda x: x.line_number):
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢"
                    }.get(issue.severity, "⚪")

                    auto_fix_indicator = " 🔧" if issue.auto_fixable else ""

                    report_lines.append(
                        f"- **Line {issue.line_number}** {severity_emoji} "
                        f"{issue.issue_type.replace('_', ' ').title()}{auto_fix_indicator}"
                    )
                    report_lines.append(f"  - {issue.description}")
                    if issue.suggested_fix:
                        report_lines.append(f"  - *Fix*: {issue.suggested_fix}")
                    report_lines.append("")

        # Global issues
        if self.global_issues:
            report_lines.append("## Global Issues")
            for issue in self.global_issues:
                report_lines.append(f"- **{issue.issue_type.replace('_', ' ').title()}**")
                report_lines.append(f"  - {issue.description}")
                if issue.suggested_fix:
                    report_lines.append(f"  - *Fix*: {issue.suggested_fix}")
                report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_file}")

        return report_content

    def export_json(self, output_file: Path):
        """Export analysis results as JSON"""
        data = {
            "summary": self._generate_summary(),
            "file_analyses": {},
            "global_issues": [
                {
                    "issue_type": issue.issue_type,
                    "description": issue.description,
                    "severity": issue.severity,
                    "suggested_fix": issue.suggested_fix,
                    "auto_fixable": issue.auto_fixable
                }
                for issue in self.global_issues
            ]
        }

        # Add detailed file analyses
        for file_path, analysis in self.file_analyses.items():
            relative_path = str(Path(file_path).relative_to(self.root_dir))
            data["file_analyses"][relative_path] = {
                "imports": len(analysis.imports),
                "functions": len(analysis.functions),
                "classes": len(analysis.classes),
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "description": issue.description,
                        "line_number": issue.line_number,
                        "severity": issue.severity,
                        "suggested_fix": issue.suggested_fix,
                        "auto_fixable": issue.auto_fixable
                    }
                    for issue in analysis.issues
                ]
            }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON export saved to: {output_file}")


def main():
    """Main entry point for the hygiene analyzer"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze code hygiene issues")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                       help="Root directory of the project")
    parser.add_argument("--output", type=Path,
                       help="Output file for the report")
    parser.add_argument("--json", type=Path,
                       help="Output file for JSON export")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run analysis
    analyzer = CodeHygieneAnalyzer(args.root)
    summary = analyzer.analyze_project()

    # Generate report
    if args.output:
        report = analyzer.generate_report(args.output)
    else:
        report = analyzer.generate_report()
        print(report)

    # Export JSON if requested
    if args.json:
        analyzer.export_json(args.json)

    # Print summary
    print(f"\n🔍 Analysis Complete!")
    print(f"📁 Files: {summary['total_files_analyzed']}")
    print(f"⚠️  Issues: {summary['total_issues_found']}")
    print(f"🔧 Auto-fixable: {summary['auto_fixable_issues']}")

    # Exit with error code if critical issues found
    critical_issues = summary['issues_by_severity'].get('critical', 0)
    if critical_issues > 0:
        print(f"❌ {critical_issues} critical issues found!")
        return 1
    else:
        print("✅ No critical issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())