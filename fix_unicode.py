#!/usr/bin/env python3
"""
Fix Unicode characters in automation scripts for Windows compatibility
"""

import os
import re
from pathlib import Path

def fix_unicode_in_file(file_path):
    """Replace Unicode emoji characters with ASCII alternatives"""
    
    # Define replacements
    replacements = {
        '✅': '[OK]',
        '❌': '[ERROR]',
        '💡': '[INFO]',
        '🎯': '[TARGET]',
        '📊': '[DATA]',
        '🚀': '[LAUNCH]',
        '⏰': '[TIMEOUT]',
        '🔄': '[RUNNING]',
        '🏁': '[COMPLETED]',
        '🌅': '[START]',
        '⚠️': '[WARNING]',
        '🔍': '[SEARCH]',
        '📈': '[CHART]',
        '💰': '[MONEY]',
        '🕐': '[TIME]',
        '📝': '[NOTE]',
        '🔧': '[CONFIG]',
        '📁': '[FOLDER]',
        '🎉': '[SUCCESS]',
        '⭐': '[STAR]',
        '🔗': '[LINK]',
        '📋': '[LIST]',
        '🎪': '[DEMO]',
        '🔥': '[HOT]',
        '💼': '[BUSINESS]',
        '🌟': '[HIGHLIGHT]',
        '📊': '[ANALYSIS]',
        '🎨': '[STYLE]',
        '🔐': '[SECURE]',
        '📺': '[DISPLAY]',
        '🎮': '[CONTROL]',
        '🌐': '[NETWORK]',
        '⚡': '[FAST]',
        '🔄': '[REFRESH]',
        '📤': '[SEND]',
        '📥': '[RECEIVE]',
        '🎊': '[CELEBRATION]'
    }
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track if any changes were made
        original_content = content
        
        # Apply replacements
        for emoji, ascii_alt in replacements.items():
            content = content.replace(emoji, ascii_alt)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix Unicode characters in all Python files"""
    
    # Get the base directory
    base_dir = Path(__file__).parent
    
    # Files to process
    files_to_fix = [
        'scripts/automation_manager.py',
        'scripts/simple_automation.py',
        'scripts/smart_scheduler.py',
        'scripts/enterprise_dashboard.py',
        'scripts/event_driven_automation.py',
        'scripts/multi_market_automation.py',
        'scripts/sector_automation.py',
        'scripts/generate_summary.py',
        'cli_analyzer.py',
        'automation_demo.py'
    ]
    
    fixed_count = 0
    
    print("[CONFIG] Fixing Unicode characters in automation scripts...")
    print("=" * 60)
    
    for file_path in files_to_fix:
        full_path = base_dir / file_path
        
        if full_path.exists():
            if fix_unicode_in_file(full_path):
                print(f"[OK] Fixed: {file_path}")
                fixed_count += 1
            else:
                print(f"[SKIP] No changes: {file_path}")
        else:
            print(f"[ERROR] Not found: {file_path}")
    
    print("=" * 60)
    print(f"[COMPLETED] Fixed {fixed_count} files")
    
    if fixed_count > 0:
        print("\n[OK] Unicode characters have been replaced with ASCII alternatives")
        print("[INFO] The automation system should now work properly in Windows terminal")
    else:
        print("\n[INFO] No Unicode characters found to fix")

if __name__ == "__main__":
    main()