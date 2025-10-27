import re

def clean_python_file(file_path):
    """Clean Python file by removing trailing whitespace and fixing blank lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # Remove trailing whitespace
        cleaned_line = line.rstrip() + '\n'
        cleaned_lines.append(cleaned_line)
    
    # Write back cleaned content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"✅ Cleaned whitespace issues in {file_path}")

if __name__ == "__main__":
    clean_python_file("scripts/comprehensive_bi_analysis.py")