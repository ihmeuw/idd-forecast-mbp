#!/usr/bin/env python3
"""
Simple markdown to PDF converter using weasyprint or reportlab
"""

import sys
from pathlib import Path

def convert_md_to_pdf_simple(md_file, output_file=None):
    """Convert markdown to PDF using basic HTML conversion"""
    
    if output_file is None:
        output_file = md_file.replace('.md', '.pdf')
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple markdown to HTML conversion (basic)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
            h3 {{ color: #888; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .tree {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <pre>{content}</pre>
    </body>
    </html>
    """
    
    # Write HTML file
    html_file = md_file.replace('.md', '.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created HTML file: {html_file}")
    print("You can open this in a browser and print to PDF, or use wkhtmltopdf if available")
    
    return html_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python md_to_pdf.py <markdown_file>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    if not Path(md_file).exists():
        print(f"File {md_file} not found")
        sys.exit(1)
    
    convert_md_to_pdf_simple(md_file)
