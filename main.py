"""
BLIP Tagging Tool
"""

import argparse
import sys
from pathlib import Path
from src.tagger import AITagger


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images and videos using BLIP model.')
    parser.add_argument('input_path', help='File or directory to process')
    parser.add_argument('-o', '--output', default='captions.csv', help='Output CSV file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--batch', action='store_true', help='Optimize for processing many files (enables faster tokenizer)')
    parser.add_argument('--no-skip', action='store_true', help='Process all files even if already processed in CSV (default: skip existing)')
    parser.add_argument('--no-continuous', action='store_true', help='Disable continuous saving (default: save after each file)')
    parser.add_argument('--format', choices=['basic', 'detailed'], default='detailed', 
                       help='CSV output format: basic (filename, caption) or detailed (includes file info, dimensions, date) (default: detailed)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    tagger = AITagger(
        verbose=args.verbose, 
        batch_mode=args.batch,
        skip_existing=not args.no_skip,  
        continuous_save=not args.no_continuous,  
        format_type=args.format
    )
    tagger.process(input_path, args.output)


if __name__ == '__main__':
    main()