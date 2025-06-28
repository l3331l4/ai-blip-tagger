import csv
from pathlib import Path
from typing import List, Tuple, Set, Dict
import logging
import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from .models.captioner import ImageCaptioner
from .processors.image_processor import ImageProcessor
from .processors.video_processor import VideoProcessor


class AITagger:
    """Tags images and videos."""
    
    SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    SUPPORTED_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.gif', '.webm'}
    
    def __init__(self, verbose: bool = False, batch_mode: bool = False, 
                 skip_existing: bool = True, continuous_save: bool = True, format_type: str = 'detailed'):
        self.verbose = verbose
        self.use_fast = batch_mode 
        self.batch_mode = batch_mode
        self.skip_existing = skip_existing
        self.continuous_save = continuous_save
        self.format_type = format_type  
        self._setup_logging()
        
        if self.verbose:
            if self.batch_mode:
                print("Batch mode on")
            if self.skip_existing:
                print("Will skip files already in CSV")
            if self.continuous_save:
                print("Saving as we go")
        
        print("Loading AI models...")
        self.captioner = ImageCaptioner(use_fast=self.use_fast)
        self.image_processor = ImageProcessor(self.captioner)
        self.video_processor = VideoProcessor(self.captioner)
        print("Models loaded!")
        
    def _setup_logging(self):
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=level, format='%(message)s')
        self.logger = logging.getLogger(__name__)
    
    def process(self, input_path: Path, output_file: str):
        """Process files and write results to CSV."""

        if self.continuous_save:
            self._ensure_csv_headers(output_file)
        
        existing_files = set()
        if self.skip_existing:
            existing_files = self._get_existing_files(output_file)
        
        files_to_process = self._get_files(input_path)
        
        if not files_to_process:
            print("No supported files found")
            return
        
        if self.skip_existing and existing_files:
            original_count = len(files_to_process)
            files_to_process = [f for f in files_to_process if f.name not in existing_files]
            skipped_count = original_count - len(files_to_process)
            
            if skipped_count > 0 and self.verbose:
                print(f"Skipping {skipped_count} files (already processed)")
            
            if not files_to_process:
                print("All files already processed!")
                return
        
        results = []
        total_files = len(files_to_process)
        
        for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            try:
                caption = self._process_single_file(file_path)
                
                if self.format_type == 'detailed':
                    file_info = self._get_file_info(file_path)
                    result = (file_path.name, caption, file_info['size_kb'], 
                             file_info['dimensions'], file_info['extension'], 
                             file_info['date_processed'])
                else:
                    result = (file_path.name, caption)
                
                if self.continuous_save:
                    self._save_single_result(result, output_file)
                    if self.verbose:
                        print(f"✓ {file_path.name}")
                else:
                    results.append(result)
                    if self.verbose:
                        print(f"✓ Queued: {file_path.name}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"✗ {file_path.name}: {e}")
                else:
                    print(f"Error processing {file_path.name}")
                if self.format_type == 'detailed':
                    error_result = (file_path.name, "ERROR", 0, "unknown", 
                                   file_path.suffix.lower(), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    error_result = (file_path.name, "ERROR")
                
                if self.continuous_save:
                    self._save_single_result(error_result, output_file)
                else:
                    results.append(error_result)
        
        if not self.continuous_save:
            if self.verbose:
                print("Saving results...")
            if self.skip_existing and existing_files:
                self._append_csv(results, output_file)
            else:
                self._write_csv(results, output_file)
            
        print(f"Done! Check {output_file}")
    
    def _get_files(self, path: Path) -> List[Path]:
        """Get all supported files from path."""

        if path.is_file():
            if self._is_supported(path):
                return [path]
            else:
                print(f"Unsupported file type: {path.suffix}")
                return []
        
        files = []
        for file_path in path.rglob('*'):
            if file_path.is_file() and self._is_supported(file_path):
                files.append(file_path)
        
        return sorted(files)
    
    def _is_supported(self, file_path: Path) -> bool:
        """Check if file type is supported."""

        suffix = file_path.suffix.lower()
        return suffix in self.SUPPORTED_IMAGES or suffix in self.SUPPORTED_VIDEOS
    
    def _process_single_file(self, file_path: Path) -> str:
        """Process a single file and return caption."""

        suffix = file_path.suffix.lower()
        
        if suffix in self.SUPPORTED_IMAGES:
            return self.image_processor.process(file_path)
        elif suffix in self.SUPPORTED_VIDEOS:
            return self.video_processor.process(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _append_csv(self, results: List[Tuple], output_file: str):
        """Append results to existing CSV file."""

        if not results:
            return
            
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(results)

    def _write_csv(self, results: List[Tuple], output_file: str):
        """Write results to CSV file."""

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            if self.format_type == 'detailed':
                writer.writerow(['filename', 'caption', 'size_kb', 'dimensions', 'file_type', 'date_processed'])
            else:
                writer.writerow(['filename', 'caption'])
            writer.writerows(results)
    
    def _ensure_csv_headers(self, output_file: str):
        """Ensure CSV file exists with headers."""

        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
                if self.format_type == 'detailed':
                    writer.writerow(['filename', 'caption', 'size_kb', 'dimensions', 'file_type', 'date_processed'])
                else:
                    writer.writerow(['filename', 'caption'])
            if self.verbose:
                print(f"Created new CSV file: {output_file}")

    def _save_single_result(self, result: Tuple, output_file: str):
        """Save a single result to CSV file."""

        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(result)

    def _get_existing_files(self, output_file: str) -> Set[str]:
        """Get set of filenames that have already been processed."""

        existing_files = set()
        
        if not os.path.exists(output_file):
            return existing_files
            
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                next(reader, None)  
                for row in reader:
                    if row and len(row) >= 1:  
                        existing_files.add(row[0])
            
            if self.verbose:
                print(f"Found {len(existing_files)} files already in CSV")
            
        except Exception as e:
            if self.verbose:
                print(f"Could not read existing CSV: {e}")
            
        return existing_files
    
    def _get_file_info(self, file_path: Path) -> Dict[str, str]:
        """Get additional file information."""

        try:
            stats = file_path.stat()
            info = {
                'size_kb': round(stats.st_size / 1024, 1),
                'extension': file_path.suffix.lower(),
                'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dimensions': 'N/A'
            }
            
            if file_path.suffix.lower() in self.SUPPORTED_IMAGES:
                try:
                    with Image.open(file_path) as img:
                        info['dimensions'] = f"{img.width}x{img.height}"
                except Exception:
                    info['dimensions'] = 'unknown'
            
            return info
        
        except Exception:
            return {
                'size_kb': 0,
                'extension': file_path.suffix.lower(),
                'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dimensions': 'unknown'
            }