import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def setup_logging():
    """Configure logging to both file and console"""
    # Create logs directory if it doesn't exist
    log_dir = Path("extraction_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"results_extraction_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")

def find_analysis_folders(results_dir: Path) -> List[Path]:
    """Find all analysis run folders in the results directory"""
    logging.info(f"Searching for analysis folders in {results_dir}")
    
    # Get all directories that contain timestamp pattern (_YYYYMMDD_)
    folders = [f for f in results_dir.iterdir() if f.is_dir() and '_202' in f.name]
    
    logging.info(f"Found {len(folders)} analysis folders")
    for folder in folders:
        logging.debug(f"Found folder: {folder.name}")
    
    return folders

def read_csv_safe(file_path: Path) -> pd.DataFrame:
    """Safely read a CSV file with error handling"""
    try:
        logging.debug(f"Reading file: {file_path.name}")
        df = pd.read_csv(file_path)
        logging.debug(f"Successfully read {file_path.name} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path.name}: {str(e)}")
        return pd.DataFrame()

def process_analysis_folder(folder: Path) -> Dict:
    """Process all CSV files in an analysis folder"""
    logging.info(f"\nProcessing folder: {folder.name}")
    
    # List all files in the folder for debugging
    logging.debug(f"Contents of {folder.name}:")
    for item in folder.iterdir():
        logging.debug(f"- {item.name}")
    
    results = {}
    
    # Expected CSV files
    expected_files = [
        "tdd_analysis_summary.csv",
        "commit_size_analysis.csv",
        "commit_message_patterns.csv",
        "commit_patterns.csv",
        "context_analysis.csv",
        "combined_analysis.csv"
    ]
    
    for filename in expected_files:
        file_path = folder / filename
        if file_path.exists():
            df = read_csv_safe(file_path)
            if not df.empty:
                results[filename.replace('.csv', '')] = df
                logging.debug(f"Processed {filename} - shape: {df.shape}")
        else:
            logging.warning(f"Missing file: {filename} in {folder.name}")
    
    return results

def aggregate_results(all_results: List[Dict]) -> Dict:
    """Aggregate results across all analysis folders"""
    logging.info("\nAggregating results across all folders")
    
    aggregated = {
        'tdd_patterns': {
            'test_first': [],
            'same_commit': [],
            'test_after': []
        },
        'commit_sizes': {
            'small': {'test_first': [], 'same_commit': [], 'test_after': []},
            'medium': {'test_first': [], 'same_commit': [], 'test_after': []},
            'large': {'test_first': [], 'same_commit': [], 'test_after': []}
        },
        'relationships': {
            'same_directory_rate': [],
            'name_pattern_match': [],
            'framework_match': [],
            'confidence_scores': []
        },
        'adoption': {
            'adoption_rates': []
        }
    }
    
    for result in all_results:
        try:
            # Process TDD analysis summary
            if 'tdd_analysis_summary' in result:
                summary = result['tdd_analysis_summary']
                aggregated['tdd_patterns']['test_first'].append(summary['test_first'].iloc[0])
                aggregated['tdd_patterns']['same_commit'].append(summary['same_commit_tdd'].iloc[0])
                aggregated['tdd_patterns']['test_after'].append(summary['test_after'].iloc[0])
                
                # Calculate and store adoption rate
                if 'tdd_adoption_rate' in summary.columns:
                    aggregated['adoption']['adoption_rates'].append(summary['tdd_adoption_rate'].iloc[0])
            
            # Process commit size analysis
            if 'commit_size_analysis' in result:
                size_df = result['commit_size_analysis']
                for _, row in size_df.iterrows():
                    size = row['size_category'].lower()
                    if size in aggregated['commit_sizes']:
                        aggregated['commit_sizes'][size]['test_first'].append(row['test_first'])
                        aggregated['commit_sizes'][size]['same_commit'].append(row['same_commit'])
                        aggregated['commit_sizes'][size]['test_after'].append(row['test_after'])
            
            logging.debug(f"Processed results from one analysis folder")
            
        except Exception as e:
            logging.error(f"Error processing results: {str(e)}")
    
    return aggregated

def calculate_statistics(aggregated: Dict) -> Dict:
    """Calculate final statistics for reporting"""
    logging.info("\nCalculating final statistics")
    
    stats = {
        'tdd_pattern_distribution': {},
        'commit_analysis': {},
        'test_source_relationships': {},
        'project_adoption': {}
    }
    
    # Check if we have any data to process
    if not any(aggregated['tdd_patterns'].values()) and not aggregated['adoption']['adoption_rates']:
        logging.warning("No data found to analyze. Setting default values.")
        stats['tdd_pattern_distribution'] = {
            'test_first': {'count': 0, 'percentage': 0},
            'same_commit': {'count': 0, 'percentage': 0},
            'test_after': {'count': 0, 'percentage': 0}
        }
        stats['project_adoption'] = {
            'mean_rate': 0,
            'std_dev': 0,
            'max_rate': 0,
            'min_rate': 0
        }
        return stats
        
    # 1. TDD Pattern Distribution
    pattern_counts = {
        'test_first': sum(aggregated['tdd_patterns']['test_first']),
        'same_commit': sum(aggregated['tdd_patterns']['same_commit']),
        'test_after': sum(aggregated['tdd_patterns']['test_after'])
    }
    
    total_commits = sum(pattern_counts.values())
    
    for pattern, count in pattern_counts.items():
        percentage = (count / total_commits * 100) if total_commits > 0 else 0
        stats['tdd_pattern_distribution'][pattern] = {
            'count': count,
            'percentage': percentage
        }
        logging.debug(f"{pattern}: count={count}, percentage={percentage:.2f}%")
    
    # 1. TDD Pattern Distribution
    total_patterns = sum(map(sum, [
        aggregated['tdd_patterns']['test_first'],
        aggregated['tdd_patterns']['same_commit'],
        aggregated['tdd_patterns']['test_after']
    ]))
    
    for pattern in ['test_first', 'same_commit', 'test_after']:
        count = sum(aggregated['tdd_patterns'][pattern])
        percentage = (count / total_patterns * 100) if total_patterns > 0 else 0
        stats['tdd_pattern_distribution'][pattern] = {
            'count': count,
            'percentage': percentage
        }
        logging.debug(f"{pattern}: count={count}, percentage={percentage:.2f}%")
    
    # 2. Commit Analysis
    for size in ['small', 'medium', 'large']:
        test_first = sum(aggregated['commit_sizes'][size]['test_first'])
        same_commit = sum(aggregated['commit_sizes'][size]['same_commit'])
        test_after = sum(aggregated['commit_sizes'][size]['test_after'])
        
        total = test_first + same_commit + test_after
        
        stats['commit_analysis'][size] = {
            'total': total,
            'test_first_pct': (test_first / total * 100) if total > 0 else 0,
            'same_commit_pct': (same_commit / total * 100) if total > 0 else 0,
            'test_after_pct': (test_after / total * 100) if total > 0 else 0
        }
        logging.debug(f"{size} commits analysis completed - Total: {total}, TF: {test_first}, SC: {same_commit}, TA: {test_after}")
    
    # 3. Project-Level Adoption
    adoption_rates = aggregated['adoption']['adoption_rates']
    stats['project_adoption'] = {
        'mean_rate': np.mean(adoption_rates) * 100,
        'std_dev': np.std(adoption_rates) * 100,
        'max_rate': np.max(adoption_rates) * 100,
        'min_rate': np.min(adoption_rates) * 100
    }
    logging.debug(f"Project adoption metrics calculated")
    
    return stats

def generate_reports(stats: Dict, output_dir: Path):
    """Generate final report tables"""
    logging.info("\nGenerating final reports")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. TDD Pattern Distribution Table
    pattern_df = pd.DataFrame([
        {
            'Development Pattern': pattern.replace('_', ' ').title(),
            'Count': data['count'],
            'Percentage': f"{data['percentage']:.2f}%"
        }
        for pattern, data in stats['tdd_pattern_distribution'].items()
    ])
    pattern_df.to_csv(output_dir / f'tdd_pattern_distribution_{timestamp}.csv', index=False)
    logging.info("\nTDD Pattern Distribution:")
    logging.info("\n" + pattern_df.to_string())
    
    # 2. Commit Analysis Table
    commit_df = pd.DataFrame([
        {
            'Size': size.title(),
            'Total': data['total'],
            'Test-First%': f"{data['test_first_pct']:.2f}%",
            'Same-Commit%': f"{data['same_commit_pct']:.2f}%",
            'Test-After%': f"{data['test_after_pct']:.2f}%"
        }
        for size, data in stats['commit_analysis'].items()
    ])
    commit_df.to_csv(output_dir / f'commit_analysis_{timestamp}.csv', index=False)
    logging.info("\nCommit Analysis:")
    logging.info("\n" + commit_df.to_string())
    
    # 3. Project-Level Adoption Table
    adoption_df = pd.DataFrame([{
        'Metric': metric.replace('_', ' ').title(),
        'Value': f"{value:.2f}%"
    } for metric, value in stats['project_adoption'].items()])
    adoption_df.to_csv(output_dir / f'project_adoption_{timestamp}.csv', index=False)
    logging.info("\nProject-Level Adoption:")
    logging.info("\n" + adoption_df.to_string())

def main():
    """Main execution function"""
    # Setup logging
    setup_logging()
    logging.info("Starting results extraction process")
    
    try:
        # Setup directories
        base_dir = Path(os.getcwd())
        results_dir = base_dir / "results"
        output_dir = base_dir / "extracted_results"
        
        logging.info(f"Base directory: {base_dir}")
        logging.info(f"Results directory: {results_dir}")
        logging.info(f"Output directory: {output_dir}")
        
        # Find all analysis folders
        folders = find_analysis_folders(results_dir)
        
        # Process each folder
        all_results = []
        for folder in folders:
            results = process_analysis_folder(folder)
            if results:
                all_results.append(results)
        
        # Aggregate results
        logging.info(f"Aggregating results from {len(all_results)} folders")
        aggregated = aggregate_results(all_results)
        
        # Calculate statistics
        stats = calculate_statistics(aggregated)
        
        # Generate reports
        generate_reports(stats, output_dir)
        
        logging.info("Results extraction completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()