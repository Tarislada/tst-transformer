"""
Convert TST Excel labels to standardized per-second CSV format.

Input: Excel files with 'Filtered Data' sheet containing per-second labels
Output: CSV files with columns [second, label] where label is 0 (mobile) or 1 (immobile)
"""

import os
import glob
import pandas as pd
import typer
from rich import print
from typing import Optional

app = typer.Typer()


def convert_single_excel(
    excel_path: str,
    out_csv: str,
    sheet_name: str = "Filtered Data",
    trim_to_seconds: Optional[int] = None,
):
    """
    Convert a single Excel file to per-second CSV format.
    
    Args:
        excel_path: Path to Excel file
        out_csv: Output CSV path
        sheet_name: Name of sheet with per-second data
        trim_to_seconds: If set, only keep first N seconds
    """
    try:
        # Read the data sheet
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Check required columns
        if 'Second' not in df.columns:
            raise ValueError(f"Missing 'Second' column in {excel_path}")
        
        # Get label column (try multiple possible names)
        label_col = None
        for col_name in ['Mobility Status_num', 'Mobility Status', 'label']:
            if col_name in df.columns:
                label_col = col_name
                break
        
        if label_col is None:
            raise ValueError(f"No label column found in {excel_path}. Available: {df.columns.tolist()}")
        
        # Extract and standardize
        out_df = pd.DataFrame()
        out_df['second'] = df['Second'].astype(int)
        
        # Convert to numeric labels (0=mobile, 1=immobile)
        if df[label_col].dtype == 'object':  # String labels
            label_map = {'mobile': 0, 'immobile': 1}
            out_df['label'] = df[label_col].str.lower().map(label_map).astype(int)
        else:  # Already numeric
            out_df['label'] = df[label_col].astype(int)
        
        # Ensure labels are 0/1
        unique_labels = set(out_df['label'].unique())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, got: {unique_labels}")
        
        # Sort by second (should already be sorted, but ensure)
        out_df = out_df.sort_values('second').reset_index(drop=True)
        
        # Trim if requested
        if trim_to_seconds is not None:
            out_df = out_df[out_df['second'] <= trim_to_seconds].reset_index(drop=True)
        
        # Renumber seconds from 0 (for consistency with your pipeline)
        out_df['second'] = range(len(out_df))
        
        # Save
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        
        print(f"[green]✓[/green] Converted {os.path.basename(excel_path)}")
        print(f"  → {out_csv}")
        print(f"  Seconds: {len(out_df)}, Mobile: {(out_df['label']==0).sum()}, Immobile: {(out_df['label']==1).sum()}")
        
        return out_df
        
    except Exception as e:
        print(f"[red]✗[/red] Failed to convert {excel_path}: {e}")
        raise


@app.command()
def convert_one(
    excel_path: str = typer.Option(..., help="Path to Excel file"),
    out_csv: str = typer.Option(..., help="Output CSV path"),
    sheet_name: str = typer.Option("Filtered Data", help="Sheet name with labels"),
    trim_to_seconds: Optional[int] = typer.Option(None, help="Trim to first N seconds"),
):
    """Convert a single Excel label file to CSV."""
    convert_single_excel(excel_path, out_csv, sheet_name, trim_to_seconds)


@app.command()
def convert_batch(
    excel_glob: str = typer.Option(..., help="Glob pattern for Excel files (e.g., 'labels/*.xlsx')"),
    out_dir: str = typer.Option("labels_csv", help="Output directory for CSV files"),
    sheet_name: str = typer.Option("Filtered Data", help="Sheet name with labels"),
    trim_to_seconds: Optional[int] = typer.Option(None, help="Trim all to first N seconds"),
):
    """Convert multiple Excel label files to CSV format."""
    
    # Find all Excel files
    excel_files = glob.glob(excel_glob, recursive=True)
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]  # Skip lock files
    
    if not excel_files:
        print(f"[yellow]⚠[/yellow] No Excel files found matching: {excel_glob}")
        return
    
    print(f"Found {len(excel_files)} Excel file(s)")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert each file
    success = 0
    failed = 0
    
    for excel_path in excel_files:
        # Generate output name
        base_name = os.path.splitext(os.path.basename(excel_path))[0]
        # Remove common prefixes
        base_name = base_name.replace('Filtered_TST_Manual_Scoring_', '')
        base_name = base_name.replace('TST_Manual_Scoring_', '')
        
        out_csv = os.path.join(out_dir, f"{base_name}.labels.csv")
        
        try:
            convert_single_excel(excel_path, out_csv, sheet_name, trim_to_seconds)
            success += 1
        except Exception:
            failed += 1
    
    print(f"\n[bold]Summary:[/bold] {success} successful, {failed} failed")


if __name__ == "__main__":
    app()
