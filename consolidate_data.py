import os
import shutil
from pathlib import Path

def consolidate_features():
    """Copy and rename feature files from KHC2 and KHC7 to consolidated directory."""
    
    # Source directories
    khc2_features = Path("../data/KHC2/features")
    khc7_features = Path("../data/KHC7/features")
    
    # Destination directory
    dest_features = Path("../data/features")
    dest_features.mkdir(parents=True, exist_ok=True)
    
    print("=== Consolidating Features ===\n")
    
    # Process KHC2 features (already have KHC2_ prefix)
    print("Processing KHC2 features...")
    for feature_file in khc2_features.glob("*_features.pt"):
        dest_path = dest_features / feature_file.name
        print(f"  {feature_file.name} -> {feature_file.name}")
        shutil.copy2(feature_file, dest_path)
    
    # Process KHC7 features (already have KHC7_ prefix)
    print("\nProcessing KHC7 features...")
    for feature_file in khc7_features.glob("*_features.pt"):
        dest_path = dest_features / feature_file.name
        print(f"  {feature_file.name} -> {feature_file.name}")
        shutil.copy2(feature_file, dest_path)
    
    print(f"\n✓ Features consolidated to {dest_features}")


def consolidate_labels():
    """Copy and rename label files from KHC2 and KHC7 to consolidated directory."""
    
    # Source directories
    khc2_labels = Path("../data/KHC2/labels_csv")
    khc7_labels = Path("../data/KHC7/labels_csv")
    
    # Destination directory
    dest_labels = Path("../data/labels")
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Consolidating Labels ===\n")
    
    # Process KHC2 labels
    print("Processing KHC2 labels...")
    if khc2_labels.exists():
        for label_file in khc2_labels.glob("*.csv"):
            # Original: D10_M1.labels.csv or post_M1.labels.csv or just M1.labels.csv
            name = label_file.stem
            
            # Remove .labels if present
            if name.endswith(".labels"):
                name = name[:-7]  # Remove ".labels"
            
            # Convert D10_M1 -> KHC2_M1_D10 to match feature naming
            if name.startswith("D10_"):
                # D10_M1 -> KHC2_M1_D10
                parts = name.split("_")  # ['D10', 'M1']
                new_name = f"KHC2_{parts[1]}_D10.labels.csv"
            elif name.startswith("pre_"):
                # pre_M1 -> KHC2_M1_pre
                parts = name.split("_")  # ['pre', 'M1']
                new_name = f"KHC2_{parts[1]}_pre.labels.csv"
            elif name.startswith("post_"):
                # post_M1 -> KHC2_M1_post
                parts = name.split("_")  # ['post', 'M1']
                new_name = f"KHC2_{parts[1]}_post.labels.csv"
            else:
                print(f"  Skipping unknown pattern: {label_file.name}")
                continue
            
            dest_path = dest_labels / new_name
            print(f"  {label_file.name} -> {new_name}")
            shutil.copy2(label_file, dest_path)
    else:
        print(f"  Directory not found: {khc2_labels}")
    
    # Process KHC7 labels
    print("\nProcessing KHC7 labels...")
    if khc7_labels.exists():
        for label_file in khc7_labels.glob("*.csv"):
            # Original: D10_M1.labels.csv or post_M1.labels.csv
            name = label_file.stem
            
            # Remove .labels if present
            if name.endswith(".labels"):
                name = name[:-7]  # Remove ".labels"
            
            # Convert D10_M1 -> KHC7_TST_D10_M1 to match feature naming
            if name.startswith("D10_"):
                # D10_M1 -> KHC7_TST_D10_M1
                parts = name.split("_")  # ['D10', 'M1']
                new_name = f"KHC7_TST_D10_{parts[1]}.labels.csv"
            elif name.startswith("pre_"):
                # pre_M1 -> KHC7_TST_pre_M1
                parts = name.split("_")  # ['pre', 'M1']
                new_name = f"KHC7_TST_pre_{parts[1]}.labels.csv"
            elif name.startswith("post_"):
                # post_M1 -> KHC7_TST_post_M1
                parts = name.split("_")  # ['post', 'M1']
                new_name = f"KHC7_TST_post_{parts[1]}.labels.csv"
            else:
                print(f"  Skipping unknown pattern: {label_file.name}")
                continue
            
            dest_path = dest_labels / new_name
            print(f"  {label_file.name} -> {new_name}")
            shutil.copy2(label_file, dest_path)
    else:
        print(f"  Directory not found: {khc7_labels}")
    
    print(f"\n✓ Labels consolidated to {dest_labels}")


def main():
    print("Consolidating features and labels from KHC2 and KHC7...\n")
    
    consolidate_features()
    consolidate_labels()
    
    # Count final files
    features_count = len(list(Path("../data/features").glob("*_features.pt")))
    labels_count = len(list(Path("../data/labels").glob("*.labels.csv")))
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total features: {features_count}")
    print(f"  Total labels: {labels_count}")
    print(f"{'='*50}")
    print("\nYou can now run training with:")
    print("  python train.py --feature_dir ../data/features --label_dir ../data/labels ...")


if __name__ == "__main__":
    main()