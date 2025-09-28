#!/usr/bin/env python3
"""
Analyze Tracking Behavior
=========================

This script analyzes the coordinate data to understand the erratic behavior
before the reference frame vs stability after.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def find_latest_debug_files():
    """Find the most recent debug files."""
    temp_dir = Path("temp")
    
    # Find latest CSV file
    csv_files = list(temp_dir.glob("full_coords_*.csv"))
    if not csv_files:
        return None, None
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    # Find corresponding JSON file
    timestamp = latest_csv.stem.split('_')[-1]  # Extract timestamp
    json_file = temp_dir / f"full_grid_coords_{timestamp}.json"
    
    return latest_csv, json_file if json_file.exists() else None


def analyze_coordinate_stability(csv_file, json_file):
    """Analyze coordinate stability before and after reference frame."""
    print(f"Analyzing coordinate data from: {csv_file}")
    
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Load metadata from JSON
    if json_file:
        with open(json_file, 'r') as f:
            data = json.load(f)
        metadata = data['metadata']
        reference_frame = metadata['reference_frame']
        grid_size = metadata['grid_size']
        total_frames = metadata['total_frames']
    else:
        # Try to determine reference frame from CSV
        reference_frame = df[df['is_reference_frame'] == True]['frame'].iloc[0] if any(df['is_reference_frame']) else 0
        grid_size = len(df[df['frame'] == 0])
        total_frames = df['frame'].max() + 1
    
    print(f"Reference frame: {reference_frame}")
    print(f"Total frames: {total_frames}")
    print(f"Grid size: {grid_size} points")
    
    # Analyze each point's behavior
    results = []
    
    for point_id in df['point_id'].unique():
        point_data = df[df['point_id'] == point_id].sort_values('frame')
        
        # Split into before/after reference frame
        before_ref = point_data[point_data['frame'] < reference_frame]
        after_ref = point_data[point_data['frame'] > reference_frame]
        at_ref = point_data[point_data['frame'] == reference_frame]
        
        if len(before_ref) == 0 or len(after_ref) == 0:
            continue
        
        # Calculate movement/stability metrics
        def calculate_movement_stats(data):
            if len(data) < 2:
                return {'displacement': 0, 'jitter': 0, 'avg_confidence': 0}
            
            # Total displacement
            start_pos = np.array([data.iloc[0]['x'], data.iloc[0]['y']])
            end_pos = np.array([data.iloc[-1]['x'], data.iloc[-1]['y']])
            displacement = np.linalg.norm(end_pos - start_pos)
            
            # Jitter (frame-to-frame movement variation)
            positions = data[['x', 'y']].values
            frame_movements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            jitter = np.std(frame_movements) if len(frame_movements) > 0 else 0
            
            # Average confidence
            avg_confidence = data['confidence'].mean()
            
            return {
                'displacement': displacement,
                'jitter': jitter,
                'avg_confidence': avg_confidence,
                'frames': len(data)
            }
        
        before_stats = calculate_movement_stats(before_ref)
        after_stats = calculate_movement_stats(after_ref)
        
        # Reference frame position
        if len(at_ref) > 0:
            ref_pos = at_ref[['x', 'y']].iloc[0]
            ref_x, ref_y = ref_pos.iloc[0], ref_pos.iloc[1]
        else:
            ref_x, ref_y = 0, 0
        
        results.append({
            'point_id': point_id,
            'ref_x': ref_x,
            'ref_y': ref_y,
            'before_displacement': before_stats['displacement'],
            'after_displacement': after_stats['displacement'],
            'before_jitter': before_stats['jitter'],
            'after_jitter': after_stats['jitter'],
            'before_confidence': before_stats['avg_confidence'],
            'after_confidence': after_stats['avg_confidence'],
            'before_frames': before_stats['frames'],
            'after_frames': after_stats['frames']
        })
    
    return pd.DataFrame(results), reference_frame, total_frames


def visualize_tracking_behavior(df, csv_file, reference_frame):
    """Create visualizations of the tracking behavior."""
    
    # Load original data for plotting
    original_df = pd.read_csv(csv_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Tracking Behavior Analysis (Reference Frame: {reference_frame})', fontsize=16)
    
    # 1. Jitter comparison
    axes[0, 0].bar(['Before Ref', 'After Ref'], 
                   [df['before_jitter'].mean(), df['after_jitter'].mean()],
                   color=['red', 'green'], alpha=0.7)
    axes[0, 0].set_title('Average Jitter (Frame-to-Frame Movement Variation)')
    axes[0, 0].set_ylabel('Pixels')
    
    # 2. Displacement comparison
    axes[0, 1].bar(['Before Ref', 'After Ref'], 
                   [df['before_displacement'].mean(), df['after_displacement'].mean()],
                   color=['red', 'green'], alpha=0.7)
    axes[0, 1].set_title('Average Total Displacement')
    axes[0, 1].set_ylabel('Pixels')
    
    # 3. Confidence comparison
    axes[0, 2].bar(['Before Ref', 'After Ref'], 
                   [df['before_confidence'].mean(), df['after_confidence'].mean()],
                   color=['red', 'green'], alpha=0.7)
    axes[0, 2].set_title('Average Confidence')
    axes[0, 2].set_ylabel('Confidence (0-1)')
    
    # 4. Individual point jitter
    point_ids = df['point_id'][:10]  # Show first 10 points
    x_pos = np.arange(len(point_ids))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, df[df['point_id'].isin(point_ids)]['before_jitter'], 
                   width, label='Before Ref', color='red', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, df[df['point_id'].isin(point_ids)]['after_jitter'], 
                   width, label='After Ref', color='green', alpha=0.7)
    axes[1, 0].set_title('Jitter by Point (First 10 Points)')
    axes[1, 0].set_xlabel('Point ID')
    axes[1, 0].set_ylabel('Jitter (pixels)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(point_ids)
    axes[1, 0].legend()
    
    # 5. Sample trajectory plot
    sample_points = [0, len(df)//4, len(df)//2, 3*len(df)//4] if len(df) > 4 else [0]
    
    for i, point_id in enumerate(sample_points[:4]):
        if point_id >= len(df):
            continue
        actual_point_id = df.iloc[point_id]['point_id']
        point_data = original_df[original_df['point_id'] == actual_point_id].sort_values('frame')
        
        # Split by reference frame
        before = point_data[point_data['frame'] < reference_frame]
        after = point_data[point_data['frame'] >= reference_frame]
        
        if len(before) > 0:
            axes[1, 1].plot(before['frame'], before['x'], 'r-', alpha=0.7, linewidth=1)
        if len(after) > 0:
            axes[1, 1].plot(after['frame'], after['x'], 'g-', alpha=0.7, linewidth=1)
    
    axes[1, 1].axvline(x=reference_frame, color='blue', linestyle='--', alpha=0.8, label='Reference Frame')
    axes[1, 1].set_title('Sample X-Coordinate Trajectories')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('X Coordinate')
    axes[1, 1].legend()
    
    # 6. Stability ratio scatter plot
    stability_ratio = df['after_jitter'] / (df['before_jitter'] + 0.001)  # Add small value to avoid division by zero
    axes[1, 2].scatter(df['before_jitter'], df['after_jitter'], alpha=0.6)
    axes[1, 2].plot([0, df['before_jitter'].max()], [0, df['before_jitter'].max()], 'r--', alpha=0.5, label='Equal Jitter')
    axes[1, 2].set_xlabel('Jitter Before Reference Frame')
    axes[1, 2].set_ylabel('Jitter After Reference Frame')
    axes[1, 2].set_title('Jitter Comparison Scatter')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('temp/tracking_behavior_analysis.png', dpi=150, bbox_inches='tight')
    print("Analysis plot saved to: temp/tracking_behavior_analysis.png")
    
    return fig


def print_analysis_summary(df, reference_frame):
    """Print a summary of the analysis."""
    print("\n" + "="*60)
    print("TRACKING BEHAVIOR ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Reference frame: {reference_frame}")
    print(f"Points analyzed: {len(df)}")
    
    # Calculate statistics
    avg_before_jitter = df['before_jitter'].mean()
    avg_after_jitter = df['after_jitter'].mean()
    jitter_improvement = ((avg_before_jitter - avg_after_jitter) / avg_before_jitter) * 100
    
    avg_before_confidence = df['before_confidence'].mean()
    avg_after_confidence = df['after_confidence'].mean()
    confidence_change = ((avg_after_confidence - avg_before_confidence) / avg_before_confidence) * 100
    
    print(f"\nJITTER ANALYSIS:")
    print(f"  Before reference frame: {avg_before_jitter:.2f} pixels (average)")
    print(f"  After reference frame:  {avg_after_jitter:.2f} pixels (average)")
    print(f"  Improvement: {jitter_improvement:.1f}% {'reduction' if jitter_improvement > 0 else 'increase'}")
    
    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Before reference frame: {avg_before_confidence:.3f} (average)")
    print(f"  After reference frame:  {avg_after_confidence:.3f} (average)")
    print(f"  Change: {confidence_change:+.1f}%")
    
    # Find most problematic points
    worst_before = df.nlargest(3, 'before_jitter')[['point_id', 'before_jitter', 'after_jitter']]
    print(f"\nMOST ERRATIC POINTS BEFORE REFERENCE FRAME:")
    for _, row in worst_before.iterrows():
        improvement = ((row['before_jitter'] - row['after_jitter']) / row['before_jitter']) * 100
        print(f"  Point {int(row['point_id']):2d}: {row['before_jitter']:.1f} → {row['after_jitter']:.1f} pixels ({improvement:+.0f}%)")
    
    # Points with biggest improvement
    df['improvement'] = ((df['before_jitter'] - df['after_jitter']) / (df['before_jitter'] + 0.001)) * 100
    best_improvement = df.nlargest(3, 'improvement')[['point_id', 'before_jitter', 'after_jitter', 'improvement']]
    print(f"\nBIGGEST IMPROVEMENTS:")
    for _, row in best_improvement.iterrows():
        print(f"  Point {int(row['point_id']):2d}: {row['before_jitter']:.1f} → {row['after_jitter']:.1f} pixels ({row['improvement']:+.0f}%)")


def main():
    """Run the tracking behavior analysis."""
    print("TRACKING BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    # Find latest debug files
    csv_file, json_file = find_latest_debug_files()
    
    if not csv_file:
        print("No debug files found. Please run the app first.")
        return
    
    # Analyze coordinate stability
    df, reference_frame, total_frames = analyze_coordinate_stability(csv_file, json_file)
    
    if df.empty:
        print("No coordinate data found for analysis.")
        return
    
    # Print summary
    print_analysis_summary(df, reference_frame)
    
    # Create visualizations
    fig = visualize_tracking_behavior(df, csv_file, reference_frame)
    
    print(f"\nAnalysis complete. Files generated:")
    print(f"  - temp/tracking_behavior_analysis.png")


if __name__ == "__main__":
    main()
