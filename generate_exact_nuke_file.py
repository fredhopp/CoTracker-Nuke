#!/usr/bin/env python
"""
CoTracker CSV to Exact Nuke .nk File Generator
Creates 1:1 match with ground truth .nk file structure
"""

import csv
import os
from datetime import datetime

def generate_exact_nuke_file(csv_path, output_path=None, image_height=1080, min_confidence=0.5):
    """
    Generate exact .nk file matching the ground truth structure.
    
    Args:
        csv_path: Path to CoTracker CSV file
        output_path: Output .nk file path (auto-generated if None)
        image_height: Image height for coordinate conversion (default 1080)
        min_confidence: Minimum confidence threshold (default 0.5)
    """
    
    # Auto-generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"cotracker_exact_{base_name}_{timestamp}.nk"
    
    print(f"CoTracker CSV to Exact Nuke .nk Generator")
    print(f"=" * 50)
    print(f"Input CSV: {csv_path}")
    print(f"Output .nk: {output_path}")
    print(f"Image Height: {image_height}")
    print(f"Min Confidence: {min_confidence}")
    print()
    
    # Read and process CSV data
    tracker_dict = {}
    total_rows = 0
    filtered_rows = 0
    
    print("Reading CSV data...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 6:
                continue
                
            total_rows += 1
            frame = int(row[0])
            point_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            visible = row[4].lower() == 'true'
            confidence = float(row[5])
            
            # Initialize point if not exists
            if point_id not in tracker_dict:
                tracker_dict[point_id] = {}
            
            # Filter based on visibility and confidence
            if not visible or confidence < min_confidence:
                filtered_rows += 1
                continue
            
            # Convert Y coordinate: Nuke uses bottom-left origin, CoTracker uses top-left
            y_nuke = image_height - y
            
            # Store valid keyframe by frame number
            tracker_dict[point_id][frame] = [x, y_nuke]
    
    print(f"CSV Processing Complete:")
    print(f"  Total rows: {total_rows}")
    print(f"  Filtered out: {filtered_rows} (invisible or confidence < {min_confidence})")
    print(f"  Valid keyframes: {total_rows - filtered_rows}")
    print(f"  Tracking points: {len(tracker_dict)}")
    print()
    
    # Generate .nk file content
    print("Generating exact .nk file...")
    
    # Sort point IDs for consistent ordering
    point_ids = sorted(tracker_dict.keys())
    num_tracks = len(point_ids)
    
    # Determine frame range (0 to max frame)
    all_frames = set()
    for point_data in tracker_dict.values():
        all_frames.update(point_data.keys())
    
    if not all_frames:
        print("ERROR: No valid keyframes found!")
        return None
        
    min_frame = min(all_frames)
    max_frame = max(all_frames)
    frame_range = list(range(min_frame, max_frame + 1))
    
    print(f"Frame range: {min_frame} to {max_frame} ({len(frame_range)} frames)")
    
    # Build track data strings with exact format
    track_data_lines = []
    
    for i, point_id in enumerate(point_ids):
        keyframes = tracker_dict[point_id]
        
        # Build X and Y coordinate curves with missing frame markers
        x_values = []
        y_values = []
        
        for frame in frame_range:
            if frame in keyframes:
                x_val, y_val = keyframes[frame]
                x_values.append(str(x_val))
                y_values.append(str(y_val))
            else:
                # Use 'x{frame}' marker for missing frames (like in ground truth)
                x_values.append(f"x{frame}")
                y_values.append(f"x{frame}")
        
        x_curve = " ".join(x_values)
        y_curve = " ".join(y_values)
        
        # Create track line exactly matching ground truth format
        track_line = f' {{ {{curve K 1}} "track {i+1}" {{curve {x_curve}}} {{curve {y_curve}}} {{curve K 0}} {{curve K 0}} 1 0 0 {{curve 0}} 1 0 -32 -32 32 32 -22 -22 22 22 {{}} {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}   }}'
        track_data_lines.append(track_line)
    
    # Calculate center point (average of all first frame positions)
    center_x = 0
    center_y = 0
    valid_points = 0
    
    for point_id in point_ids:
        if min_frame in tracker_dict[point_id]:
            x_val, y_val = tracker_dict[point_id][min_frame]
            center_x += x_val
            center_y += y_val
            valid_points += 1
    
    if valid_points > 0:
        center_x /= valid_points
        center_y /= valid_points
    
    # Generate translate curves (difference from center for first point)
    translate_x_values = []
    translate_y_values = []
    
    if point_ids and min_frame in tracker_dict[point_ids[0]]:
        first_point_data = tracker_dict[point_ids[0]]
        for frame in frame_range:
            if frame in first_point_data:
                x_val, y_val = first_point_data[frame]
                # Calculate offset from center
                offset_x = x_val - center_x
                offset_y = y_val - center_y
                translate_x_values.append(str(offset_x))
                translate_y_values.append(str(offset_y))
            else:
                translate_x_values.append("0")
                translate_y_values.append("0")
    
    translate_x_curve = " ".join(translate_x_values)
    translate_y_curve = " ".join(translate_y_values)
    
    # Generate center curves (constant values)
    center_x_curve = " ".join([str(center_x)] * len(frame_range))
    center_y_curve = " ".join([str(center_y)] * len(frame_range))
    
    # Generate complete .nk file content matching exact structure
    nk_content = f'''Root {{
inputs 0
name {output_path}
frame {min_frame}
format "2048 1556 0 0 2048 1556 1 2K_Super_35(full-ap)"
proxy_type scale
proxy_format "1024 778 0 0 1024 778 1 1K_Super_35(full-ap)"
colorManagement Nuke
OCIO_config aces_1.2
workingSpaceLUT linear
monitorLut sRGB
monitorOutLUT rec709
int8Lut sRGB
int16Lut sRGB
logLut Cineon
floatLut linear
}}
Tracker4 {{
inputs 0
tracks {{ {{ 1 31 {num_tracks} }} 
{{ {{ 5 1 20 enable e 1 }} 
{{ 3 1 75 name name 1 }} 
{{ 2 1 58 track_x track_x 1 }} 
{{ 2 1 58 track_y track_y 1 }} 
{{ 2 1 63 offset_x offset_x 1 }} 
{{ 2 1 63 offset_y offset_y 1 }} 
{{ 4 1 27 T T 1 }} 
{{ 4 1 27 R R 1 }} 
{{ 4 1 27 S S 1 }} 
{{ 2 0 45 error error 1 }} 
{{ 1 1 0 error_min error_min 1 }} 
{{ 1 1 0 error_max error_max 1 }} 
{{ 1 1 0 pattern_x pattern_x 1 }} 
{{ 1 1 0 pattern_y pattern_y 1 }} 
{{ 1 1 0 pattern_r pattern_r 1 }} 
{{ 1 1 0 pattern_t pattern_t 1 }} 
{{ 1 1 0 search_x search_x 1 }} 
{{ 1 1 0 search_y search_y 1 }} 
{{ 1 1 0 search_r search_r 1 }} 
{{ 1 1 0 search_t search_t 1 }} 
{{ 2 1 0 key_track key_track 1 }} 
{{ 2 1 0 key_search_x key_search_x 1 }} 
{{ 2 1 0 key_search_y key_search_y 1 }} 
{{ 2 1 0 key_search_r key_search_r 1 }} 
{{ 2 1 0 key_search_t key_search_t 1 }} 
{{ 2 1 0 key_track_x key_track_x 1 }} 
{{ 2 1 0 key_track_y key_track_y 1 }} 
{{ 2 1 0 key_track_r key_track_r 1 }} 
{{ 2 1 0 key_track_t key_track_t 1 }} 
{{ 2 1 0 key_centre_offset_x key_centre_offset_x 1 }} 
{{ 2 1 0 key_centre_offset_y key_centre_offset_y 1 }} 
}} 
{{ 
{chr(10).join(track_data_lines)}
}} 
}}
reference_frame {min_frame}
translate {{{{curve {translate_x_curve}}}}} {{{{curve {translate_y_curve}}}}}
center {{{{curve {center_x_curve}}}}} {{{{curve {center_y_curve}}}}}
selected_tracks {num_tracks - 1}
name CoTracker_Exact_{num_tracks}pts
selected true
xpos 191
ypos 57
}}
'''
    
    # Write .nk file
    with open(output_path, 'w') as f:
        f.write(nk_content)
    
    print(f"SUCCESS: Generated {output_path}")
    print(f"   {num_tracks} tracks with {total_rows - filtered_rows} total keyframes")
    print(f"   Frame range: {min_frame}-{max_frame}")
    print(f"   Center point: ({center_x:.2f}, {center_y:.2f})")
    print(f"   Ready to load in Nuke!")
    print()
    
    return output_path

if __name__ == "__main__":
    # Configuration
    csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_171202.csv"
    image_height = 1080
    min_confidence = 0.5
    
    # Generate exact .nk file
    output_file = generate_exact_nuke_file(
        csv_path=csv_path,
        image_height=image_height,
        min_confidence=min_confidence
    )
    
    if output_file:
        print(f"TO USE:")
        print(f"   1. Open Nuke")
        print(f"   2. File > Open > {output_file}")
        print(f"   3. Tracker4 node will be loaded with exact structure match")
