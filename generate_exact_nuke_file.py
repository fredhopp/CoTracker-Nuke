#!/usr/bin/env python
"""
CoTracker CSV to Exact Nuke .nk File Generator
Creates 1:1 match with ground truth .nk file structure
"""

import csv
import os
from datetime import datetime

def generate_exact_nuke_file(csv_path, output_path=None, image_height=1080, min_confidence=0.5, 
                            tracker_node_name=None, frame_offset=0, reference_frame=0):
    """
    Generate exact .nk file matching the ground truth structure.
    
    Args:
        csv_path: Path to CoTracker CSV file
        output_path: Output .nk file path (auto-generated if None)
        image_height: Image height for coordinate conversion (default 1080)
        min_confidence: Minimum confidence threshold (default 0.5)
        tracker_node_name: Name for the tracker node (auto-generated if None)
        frame_offset: Offset to add to all frame numbers (default 0)
        reference_frame: User's chosen reference frame (default 0)
    """
    
    # Auto-generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_path = f"outputs/CoTracker_{timestamp}.nk"
    
    # Auto-generate tracker node name if not provided
    if tracker_node_name is None:
        # Extract filename without extension from output path
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        tracker_node_name = base_name
    
    print(f"CoTracker CSV to Exact Nuke .nk Generator")
    print(f"=" * 50)
    print(f"Input CSV: {csv_path}")
    print(f"Output .nk: {output_path}")
    print(f"Tracker Node Name: {tracker_node_name}")
    print(f"Image Height: {image_height}")
    print(f"Min Confidence: {min_confidence}")
    print(f"Frame Offset: {frame_offset}")
    print(f"Reference Frame: {reference_frame}")
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
            frame = int(row[0]) + frame_offset  # Apply frame offset
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
    
    for track_idx, point_id in enumerate(point_ids):
        keyframes = tracker_dict[point_id]
        
        # Build X and Y coordinate curves with missing frame markers
        x_values = []
        y_values = []
        
        # Get sorted frames for this point
        sorted_frames = sorted(keyframes.keys())
        
        # Find gaps to determine which frames need x-markers
        frames_after_gaps = set()
        for i in range(len(sorted_frames)):
            frame = sorted_frames[i]
            prev_frame = frame - 1
            
            # If previous frame is missing, this frame comes after a gap
            if prev_frame not in keyframes and frame > 0:
                frames_after_gaps.add(frame)
                
                # Also check if next frame is consecutive (both frames after same gap)
                if i + 1 < len(sorted_frames):
                    next_frame = sorted_frames[i + 1]
                    if next_frame == frame + 1:
                        frames_after_gaps.add(next_frame)
        
        for frame in frame_range:
            if frame in keyframes:
                x_val, y_val = keyframes[frame]
                # Format numbers to match ground truth (remove unnecessary decimals)
                x_str = f"{x_val:g}" if x_val != int(x_val) else str(int(x_val))
                y_str = f"{y_val:g}" if y_val != int(y_val) else str(int(y_val))
                
                # Add x-marker if this frame comes after a gap
                if frame in frames_after_gaps:
                    x_values.append(f"x{frame}")
                    y_values.append(f"x{frame}")
                
                x_values.append(x_str)
                y_values.append(y_str)
        
        x_curve = " ".join(x_values)
        y_curve = " ".join(y_values)
        
        # Create track line exactly matching ground truth format  
        track_line = f' {{ {{curve K 1}} "track {track_idx+1}" {{curve {x_curve}}} {{curve {y_curve}}} {{curve K 0}} {{curve K 0}} 1 0 0 {{curve 0}} 1 0 -32 -32 32 32 -22 -22 22 22 {{}} {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}   }}'
        track_data_lines.append(track_line)
    
    # Note: translate and center curves removed as per user request
    
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

reference_frame {reference_frame + frame_offset}
selected_tracks {num_tracks - 1}
name {tracker_node_name}
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
    print(f"   Reference frame: {reference_frame + frame_offset}")
    print(f"   Ready to load in Nuke!")
    print()
    
    return output_path

if __name__ == "__main__":
    # Configuration
    csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_214434.csv"
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
