#!/usr/bin/env python
"""
CoTracker CSV to Nuke .nk File Generator
Direct .nk file creation based on analysis of working tracker
"""

import csv
import os
from datetime import datetime

def generate_nuke_file(csv_path, output_path=None, image_height=1080, min_confidence=0.5, frame_offset=0, reference_frame=0):
    """
    Generate a complete .nk file with Tracker4 node from CoTracker CSV data.
    
    Args:
        csv_path: Path to CoTracker CSV file
        output_path: Output .nk file path (auto-generated if None)
        image_height: Image height for coordinate conversion (default 1080)
        min_confidence: Minimum confidence threshold (default 0.5)
        frame_offset: Frame offset to apply to frame numbers (default 0)
        reference_frame: Reference frame number for the tracker (default 0)
    """
    
    # Auto-generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"cotracker_{base_name}_{timestamp}.nk"
    
    print(f"CoTracker CSV to Nuke .nk Generator")
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
                tracker_dict[point_id] = []
            
            # Filter based on visibility and confidence
            if not visible or confidence < min_confidence:
                filtered_rows += 1
                continue
            
            # Convert Y coordinate: Nuke uses bottom-left origin, CoTracker uses top-left
            y_nuke = image_height - y
            
            # Apply frame offset and store valid keyframe
            frame_with_offset = frame + frame_offset
            tracker_dict[point_id].append([frame_with_offset, x, y_nuke])
    
    print(f"CSV Processing Complete:")
    print(f"  Total rows: {total_rows}")
    print(f"  Filtered out: {filtered_rows} (invisible or confidence < {min_confidence})")
    print(f"  Valid keyframes: {total_rows - filtered_rows}")
    print(f"  Tracking points: {len(tracker_dict)}")
    print()
    
    # Generate .nk file content
    print("Generating .nk file...")
    
    # Sort point IDs for consistent ordering
    point_ids = sorted(tracker_dict.keys())
    num_tracks = len(point_ids)
    
    # Build track data strings
    track_data_lines = []
    
    for i, point_id in enumerate(point_ids):
        keyframes = tracker_dict[point_id]
        
        # Build X and Y coordinate curves
        x_values = []
        y_values = []
        
        for frame, x, y in keyframes:
            x_values.append(str(x))
            y_values.append(str(y))
        
        x_curve = " ".join(x_values)
        y_curve = " ".join(y_values)
        
        # Create track line (based on analysis of working .nk file)
        track_line = f' {{ {{curve K 1}} "track {i+1}" {{curve {x_curve}}} {{curve {y_curve}}} {{curve K 0}} {{curve K 0}} 1 0 0 {{curve 0}} 1 0 -32 -32 32 32 -22 -22 22 22 {{}} {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}  {{}}   }}'
        track_data_lines.append(track_line)
    
    # Generate complete .nk file content
    nk_content = f'''#! C:/Program Files/Nuke16.0v5/nuke-16.0.5.dll -nx
version 16.0 v5
define_window_layout_xml {{<?xml version="1.0" encoding="UTF-8"?>
<layout version="1.0">
    <window x="-2556" y="-5" w="5114" h="1361" screen="1">
        <splitter orientation="1">
            <split size="40"/>
            <dock id="" hideTitles="1" activePageId="Toolbar.1">
                <page id="Toolbar.1"/>
            </dock>
            <split size="2512" stretch="1"/>
            <splitter orientation="1">
                <split size="2512"/>
                <dock id="" activePageId="Viewer.1">
                    <page id="Viewer.1"/>
                </dock>
            </splitter>
            <split size="2554"/>
            <splitter orientation="2">
                <split size="756"/>
                <splitter orientation="1">
                    <split size="603"/>
                    <dock id="" activePageId="Properties.1">
                        <page id="Properties.1"/>
                    </dock>
                    <split size="1947"/>
                    <dock id="" activePageId="DAG.1" focus="true">
                        <page id="DAG.1"/>
                    </dock>
                </splitter>
                <split size="563"/>
                <dock id="" activePageId="uk.co.thefoundry.scripteditor.1">
                    <page id="DopeSheet.1"/>
                    <page id="Curve Editor.1"/>
                    <page id="uk.co.thefoundry.backgroundrenderview.1"/>
                    <page id="uk.co.thefoundry.scripteditor.1"/>
                </dock>
            </splitter>
        </splitter>
    </window>
</layout>
}}
Root {{
inputs 0
name {output_path}
frame 0
format "2048 1556 0 0 2048 1556 1 2K_Super_35(full-ap)"
proxy_type scale
proxy_format "1024 778 0 0 1024 778 1 1K_Super_35(full-ap)"
colorManagement Nuke
workingSpaceLUT linear
monitorLut sRGB
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
reference_frame {reference_frame}
name CoTracker_Generated_{num_tracks}pts
selected true
xpos 0
ypos 0
}}

'''
    
    # Write .nk file
    with open(output_path, 'w') as f:
        f.write(nk_content)
    
    print(f"SUCCESS: Generated {output_path}")
    print(f"   {num_tracks} tracks with {total_rows - filtered_rows} total keyframes")
    print(f"   Ready to load in Nuke!")
    print()
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Check if called with command line arguments (from nuke_exporter.py)
    if len(sys.argv) >= 6:
        csv_path = sys.argv[1]
        output_path = sys.argv[2]
        frame_offset = int(sys.argv[3])
        reference_frame_video = int(sys.argv[4])  # 0-based video frame index
        image_height = int(sys.argv[5])
        
        # Convert 0-based video reference frame to user's actual reference frame number
        reference_frame = reference_frame_video + frame_offset
        
        # Generate .nk file with provided parameters
        output_file = generate_nuke_file(
            csv_path=csv_path,
            output_path=output_path,
            image_height=image_height,
            min_confidence=0.5,
            frame_offset=frame_offset,
            reference_frame=reference_frame
        )
        
        # Print the absolute path (nuke_exporter.py expects this on the last line)
        print(os.path.abspath(output_file))
        
    else:
        # Configuration for standalone use
        csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_171202.csv"
        image_height = 1080
        min_confidence = 0.5
        
        # Generate .nk file
        output_file = generate_nuke_file(
            csv_path=csv_path,
            image_height=image_height,
            min_confidence=min_confidence
        )
        
        print(f"TO USE:")
        print(f"   1. Open Nuke")
        print(f"   2. File > Open > {output_file}")
        print(f"   3. Tracker4 node will be loaded with all tracking data")
