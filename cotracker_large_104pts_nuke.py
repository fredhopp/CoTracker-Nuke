#!/usr/bin/env python
"""
CoTracker to Nuke - LARGE DATASET (104 points)
Direct CSV loading approach for better performance
Source: Z:/Dev/Cotracker/temp/full_coords_20250928_165315.csv
Points: 104, Valid keyframes: 11275
"""

import nuke
import csv
import time

def create_cotracker_tracks():
    """Create Tracker4 node with all points preserved - loads directly from CSV."""
    
    start_time = time.time()
    print("CoTracker: Loading large dataset with 104 points...")
    
    # Configuration
    csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_165315.csv"
    min_confidence = 0.5
    image_height = 1080
    
    # Read and filter data directly in Nuke
    tracker_dict = {}
    total_frames = 0
    skipped_frames = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 6:
                continue
                
            total_frames += 1
            frame = int(row[0])
            point_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            visible = row[4].lower() == 'true'
            confidence = float(row[5])
            
            # Initialize ALL points
            if point_id not in tracker_dict:
                tracker_dict[point_id] = []
            
            # Skip only individual bad keyframes
            if not visible or confidence < min_confidence:
                skipped_frames += 1
                continue
            
            tracker_dict[point_id].append([frame, x, y])
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Points: {len(tracker_dict)}, Valid keyframes: {total_frames - skipped_frames}")
    
    # Create Tracker4 node
    tracker_node = nuke.createNode("Tracker4")
    tracker_node.setName("CoTracker_Large_{}pts".format(len(tracker_dict)))
    tracker_node.knob("selected").setValue(True)
    
    # Access tracks knob
    tracks = tracker_node["tracks"]
    columns = 31
    
    # Create tracks for all points
    point_ids = sorted(tracker_dict.keys())
    
    creation_start = time.time()
    for tracker_id, point_id in enumerate(point_ids):
        track_data = tracker_dict[point_id]
        
        # Progress indicator for large datasets
        if tracker_id % 20 == 0 or tracker_id == len(point_ids) - 1:
            print(f"Creating track {tracker_id + 1}/{len(point_ids)}...")
        
        # Add new track
        tracker_node['add_track'].execute()
        
        # Calculate knob indices
        track_x_knob = tracker_id * columns + 2
        track_y_knob = tracker_id * columns + 3
        t_option = tracker_id * columns + 6
        r_option = tracker_id * columns + 7
        s_option = tracker_id * columns + 8
        
        # Set T/R/S options
        tracks.setValue(1, t_option)  # Enable translate
        tracks.setValue(0, r_option)  # Disable rotate
        tracks.setValue(0, s_option)  # Disable scale
        
        # Set keyframes with coordinate conversion
        for frame_data in track_data:
            frame_number = int(frame_data[0])
            x_value = float(frame_data[1])
            y_value = float(frame_data[2])
            
            # Convert coordinate system: Nuke uses bottom-left origin, CoTracker uses top-left
            y_value_nuke = image_height - y_value
            
            tracks.setValueAt(x_value, frame_number, track_x_knob)
            tracks.setValueAt(y_value_nuke, frame_number, track_y_knob)
    
    creation_time = time.time() - creation_start
    total_time = time.time() - start_time
    
    # Set reference frame
    reference_frame = 56
    tracker_node["reference_frame"].setValue(reference_frame)
    
    print(f"CoTracker: SUCCESS! Created {len(point_ids)} tracks in {total_time:.2f} seconds")
    print(f"  Data loading: {load_time:.2f}s")
    print(f"  Track creation: {creation_time:.2f}s")
    
    nuke.message("CoTracker Large Dataset Complete!\\n{} tracks created\\n{} quality keyframes\\nCompleted in {:.1f} seconds".format(
        len(point_ids), total_frames - skipped_frames, total_time))

# Run immediately
create_cotracker_tracks()