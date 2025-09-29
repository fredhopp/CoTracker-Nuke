#!/usr/bin/env python
"""
CoTracker to Nuke - ULTRA MINIMAL VERSION
No printing, maximum performance for large datasets
104 points, 11,275 keyframes
"""

import nuke
import csv

def create_cotracker_tracks():
    """Create Tracker4 node - ultra minimal, no printing."""
    
    # Configuration
    csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_165315.csv"
    min_confidence = 0.5
    image_height = 1080
    
    # Read and filter data
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
    
    # Create Tracker4 node
    tracker_node = nuke.createNode("Tracker4")
    tracker_node.setName("CoTracker_Minimal_{}pts".format(len(tracker_dict)))
    tracker_node.knob("selected").setValue(True)
    
    # Access tracks knob
    tracks = tracker_node["tracks"]
    columns = 31
    point_ids = sorted(tracker_dict.keys())
    
    # Create tracks - ZERO PRINTING FOR MAXIMUM SPEED
    for tracker_id, point_id in enumerate(point_ids):
        track_data = tracker_dict[point_id]
        
        tracker_node['add_track'].execute()
        
        track_x_knob = tracker_id * columns + 2
        track_y_knob = tracker_id * columns + 3
        t_option = tracker_id * columns + 6
        r_option = tracker_id * columns + 7
        s_option = tracker_id * columns + 8
        
        tracks.setValue(1, t_option)
        tracks.setValue(0, r_option)
        tracks.setValue(0, s_option)
        
        # Set keyframes
        for frame_data in track_data:
            frame_number = int(frame_data[0])
            x_value = float(frame_data[1])
            y_value = float(frame_data[2])
            y_value_nuke = image_height - y_value
            
            tracks.setValueAt(x_value, frame_number, track_x_knob)
            tracks.setValueAt(y_value_nuke, frame_number, track_y_knob)
    
    # Set reference frame
    tracker_node["reference_frame"].setValue(56)
    
    # Single message at end only
    nuke.message("CoTracker: {} tracks created".format(len(point_ids)))

# Run immediately
create_cotracker_tracks()
