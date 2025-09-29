#!/usr/bin/env python
"""
CoTracker to Nuke - SMALL DATASET TEST
15 points, minimal code to avoid hanging
"""

import nuke
import csv

def create_cotracker_tracks():
    """Create Tracker4 node - absolute minimal version."""
    
    csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_171202.csv"
    min_confidence = 0.5
    image_height = 1080
    
    # Read data
    tracker_dict = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 6:
                continue
                
            frame = int(row[0])
            point_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            visible = row[4].lower() == 'true'
            confidence = float(row[5])
            
            if point_id not in tracker_dict:
                tracker_dict[point_id] = []
            
            if not visible or confidence < min_confidence:
                continue
            
            tracker_dict[point_id].append([frame, x, y])
    
    # Create node
    tracker_node = nuke.createNode("Tracker4")
    tracker_node.setName("CoTracker_Small_15pts")
    
    # Get tracks
    tracks = tracker_node["tracks"]
    columns = 31
    point_ids = sorted(tracker_dict.keys())
    
    # Create tracks - absolutely minimal
    for tracker_id, point_id in enumerate(point_ids):
        track_data = tracker_dict[point_id]
        
        tracker_node['add_track'].execute()
        
        # Set keyframes
        for frame_data in track_data:
            frame_number = int(frame_data[0])
            x_value = float(frame_data[1])
            y_value = float(frame_data[2])
            y_nuke = image_height - y_value
            
            tracks.setValueAt(x_value, frame_number, tracker_id * columns + 2)
            tracks.setValueAt(y_nuke, frame_number, tracker_id * columns + 3)
        
        # Enable translate only
        tracks.setValue(1, tracker_id * columns + 6)
        tracks.setValue(0, tracker_id * columns + 7)
        tracks.setValue(0, tracker_id * columns + 8)
    
    # Done
    nuke.message("Created {} tracks".format(len(point_ids)))

# Run
create_cotracker_tracks()
