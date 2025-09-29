#!/usr/bin/env python
"""
CoTracker to Nuke - SINGLE TRACK TEST
Test with just one track to isolate the hanging issue
"""

import nuke

def create_single_track_test():
    """Create just one track to test if the basic operation works."""
    
    # Create node
    tracker_node = nuke.createNode("Tracker4")
    tracker_node.setName("CoTracker_SingleTest")
    
    # Get tracks
    tracks = tracker_node["tracks"]
    columns = 31
    
    # Add one track
    tracker_node['add_track'].execute()
    
    # Set a few test keyframes manually
    image_height = 1080
    test_data = [
        [0, 100.0, 200.0],
        [10, 110.0, 210.0], 
        [20, 120.0, 220.0],
        [30, 130.0, 230.0]
    ]
    
    for frame_data in test_data:
        frame_number = int(frame_data[0])
        x_value = float(frame_data[1])
        y_value = float(frame_data[2])
        y_nuke = image_height - y_value
        
        tracks.setValueAt(x_value, frame_number, 2)  # track_x
        tracks.setValueAt(y_nuke, frame_number, 3)   # track_y
    
    # Enable translate
    tracks.setValue(1, 6)  # T
    tracks.setValue(0, 7)  # R
    tracks.setValue(0, 8)  # S
    
    nuke.message("Single track test complete")

# Run
create_single_track_test()
