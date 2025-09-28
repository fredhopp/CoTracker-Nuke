#!/usr/bin/env python
"""
CoTracker to Nuke Tracker Script
This script reads CoTracker CSV data and creates tracks in a Nuke Tracker4 node.
Based on the DL_Syn2Trackers approach but adapted for CoTracker CSV format.

Usage in Nuke:
1. Load this script in Nuke's Script Editor
2. Set the csv_path variable to your CoTracker CSV file
3. Run the script
"""

import nuke
import time

def cotracker_to_nuke_tracker(csv_path, time_offset=0, enable_t=True, enable_r=False, enable_s=False):
    """
    Create Nuke Tracker4 node from CoTracker CSV data.
    
    Args:
        csv_path: Path to CoTracker CSV file
        time_offset: Frame offset to apply
        enable_t: Enable translate tracking
        enable_r: Enable rotate tracking  
        enable_s: Enable scale tracking
    """
    
    try:
        print(f"Loading CoTracker data from: {csv_path}")
        
        if not csv_path.endswith('.csv'):
            nuke.message('Invalid CSV file! Please select a CoTracker CSV file.')
            return
        
        # Read and parse CSV data
        tracker_dict = {}
        
        with open(csv_path, "r") as csv_file:
            # Skip header line
            header = csv_file.readline().strip()
            print(f"CSV Header: {header}")
            
            for line in csv_file:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse CSV: frame,point_id,x,y,visible,confidence,is_reference_frame
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                    
                frame = int(parts[0])
                point_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                visible = parts[4].lower() == 'true' if len(parts) > 4 else True
                
                # Only include visible points
                if not visible:
                    continue
                
                # Group by point_id
                if point_id not in tracker_dict:
                    tracker_dict[point_id] = []
                
                tracker_dict[point_id].append([frame, x, y])
        
        if not tracker_dict:
            nuke.message('No valid tracking data found in CSV file!')
            return
        
        print(f"Loaded {len(tracker_dict)} tracks")
        for point_id, data in tracker_dict.items():
            print(f"  Track {point_id}: {len(data)} keyframes")
        
        # Create Tracker4 node
        print("Creating Tracker4 node...")
        tracker_node = nuke.createNode("Tracker4")
        tracker_node.setName("CoTracker_Import")
        tracker_node.knob("selected").setValue(True)
        
        # Access tracks knob
        tracks = tracker_node["tracks"]
        columns = 31  # Number of columns per track in Nuke
        
        # Create progress task
        task = nuke.ProgressTask("CoTracker Import")
        task.setMessage("Creating tracks from CoTracker data...")
        
        tracker_id = -1
        total_tracks = len(tracker_dict)
        
        for point_id in sorted(tracker_dict.keys()):
            track_data = tracker_dict[point_id]
            
            # Update progress
            tracker_id += 1
            progress = int((tracker_id / total_tracks) * 100)
            task.setProgress(progress)
            task.setMessage(f"Creating track {tracker_id + 1}/{total_tracks} (Point ID: {point_id})")
            
            if task.isCancelled():
                break
            
            # Add new track
            tracker_node['add_track'].execute()
            
            # Minimal printing - only every 10th track to avoid spam
            if tracker_id % 10 == 0 or tracker_id == len(sorted(tracker_dict.keys())) - 1:
                print(f"Processing track {tracker_id + 1}/{len(sorted(tracker_dict.keys()))}...")
            
            # Calculate knob indices for this track (exact same as DL_Syn2Trackers)
            track_x_knob = tracker_id * columns + 2  # track_x column
            track_y_knob = tracker_id * columns + 3  # track_y column
            
            # T/R/S options
            t_option = tracker_id * columns + 6
            r_option = tracker_id * columns + 7
            s_option = tracker_id * columns + 8
            
            # Set T/R/S options (exact same logic as DL_Syn2Trackers)
            if enable_t:
                tracks.setValue(1, t_option)
            else:
                tracks.setValue(0, t_option)
            if enable_r:
                tracks.setValue(1, r_option)
            if enable_s:
                tracks.setValue(1, s_option)
            
            # Set keyframes for this track
            for frame_data in track_data:
                frame_number = int(frame_data[0]) + time_offset
                x_value = float(frame_data[1])
                y_value = float(frame_data[2])
                
                # Set keyframes
                tracks.setValueAt(x_value, frame_number, track_x_knob)
                tracks.setValueAt(y_value, frame_number, track_y_knob)
        
        # Cleanup
        del task
        
        print(f"✅ Successfully created {tracker_id + 1} tracks in Tracker4 node!")
        nuke.message(f'CoTracker import completed!\\n\\nCreated {tracker_id + 1} tracks from {len(tracker_dict)} points.')
        
    except Exception as e:
        error_msg = f"Error importing CoTracker data: {str(e)}"
        print(error_msg)
        nuke.message(error_msg)
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Configuration - EDIT THESE VALUES
    csv_path = "Z:/Dev/Cotracker/temp/full_coords_20250928_161516.csv"  # Path to your CoTracker CSV (smaller dataset)
    time_offset = 0      # Frame offset (usually 0 for CoTracker)
    enable_t = True      # Enable translate tracking
    enable_r = False     # Enable rotate tracking
    enable_s = False     # Enable scale tracking
    
    print("CoTracker CSV to Nuke Tracker Import")
    print("=" * 40)
    print(f"CSV File: {csv_path}")
    print(f"Settings: T={enable_t}, R={enable_r}, S={enable_s}")
    print(f"Time Offset: {time_offset}")
    print()
    
    # Run the import
    cotracker_to_nuke_tracker(csv_path, time_offset, enable_t, enable_r, enable_s)
