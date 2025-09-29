#!/usr/bin/env python3
"""
Nuke Tracker Export using Nuke Python API
This script creates a Tracker4 node and populates it with CoTracker data
"""

import os
import sys
import numpy as np

def create_nuke_tracker_script(coords, visibility, output_path, nuke_executable=None):
    """
    Create a Nuke script that uses the Python API to build a Tracker4 node.
    This script can be executed by Nuke in non-interactive mode.
    """
    
    num_frames, num_points, _ = coords.shape
    
    # Create the Nuke Python script content
    nuke_python_script = f'''#!/usr/bin/env python
"""
Auto-generated Nuke script to create Tracker4 node with CoTracker data
"""

import nuke
import numpy as np

def create_tracker_with_data():
    """Create Tracker4 node and populate with tracking data."""
    
    # Clear existing nodes
    nuke.selectAll()
    nuke.delete()
    
    # Create Tracker4 node
    tracker = nuke.createNode("Tracker4")
    tracker.setName("CoTracker_Export")
    
    # Set basic tracker properties
    tracker['grablink_error_above'].setValue(0.01)
    
    # Tracking data
    coords = np.array({coords.tolist()})
    visibility = np.array({visibility.tolist()})
    
    num_frames, num_points, _ = coords.shape
    print(f"Creating tracker with {{num_points}} tracks across {{num_frames}} frames")
    
    # Add tracks to the tracker
    for point_id in range(num_points):
        track_name = f"track_{{point_id + 1}}"
        
        # Add a new track
        tracker['tracks'].addTrack()
        track_index = tracker['tracks'].getNumTracks() - 1
        
        # Set track name
        tracker['tracks'][track_index]['name'].setValue(track_name)
        
        # Set track data for each frame
        for frame in range(num_frames):
            if visibility[frame, point_id]:
                x = float(coords[frame, point_id, 0])
                y = float(coords[frame, point_id, 1])
                
                # Set keyframes for track position
                tracker['tracks'][track_index]['track_x'].setValueAt(x, frame)
                tracker['tracks'][track_index]['track_y'].setValueAt(y, frame)
                
                # Enable the track for this frame
                tracker['tracks'][track_index]['enable'].setValueAt(1.0, frame)
            else:
                # Disable track for invisible frames
                tracker['tracks'][track_index]['enable'].setValueAt(0.0, frame)
    
    # Set reference frame to middle frame
    reference_frame = num_frames // 2
    tracker['reference_frame'].setValue(reference_frame)
    
    print(f"Tracker created successfully with {{num_points}} tracks")
    print(f"Reference frame set to: {{reference_frame}}")
    
    return tracker

def main():
    """Main function to create and save the tracker."""
    try:
        # Create the tracker
        tracker = create_tracker_with_data()
        
        # Save the script
        output_file = "{output_path}"
        nuke.scriptSaveAs(output_file)
        print(f"Nuke script saved to: {{output_file}}")
        
        return True
        
    except Exception as e:
        print(f"Error creating tracker: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Nuke tracker export completed successfully!")
    else:
        print("❌ Nuke tracker export failed!")
        sys.exit(1)
'''
    
    # Write the Nuke Python script to a temporary file
    script_path = output_path.replace('.nk', '_generator.py')
    with open(script_path, 'w') as f:
        f.write(nuke_python_script)
    
    return script_path

def export_to_nuke_via_api(coords, visibility, output_path, nuke_executable=None):
    """
    Export tracking data to Nuke using the Nuke Python API.
    
    Args:
        coords: numpy array of shape (num_frames, num_points, 2)
        visibility: numpy array of shape (num_frames, num_points)
        output_path: path where to save the .nk file
        nuke_executable: path to Nuke executable (optional)
    """
    
    # Try to find Nuke executable if not provided
    if nuke_executable is None:
        possible_paths = [
            "C:/Program Files/Nuke16.0v5/Nuke16.0.exe",
            "C:/Program Files/Nuke15.1v1/Nuke15.1.exe", 
            "C:/Program Files/Nuke14.0v5/Nuke14.0.exe",
            "C:/Program Files/Nuke13.2v7/Nuke13.2.exe",
            "/usr/local/Nuke16.0v5/Nuke16.0",
            "/Applications/Nuke16.0v5/Nuke16.0v5.app/Contents/MacOS/Nuke16.0",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                nuke_executable = path
                break
        
        if nuke_executable is None:
            raise Exception("Could not find Nuke executable. Please specify the path manually.")
    
    print(f"Using Nuke executable: {{nuke_executable}}")
    
    # Create the Nuke Python script
    script_path = create_nuke_tracker_script(coords, visibility, output_path)
    print(f"Generated Nuke Python script: {{script_path}}")
    
    # Execute Nuke in non-interactive mode
    import subprocess
    
    cmd = [
        nuke_executable,
        "-t",  # Non-interactive mode
        script_path
    ]
    
    print(f"Executing: {{' '.join(cmd)}}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Nuke execution successful!")
            print("STDOUT:", result.stdout)
            return output_path
        else:
            print("❌ Nuke execution failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            raise Exception(f"Nuke execution failed with return code {{result.returncode}}")
            
    except subprocess.TimeoutExpired:
        raise Exception("Nuke execution timed out after 60 seconds")
    except Exception as e:
        raise Exception(f"Error executing Nuke: {{e}}")

def test_nuke_api_export():
    """Test the Nuke API export with synthetic data."""
    
    print("🎬 Testing Nuke API export...")
    
    # Create simple test data (10 frames, 4 points)
    num_frames = 10
    num_points = 4
    
    # Create some simple moving points
    coords = np.zeros((num_frames, num_points, 2))
    visibility = np.ones((num_frames, num_points), dtype=bool)
    
    # Generate simple motion paths
    for frame in range(num_frames):
        for point in range(num_points):
            # Simple circular motion for each point
            angle = (frame / num_frames) * 2 * np.pi + (point * np.pi / 2)
            radius = 50 + point * 20
            center_x = 200 + point * 100
            center_y = 200
            
            coords[frame, point, 0] = center_x + radius * np.cos(angle)
            coords[frame, point, 1] = center_y + radius * np.sin(angle)
    
    print(f"📊 Generated test data: {{num_points}} points across {{num_frames}} frames")
    
    # Test the export
    output_path = "temp/nuke_api_export.nk"
    
    try:
        result_path = export_to_nuke_via_api(coords, visibility, output_path)
        print(f"✅ Nuke API export successful: {{result_path}}")
        
        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📄 Generated file size: {{file_size}} bytes")
            
            # Show first few lines
            with open(output_path, 'r') as f:
                lines = f.readlines()[:10]
                print("📝 First 10 lines:")
                for i, line in enumerate(lines):
                    print(f"   {{i+1:2d}}: {{line.rstrip()}}")
        
        return True
        
    except Exception as e:
        print(f"❌ Nuke API export failed: {{e}}")
        return False

if __name__ == "__main__":
    success = test_nuke_api_export()
    if success:
        print("\\n🎉 Test completed successfully!")
    else:
        print("\\n💥 Test failed!")
