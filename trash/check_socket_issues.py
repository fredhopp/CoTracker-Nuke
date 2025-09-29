#!/usr/bin/env python3
"""
Check for Socket Issues
=======================

This script investigates potential socket-related errors in the CoTracker app.
"""

import socket
import subprocess
import sys
import time


def check_port_usage(port=7860):
    """Check if a port is in use and by what process."""
    print(f"Checking port {port} usage...")
    
    # Try to bind to the port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print(f"✓ Port {port} is in use (something is listening)")
            return True
        else:
            print(f"✓ Port {port} is available")
            return False
    except Exception as e:
        print(f"⚠ Error checking port {port}: {e}")
        return False


def check_gradio_socket_handling():
    """Test Gradio socket handling."""
    print("\nTesting Gradio socket behavior...")
    
    try:
        import gradio as gr
        
        # Create a simple interface
        def test_function(x):
            return f"Echo: {x}"
        
        interface = gr.Interface(fn=test_function, inputs="text", outputs="text")
        
        print("✓ Gradio interface created successfully")
        
        # Try to launch and immediately close
        try:
            # Launch in a non-blocking way
            interface.launch(prevent_thread_lock=True, show_error=True, quiet=True)
            print("✓ Gradio launched successfully")
            
            # Close immediately
            interface.close()
            print("✓ Gradio closed successfully")
            
        except Exception as e:
            print(f"⚠ Gradio launch/close error: {e}")
            
    except ImportError:
        print("⚠ Gradio not available for testing")
    except Exception as e:
        print(f"⚠ Gradio test error: {e}")


def check_socket_shutdown_issue():
    """Test socket shutdown behavior that might cause the error you mentioned."""
    print("\nTesting socket shutdown behavior...")
    
    try:
        # Create a socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to a test port
        test_port = 7861  # Use different port to avoid conflicts
        sock.bind(('127.0.0.1', test_port))
        sock.listen(1)
        
        print(f"✓ Socket created and listening on port {test_port}")
        
        # Try different shutdown methods
        try:
            sock.shutdown(socket.SHUT_RDWR)
            print("✓ socket.SHUT_RDWR shutdown successful")
        except Exception as e:
            print(f"⚠ socket.SHUT_RDWR shutdown error: {e}")
        
        try:
            sock.close()
            print("✓ Socket close successful")
        except Exception as e:
            print(f"⚠ Socket close error: {e}")
            
    except Exception as e:
        print(f"⚠ Socket test error: {e}")


def check_running_python_processes():
    """Check for running Python processes that might be holding sockets."""
    print("\nChecking for running Python processes...")
    
    try:
        # Try to find Python processes (Windows-compatible)
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            python_processes = [line for line in lines if 'python.exe' in line.lower()]
            
            if python_processes:
                print(f"Found {len(python_processes)} Python processes:")
                for proc in python_processes[:5]:  # Show first 5
                    print(f"  {proc}")
            else:
                print("✓ No Python processes found")
        else:
            print("⚠ Could not check running processes (tasklist failed)")
            
    except subprocess.TimeoutExpired:
        print("⚠ Process check timed out")
    except FileNotFoundError:
        print("⚠ tasklist command not found (not Windows?)")
    except Exception as e:
        print(f"⚠ Process check error: {e}")


def test_cotracker_app_socket_behavior():
    """Test the CoTracker app's socket behavior."""
    print("\nTesting CoTracker app socket behavior...")
    
    try:
        # Import the app
        from cotracker_nuke_app import create_gradio_interface
        
        print("✓ CoTracker app imported successfully")
        
        # Create interface
        interface = create_gradio_interface()
        print("✓ Interface created successfully")
        
        # Try to launch briefly
        try:
            interface.launch(prevent_thread_lock=True, show_error=True, quiet=True, server_port=7862)
            print("✓ App launched successfully on port 7862")
            
            # Wait a moment
            time.sleep(2)
            
            # Close
            interface.close()
            print("✓ App closed successfully")
            
        except Exception as e:
            print(f"⚠ App launch/close error: {e}")
            # Try to close anyway
            try:
                interface.close()
            except:
                pass
            
    except Exception as e:
        print(f"⚠ CoTracker app test error: {e}")


def main():
    """Run all socket-related checks."""
    print("SOCKET ISSUE INVESTIGATION")
    print("=" * 50)
    
    # Check port usage
    check_port_usage(7860)
    
    # Check running processes
    check_running_python_processes()
    
    # Test Gradio socket handling
    check_gradio_socket_handling()
    
    # Test socket shutdown behavior
    check_socket_shutdown_issue()
    
    # Test CoTracker app
    test_cotracker_app_socket_behavior()
    
    print("\n" + "=" * 50)
    print("SOCKET INVESTIGATION COMPLETE")
    print("=" * 50)
    
    print("\nCommon causes of socket.SHUT_RDWR errors:")
    print("1. Process terminated while socket still open")
    print("2. Multiple instances trying to use same port")
    print("3. Gradio not properly cleaning up sockets")
    print("4. Windows firewall or antivirus interference")
    print("5. Port already in use by another application")
    
    print("\nRecommended solutions:")
    print("1. Always use interface.close() when done")
    print("2. Use different ports for testing (server_port parameter)")
    print("3. Check for zombie Python processes")
    print("4. Restart terminal/IDE if issues persist")


if __name__ == "__main__":
    main()
