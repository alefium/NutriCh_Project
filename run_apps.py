#!/usr/bin/env python3
"""
Simple script to run both applications for testing
"""

import subprocess
import sys
import time
import webbrowser

from threading import Thread

def run_faq_manager():
    """Run the FAQ Manager application on port 5000"""
    print("Starting FAQ Manager on http://localhost:5000")
    subprocess.run([sys.executable, "src/vectordb_app.py"], cwd=".")

def run_chatbot():
    """Run the Chatbot application on port 5001"""
    print("Starting Chatbot on http://localhost:5001")
    subprocess.run([sys.executable, "src/chatbot_app.py"], cwd=".")

def run_faq_manager_background():
    """Run the FAQ Manager application in background thread"""
    subprocess.Popen([sys.executable, "src/vectordb_app.py"], cwd=".")

def run_chatbot_background():
    """Run the Chatbot application in background thread"""
    subprocess.Popen([sys.executable, "src/chatbot_app.py"], cwd=".")

def open_browsers():
    """Open both applications in browser tabs"""
    time.sleep(3)  # Wait for servers to start
    try:
        webbrowser.open("http://localhost:5000")
        time.sleep(1)
        webbrowser.open("http://localhost:5001")
    except:
        print("Could not open browsers automatically")
        print("Please open:")
        print("- FAQ Manager: http://localhost:5000")
        print("- Chatbot: http://localhost:5001")

if __name__ == "__main__":
    print("=" * 60)
    print("VECTOR DATABASE FAQ SYSTEM")
    print("=" * 60)
    print("This will start both applications:")
    print("1. FAQ Manager (Port 5000) - Upload and manage data")
    print("2. Chatbot (Port 5001) - Search and query data")
    print("=" * 60)
    
    choice = input("Choose option:\n1. Run FAQ Manager only\n2. Run Chatbot only\n3. Run both (in separate tabs)\nEnter choice (1-3): ")
    
    if choice == "1":
        run_faq_manager()
    elif choice == "2":
        run_chatbot()
    elif choice == "3":
        print("\nStarting both applications concurrently...")
        print("- FAQ Manager will start on http://localhost:5000")
        print("- Chatbot will start on http://localhost:5001")
        print("- Both browser tabs will open automatically")
        print("- Press CTRL+C to stop both applications")
        
        # Start both applications in background
        print("\nLaunching FAQ Manager...")
        run_faq_manager_background()
        
        print("Launching Chatbot...")
        run_chatbot_background()
        
        # Start browser opener in background
        browser_thread = Thread(target=open_browsers)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Keep the main process alive
        try:
            print("\nBoth applications are running.")
            print("Visit the URLs shown above to use the applications.")
            print("Press CTRL+C to stop both applications...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping applications...")
            print("Both applications have been terminated.")
    else:
        print("Invalid choice. Please run the script again.")
