#!/usr/bin/env python3
"""
Real-time Whisper Voice-to-Text Daemon
-------------------------------------
This script performs incremental transcription while recording.
It processes audio in chunks and outputs text as it's transcribed.

Requirements:
- pyaudio: for audio recording
- whisper: for speech recognition
- xdotool: for simulating keyboard input
"""

import os
import tempfile
import subprocess
import time
import wave
import sys
import signal
import threading
import queue
from collections import deque

import numpy as np
import pyaudio
import whisper

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4096  # Larger chunk for better processing
MODEL_SIZE = "tiny"  # Use smaller model for faster processing
TEMP_DIR = tempfile.gettempdir()
CONTROL_FILE = "/tmp/whisper_control"
SEGMENT_LENGTH = 2  # Process audio in 2-second segments
BUFFER_LENGTH = 10  # Keep a 10-second rolling buffer

# Global variables
recording = False
p = pyaudio.PyAudio()
stream = None
whisper_model = None
audio_buffer = deque(maxlen=int(BUFFER_LENGTH * SAMPLE_RATE))
audio_queue = queue.Queue()
transcription_thread = None
stop_threads = False
last_text = ""

def setup():
    """Initialize the whisper model and create the control file."""
    global whisper_model
    
    # Create control file if it doesn't exist
    if not os.path.exists(CONTROL_FILE):
        with open(CONTROL_FILE, 'w') as f:
            f.write("ready\n")
    
    # Load Whisper model
    print(f"Loading Whisper model ({MODEL_SIZE})...")
    whisper_model = whisper.load_model(MODEL_SIZE)
    print("Model loaded and ready!")

def start_recording():
    """Start recording audio."""
    global recording, stream, audio_buffer, stop_threads, transcription_thread
    
    if recording:
        return
    
    # Clear the buffer
    audio_buffer.clear()
    
    # Reset stop flag
    stop_threads = False
    
    # Start transcription thread
    transcription_thread = threading.Thread(target=transcription_worker)
    transcription_thread.daemon = True
    transcription_thread.start()
    
    recording = True
    
    # Open audio stream
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("Recording started...")
    except Exception as e:
        print(f"Error starting recording: {e}")
        recording = False

def transcription_worker():
    """Worker thread to process audio chunks."""
    global audio_queue, stop_threads, last_text
    
    while not stop_threads:
        try:
            # Get audio data from queue with a timeout
            audio_data = audio_queue.get(timeout=0.5)
            
            # Convert to float32 numpy array for whisper
            audio_float32 = audio_data.astype(np.float32) / 32768.0  # Scale from int16 to float32
            
            # Transcribe the audio segment
            result = whisper_model.transcribe(
                audio_float32, 
                initial_prompt=last_text,
                language="en"
            )
            
            transcription = result["text"].strip()
            
            if transcription:
                # Only output new text, avoiding duplicates
                if transcription != last_text and not last_text.endswith(transcription):
                    # Check if new transcription is longer than last and starts similarly
                    if len(transcription) > len(last_text) and transcription.startswith(last_text[:10]):
                        # Only output the new part
                        new_text = transcription[len(last_text):]
                    else:
                        # Output full text with a space
                        new_text = " " + transcription
                        
                    print(f"Transcribed: {new_text}")
                    
                    # Type the transcribed text using xdotool
                    try:
                        subprocess.run(["xdotool", "type", new_text], check=True)
                    except Exception as e:
                        print(f"Error typing text: {e}")
                    
                    # Update last text
                    last_text = transcription
                
            # Mark task as done
            audio_queue.task_done()
            
        except queue.Empty:
            # No audio data available, just continue
            pass
        except Exception as e:
            print(f"Error in transcription worker: {e}")
            time.sleep(0.1)

def stop_recording():
    """Stop recording."""
    global recording, stream, stop_threads, transcription_thread
    
    if not recording:
        return
    
    recording = False
    stop_threads = True
    
    # Close the stream
    if stream:
        stream.stop_stream()
        stream.close()
    
    # Wait for transcription thread to finish
    if transcription_thread:
        transcription_thread.join(timeout=2)
    
    print("Recording stopped.")
    
    # Type a newline to end the transcription
    try:
        subprocess.run(["xdotool", "key", "Return"], check=True)
    except Exception as e:
        print(f"Error typing newline: {e}")

def process_audio_chunk(chunk):
    """Process a chunk of audio data and add to buffer."""
    global audio_buffer, audio_queue
    
    # Convert bytes to numpy array
    audio_data = np.frombuffer(chunk, dtype=np.int16)
    
    # Add to rolling buffer
    audio_buffer.extend(audio_data)
    
    # If we have enough data, send to the transcription queue
    if len(audio_buffer) >= SEGMENT_LENGTH * SAMPLE_RATE:
        # Get the last N seconds of audio
        segment_audio = np.array(list(audio_buffer))
        
        # Add to processing queue
        audio_queue.put(segment_audio)

def monitor_control_file():
    """Monitor the control file for commands."""
    last_command = ""
    
    while True:
        try:
            with open(CONTROL_FILE, 'r') as f:
                command = f.read().strip()
            
            if command != last_command:
                if command == "start":
                    start_recording()
                elif command == "stop":
                    stop_recording()
                
                last_command = command
            
            # If recording, collect and process audio
            if recording and stream:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    process_audio_chunk(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
            
            time.sleep(0.01)  # Short sleep to prevent high CPU usage
            
        except Exception as e:
            print(f"Error monitoring control file: {e}")
            time.sleep(1)  # Longer sleep on error

def cleanup(signum, frame):
    """Clean up resources when exiting."""
    global p, stream, recording, stop_threads
    
    print("\nShutting down...")
    
    stop_threads = True
    
    if recording:
        recording = False
        if stream:
            stream.stop_stream()
            stream.close()
    
    p.terminate()
    sys.exit(0)

def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Setup whisper and control file
    setup()
    
    print("Real-time Whisper Voice-to-Text Daemon started")
    print("Press Ctrl+C to exit")
    
    # Monitor control file
    monitor_control_file()

if __name__ == "__main__":
    main()
