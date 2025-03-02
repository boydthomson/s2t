#!/usr/bin/env python3
"""
Whisper Voice-to-Text Daemon
---------------------------
This script runs in the background and monitors a control file
to start/stop recording and transcription.

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

import pyaudio
import whisper

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
TEMP_DIR = tempfile.gettempdir()
AUDIO_FILE = os.path.join(TEMP_DIR, "speech_recording.wav")
CONTROL_FILE = "/tmp/whisper_control"

# Global variables
recording = False
frames = []
p = pyaudio.PyAudio()
stream = None
whisper_model = None

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
    global recording, stream, frames
    
    if recording:
        return
    
    frames = []
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

def stop_recording_and_process():
    """Stop recording and process the audio to text."""
    global recording, stream, frames
    
    if not recording:
        return
    
    recording = False
    
    # Close the stream
    if stream:
        stream.stop_stream()
        stream.close()
    
    print("Recording stopped. Processing...")
    
    # Save the recorded audio to a WAV file
    if frames:
        with wave.open(AUDIO_FILE, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        
        # Transcribe the audio
        try:
            result = whisper_model.transcribe(AUDIO_FILE)
            transcribed_text = result["text"].strip()
            
            if transcribed_text:
                print(f"Transcribed: {transcribed_text}")
                
                # Type the transcribed text using xdotool
                try:
                    subprocess.run(["xdotool", "type", transcribed_text], check=True)
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    print(f"xdotool error: {e}")
                    # Fallback: try to use xclip to place text in clipboard
                    try:
                        process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                        process.communicate(input=transcribed_text.encode())
                        print("Text placed in clipboard - please paste manually")
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print("Both xdotool and xclip failed. Please install either package.")
            else:
                print("No speech detected or transcription failed.")
        except Exception as e:
            print(f"Error during transcription: {e}")
    else:
        print("No audio recorded.")

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
                    stop_recording_and_process()
                
                last_command = command
            
            # If recording, collect frames
            if recording and stream:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
            
            time.sleep(0.01)  # Short sleep to prevent high CPU usage
            
        except Exception as e:
            print(f"Error monitoring control file: {e}")
            time.sleep(1)  # Longer sleep on error

def cleanup(signum, frame):
    """Clean up resources when exiting."""
    global p, stream, recording
    
    print("\nShutting down...")
    
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
    
    print("Whisper Voice-to-Text Daemon started")
    print("Press Ctrl+C to exit")
    
    # Monitor control file
    monitor_control_file()

if __name__ == "__main__":
    main()
