#!/bin/bash
# Save as ~/dev/s2t/toggle-recording.sh and make executable with chmod +x

# Check current state
if [ -f /tmp/whisper_recording ]; then
    # Stop recording
    rm /tmp/whisper_recording
    echo stop > /tmp/whisper_control
    
    # Visual notification
    if command -v notify-send &> /dev/null; then
        notify-send -t 1000 "Speech recording stopped"
    fi
else
    # Start recording
    touch /tmp/whisper_recording
    echo start > /tmp/whisper_control
    
    # Visual notification
    if command -v notify-send &> /dev/null; then
        notify-send -t 1000 "Speech recording started" -h string:bgcolor:#FF0000
    fi
fi
