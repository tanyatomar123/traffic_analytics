#!/bin/bash
# setup.sh for Streamlit deployment

# Create necessary directories
mkdir -p baggage_detection/yolov8_baggage2/weights

# Install required packages
pip install -r requirements.txt

echo "Setup completed successfully!"
