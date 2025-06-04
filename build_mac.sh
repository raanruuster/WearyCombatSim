#!/bin/bash
# Script to build a standalone macOS application using PyInstaller
# Usage: ./build_mac.sh
set -e

# Ensure PyInstaller is available
pip install --upgrade pyinstaller

pyinstaller \
    --onefile \
    --windowed \
    --name WearyCombatSim \
    encounter_sim.py

# Output will be in the dist/ directory
