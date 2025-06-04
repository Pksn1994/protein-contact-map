#!/bin/bash
# set -e

# echo "Installing dependencies"
# pip install -r requirements.txt
# echo "Navigating to main.py path"
# cd /protein_contact_map/contactmap
# echo "Running main.py"
# python main.py




Write-Host "Installing dep"
pip install -r requirements.txt
Write-Host "set main.py path"
Set-Location "protein_contact_map\contactmap"
Write-Host "Running main.py"
python main.py