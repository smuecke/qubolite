#!/bin/bash
echo "Uploading to PyPi"
python3.10 setup.py sdist
python3.10 -m twine upload dist/*
echo "Done"