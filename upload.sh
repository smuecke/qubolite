#!/bin/bash
echo "Uploading to PyPi"
#python3.10 setup.py sdist
python3.10 -m build --sdist
python3.10 -m twine upload dist/qubolite-*.tar.gz --repository qubolite
echo "Done"
