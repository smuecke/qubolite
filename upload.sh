#!/bin/bash
echo "Uploading to PyPi"
python3 -m pip install --upgrade build twine
#python3.10 setup.py sdist
python3 -m build --sdist
python3 -m twine upload dist/qubolite-*.tar.gz --repository pypi
echo "Done"
