#!/bin/bash

FILES=`find mlxtk -name "*.py"`

echo "Run autopep8"
autopep8 --in-place ${FILES}

echo "Run black"
black --skip-string-normalization ${FILES}

echo "Run yapf"
yapf --in-place ${FILES}
