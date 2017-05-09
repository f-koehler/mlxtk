#!/bin/bash
FILES=$(find -name "*.py")
autopep8 -j 4 -i -a -a -v -v --max-line-length 99 $FILES
yapf -i --style google $FILES
