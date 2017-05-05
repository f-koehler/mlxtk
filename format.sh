#!/bin/bash
FILES=$(find -name "*.py")
autopep8 -j 4 -i -a -a -v -v $FILES
yapf -i --style google $FILES
