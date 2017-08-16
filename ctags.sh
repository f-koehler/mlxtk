#!/bin/bash
PYTHON_SOURCES=$(find mlxtk -iname "*.py")
rm -f ./.ctags
ctags -f .ctags --totals=yes $FORTRAN_SOURCES $PYTHON_SOURCES
