import sys

# pylint: disable=unused-import
if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO
