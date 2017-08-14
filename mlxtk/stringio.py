import sys
if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO
