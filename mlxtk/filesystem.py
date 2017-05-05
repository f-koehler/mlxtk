import os
import subprocess


def relative_symlink(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    subprocess.check_output(["ln", "-sr", src, dst])
