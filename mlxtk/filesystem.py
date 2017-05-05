import logging
import os
import subprocess


def relative_symlink(src, dst):
    if os.path.lexists(dst):
        logging.debug("Remove symlink: %s", dst)
        os.unlink(dst)

    logging.debug("Create symlink: %s -> %s", src, dst)
    subprocess.check_output(["ln", "-sr", src, dst])
