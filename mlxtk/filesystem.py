import logging
import os
import subprocess


def relative_symlink(src, dst):
    unlink_if_present(dst)
    logging.debug("Create symlink: %s -> %s", src, dst)
    subprocess.check_output(["ln", "-sr", src, dst])


def unlink_if_present(link):
    if os.path.islink(link):
        logging.debug("Remove symlink: %s", link)
        os.unlink(link)
