import subprocess

def compress_file_gzip(path, keep_original=True):
    if not keep_original:
        subprocess.check_output(["gzip", path])
        return

    with open(path) as fh_in:
        with open(path + ".gz", "w") as fh_out:
            p = subprocess.Popen(["gzip"], stdout=fh_out, stdin=subprocess.PIPE)
            p.communicate(fh_in.read().encode())

def decompress_file_gzip(path):
    subprocess.check_output(["gzip", "-d", path])
