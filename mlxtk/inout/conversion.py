import os.path
import shutil

import inout.gpop
import inout.compression

def convert_default_data(simulation_dir, output_dir):
    # compress output
    inout.compression.compress_file_gzip(
        os.path.join(simulation_dir, "output"),
        keep_original=True
    )
    shutil.move(
        os.path.join(simulation_dir, "output.gz"),
        os.path.join(output_dir, "output.gz")
    )

    # read gpop data and write compressed files for each DOF
    inout.gpop.write(
        inout.gpop.read_raw(path),
        os.path.join(output_dir, "gpop")
    )
