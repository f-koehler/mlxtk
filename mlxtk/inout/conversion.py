import os.path
import shutil

import mlxtk.inout.compression
import mlxtk.inout.gpop
import mlxtk.inout.natpop

def convert_default_data(simulation_dir, output_dir):
    # compress output
    mlxtk.inout.compression.compress_file_gzip(
        os.path.join(simulation_dir, "output"),
        keep_original=True
    )
    shutil.move(
        os.path.join(simulation_dir, "output.gz"),
        os.path.join(output_dir, "output.gz")
    )

    # read gpop data and write compressed files for each DOF
    mlxtk.inout.gpop.write(
        mlxtk.inout.gpop.read_raw(os.path.join(simulation_dir, "gpop")),
        os.path.join(output_dir, "gpop")
    )

    # convert natpop
    mlxtk.inout.natpop.write(
        mlxtk.inout.natpop.read_raw(os.path.join(simulation_dir, "natpop")),
        os.path.join(output_dir, "natpop")
    )
