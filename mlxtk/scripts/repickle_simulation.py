#!/usr/bin/env python
import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Optional

import tqdm
import tqdm.auto

from mlxtk.cwd import WorkingDir


def repickle_simulation(simulation_dir: str, bar_offset: Optional[int] = 0):
    with WorkingDir(Path(simulation_dir).resolve()):
        with open("doit.json", "r") as fptr:
            state = json.load(fptr)

        pickle_extensions = {
            ".mb_opr_pickle",
            ".prop_pickle",
            ".opr_pickle",
            ".wfn_pickle",
        }

        for task_name in tqdm.tqdm(
            state, position=bar_offset, leave=False, desc="tasks"
        ):
            if "deps:" not in state[task_name]:
                continue

            for dep in tqdm.tqdm(
                state[task_name]["deps:"],
                position=bar_offset + 1,
                leave=False,
                desc="dependencies",
            ):
                if Path(dep).suffix not in pickle_extensions:
                    continue

                with open(dep, "rb") as fptr:
                    obj = pickle.load(fptr)

                with open(dep, "wb") as fptr:
                    pickle.dump(obj, fptr, protocol=3)

                new_hash = subprocess.check_output(["md5sum", dep]).decode().split()[0]
                old_mtime, old_size, old_hash = state[task_name][dep]
                stat = os.stat(dep)
                new_mtime = stat.st_mtime
                new_size = stat.st_size

                tqdm.auto.tqdm.write("\t" + dep + ":")
                tqdm.auto.tqdm.write("\t\tsize:  {} -> {}".format(old_size, new_size))
                tqdm.auto.tqdm.write("\t\tmtime: {} -> {}".format(old_mtime, new_mtime))
                tqdm.auto.tqdm.write("\t\tmd5:   {} -> {}".format(old_hash, new_hash))

                state[task_name][dep] = [new_mtime, new_size, new_hash]

        with open("doit.json", "w") as fptr:
            json.dump(state, fptr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_dir", type=str)
    args = parser.parse_args()
    repickle_simulation(args.simulation_dir)


if __name__ == "__main__":
    main()
