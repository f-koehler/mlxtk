import argparse
from pathlib import Path

import tqdm
import tqdm.auto

from mlxtk.scripts.repickle_simulation import repickle_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=str)
    args = parser.parse_args()

    for sim_dir in tqdm.tqdm(
        [str(sim) for sim in (Path(args.scan_dir) / "sim").iterdir() if sim.is_dir()],
        leave=False,
        position=0,
        desc="simulations",
    ):
        tqdm.auto.tqdm.write("================================================")
        tqdm.auto.tqdm.write("Entering: " + sim_dir)
        repickle_simulation(sim_dir, bar_offset=1)
        tqdm.auto.tqdm.write("================================================")


if __name__ == "__main__":
    pass
