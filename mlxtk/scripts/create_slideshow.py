import argparse
from pathlib import Path

from mlxtk.tools.video import create_slideshow
from mlxtk.util import list_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="folder with the input files")
    parser.add_argument("output", type=Path, help="path to the output file")
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default=".png",
        help="extension of the input files",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=20, help="duration for the video"
    )
    args = parser.parse_args()

    files = list_files(args.folder, extensions=[args.extension])
    create_slideshow(files, args.output, duration=args.duration)


if __name__ == "__main__":
    main()
