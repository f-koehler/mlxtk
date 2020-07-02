import pickle
from pathlib import Path
from typing import List, Union

from mlxtk.tools import video
from mlxtk.util import make_path


def create_slideshow(
    images: List[Union[Path, str]],
    output_file: Union[Path, str],
    duration: float = 20.0,
):
    output_file = make_path(output_file)
    pickle_file = output_file.with_suffix(output_file.suffix + ".pickle")

    def action_write_pickle(duration, targets):
        make_path(targets[0]).parent.mkdir(parents=True, exist_ok=True)
        obj = {"duration": duration}
        with open(targets[0], "wb") as fptr:
            pickle.dump(obj, fptr, protocol=3)

    yield {
        "name": "slideshow:{}:pickle".format(str(pickle_file)).replace("=", "_"),
        "targets": [pickle_file,],
        "clean": True,
        "actions": [(action_write_pickle, [duration])],
    }

    def action_create_slideshow(duration, images, targets):
        video.create_slideshow(images, targets[0], duration)

    yield {
        "name": "slideshow:{}:create".format(str(output_file)).replace("=", "_"),
        "targets": [output_file,],
        "clean": True,
        "actions": [(action_create_slideshow, [duration, images])],
    }
