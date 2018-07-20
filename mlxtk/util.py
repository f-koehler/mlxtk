import multiprocessing
import os
import shutil

from .log import get_logger

LOGGER = get_logger(__name__)


def copy(src, dst, force=False):
    if (not force) and os.path.exists(dst):
        raise FileExistsError(dst)

    LOGGER.debug("copy: %s -> %s", src, dst)
    shutil.copy2(src, dst)


def create_fresh_symlink(path):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.exists(path):
        os.remove(path)
    os.symlink(os.devnull, path)


class TemporaryCopy(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.copy = False

    def __enter__(self):
        if not os.path.exists(self.dst):
            LOGGER.debug("create temporary copy: %s -> %s", self.src, self.dst)
            copy(self.src, self.dst)
            self.copy = True
        else:
            LOGGER.debug("no creation of temporary file necessary")
            self.copy = False

    def __exit__(self, type, value, traceback):
        if self.copy:
            LOGGER.debug("remove temporary copy: %s", self.dst)
            os.remove(self.dst)
        else:
            LOGGER.debug("no clean-up of temporary copy necessary")


def parallel_map_helper(function, queue_in, queue_out):
    while True:
        # get new set of arguments for the function f
        # i denotes the index of this arugment set
        i, args = queue_in.get()

        # if there is no more work left to do, a tuple (None, None) is passed
        # we can stop in this case
        if i is None:
            break

        # the index of the given argument set and the result of the
        # corresponding function call are stored in the output queue
        queue_out.put((i, function(args)))


def parallel_map(function, arguments, num_workers=multiprocessing.cpu_count()):
    queue_in = multiprocessing.Queue(1)
    queue_out = multiprocessing.Queue()

    # start the desired number of worker processes (specified by num_workers)
    # each process will run will run the parallel_map_helper function
    processes = [
        multiprocessing.Process(
            target=parallel_map_helper, args=(function, queue_in, queue_out))
        for _ in range(num_workers)
    ]

    for process in processes:
        # set all processes to daemon mode
        # this will stop all child processes when the creating process is
        # stopped (e.g. by  C-c on the keyboard)
        process.daemon = True

        # start all worker processes
        process.start()

    # fill the input queue with all requested function arguments
    # the arguments are passed in numbered tuples
    sent = [queue_in.put((i, args)) for i, args in enumerate(arguments)]

    # Each worker process will stop once it gets passed the tuple (None, None)
    # for the new arguments. This indicated that there is no more work left
    # to do and the process will stop. Each worker process gets exactly one
    # of such end indicators
    for _ in range(num_workers):
        queue_in.put((None, None))

    # the results of the function calls are stored in the output queue
    results = [queue_out.get() for _ in range(len(sent))]

    # wait for all processes to stop
    for process in processes:
        process.join()

    # Return a the list of all function call return values which are sorted
    # by the index of argument index. This ensures that the results are in the
    # same order as the corresponding arguments.
    return [result for i, result in sorted(results)]
