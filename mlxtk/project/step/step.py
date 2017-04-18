import logging
import os
import timeit


class Step:

    def __init__(self, name, input_files, output_files):
        self.input = input_files
        self.name = name
        self.output = output_files
        self.success = False
        self.time = None

    def is_outdated(self):
        if not self.success:
            return True
        if not self.input:
            return True
        if not self.output:
            return True

        newest_input = self.input[0]
        mtime = os.path.getmtime(self.input[0])
        for i in range(1, len(self.input)):
            new_mtime = os.path.getmtime(self.input[i])
            if new_mtime > mtime:
                newest_input = self.input[i]
                mtime = new_mtime

        oldest_output = self.output[0]
        mtime = os.path.getmtime(self.output[0])
        for i in range(1, len(self.output)):
            new_mtime = os.path.getmtime(self.output[i])
            if new_mtime > mtime:
                oldest_output = self.output[i]
                mtime = new_mtime

        return oldest_output <= newest_input

    def clean(self):
        for o in self.output:
            if os.path.exists(o):
                os.remove(o)

    def run(self):
        if not self.is_outdated():
            logging.info("Step {} is up-to-date, skipping".format(self.name))

        logging.info("Run Step {}".format(name))
        self.time = timeit.default_timer()
        self.success = self.execute()
        self.time = timeit.default_timer() - self.time

        if not self.success:
            logging.error("Step {} failed after {}s!".format(self.name, time))
            exit(1)
        logging.info("Finished Step {} in {}s".format(name, time))

    def execute(self):
        return False
