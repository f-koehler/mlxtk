from time import monotonic, perf_counter, process_time


class Timer:
    def __init__(self):
        self.monotonic_stop = 0.0
        self.process_stop = 0.0
        self.perf_stop = 0.0
        self.monotonic_start = monotonic()
        self.process_start = process_time()
        self.perf_start = perf_counter()

    def stop(self):
        self.perf_stop = perf_counter()
        self.process_stop = process_time()
        self.monotonic_stop = monotonic()

    def get_monotonic_time(self) -> float:
        return self.monotonic_stop - self.monotonic_start

    def get_process_time(self) -> float:
        return self.process_stop - self.process_start

    def get_perf_time(self) -> float:
        return self.perf_stop - self.perf_start
