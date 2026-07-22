#!/usr/bin/env python3
"""Samples /proc/stat into a CSV until killed.

Spawned as a separate process by benchmark.py, because add_model() holds the GIL
for the whole model load and would freeze an in-process sampler thread exactly
where it matters.

Usage: sampler.py CPU_CSV [INTERVAL_SECONDS]
"""
import os
import sys
import time


def main():
    cpu_out = open(sys.argv[1], "w", buffering=1)
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    stat_fd = os.open("/proc/stat", os.O_RDONLY)

    while True:
        now = time.time()
        row = [f"{now:.3f}"]
        # 4096 bytes covers "cpu" + the per-core lines; "intr" follows them
        for line in os.pread(stat_fd, 4096, 0).decode().splitlines():
            field = line.split()
            if not field[0].startswith("cpu"):
                break
            raw = [int(v) for v in field[1:9]]  # user..steal; guest double-counts
            if field[0] == "cpu":
                row += [str(v) for v in raw]
            else:
                total = sum(raw)
                row += [str(total - raw[3] - raw[4]), str(total)]  # busy, total
        cpu_out.write(",".join(row) + "\n")
        time.sleep(max(0.0, interval - (time.time() - now)))


if __name__ == "__main__":
    main()
