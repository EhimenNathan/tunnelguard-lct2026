"""Give ROS launch an explicit result for a user-cancelled bag player.

Humble's CLI can terminate by SIGINT after playback completes. Only that signal,
after a shutdown request to this supervisor, is an expected cancellation. All
other exits, including crashes and playback errors, retain their failure status.
"""
import os
import signal
import subprocess
import sys
import time


def exit_status(code, requested):
    if requested and code == -signal.SIGINT:
        return 0
    return code if code >= 0 else 128 - code


def main():
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    while child.poll() is None and not requested:
        time.sleep(.05)
    if requested and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    code = child.wait()
    if requested and code == -signal.SIGINT:
        print('Bag playback cancelled by requested SIGINT', flush=True)
    raise SystemExit(exit_status(code, requested))


if __name__ == '__main__':
    main()
