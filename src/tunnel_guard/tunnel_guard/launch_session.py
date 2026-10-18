"""Restore terminal signals before starting a detached ROS launch process.

Bash background jobs can inherit SIGINT ignored. Keeping that disposition makes
tg stop wait until escalation instead of letting launch shut down its children.
"""
import os
import signal
import sys


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()
