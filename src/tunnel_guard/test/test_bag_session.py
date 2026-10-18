import signal
import os
import subprocess
import sys
import time
import pytest
from tunnel_guard.bag_session import exit_status


def test_only_requested_interrupt_is_expected_cancellation():
    assert exit_status(-signal.SIGINT, True) == 0
    assert exit_status(-signal.SIGINT, False) != 0
    for requested in (True, False):
        assert exit_status(0, requested) == 0
        assert exit_status(1, requested) == 1
        assert exit_status(-signal.SIGSEGV, requested) != 0
        assert exit_status(-9, requested) != 0  # Linux SIGKILL; absent from Windows signal module


@pytest.mark.skipif(os.name != 'posix', reason='Linux ROS process signalling')
def test_supervisor_cancels_real_child_and_preserves_crash(tmp_path):
    ready = tmp_path / 'ready'
    script = ('import signal,time;from pathlib import Path;'
              'signal.signal(signal.SIGINT,signal.SIG_DFL);'
              f'Path({str(ready)!r}).touch();time.sleep(30)')
    proc = subprocess.Popen([sys.executable, '-m', 'tunnel_guard.bag_session',
                             sys.executable, '-c', script])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=9) == 0
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()
    result = subprocess.run([sys.executable, '-m', 'tunnel_guard.bag_session',
                             sys.executable, '-c', 'raise SystemExit(7)'], timeout=10)
    assert result.returncode == 7


@pytest.mark.skipif(os.name != 'posix', reason='Linux detached launch signals')
def test_detached_launch_restores_inherited_sigint():
    result = subprocess.run(
        [sys.executable, '-m', 'tunnel_guard.launch_session', sys.executable, '-c',
         'import os,signal;os.kill(os.getpid(),signal.SIGINT);raise SystemExit(0)'],
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        capture_output=True, timeout=10)
    assert result.returncode == -signal.SIGINT
