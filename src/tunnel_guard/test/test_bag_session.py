import signal
from tunnel_guard.bag_session import exit_status


def test_only_requested_interrupt_is_expected_cancellation():
    assert exit_status(-signal.SIGINT, True) == 0
    assert exit_status(-signal.SIGINT, False) != 0
    for requested in (True, False):
        assert exit_status(0, requested) == 0
        assert exit_status(1, requested) == 1
        assert exit_status(-signal.SIGSEGV, requested) != 0
        assert exit_status(-signal.SIGKILL, requested) != 0
