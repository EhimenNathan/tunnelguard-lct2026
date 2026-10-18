"""Close RViz through its GUI event loop before escalating to process signals.

ROS launch prefixes the real RViz executable with this module. A separate child
session prevents launch SIGINT from racing OGRE/Qt teardown in two threads.
The actual exit status is preserved; crashes are never hidden.
"""
import os
import shutil
import signal
import subprocess
import sys
import time


def request_window_close(pid):
    if not os.environ.get('DISPLAY') or not shutil.which('wmctrl'):
        return False
    windows=subprocess.run(['wmctrl','-lp'],capture_output=True,text=True,timeout=2)
    for line in windows.stdout.splitlines():
        fields=line.split(None,4)
        if len(fields)>=3 and fields[2]==str(pid):
            # RViz prepareToExit opens a modal Save/Discard/Cancel dialog
            # when its camera view changes. Save only the temporary session
            # config before closing; never discard a modified view silently.
            if shutil.which('xdotool'):
                subprocess.run(['wmctrl','-ia',fields[0]],timeout=2)
                subprocess.run(['xdotool','key','--window',fields[0],'--clearmodifiers','ctrl+s'],timeout=2)
                time.sleep(.25)
            print('RViz: saved session view; requesting GUI close',flush=True)
            return subprocess.run(['wmctrl','-ic',fields[0]],timeout=2).returncode==0
    return False


def main():
    if len(sys.argv)<2:raise SystemExit('Expected RViz executable and arguments')
    closing=False
    def stop(signum,frame):
        nonlocal closing
        closing=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    child=subprocess.Popen(sys.argv[1:],start_new_session=True)
    while not closing:
        try:raise SystemExit(child.wait(timeout=.1))
        except subprocess.TimeoutExpired:pass
    try:
        if request_window_close(child.pid):
            try:raise SystemExit(child.wait(timeout=3))
            except subprocess.TimeoutExpired:pass
    except (OSError,subprocess.SubprocessError):pass
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGKILL]:
        if child.poll() is not None:break
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        try:raise SystemExit(child.wait(timeout=5))
        except subprocess.TimeoutExpired:pass
    raise SystemExit(child.returncode)


if __name__=='__main__':main()
