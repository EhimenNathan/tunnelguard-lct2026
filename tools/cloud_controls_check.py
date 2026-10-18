"""Сквозной тест tg: пауза, шаг, скорость и смена топика без перезапуска узла."""
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import tempfile
import yaml
import rclpy
from rclpy.node import Node
from rosbag2_interfaces.srv import Pause, Resume, PlayNext, SetRate, GetRate
from tunnel_guard_msgs.msg import ObstacleStatus

def main(bag,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Raw organizer data must never be included in uploaded test artifacts,
    # including when a later service call or cleanup fails.
    alt=Path(tempfile.mkdtemp(prefix='tg_controls_switch_'));shutil.copytree(bag,alt,dirs_exist_ok=True)
    for db in alt.glob('*.db3'):
        with sqlite3.connect(db) as c:c.execute("UPDATE topics SET name='/falcon_secondary' WHERE type='sensor_msgs/msg/PointCloud2'")
    mp=alt/'metadata.yaml';meta=yaml.safe_load(mp.read_text())
    for t in meta['rosbag2_bagfile_information']['topics_with_message_count']:
        if t['topic_metadata']['type']=='sensor_msgs/msg/PointCloud2':t['topic_metadata']['name']='/falcon_secondary'
    mp.write_text(yaml.safe_dump(meta))
    rclpy.init();node=Node('controls_probe');messages=[]
    sub=node.create_subscription(ObstacleStatus,'/tunnel_guard/status',lambda m:messages.append(m),100)
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
    def call(name,kind,**kwargs):
        c=node.create_client(kind,'/rosbag2_player/'+name)
        if not c.wait_for_service(timeout_sec=15):raise RuntimeError('no player service '+name)
        req=kind.Request()
        for k,v in kwargs.items():setattr(req,k,v)
        fut=c.call_async(req);rclpy.spin_until_future_complete(node,fut,timeout_sec=10)
        if not fut.done():raise RuntimeError('service timeout '+name)
        result=fut.result();node.destroy_client(c);return result
    def pid():
        return subprocess.check_output(['pgrep','-f','/tunnel_guard/detector_node'],text=True).strip()
    subprocess.run(['tg','start','false'],check=True,timeout=40)
    before=pid();checks={};player=None
    try:
        with (out/'controls_play.log').open('w') as log:
            player=subprocess.Popen(['tg','play',bag,'0.5','--start-paused'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            spin(6);checks['starts_paused']=len(messages)==0
            call('resume',Resume);spin(4);checks['resume']=len(messages)>=10
            checks['initial_rate_half']=abs(call('get_rate',GetRate).rate-.5)<1e-6
            call('pause',Pause);spin(1);count=len(messages);spin(1.5);checks['pause']=len(messages)==count
            call('play_next',PlayNext);spin(2);checks['single_step']=len(messages)==count+1
            call('set_rate',SetRate,rate=.25);checks['change_rate']=abs(call('get_rate',GetRate).rate-.25)<1e-6
            os.killpg(player.pid,signal.SIGINT);player.wait(timeout=10);player=None;spin(2)
            count=len(messages)
            player=subprocess.Popen(['tg','play',str(alt),'1.0','--start-paused'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            spin(6);call('resume',Resume);spin(5)
            checks['second_bag_output']=len(messages)>count+20
            checks['same_detector_process']=pid()==before
            node_log=Path('/tmp/tunnel_guard.log').read_text()
            checks['new_topic_selected']='subscribed to /falcon_secondary' in node_log
            (out/'controls_node.log').write_text(node_log)
    finally:
        if player is not None and player.poll() is None:
            os.killpg(player.pid,signal.SIGINT);player.wait(timeout=10)
        try:
            # tg stop allows up to 12 s for launch/RViz shutdown, plus CLI setup.
            stopped=subprocess.run(['tg','stop'],check=False,timeout=20)
            checks['stop_command']=stopped.returncode==0
            log_text=Path('/tmp/tunnel_guard.log').read_text()
            (out/'controls_node.log').write_text(log_text)
            checks['clean_detector_shutdown']='process has died' not in log_text and 'Traceback' not in log_text
            checks['detector_stopped']=subprocess.run(['pgrep','-f','/tunnel_guard/detector_node'],capture_output=True).returncode==1
        finally:
            node.destroy_node();rclpy.shutdown()
            shutil.rmtree(alt)
    report=dict(checks=checks,passed=all(checks.values()),received=len(messages))
    (out/'controls_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit('Проверка управления не пройдена')

if __name__=='__main__':main(*sys.argv[1:3])
