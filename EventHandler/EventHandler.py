import time
from multiprocessing import Process, Pipe, Queue
from Utils.logger import get_logger
from threading import Thread
from queue import Queue
import pymysql
import requests
import pika
import json
import datetime
import copy

ENDPOINT = "https://was:40000"

class EventHandler:
    def __init__(self, root_args=None):
        if root_args == None:
            print("EventHandler initialization failed")
            exit()

        self.logger = get_logger(name= '[EventHandler]', console= True, file= True)

        self._last_action_time = {"falldown": None, "selfharm": None, "longterm_status": None, "violence": None}
        self._event_delay = root_args.event_delay
        
    def update(self, pipe):
        return -1 # TODO fix it.

        while True:
            event = pipe.recv()    
            if event is not None:
                print(event)
                url = ENDPOINT + "/message/send"

                if event['action'] != "emotion":       
                    if event['action'] == "selfharm":
                        action = "자해"
                    elif event['action'] == "falldown":
                        action = "낙상"
                    elif event['action'] == "emotion":
                        action = "감정"
                    elif event['action'] == "longterm_status":
                        action = "장시간 고정 자세"
                    elif event['action'] == "violence":
                        action = "폭행"

                if event['action'] != "emotion":
                    import time
                    timestamp = time.mktime(event['current_datetime'].timetuple())
                    requestData = {
                        "key": "event",
                        "message": {"event": action, "location": '현관', "occurred_at": timestamp}
                    }
                    response = requests.post(url, data=json.dumps(requestData), verify=False)
                else:
                    url = ENDPOINT + "/snap/update"
                    targetData = []

                    for td in event['combine_list']:
                        if td['combine_dict']['heart'] == None:
                            heart = 0
                        else:
                            heart = td['combine_dict']['heart']
                        if td['combine_dict']['breath'] == None:
                            breath = 0
                        else:
                            breath = td['combine_dict']['breath']

                        if td['bbox'] == None: # TODO fix it.
                            bbox = [100, 100, 200, 200]
                        else:
                            bbox = [100, 100, 200, 200]
                
                        form = {
                            "tid": td['id'],
                            "thumb": "", 
                            "bbox": bbox,
                            "heart": heart, 
                            "breath": breath, 
                            "temp": 32.5,
                            "emotion": td['emotion_index']}
                        targetData.append(form)

                    requestData = {
                        "target": "현관",
                        "url": "rtsp://admin:wonwoo0!23@172.30.1.42/stream1",
                        "data": targetData
                    }
                    print(requestData)
                    response = requests.post(url, data=json.dumps(requestData), verify=False)
                    print(response)
            else:
                time.sleep(0.0001)