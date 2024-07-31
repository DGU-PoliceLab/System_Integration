from Service.was import sendMessage, updateSnap
from Utils.snap import extract_face
from variable import get_arg
import copy
import time
from datetime import datetime

DELAY = 15

def check_time_gap(pre):
    now = datetime.now()
    gap = now - pre
    gap_seconds = gap.seconds
    if gap_seconds > DELAY:
        return True
    else:
        return False



def update(pipe): 
    last_occured = {"violence": None, "selfharm": None, "falldown": None, "longterm_status": None}
    while True:
        event = pipe.recv()
        if event is not None:
            if event['action'] != "emotion":
            
                current_datetime = event["meta_data"]["current_datetime"]
                target = event["meta_data"]["location_name"]

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
                if last_occured[event['action']] == None:
                    last_occured[event['action']] = datetime.now()
                    sendMessage(target, action, current_datetime)
                else:
                    if check_time_gap(last_occured[event['action']]):
                        last_occured[event['action']] = datetime.now()
                        sendMessage(target, action, current_datetime)
                        
            else:
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
                    if td['bbox'] == None:
                        bbox = [100, 100, 200, 200]
                    else:
                        bbox = td['bbox']
                    thumb = extract_face(event['meta_data']['frame'], bbox)
                    form = {
                        "tid": td['id'],
                        "thumb": thumb, 
                        "heart": heart, 
                        "breath": breath, 
                        "temp": 32.5,
                        "emotion": td['emotion_index']}
                    targetData.append(form)
                updateSnap(event['meta_data']['location_name'], targetData)
        else:
            time.sleep(0.0001)