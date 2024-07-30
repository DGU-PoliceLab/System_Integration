from Service.was import sendMessage, updateSnap
from Utils.snap import extract_face
from variable import get_arg
import copy

LAST_EVENT_TIME = {"falldown": None, "selfharm": None, "longterm_status": None, "violence": None}
DELAY_TIME = get_arg('root', 'event_delay')

def update(pipe): 
    def str_to_second(time_str):
        tl = list(map(int, time_str.split(":")))
        tn = tl[0] * 3600 + tl[1] * 60 + tl[2]
        return tn

    def check(event, cur_time):
        last_time = LAST_EVENT_TIME[event]
        if last_time == None:
            return True
        else:
            diff = cur_time - last_time
            if diff > DELAY_TIME:
                return True
            else:
                return False
    
    while True:
        event = pipe.recv()
        if event is not None:
            current_datetime = event["meta_data"]["current_datetime"]
            if event['action'] != "emotion":   
                event_type = event['action']
                event_time = copy.deepcopy(str(current_datetime)[11:19])            
                event_cur_time = str_to_second(event_time) # event insert delay code use here
                if check(event_type, event_cur_time): # event insert delay code use here
                    LAST_EVENT_TIME[event['action']] = event_cur_time

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
                target = event["meta_data"]["location_name"]
                # sendMessage(target, action, current_datetime)
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