import requests
import json
import time
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ENDPOINT = "https://was:40000"
ENDPOINT = "https://host.docker.internal:40000" #컨테이너가 아닌 로컬에서 WAS 실행시

def check():
    try:
        response = requests.get(ENDPOINT, verify=False)
        print(response.json())
    except Exception as e:
        print(e)

def readActiveCctvList(debug):
    try:
        if debug:
            return [{
            "cctv_id": 0,
            "cctv_name": "엣지카메라",
            "cctv_ip": "/System_Integration/Input/videos/mhn_demo_1.mp4",
            "location_id": 0,
            "location_name": '유치실1',
            "thermal_ip": None,
            "thermal_port": None,
            "rader_ip": None,
            "rader_port": None,
        }]
        else:
            cctv_data = []
            api_url = ENDPOINT + "/location/read/cctv"
            response = requests.post(api_url, verify=False)
            data = response.json()
            for row in data:
                cctv_info = {}
                cctv_info['cctv_id'] = row[2]
                cctv_info['cctv_name'] = row[3]
                cctv_info['cctv_ip'] = row[4]
                cctv_info['location_id'] = row[0]
                cctv_info['location_name'] = row[1]
                cctv_info['thermal_ip'] = row[6]
                cctv_info['thermal_port'] = row[7]
                cctv_info['rader_ip'] = row[8]
                cctv_info['rader_port'] = row[9]
                cctv_info['toilet_rader_ip'] = row[10]
                cctv_info['toilet_rader_port'] = row[11]
                cctv_data.append(cctv_info)
            return cctv_data
    except Exception as e:
        print(e)
        return [{
            "cctv_id": 0,
            "cctv_name": "엣지카메라",
            "cctv_ip": "/System_Integration/Input/videos/mhn_demo_1.mp4",
            "location_id": 0,
            "location_name": '유치실1',
            "thermal_ip": None,
            "thermal_port": None,
            "rader_ip": None,
            "rader_port": None,
        }]

def sendMessage(target, action, timedata):
    try:
        api_url = ENDPOINT + "/message/send"
        timestamp = time.mktime(timedata.timetuple())
        requestData = {
            "key": "event",
            "message": {"event": action, "location": target, "occurred_at": timestamp}
        }
        response = requests.post(api_url, data=json.dumps(requestData), verify=False)
        return True
    except:
        return False

def updateSnap(target, data):
    try:
        api_url = ENDPOINT + "/snap/update"
        requestData = {
            "target": target,
            "data": data
        }
        response = requests.post(api_url, data=json.dumps(requestData), verify=False)
        return True
    except:
        return False