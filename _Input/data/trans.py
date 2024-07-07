import time
import json

base = []
cur_track_info = {}
cur_vital_info = []
cur_frame = 0
raw_data = open("rader_raw_data.txt", "r")
result = []
for i, row in enumerate(raw_data):
    if i % 2 == 0:
        row = row.replace("\n", "")
        row = row.replace("Track_info : ", "")
        cur_track_info = eval(row)
    else:
        row = row.replace("\n", "")
        temp_viatal_info = row.split(", ")
        vital_id = int(temp_viatal_info[0].split(" : ")[-1])
        heartbeat_rate = int(temp_viatal_info[1].split(" : ")[-1])
        breath_rate = int(temp_viatal_info[2].split(" : ")[-1])
        
        if vital_id not in base:
            base.append(vital_id)
            pos = cur_track_info.get(vital_id)
            if pos != None:
                cur_vital_info.append({"id": vital_id, "pos": pos, "heartbeat_rate": heartbeat_rate, "breath_rate": breath_rate})
        else:
            print(cur_frame, cur_track_info, cur_vital_info)
            result.append({"frame": cur_frame, "track_info": cur_track_info, "vital_info": cur_vital_info})
            cur_frame += 1
            base = [vital_id]
            cur_vital_info = []
            pos = cur_track_info.get(vital_id)
            if pos != None:
                cur_vital_info.append({"id": vital_id, "pos": pos, "heartbeat_rate": heartbeat_rate, "breath_rate": breath_rate})
with open('rader_data.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent="\t")