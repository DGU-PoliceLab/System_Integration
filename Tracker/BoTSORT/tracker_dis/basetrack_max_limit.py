import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0
    _max_count = 3  # 최대 객체 수 설정
    person_track_count = 0
    head_track_count = 0
    lp_track_count = 0
    _available_ids = set()  # 재사용 가능한 ID를 저장할 집합
    
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id(tracker_num):
        if tracker_num == 0:
            if BaseTrack._available_ids:
                return BaseTrack._available_ids.pop()  # 재사용 가능한 ID가 있으면 반환
            if BaseTrack._count >= BaseTrack._max_count:
                print("basetrack.py : 최대 객체 수를 초과했습니다. 일단 넘어감.")
                return BaseTrack.person_track_count
                
            BaseTrack.person_track_count += 1
            return BaseTrack.person_track_count
        
        elif tracker_num == 1:
            if BaseTrack._available_ids:
                return BaseTrack._available_ids.pop()  # 재사용 가능한 ID가 있으면 반환
            if BaseTrack._count >= BaseTrack._max_count:
                print("basetrack.py : 최대 객체 수를 초과했습니다. 일단 넘어감.")
                return BaseTrack.person_track_count
            
            BaseTrack.head_track_count += 1
            return BaseTrack.head_track_count
        elif tracker_num == 2:
            if BaseTrack._available_ids:
                return BaseTrack._available_ids.pop()  # 재사용 가능한 ID가 있으면 반환
            if BaseTrack._count >= BaseTrack._max_count:
                print("basetrack.py : 최대 객체 수를 초과했습니다. 일단 넘어감.")
                return BaseTrack.person_track_count
            
            BaseTrack.lp_track_count += 1
            return BaseTrack.lp_track_count
        
    @staticmethod
    def release_id(track_id):
        BaseTrack._available_ids.add(track_id)  # ID를 재사용 가능하도록 추가
    
    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack.person_track_count = 0
        BaseTrack.head_track_count = 0
        BaseTrack.lp_track_count = 0
        BaseTrack._available_ids.clear()