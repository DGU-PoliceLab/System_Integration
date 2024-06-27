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

    person_track_count = 0
    head_track_count = 0
    lp_track_count = 0

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
            BaseTrack.person_track_count += 1
            return BaseTrack.person_track_count
        elif tracker_num == 1:
            BaseTrack.head_track_count += 1
            return BaseTrack.head_track_count
        elif tracker_num == 2:
            BaseTrack.lp_track_count += 1
            return BaseTrack.lp_track_count
        

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
