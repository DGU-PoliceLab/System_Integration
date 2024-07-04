import argparse
import sys
sys.path.insert(0, '/System_Integration/_HAR/MHNCITY/longterm')
import numpy as np
import time
from _Utils.logger import get_logger

# (동일한 이미지)long_term_demo_0.mp4에 대해 (최소 0.1, 최대 3.5, 평균 1.7)
# (앉은 자세 유지)long_term_demo_1.mp4에 대해 (최소 0.0, 최대 18.5, 평균 1.7)
# (앉은 자세에서 숨쉬기)long_term_demo_2.mp4에 대해 (최소 0.0, 최대 26.7, 평균 8.2)
# (앉은 자세에서 발목 돌리기)long_term_demo_3.mp4에 대해 (최소 0.1, 최대 133.4, 평균 18.0)
# (앉은 자세에서 움직임)long_term_demo_4.mp4에 대해 (최소 0.1, 최대 274.9, 평균 83.97)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshhold', type=float, default=15.0, help='longterm threshhold')
    parser.add_argument('--hold_time', type=int, default=10, help='hold time (seconds)')
    parser.add_argument('--fps', type=int, default=30, help='frame num')
    parser.add_argument('--max_person', type=int, default=10, help='max person num')
    args = parser.parse_args()
    return args

def check_longterm(confidence, threshhold):
    if threshhold < confidence:
        return True
    return False

def Longterm(data_pipe, event_pipe):
    logger = get_logger(name="[MhnCity.Longterm]", console=True, file=True)
    args = parse_args()
    hold_frames = [[] for x in range(args.max_person)]
    count = [0 for x in range(args.max_person)]
    data_pipe.send(True)
    while True:
        data = data_pipe.recv()
        if data:
            tracks, meta_data = data
            tids = []
            for track in tracks:
                tid = track.track_id
                tids.append(tid)
                skeleton = track.skeletons[0]
                hold_frames[tid % args.max_person].append(skeleton)
                if len(hold_frames[tid % args.max_person]) > args.fps * args.hold_time:
                    hold_frames[tid % args.max_person].pop(0)
            for i in range(args.max_person):
                if i not in tids:
                    hold_frames[i] = []
                    count[i] = 0
            for i, hold in enumerate(hold_frames):
                if len(hold) >= args.fps * args.hold_time:
                    cur = None
                    similarity = 0
                    for skeletons in hold:
                        if cur is None:
                            cur = skeletons
                        else:
                            simil = np.linalg.norm(cur - skeletons)
                            similarity += simil
                    confidence = similarity/(args.fps * args.hold_time)
                    if confidence < args.threshhold:
                        count[i] += 1
                    else:
                        count[i] = 0
            for i, c in enumerate(count):
                if c > args.hold_time * args.fps:
                    logger.info(f"action: longterm, tid: {i}")
                    count[i] = 0
        else:
            time.sleep(0.0001)
