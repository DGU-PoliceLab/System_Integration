# import sys
# sys.path.insert(0, '/System_Integration/HAR/MHNCITY/longterm')
# import numpy as np
# import time
# from Utils.logger import get_logger
# from Utils._visualize import Visualizer
# from variable import get_longterm_args, get_debug_args

# # (동일한 이미지)long_term_demo_0.mp4에 대해 (최소 0.1, 최대 3.5, 평균 1.7)
# # (앉은 자세 유지)long_term_demo_1.mp4에 대해 (최소 0.0, 최대 18.5, 평균 1.7)
# # (앉은 자세에서 숨쉬기)long_term_demo_2.mp4에 대해 (최소 0.0, 최대 26.7, 평균 8.2)
# # (앉은 자세에서 발목 돌리기)long_term_demo_3.mp4에 대해 (최소 0.1, 최대 133.4, 평균 18.0)
# # (앉은 자세에서 움직임)long_term_demo_4.mp4에 대해 (최소 0.1, 최대 274.9, 평균 83.97)

# def check_longterm(confidence, threshhold):
#     if threshhold < confidence:
#         return True
#     return False

# def Longterm(data_pipe, event_pipe):
#     logger = get_logger(name="[MhnCity.Longterm]", console=True, file=True)
#     args = get_longterm_args()
#     debug_args = get_debug_args()

#     debug_args.visualize = False #TODO. TEMP. 해당 기능에 문제가 있어서 비활성화 시킴. !김현수 작업 요망!

#     if debug_args.visualize:
#         visualizer = Visualizer("longterm")
#         init_flag = True
#     hold_frames = [[] for x in range(args.max_person)]
#     count = [0 for x in range(args.max_person)]
#     data_pipe.send(True)
#     while True:
#         try:
#             event = ['normal', 1.0]
#             data = data_pipe.recv()
#             if data:
#                 if data == "end_flag":
#                     break
#                 tracks, meta_data = data
#                 tids = []
#                 for track in tracks:
#                     tid = track.track_id
#                     tids.append(tid)
#                     skeleton = track.skeletons[0]
#                     hold_frames[tid % args.max_person].append(skeleton)
#                     if len(hold_frames[tid % args.max_person]) > args.fps * args.hold_time:
#                         hold_frames[tid % args.max_person].pop(0)
#                 for i in range(args.max_person):
#                     if i not in tids:
#                         hold_frames[i] = []
#                         count[i] = 0
#                 for i, hold in enumerate(hold_frames):
#                     if len(hold) >= args.fps * args.hold_time:
#                         cur = None
#                         similarity = 0
#                         for skeletons in hold:
#                             if cur is None:
#                                 cur = skeletons
#                             else:
#                                 simil = np.linalg.norm(cur - skeletons)
#                                 similarity += simil
#                         confidence = similarity/(args.fps * args.hold_time)
#                         if confidence < args.threshhold:
#                             count[i] += 1
#                             event[0] = 'normal'
#                             event[1] = str(count)
#                         else:
#                             count[i] = 0
#                             event[0] = 'longterm(counting)'
#                             event[1] = str(count)
#                 for i, c in enumerate(count):
#                     if c > args.hold_time * args.fps:
#                         event[0] = 'longterm(detect)'
#                         logger.info(f"action: longterm, tid: {i}")
#                         count[i] = 0
#                 if debug_args.visualize:
#                     if init_flag == True:
#                         visualizer.mkdir(meta_data['timestamp'])
#                         init_flag = False
#                     visualizer.save_temp_image([meta_data["v_frame"], event[0], event[1]], meta_data["num_frame"])
#             else:
#                 time.sleep(0.0001)
#         finally:
#             logger.warning("Longterm process end.")
