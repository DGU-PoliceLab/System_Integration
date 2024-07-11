 

####################################################################    
    # 객체 오버뷰 6/20 시연버전 급하게 붙임.
    # Sensor Queue
    radar_queue = Queue()
    emotion_result_queue = Queue()
    realtime_status_queue = Queue()
    realtime_status_conn = connect_db("mysql-pls")


    # 프로세스간 데이터 전달을 위한 파이프라인 생성
    if 'snapshot' in args.modules:
        snapshot_input_pipe, snapshot_output_pipe = Pipe()
    if 'thermal' in args.modules:
        thermal_input_pipe, thermal_output_pipe = Pipe()
##        #3##

    if 'selfharm' in args.modules:
        selfharm_input_pipe, selfharm_output_pipe = Pipe()
    if 'falldown' in args.modules:
        falldown_input_pipe, falldown_output_pipe = Pipe()
    if 'emotion' in args.modules:
        emotion_input_pipe, emotion_output_pipe = Pipe()
    if 'violence' in args.modules:
        violence_input_pipe, violence_output_pipe = Pipe()

    # if 'longterm' in args.modules:
    #     longterm_input_pipe, longterm_output_pipe = Pipe()

    # 이벤트 처리를 위한 수집을 위한 파이프라인 생성
    event_input_pipe, event_output_pipe = Pipe()
    # 센서 데이터 + 감정 = 실시간 객체 정보 수집을 위한 파이프라인 생성
    realtime_sensor_input_pipe, realtime_sensor_output_pipe = Pipe()
    # 이벤트 프로세스
    event_process = Process(target=collect_evnet, args=(event_output_pipe,))
    event_process.start()
    # 센서 데이터 + 감정 = 실시간 객체 정보 수집을 위한 프로세스
    realtime_sensor_process = Process(target=collect_realtime, args=(realtime_sensor_output_pipe, emotion_result_queue, radar_queue, realtime_status_queue,))
    realtime_sensor_process.start()
    



    # 모듈별 프로세스
    if 'snapshot' in args.modules:
        snapshot_process = Process(target=object_snapshot_control, args=(snapshot_output_pipe,))
        snapshot_process.start()
    if 'selfharm' in args.modules:
        selfharm_process = Process(target=Selfharm, args=(selfharm_output_pipe, event_input_pipe,))
        selfharm_process.start()
    if 'falldown' in args.modules:
        falldown_process = Process(target=Falldown, args=(falldown_output_pipe, event_input_pipe,))
        falldown_process.start()
    if 'emotion' in args.modules:
        emotion_process = Process(target=Emotion, args=(emotion_output_pipe, event_input_pipe, emotion_result_queue,))
        emotion_process.start()
    if 'violence' in args.modules:
        violence_process = Process(target=Violence, args=(violence_output_pipe, event_input_pipe,))
        violence_process.start()
    if 'thermal' in args.modules:
        thermal_process = Process(target=Thermal, args=(cctv_info['thermal_ip'],thermal_output_pipe, realtime_sensor_input_pipe,))
        thermal_process.start()

    radar_thread = Thread(target=radar_start, args=(radar_queue, cctv_info['rader_ip'], cctv_info['cctv_id'],)).start()
    insert_realtime_thread = Thread(target=insert_realtime, args=(realtime_status_queue, realtime_status_conn),daemon=False).start()
    
####################################################################


    # CCTV 정보 출력
    logger.info(f"cctv_info : {cctv_info}")

    # 사람 감지 및 추적을 위한 모델 로드
    inferencer, init_args, call_args, display_alias = get_model()
    # 얼굴 감지 모델 로드
    face_detector = face_detection.build_detector('RetinaNetResNet50', confidence_threshold=.5, nms_iou_threshold=.3)

    # 동영상 관련 설정
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H_%M_%S")
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    fps = 30
    num_frame = 0
    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

    # 디버그(시각화, 동영상 저장) 
    if debug_args.visualize:    
        out = cv2.VideoWriter(f'/System_Integration/_Output/video_clip_{timestamp}.mp4', fourcc, fps, (int(w), int(h))) 

    
    # _HAR 모듈 실행 대기
    if 'selfharm' in args.modules:
        wait_subprocess_ready("Selfharm", selfharm_input_pipe, logger)
    if 'falldown' in args.modules:
        wait_subprocess_ready("Falldown", falldown_input_pipe, logger)
    if 'emotion' in args.modules:
        wait_subprocess_ready("Emotion", emotion_input_pipe, logger)
    if 'violence' in args.modules:
        wait_subprocess_ready("Violence", violence_input_pipe, logger)
    # if 'longterm' in args.modules:
    #     wait_subprocess_ready("Longterm", longterm_input_pipe, logger)

    # 사람 감지 및 추적
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            draw_frame = frame.copy()
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            detections = []
            skeletons = []
            temp_call_args = copy.deepcopy(call_args)
            temp_call_args['inputs'] = frame         

            for _ in inferencer(**temp_call_args):
                pred = _['predictions'][0]
                l_p = len(pred)
                logger.info(f'frame #{num_frame} pose_results- {l_p} person detect!')
                n_person = 1
                pred.sort(key = lambda x: x['bbox'][0][0])

                for p in pred:
                    keypoints = p['keypoints']
                    keypoints_scores = p['keypoint_scores']
                    detection = [*p['bbox'][0], p['bbox_score']]
                    detections.append(detection)
                    skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])
                    if debug_args.visualize:
                        draw_frame = draw_bbox_skeleton.draw(draw_frame, n_person, detection, keypoints)
                    n_person += 1   
            
            detections = np.array(detections, dtype=np.float32)
            skeletons = np.array(skeletons, dtype=np.float32)
            online_targets = tracker.update(detections, skeletons, frame)
            if debug_args.debug:
                # if num_frame % fps == 0:
                #     face_detections = face_detector.detect(frame)
                #     temperature = Thermal(thermal_info, frame, face_detections)
                if num_frame < len(rader_data):
                    cur_rader_data = rader_data[num_frame]
                    vital_data = cur_rader_data["vital_info"]
                    target_data = []
                    for track in online_targets:
                        tid = track.track_id
                        x1, y1, x2, y2 = track.tlbr
                        target_data.append({"id": tid, "range": [x1, x2, y1, y2]})

                    for vital in vital_data:
                        pos, depth = vital["pos"]
                        heartbeat_rate = vital["heartbeat_rate"]
                        breath_rate = vital["breath_rate"]
                        offset = (int(pos) + 200) / 400 * int(w)
                        for target in target_data:
                            tid = target["id"]
                            pos_range = target["range"]
                            if offset >= pos_range[0] and offset <= pos_range[1]:
                                logger.info(f"tid:{tid}, heartbeat_rate: {heartbeat_rate}, breath_rate: {breath_rate}")
                                if debug_args.visualize:
                                    draw_frame = draw_vital.draw(draw_frame, int(pos_range[0]), int(pos_range[2]), heartbeat_rate, breath_rate)
                                    
            num_frame += 1
            tracks = online_targets # 모듈로 전달할 감지 결과
            
            if debug_args.visualize:
                meta_data = {'cctv_id': cctv_info['cctv_id'], 'current_datetime': current_datetime, 'cctv_name': cctv_info['cctv_name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h)), 'frame': draw_frame} # 모듈로 전달할 메타데이터
            else:
                meta_data = {'cctv_id': cctv_info['cctv_id'], 'current_datetime': current_datetime, 'cctv_name': cctv_info['cctv_name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h))} # 모듈로 전달할 메타데이터
            input_data = [tracks, meta_data, num_frame] # 모듈로 전달할 데이터
            e_input_data = [frame, meta_data, num_frame]
            snapshot_input_data = [tracks, meta_data, frame, num_frame] 
            thermal_input_data = [tracks, meta_data, frame, num_frame] # snapshot_input_data랑 똑같지만, thermal_input_pipe에 snapshot_input_data을 넣으면 보는 사람이 헷갈릴까봐 의도적으로 구분해놓음. 
            # 모듈로 데이터 전송

            if 'thermal' in args.modules:
                thermal_input_pipe.send(thermal_input_data)
            if 'snapshot' in args.modules:
                snapshot_input_pipe.send(snapshot_input_data)
            # if 'longterm' in args.modules:
            #     longterm_input_pipe.send(input_data)
