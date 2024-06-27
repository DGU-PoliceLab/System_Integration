import os
import numpy as np
import cv2
from keras.models import load_model
from collections import deque
import pymysql
import pika
from datetime import datetime
import json
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
import configparser

def apikey_dec(prkeyloc,encdata):
    # prkeyloc = os.environ['DIS_SECRET_PATH'] +"/"+ prkeyloc
    # encdata= os.environ['DIS_SECRET_PATH'] +"/"+ encdata
    prkeyloc = "/home/mhncity/violence_classification/config/secret_key/" + prkeyloc
    encdata= "/home/mhncity/violence_classification/config/secret_key/" + encdata
    with open(encdata, "rb") as f:
        private_key = RSA.import_key(open(prkeyloc).read())
        enc_session_key, nonce, tag, ciphertext = \
        [ f.read(x) for x in (private_key.size_in_bytes(), 16, 16, -1) ]
        cipher_rsa = PKCS1_OAEP.new(private_key)
        session_key = cipher_rsa.decrypt(enc_session_key)
        cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
        data = cipher_aes.decrypt_and_verify(ciphertext, tag) 
    return data


##### main.py에서 전달해서 쓸지 판단하여 사용
config = configparser.ConfigParser()
config.read('/home/mhncity/violence_classification/config/config.ini')

# MySQL 연결 설정
db = config["DATABASE"]
db_host = db['endpoint_url']
db_port = db['port']
db_user = db['user']
## config.ini 파일 안에 경로를 읽어 비밀번호 복원
db_password = apikey_dec(db['private_key'],db['passwd']).decode('utf-8')
db_name = db['db_name']
charset = 'utf8'


# RabbitMQ 연결 설정
mq = config["RABBIT_MQ"]
mq_host = mq['endpoint_url']
mq_port = mq['port']
mq_user = mq['user']
mq_password = apikey_dec(mq['private_key'],mq['passwd']).decode('utf-8')
mq_exchange = mq['exchange']

#####

# imgSize = 128
# frames = []

# # MySQL 연결 함수
# def connect_to_db():
#     connection = pymysql.connect(host=db_host,
#                                  port=db_port,
#                                  user=db_user,
#                                  password=db_password,
#                                  db=db_name)
#     return connection

# # RabbitMQ 연결 함수
# def connect_to_mq():
#     credentials = pika.PlainCredentials(username=mq_user, password=mq_password)
#     # print(111)
#     connection_params = pika.ConnectionParameters(host=mq_host, credentials=credentials)
#     # print(222222222222222222222222222222222222222222222222)
#     connection = pika.BlockingConnection(connection_params)
#     # print(3333333333333333333333333333333333333333333)
#     channel = connection.channel()
#     # print(444444444444444444444444444444444444444444444)
#     # channel.exchange_declare(exchange=mq_exchange, exchange_type='direct')
#     # print(555555555555555555555555555555555555555555)
#     return channel

# # DB에 삽입 및 MQ에 메시지 발행
# def insert_to_db_and_publish(violence_label, preds):
#     # DB 연결
#     db_connection = connect_to_db()

#     try:
#         with db_connection.cursor() as cursor:
#             # DB에 데이터 insert
#             sql = """
#             INSERT INTO event 
#             (cctv_id, event_type, event_location, event_detection_people, 
#             event_date, event_time, event_clip_directory, 
#             event_confirm_date, event_confirm_time, event_check, 
#             event_start, event_end) 
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#             """
            
#             # 현재 날짜 및 시간 가져오기
#             current_date = datetime.now().strftime('%Y-%m-%d')
#             current_time = datetime.now().strftime('%H:%M:%S')

#             # DB에 삽입할 값 설정
#             values = [27, "violence", "연구실", 1,  current_date, current_time,
#                       "/home/mhncity", current_date, current_time, 0,
#                       current_date + ' ' + current_time, current_date + ' ' + current_time]

#             cursor.execute(sql, values)
            
#             # 삽입된 데이터의 event_id 가져오기
#             event_id = cursor.lastrowid
            
#             # MQ에 메시지 발행
#             publish_message(event_id)
            
#         db_connection.commit()
        
#         # # MQ에 메시지 발행
#         # publish_message(violence_label, preds)
        
#     except Exception as e:
#         print(f"Error inserting data into DB: {e}")
#     finally:
#         db_connection.close()

# # RabbitMQ에 메시지 발행
# def publish_message(event_id):
#     # RabbitMQ 연결
#     mq_channel = connect_to_mq()

#     try:
#         # 메시지 발행
#         # message = f"Event ID:{event_id}"
#         message = {"Event ID": event_id}
#         mq_channel.basic_publish(exchange=mq_exchange, body=json.dumps(message), routing_key='')
#         print(f" [x] Sent '{message}' to '{mq_exchange}'")
#     except Exception as e:
#         print(f"Error publishing message to RabbitMQ: {e}")
#     finally:
#         mq_channel.close()

# # 실시간 분석
# def real_time_analysis(rtsp_url):
#     if not os.path.exists('output'):
#         os.mkdir('output')

#     model = load_model('./model.h5')
#     Q = deque(maxlen=128)
#     vs = cv2.VideoCapture(rtsp_url)
#     writer = None
#     (W, H) = (None, None)
#     count = 0

#     while True:
#         (grabbed, frame) = vs.read()
#         if not grabbed:
#             break

#         try:
#             count += 1

#             if W is None or H is None:
#                 (H, W) = vs.get(cv2.CAP_PROP_FRAME_HEIGHT), vs.get(cv2.CAP_PROP_FRAME_WIDTH)

#             oriFrame = frame
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             output = cv2.resize(frame, (512, 360)).copy()
#             frame = cv2.resize(frame, (128, 128)).astype("float32")
#             frame = frame.reshape(imgSize, imgSize, 3) / 255
#             preds = model.predict(np.expand_dims(frame, axis=0))[0]

#             results = np.array(Q).mean(axis=0)
#             i = (preds > 0.11)[0]
#             label = i
#             if label:
#                 # 강제로 30초에 한 번 violence가 True로 발생하고, DB에 데이터 insert 및 message 발행
#                 if count % 30 == 0:
#                     insert_to_db_and_publish(True, preds)
#             else:
#                 label = False

#             text = "Violence: {}".format(label) + str(preds)

#             color = (0, 255, 0)
#             if label:
#                 color = (255, 0, 0)
#             else:
#                 color = (0, 255, 0)

#             cv2.putText(oriFrame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
#             frames.append(oriFrame)

#             # 결과를 실시간으로 보여주기 위해 추가한 부분
#             cv2.imshow("Real-time Analysis", oriFrame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord("q"):
#                 break

#             if writer is None:
#                 # Create VideoWriter object on the first frame
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 writer = cv2.VideoWriter("./output/lockup_8_1128.mp4", fourcc, 30.0, (int(W), int(H)), True)

#             writer.write(oriFrame)

#         except Exception as e:
#             print(f"Error processing frame: {e}")
#             break

#     if writer is not None:
#         writer.release()
#     vs.release()
#     cv2.destroyAllWindows()

# # RTSP URL 예시
# rtsp_url = "rtsp://admin:dpadpdlcl-1@172.30.1.2:554/cam/realmonitor?channel=1&subtype=0"
# real_time_analysis(rtsp_url)
# print('complete')