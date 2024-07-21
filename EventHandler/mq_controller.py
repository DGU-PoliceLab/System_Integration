import os
import pika
import json
from EventHandler.config import mq_config as CONFIG
from Utils.logger import get_logger

LOGGER = get_logger(name = '[MQ]', console=True, file=False)

def connect_mq():
    credentials = pika.PlainCredentials(username=CONFIG['user'], password=CONFIG['password'])
    # connection_params = pika.ConnectionParameters(host=CONFIG['host'], credentials=CONFIG['credentials']) # 현수님 버전 (에러)
    connection_params = pika.ConnectionParameters(host=CONFIG['host'], credentials=credentials)
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    if channel.is_open:
        print("MQ 연결")
    else:
        LOGGER.error("Failed to open MQ channel.")
    return channel

def publish_message(message, channel):
    assert channel is not None, 'FUNCTION publish_message : channel object does not exist'
    LOGGER.info("publish_message 실행")
    try:
        channel.basic_publish(exchange=CONFIG['exchange'], body=json.dumps(message), routing_key='')

    except Exception as e:
        LOGGER.warning(f'FUNCTION publish_message : Error occurred during message publishing, error: {e}')

# def test():
#     mq_config = {
#     "host": '172.30.1.40',
#     "port": 9090,
#     "user": 'mhncity',
#     "password": 'mhncity@364',
#     "exchange": 'event_exchange'
#     }
#     credentials = pika.PlainCredentials(username=mq_config['user'], password=mq_config['password'])
#     # connection_params = pika.ConnectionParameters(host=CONFIG['host'], credentials=CONFIG['credentials']) # 현수님 버전 (에러)
#     connection_params = pika.ConnectionParameters(host=mq_config['host'], credentials=credentials)
#     connection = pika.BlockingConnection(connection_params)
#     channel = connection.channel()
#     message = {"alert_type": "event", "event_id": 30006}
#     channel.basic_publish(exchange=mq_config['exchange'], body=json.dumps(message), routing_key='')


# if __name__ == "__main__":
#     test()