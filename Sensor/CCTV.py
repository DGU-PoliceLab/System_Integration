import json
import pymysql
from Utils.logger import get_logger

class CCTV:
    def __init__(self, debug_args):
        self.logger = get_logger(name= '[THERMAL]', console= True, file= False)

        if debug_args.debug == True:
            self.cctv_info = dict()
            self.cctv_info['cctv_id'] = debug_args.cctv_id
            self.cctv_info['ip'] = debug_args.cctv_ip
            self.cctv_info['cctv_name'] = debug_args.cctv_name
            self.cctv_info['source'] = debug_args.source
        else:
            import config
            conn = self.connect_db(config.db_config)
            if conn.open:
                cctv_info = self.get_cctv_info(conn)
                print(f"cctv_info : {cctv_info}")
                cctv_info = cctv_info[1]
                source = cctv_info['cctv_ip']
            else:
                self.logger.warning('RUN-CCTV Database connection is not open.')
                cctv_info = {'cctv_id': 404}

    def connect_db(self, config=None):
        if config != None:
            conn = pymysql.connect(host=config["host"],
                                port=config["port"],
                                user=config["user"],
                                password=config["password"],
                                database=config["database"],
                                charset=config["charset"])
            return conn
        else:
            return None

    def get_cctv_info(self, conn=None):
        """
        Get CCTV info (include rader, thermal) from database.
        """
        cctv_info = []

        if conn != None:
            with conn.cursor() as cur:
                sql = "SELECT * FROM cctv"
                cur.execute(sql)
                response = cur.fetchall()
            for row in response:
                cctv_info.append({'cctv_id': row[0], 'cctv_ip': row[1], 'rader_ip': row[2], 'thermal_ip': row[3], 'cctv_name': row[4]})
            return cctv_info
        else:
            return self.cctv_info