from Utils.logger import get_logger
import requests

class CCTV:
    def __init__(self, debug_args):
        self.logger = get_logger(name= '[THERMAL]', console= True, file= False)

        if debug_args.debug == True:
            self.cctv_info = dict()
            self.cctv_info['cctv_id'] = debug_args.cctv_id
            self.cctv_info['cctv_name'] = debug_args.cctv_name
            self.cctv_info['cctv_ip'] = debug_args.source
        else:
            ENDPOINT = "https://was:40000" # TODO hard coding. 
            self.cctv_info = self.get_cctv_info(ENDPOINT)

    def get_cctv_info(self, end_point):
        cctv_info = None
        if end_point != None:
            url = end_point + "/cctv/read"
            cctv_data = requests.post(url, verify=False).json() # TODO , verify=False: 인증서 문제 bypass
            cctv_info = {
                "cctv_id": cctv_data[0][0],
                "cctv_name": cctv_data[0][1],
                "cctv_ip": cctv_data[0][2]
            }            
            self.logger.info(cctv_data)
        return cctv_info