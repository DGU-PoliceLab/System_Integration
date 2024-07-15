from rtsp_validation import string_validator, stream_object_validator
from util import Database

"""
  전달받은 RTSP주소를 검증
  Returns:
  성공시
  {
      validation: 1,
      width: 1920,
      height: 1080,
      thumbnail_location: "~~~",
      heatmap_thumbnail_location: "~~~"
  }
  실패시
  {
      validation: 0,
      message: "~~~~" ,
      error_code : "0~4"
  } 
  0. url check
  1. Connect
  2. Read Frame_Info(W,H)
  """
  
rtsp_info = rtsp_verify.dict()
rtsp_url = rtsp_info["rtspIP"]

#print(rtsp_info["rtspIP"])
vaidate_result = {}
string_validation_result = string_validator(rtsp_url)
print(f"STRING VALIDATION RESULT : {string_validation_result}")

if string_validation_result is True:
    validate_info, validate_result = stream_object_validator(rtsp_url)
    print(f"STREAM OBJECT VALIDATION RESULT : {validate_info}")
    if validate_result:
        vaidate_result = {
            "validation": 1,
            "width": validate_info["width"],
            "height": validate_info["height"]
        }
    else:
        status_code = validate_info["status_code"]
        msg = validate_info["message"]
        vaidate_result = {
            "validation": 0,
            "message": msg,
            "error_code": status_code
        }
else:
    status_code = 0
    #msg = "Fail : CCTV address Validation Error"
    vaidate_result = {
        "validation": 0,
        "message": string_validation_result,
        "error_code": status_code
    }

# 유효성 검사 결과를 RtspExchange에 publish
# 실패하면 WAS에 메세지 전달 ( CCTV idx , rtsp검증 실패 ..... )
    # {'rtsp_validation_result' : 0}
    # 성공이면 {'rtsp_validation_result' : 1}
    # '/home/mhncity/docker/썸네일폴더/thumbnail.jpg'에 썸네일 저장
    # 썸네일 저장 -> 메시지 publish