import subprocess
import re

def kill_process(port):
    try:
        # 특정 포트를 사용하는 프로세스 찾기
        cmd = f"netstat -ntlp | grep :{port}"
        process = subprocess.check_output(cmd, shell=True, text=True)
        
        # PID 추출 (PID는 일반적으로 마지막 열에 위치하며 "PID/프로그램이름" 형식임)
        pid = re.search(r"\s+(\d+)/", process).group(1)
        
        # 추출한 PID를 사용하여 프로세스 종료
        subprocess.run(["kill", pid], check=True)
        print(f"Port {port}를 사용하는 프로세스({pid})가 종료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Port {port}를 사용하는 프로세스를 찾을 수 없습니다. 오류: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")