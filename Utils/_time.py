from functools import wraps
import time
def process_time_check(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                #total_time = round(end_time - start_time, 3)
                #필요시 반올림해서 시간 확인
                total_time = end_time - start_time
                print(f"{func.__qualname__}: {total_time} 초", flush=True)
                return result
        return wrapper