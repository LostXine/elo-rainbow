from main import *

import requests
import json
import time
import numpy as np
import traceback

with open('server-addr', 'r') as f:
    url = f.readline().strip()
    print("URL: " + url)
    
p_data = None

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    is_running = True

    while is_running:
        while True:
            try:
                r = requests.get(url + 'get_task')
                if r.status_code == 200:
                    break
                time.sleep(10)
            except ConnectionRefusedError:
                print("Connection Refused Error, retry in 10s.")

        p_data = json.loads(r.text)
        print(p_data)
        if 'stop' in p_data:
            is_running = False
            break

        weights = p_data['data']
        print(f"Weights: ", weights)
        assert len(weights) == 5

        try:
            results = [main(weights, tid=p_data['tid'], seed=i, task_str=r.text) for i in p_data['seed']]
            p_data['results'] = results
            p_data['status'] = 0
        except KeyboardInterrupt:
            p_data['status'] = 1
            is_running = False
        except:
            traceback.print_exc()
            p_data['status'] = 2
            time.sleep(10)
        while True:
            try:
                r = requests.post(url + 'submit_result', data = json.dumps(p_data))
                if r.status_code == 200:
                    break
                time.sleep(10)
            except ConnectionRefusedError:
                print("Connection Refused Error, retry in 10s.")

        print("Done")

