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
    while True:
        try:
            r = requests.get(url + 'get_eval')
            if r.status_code == 200:
                break
            time.sleep(10)
        except ConnectionRefusedError:
            print("Connection Refused Error, retry in 10s.")

    p_data = json.loads(r.text)
    print(p_data)

    weights = p_data['data']
    print(f"Weights: ", weights)

    main(weights, task_str=r.text)

    print("Done")


