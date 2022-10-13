from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from time import gmtime, strftime, time
from datetime import datetime
import json
import traceback
import numpy as np
import argparse
import scipy.stats

# https://gist.github.com/mdonkers/63e115cc0c79b4f6b8b3a6b797e485c7

class ParaManager:
    def __init__(self, args):
        self.repeat = args.repeat
        self.path = args.path
        self.stop = args.stop
        self.p_timeout = 3600 * args.timeout

        self.n_pop = 30
        self.v_max = 0.1
        self.f_w = 0.5
        self.f_c1 = 2
        self.f_c2 = 2

        # type, min, max
        self.paras = [
            (float, 0, 10, 'Predict Future'), # weights
            (float, 0, 10, 'Extract Reward'),
            (float, 0, 10, 'BYOL'),
            (float, 0, 10, 'AutoEncoder'),
            (float, 0, 10, 'Rotation CLS'),
        ]

        self.n_para = len(self.paras)
        
        # particle_pos
        self.p_pos = np.random.rand(self.n_pop, self.n_para)
        self.p_vel = (np.random.rand(self.n_pop, self.n_para) - 0.5) * self.v_max * 2
        
        # initialization
        self.p_pos[0] = 0
        n_weights = len(self.paras)
        self.p_pos[1:n_weights+1] = np.eye(n_weights, n_weights) * 0.1
        
        # personal best
        self.p_best = self.p_pos.copy()
        self.p_best_val = np.zeros(self.n_pop, dtype=np.float32)
        
        # global best
        self.g_best = self.p_pos[0]
        self.g_best_val = 0
        
        # particle status
        self.p_status = np.zeros(self.n_pop, dtype=np.uint8)
        self.p_iter =   np.zeros(self.n_pop, dtype=np.uint8) # iteration/generation
        self.p_idx = 0
        
        # task assgined time
        self.p_time = np.ones(self.n_pop) * time()
        
        # load previous tasks
        self.load()
       
        print("Particle status:")
        print(self.p_status[:10])
        print("*" * 8)
        print("Particle pos:")
        print(self.p_pos[:10])
        print("*" * 8)
        print("Particle vel:")
        print(self.p_vel[:10])
    

    def load(self, path=None):
        try:
            if path is None:
                path = self.path
            with open(path, 'rb') as f:
                tmp = np.load(f)
                assert tmp.shape == self.p_pos.shape
                self.p_pos = tmp
                self.p_vel = np.load(f)
                self.p_best = np.load(f)
                self.p_best_val = np.load(f)
                self.g_best = np.load(f)
                self.g_best_val = float(np.load(f))
                self.p_status = np.load(f)
                self.p_iter = np.load(f)
                self.p_idx = int(np.load(f))
                self.p_time = np.load(f)
                logging.info(f'Load PSO states from {path}')
        except ValueError:
            logging.warning("Failed to load some values")
        except FileNotFoundError:
            pass
        except AssertionError:
            logging.warning(f'Dim from {path} does not match, will not load file.')
            
    
    def save(self, path=None):
        if path is None:
            path = self.path
        with open(path, 'wb') as f:
            np.save(f, self.p_pos)
            np.save(f, self.p_vel)
            np.save(f, self.p_best)
            np.save(f, self.p_best_val)
            np.save(f, self.g_best)
            np.save(f, self.g_best_val)
            np.save(f, self.p_status)
            np.save(f, self.p_iter)
            np.save(f, self.p_idx)
            np.save(f, self.p_time)
            logging.info('Save PSO states to %s', path)
        
    def pos2data(self, pos):
        data = []
        for v, para in zip(pos, self.paras):
            t, v_min, v_max, des = para
            v = np.clip(v, 0, 1)
            val = v * (v_max - v_min) + v_min
            if t == int:
                val = int(val)
            data.append(val)
        return data
    
    def check_timeout(self):
        t_now = time()
        for i in range(self.n_pop):
            if self.p_status[i] > 0 and t_now - self.p_time[i] > self.p_timeout:
                self.p_status[i] = 0
                logging.warning(f"Particle {i} time out! {t_now} > {self.p_time[i]}")
    
    def get_status(self, per_particle=False):
        self.check_timeout()
        
        def unix2str(ts):
            return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        def clip_float(l):
            return [round(i, 3) for i in l]
        
        data = {
            'global_best_val': round(self.g_best_val, 3),
            'global_best': clip_float(self.pos2data(self.g_best)),
            'running_jobs': int(sum(self.p_status)),
            'status': [int(v) for v in self.p_status],
            'task_index': self.p_idx,
            'time': unix2str(time())
        }
        if per_particle:
            data['particles'] = [{
                'idx': i,
                'assigned_time': unix2str(self.p_time[i]),
                'iter': int(self.p_iter[i]),
                'best_val': round(float(self.p_best_val[i]), 3),
                'best': clip_float(self.pos2data(self.p_best[i])),
                'pos': clip_float(self.p_pos[i]),
                'vel': clip_float(self.p_vel[i])
            } for i in range(self.n_pop)]

        # print(data)
        return json.dumps(data)
    
    def get_eval(self):
        data = self.pos2data(self.g_best)
        p_data = {
            'id': self.p_idx,
            'tid': self.p_idx,
            'pos': [float(i) for i in self.g_best],
            'seed': list(range(self.repeat)),
            'data': data
        }
        return json.dumps(p_data)
    
    def get_next(self):
        self.check_timeout()
        if self.stop:
            return json.dumps({'stop': True})

        self.p_idx += 1
        # Better way to find next task
        min_iter = (1 - self.p_status) * (self.p_iter + 1)
        min_iter[min_iter == 0] = np.iinfo(np.uint8).max
        p_idx = int(np.argmin(min_iter))

        self.p_status[p_idx] = 1
        self.p_time[p_idx] = time()
        
        data = self.pos2data(self.p_pos[p_idx])
        p_data = {
            'id': p_idx,
            'tid': self.p_idx,
            'pos': [float(i) for i in self.p_pos[p_idx]],
            'seed': list(range(self.repeat)),
            'data': data
        }
        return json.dumps(p_data)
    
    def post_para(self, data_str):
        try:
            p = json.loads(data_str)
            logging.info(f"Parse {data_str}")
            # particle id
            p_idx = p["id"]
            
            # validate
            if "pos" in p:
                for i, value in enumerate(p["pos"]):
                    if abs(value - self.p_pos[p_idx][i]) > 1e-3:
                        logging.warning(f"Particle {p_idx} value doesn't match, submission abandoned.")
                        return 0
                
            self.p_status[p_idx] = 0
            
            if p["status"] > 0:
                # task canceled
                print(f"Particle {p_idx} canceled")
                logging.info(f"Particle {p_idx} canceled")
                return 0
            
            results_dist = np.array(p["results"])
            
            # ref: https://github.com/google-research/rliable/blob/master/rliable/metrics.py
            p_val = scipy.stats.trim_mean(results_dist, proportiontocut=0.25, axis=None)
            
            if p_val > self.p_best_val[p_idx]:
                self.p_best_val[p_idx] = p_val
                self.p_best[p_idx] = self.p_pos[p_idx].copy()
                logging.info(f"Particle {p_idx}'s local best is update to {p_val:.3f}")
                
            if p_val > self.g_best_val:
                self.g_best_val = p_val
                self.g_best = self.p_pos[p_idx].copy()
                print(f"Global best is updated to {p_val:.3f}")
                logging.info(f"Global best is updated to {p_val:.3f}")
            
            # print(p_idx, 'vel_before', self.p_vel[p_idx])
            # update velocity
            self.p_vel[p_idx] = self.p_vel[p_idx] * self.f_w + \
                                self.f_c1 * np.random.rand(self.n_para) * (self.p_best[p_idx] - self.p_pos[p_idx]) + \
                                self.f_c2 * np.random.rand(self.n_para) * (self.g_best - self.p_pos[p_idx])
            # print(p_idx, 'vel_after', self.p_vel[p_idx])
            # clip velocity
            self.p_vel[p_idx] = np.clip(self.p_vel[p_idx], -self.v_max, self.v_max)
            
            # print(p_idx, 'pos_before', self.p_pos[p_idx])
            # update position
            self.p_pos[p_idx] += self.p_vel[p_idx]
            
            # clip position
            # self.p_pos[p_idx] = np.clip(self.p_pos[p_idx], 0, 1)
            # print(p_idx, 'pos_after', self.p_pos[p_idx])
            self.p_iter[p_idx] += 1
            self.save()
            return 0
        except:
            traceback.print_exc()
            logging.warning(f"Fail to parse {data_str}")
            return 1


def make_handler(para_manager):
    class ParaRequestHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.para = para_manager
            # https://stackoverflow.com/questions/21631799/how-can-i-pass-parameters-to-a-requesthandler
            # """One thing to bear in mind is that BaseHTTPRequestHandler actually runs the handler functions like do_GET inside its __init__ method, so you have to do your initialization before calling super().__init__, contrary to more typical best-practice-by-default""" -- mtraceur
            super(ParaRequestHandler, self).__init__(*args, **kwargs)

        def _set_response(self, code=200):
            self.send_response(code)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_GET(self):
            # logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
            if self.path == '/get_task':
                data = self.para.get_next()
                logging.info("GET request from %s:%s, send task: %s", *self.client_address, data)
                self._set_response()
                self.wfile.write(data.encode('utf-8'))
            elif self.path == '/get_eval':
                data = self.para.get_eval()
                logging.info("GET request from %s:%s, send task: %s", *self.client_address, data)
                self._set_response()
                self.wfile.write(data.encode('utf-8'))
            elif self.path =='/get_status':
                data = self.para.get_status()
                self._set_response()
                self.wfile.write(data.encode('utf-8'))
            elif self.path =='/get_full_status':
                data = self.para.get_status(True)
                self._set_response()
                self.wfile.write(data.encode('utf-8'))
            else:
                self._set_response(404)

        def do_POST(self):
            if self.path == '/submit_result':
                content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
                post_data = self.rfile.read(content_length) # <--- Gets the data itself
                # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                #        str(self.path), str(self.headers), post_data.decode('utf-8'))
                data_str = post_data.decode('utf-8')
                logging.info("POST request from %s:%s, get data: %s", *self.client_address, data_str)
                if not self.para.post_para(data_str):
                    self._set_response()
                    self.wfile.write("accepted".encode('utf-8'))
                else:
                    self._set_response(400)
            else:
                self._set_response(404)

    return ParaRequestHandler

        
def run():
    parser = argparse.ArgumentParser(description='Search server')
    parser.add_argument('--port', type=int, help='port to listen (default: 8888)', default=8888)
    parser.add_argument('--repeat', type=int, help='test one set of parameters multiple time. (default: 5)', default=5)
    parser.add_argument('--timeout', type=int, help='particle busy state reset timeout (unit: hour, default: 12)', default=12)
    parser.add_argument('--path', type=str, help='where to save search result file (default: server/evol_rl.npy)', default='server/evol_rl.npy')
    parser.add_argument('--log', type=str, help='where to save log file (default: server/evol_rl.log)', default='server/evol_rl.log')
    parser.add_argument('--stop', type=bool, help='stop the client after the current task is completed (default: False)', default=False)
    args = parser.parse_args()
    
    logging.basicConfig(filename=args.log, filemode='a', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', force=True)
    
    para_manager = ParaManager(args)
    handler = make_handler(para_manager)

    server_address = ('', args.port)
    httpd = HTTPServer(server_address, handler)

    logging.info('Starting httpd...')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    logging.info('Stopping httpd...')

    httpd.server_close()
    para_manager.save()
    logging.shutdown()


if __name__ == '__main__':
    run()

