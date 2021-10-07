from oct_Utils import *
from datetime import datetime
import time
# from volume_extraction import *


if __name__ == '__main__':
    all_fdas = []
    all_volumes = []
    all_cps = []
    while True:
        new_all_fdas = get_all_fdas(cache=False)
        if len(new_all_fdas) != len(all_fdas):
            all_fdas = new_all_fdas
            print(f'All valid fdas updated at {datetime.now()}， n_files = {len(all_fdas)}')
        
        new_all_volumes = get_all_volumes(cache=False)
        if len(new_all_volumes) != len(all_volumes):
            all_volumes = new_all_volumes
            print(f'All valid volumes updated at {datetime.now()}， n_files = {len(all_volumes)}')

        new_all_cps = get_all_cps(cache=False)
        if len(new_all_cps) != len(all_cps):
            all_cps = new_all_cps
            print(f'All valid cpsd updated at {datetime.now()}， n_files = {len(all_volumes)}')
        time.sleep(60 * 60)
