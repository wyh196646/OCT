import pandas as pd
data=pd.read_csv('/home/octusr3/project/oct_wk/data.csv')

from pathlib import Path
import tarfile
targz=[]

def read_imgs_targz(path):
    result = []
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content = f.read()
            img = np.frombuffer(content, dtype=np.uint8)
            result.append(cv2.imdecode(img, cv2.IMREAD_GRAYSCALE))
    return result

for i in data['image_path']:
    #print(i)
    tar_gz_path=Path('/home/octusr2/projects/data_fast/proceeded/volume_targz')
    label_path=(tar_gz_path/i).with_suffix('.tgz')
    #print(label_path)
    targz.append(label_path)

from Utils import *
def cut_cp_from_imgs_bupt(img_array, center_x,center_y,volume_size, flip=False, radius=2, threshold=100, size=(380, 380)):
    volume = np.stack(img_array, axis=0)
    volume = volume.transpose(0,2,1)
    volume = volume[:, ::2, :]
    
    if flip:
        volume = np.flip(volume, axis=1)
    y, x = np.mgrid[255:-1:-1, 0:256:1]
    
    dis = (x - center_x) ** 2 + (y - center_y) ** 2
    valid_mask = np.abs(dis - (radius / volume_size * 256) ** 2) < threshold
    theta = np.arctan2(y - center_y, x - center_x)
    #这里好像出现了问题，需要重新切数据
    masked_theta = theta[valid_mask]
    masked_volume = volume[valid_mask, :]
    img = masked_volume[masked_theta.argsort()]
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
    return img.transpose()
    
def chunk(list, n):
    result = []
    for i in range(n):
        result.append(list[math.floor(i / n * len(list)):math.floor((i + 1) / n * len(list))])
    return result

def run_multi_process(item_list, n_proc, func, with_proc_num=False):
    tasks = chunk(item_list, n_proc)
    if with_proc_num:
        for i in range(len(tasks)):
            tasks[i] = (i, tasks[i])
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(func, tasks)
    return results

def extract_cp_mp_bupt(tasks, size):
    input_root = Path('/home/octusr2/projects/data_fast/proceeded/volume_targz')
    output_root = Path(f'/home/octusr3/project/data_fast/new_slice')
    info = pd.read_csv('/home/octusr2/projects/data_fast/proceeded/volume_info/volume_info.csv')
    info['oct_path'] = info['oct_path'].map(lambda x: str(Path(x).with_suffix('')))
    info = info.drop_duplicates(subset='oct_path')
    info = info.set_index('oct_path')
    errors = run_multi_process(tasks, 80, partial(extract_cp_bupt, input_root=input_root, output_root=output_root, info=info, size=size), with_proc_num=True)
    print('all have been done')
    errors = reduce(lambda x, y: x + y, errors)
    pkl_dump(errors, 'data/errors_cp.pkl')

def extract_cp_bupt(tasks, input_root, output_root, info, size):
    info = info.copy()
    data=pd.read_csv('/home/octusr3/project/oct_wk/data.csv')
    proc_num, tasks = tasks
    errors = []
    with Benchmark(f'Converting', print=proc_num == 0) as t:
        for i, task in enumerate(tasks):
            if i % (len(tasks) // 10 + 1) == 0:
                t.print_elapsed(f'{i}/{len(tasks)}, {i / len(tasks) * 100:.2f}%')
            task = Path(task).relative_to(input_root)
            info_key = str(task.with_suffix('').with_suffix(''))
            
            if info_key not in info.index:
                err = Exception('No oct info.')
                errors.append((task, err))
                # print(f'Error with file {str(task)}: {err}')
                continue
            volume_size = info.loc[info_key, 'oct_x']
            input_path = str(input_root / task)
            output_path = (output_root / task).with_suffix('.jpg')
            #print(3)
            key_path=task.with_suffix('.jpg')
            print(key_path)
           # print(data.loc[data['image_path']==key_path])['center_point'].values[0]
            a=data[data['image_path']==str(key_path)]['center_point'].values[0]
            center_x,center_y=[int(x) for x in a.strip('(').strip(')').split(',')]
            #print(center_x,center_y)
            eye = ''
            if 'OD' in input_path:
                eye = 'OD'
            else:
                eye = 'OS'
            if eye == '':
                err = Exception('No eye infomation (OS/OD)')
                errors.append((task, err))
                # print(f'Error with file {str(task)}: {err}')
                continue
            try:
                imgs = read_imgs_targz(input_path)

                cp = cut_cp_from_imgs_bupt(imgs, center_x,center_y,volume_size=volume_size, flip=eye=='OS', radius=1.4, threshold=100, size=(size, size))
                ensure_file(output_path)
                Image.fromarray(cp).save(output_path)
            except Exception as err:
                errors.append((task, err))
                # print(f'Error with file {str(task)}: {err}')
    return errors

extract_cp_mp_bupt(targz, size=512)