import itertools
from Utils import *
import os.path
from construct import Int8un, Int16un, Float64l, Array, Struct
from fda import FDA
from fds import FDS
import sys
import lmdb
import io
import tarfile


field_headers = {}
field_headers[b'@CAPTURE_INFO_02'] = Struct(
    'x' / Int16un,
    'zeros' / Array(52, Int16un),
    'aquisition year' / Int16un,
    'aquisition month' / Int16un,
    'aquisition day' / Int16un,
    'aquisition hour' / Int16un,
    'aquisition minute' / Int16un,
    'aquisition second' / Int16un
)

field_headers[b'@PATIENT_INFO_02'] = Struct(
    'patient id' / Array(32, Int8un),
    'patient given name' / Array(32, Int8un),
    'patient surname' / Array(32, Int8un),
    'zeros' / Array(8, Int8un),
    'birth data valid' / Int8un,
    'birth year' / Int16un,
    'birth month' / Int16un,
    'birth day' / Int16un,
    'zeros2' / Array(504, Int8un)
)

field_headers[b'@PARAM_SCAN_04'] = Struct(
    'unknown' / Array(6, Int16un),
    'x mm' / Float64l,
    'y mm' / Float64l,
    'z um' / Float64l,
)

def read_field(oct_object, key):
    with open(oct_object.filepath, 'rb') as f:
        chunk_location, chunk_size = oct_object.chunk_dict[key]
        f.seek(chunk_location)
        raw = f.read(chunk_size)
        header = field_headers[key].parse(raw)
    return header


def get_all_fdas(cache=True, limit=0):
    if cache and os.path.isfile('/home/octusr2/projects/data_fast/filelists/all_fdas.json'):
        return json_load('/home/octusr2/projects/data_fast/filelists/all_fdas.json') 
    tasks = []
    not_valid1 = {
        'wide',
        '3d wide',
    #     'oct',
        'circle disc',
        '3d external',
        'circle external',
        'radial external',
        'octa external',
        'line macule',
        'radial disc',
        'radial macule',
        'oct disc',
        'octa disc',
        'octa macula',
        'octa macular',
        'octa disc-fda',
        'octa macula-fda',
    }
    not_valid2 = {
        'oct',
    }
    for dirpath, dirnames, filenames in os.walk('/data/rawdata/OCT/data/'):
        for d in dirnames[:]:
            if '-clean-' in d or any(x in d.lower() for x in not_valid1):
                dirnames.remove(d)
        
        for filename in filenames:
            if filename.endswith('.fda') or filename.endswith('.fds'):
                tasks.append(f'{dirpath}/{filename}')
        if limit != 0 and len(tasks) > limit:
            break
        print(len(tasks), end="\r", flush=True)
    new_tasks = []
    for task in tasks:
        parent = task.split('/')[-2]
        if parent.lower() in not_valid2:
            continue
        new_tasks.append(task)
    tasks = new_tasks
    json_dump(tasks, '/home/octusr2/projects/data_fast/filelists/all_fdas.json')
    return tasks


def get_all_volumes(cache=True):
    if cache and os.path.isfile('/home/octusr2/projects/data_fast/filelists/all_volumes.json'):
        return json_load('/home/octusr2/projects/data_fast/filelists/all_volumes.json')
    tasks = []
    for dirpath, dirnames, filenames in os.walk('/home/octusr2/projects/data_fast/proceeded/volume_targz'):        
        for filename in filenames:
            if filename.endswith('.tgz'):
                tasks.append(f'{dirpath}/{filename}')
        print(len(tasks), end="\r", flush=True)
    json_dump(tasks, '/home/octusr2/projects/data_fast/filelists/all_volumes.json')
    return tasks


def get_all_cps(size, cache=True):
    if cache and os.path.isfile(f'/home/octusr2/projects/data_fast/filelists/all_cps_{size}.json'):
        return json_load(f'/home/octusr2/projects/data_fast/filelists/all_cps_{size}.json')
    tasks = []
    for dirpath, dirnames, filenames in os.walk(f'/home/octusr2/projects/data_fast/proceeded/cp_projection/{size}'):        
        for filename in filenames:
            if filename.endswith('.jpg'):
                tasks.append(f'{dirpath}/{filename}')
        print(len(tasks), end="\r", flush=True)
    json_dump(tasks, f'/home/octusr2/projects/data_fast/filelists/all_cps_{size}.json')
    return tasks

# 右眼
# 患者方向：256维的正方向是下方，512维的正方向是鼻侧，992维的正方向是后方
# 面对患者方向：256维的正方向是下方，512维的正方向是右侧，992维的正方向是前方

# 左眼
# 患者方向：256维的正方向是下方，512维的正方向是颞侧，992维的正方向是后方
# 面对患者方向：256维的正方向是下方，512维的正方向是右侧，992维的正方向是前方
def read_fdas(path):
    path = Path(path)
    fda = True
    if path.suffix == '.fda':
        fda = True
        obj = FDA(str(path))
    else:
        fda = False
        obj = FDS(str(path))
    oct_volume = obj.read_oct_volume()
    return [img.astype(np.uint8) if fda else (img/255).astype(np.uint8) for img in oct_volume.volume]


def compact_img_array(img_array):
    imgs = []
    lens = []
    max_size = 0
    for img in img_array:
        img = np.squeeze(cv2.imencode('.jpg', img)[1])
        lens.append(img.size)
        if img.size > max_size:
            max_size = img.size
        imgs.append(img)
    for i, img in enumerate(imgs):
        imgs[i] = np.pad(img, (0, max_size - img.size), constant_values=0)
    imgs = np.stack(imgs, axis=0)
    return imgs, lens


def read_compacted_imgs(path):
    obj = np.load(path)
    imgs = obj['imgs']
    lens = obj['lens']
    result = []
    for img, l in zip(imgs, lens):
        img = img[:l]
        img = np.frombuffer(img, dtype=np.uint8)
        result.append(cv2.imdecode(img, cv2.IMREAD_GRAYSCALE))
    return result


def save_as_npz(img_array, task, output_root):
    imgs, lens = compact_img_array(img_array)
    output_file = str(output_root / task.with_suffix('.npz')).replace('', ' ')
    ensure_file(output_file)
    np.savez_compressed(output_file, imgs=imgs, lens=lens)


def save_as_targz(img_array, task, output_root):
    output_file = str(output_root / (str(task) + '.tgz')).replace('', ' ')

    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode='w:gz') as tar:
        for index, img in enumerate(img_array):
            img_bytes = io.BytesIO()
            Image.fromarray(img).save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            img_value = img_bytes.getvalue()

            info = tarfile.TarInfo(f'slice_{index}.jpg')
            info.size = len(img_value)
            tar.addfile(info, img_bytes)

    ensure_file(output_file)
    with open(output_file, 'wb') as f:
        f.write(tar_bytes.getvalue())


def read_imgs_targz(path):
    result = []
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content = f.read()
            img = np.frombuffer(content, dtype=np.uint8)
            result.append(cv2.imdecode(img, cv2.IMREAD_GRAYSCALE))
    return result


# 展开方向是左上右下左（面对患者方向），只翻转左眼的话就是颞上鼻下颞
def cut_cp_from_imgs(img_array, volume_size, flip=False, radius=2.5, threshold=100, size=(380, 380)):
    volume = np.stack(img_array, axis=0)
    volume = volume.transpose(0,2,1)
    volume = volume[:, ::2, :]#下采样函数
    if flip:
        volume = np.flip(volume, axis=1)
    y, x = np.mgrid[255:-1:-1, 0:256:1]
    
    dis = (x - 255 / 2) ** 2 + (y - 255 / 2) ** 2
    valid_mask = np.abs(dis - (radius / volume_size * 256) ** 2) < threshold
    theta = np.arctan2(y - 255/2, x - 255/2)
    
    masked_theta = theta[valid_mask]
    masked_volume = volume[valid_mask, :]
    img = masked_volume[masked_theta.argsort()]
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
    return img.transpose()


def str_to_np_array(row, dtype=np.float):
    row = row.replace('[', '').replace(']', '')
    row = row.split(' ')
    row = filter(lambda x: x != '', row)
    row = map(lambda x: dtype(x), row)
    row = np.array(list(row))
    return row


def str_to_np_mat(s, dtype=np.float):
    s = s[1:-1]
    rows = s.split('\n')
    for index, row in enumerate(rows):
        rows[index] = str_to_np_array(row, dtype)
    rows = np.stack(rows, axis=0)
    return rows


def remove_empty_folders(path_abs):
    for root, dirs, files in os.walk(path_abs):
        if len(files) == 0 and len(dirs) == 0:
            os.rmdir(root)


def delete_invalid_files():
    cps_sizes = [380]
    fdas_root = Path('/data/rawdata/OCT/data/')
    volume_root = Path('/home/octusr2/projects/data_fast/proceeded/volume_targz/')
    cps_roots = [Path(f'/home/octusr2/projects/data_fast/proceeded/cp_projection/{size}') for size in cps_sizes]

    fdas = get_all_fdas(cache=False)
    volumes = get_all_volumes(cache=False)
    cpss = [get_all_cps(size=s, cache=False) for s in cps_sizes]

    fdas = [str(Path(x).relative_to(fdas_root).with_suffix('')) for x in fdas]

    volumes = [str(Path(x).relative_to(volume_root).with_suffix('').with_suffix('')) for x in volumes]
    volume_to_delete = [(volume_root / x).with_suffix('.tgz') for x in set(volumes).difference(set(fdas))]
    print(f'Delete {len(volume_to_delete)} volnme files')
    for file in volume_to_delete:
        file.unlink()
    remove_empty_folders('/home/octusr2/projects/data_fast/proceeded/volume_targz/')

    for i in range(len(cps_sizes)):
        cpss[i] = [str(Path(x).relative_to(cps_roots[i]).with_suffix('')) for x in cpss[i]]
        cps_to_delete = [(cps_roots[i] / x).with_suffix('.jpg') for x in set(cpss[i]).difference(set(fdas))]
        print(f'Delete {len(cps_to_delete)} cps files, size={cps_sizes[i]}')

        for file in cps_to_delete:
            file.unlink()
        remove_empty_folders(cps_roots[i])



def extract_volume(tasks, input_root, output_root, override=False):
    proc_num, tasks = tasks
    errors = []
    with Benchmark(f'Converting', print=proc_num == 0) as t:
        for i, task in enumerate(tasks):
            if i % (len(tasks) // 10 + 1) == 0:
                t.print_elapsed(f'{i}/{len(tasks)}, {i / len(tasks) * 100:.2f}%')
            task = Path(task).relative_to(input_root)
            output_file = str(output_root / (str(task) + '.tgz')).replace('', ' ')
            if override == False and os.path.isfile(output_file):
                continue 
            input_path = input_root / task
            try:
                img_array = read_fdas(input_path)
                save_as_targz(img_array, task, output_root)
            except Exception as err:
                errors.append((task, err))
                # print(f'Error with file {str(task)}: {err}')
    return errors


def extract_volume_mp(tasks):
    input_root = Path('/data/rawdata/OCT/data/')
    output_root = Path('/home/octusr2/projects/data_fast/proceeded/volume_targz')
    errors = run_multi_process(tasks, 80, partial(extract_volume, input_root=input_root, output_root=output_root), with_proc_num=True)
    errors = reduce(lambda x, y: x + y, errors)
    pkl_dump(errors, 'data/errors_volume.pkl')


def extract_volume_info(tasks, input_root):
    proc_num, tasks = tasks
    errors = []
    result = []

    with Benchmark(f'Converting', print=proc_num == 0) as t:
        for i, task in enumerate(tasks):
            if i % (len(tasks) // 10 + 1) == 0:
                t.print_elapsed(f'{i}/{len(tasks)}, {i / len(tasks) * 100:.2f}%')
            task = Path(task).relative_to(input_root)
            input_path = input_root / task
            try:
                row = {
                    'oct_path': str(task)
                }
                if input_path.suffix == '.fda':
                    obj = FDA(str(input_path))
                else:
                    obj = FDS(str(input_path))
                capture_info_02 = read_field(obj, b'@CAPTURE_INFO_02')
                param_scan_04 = read_field(obj, b'@PARAM_SCAN_04')
                row['oct_date_file'] = f"{int(capture_info_02['aquisition year'])}-{int(capture_info_02['aquisition month']):02d}-{int(capture_info_02['aquisition day']):02d}"
                row['oct_x'] = param_scan_04['x mm']
                row['oct_y'] = param_scan_04['y mm']
                row['oct_z'] = param_scan_04['z um']
                result.append(row)
            except Exception as err:
                errors.append((task, err))
                print(f'Error with file {str(task)}: {err}')
    return result, errors


def extract_volume_info_mp(tasks):
    input_root = Path('/data/rawdata/OCT/data/')
    outputs = run_multi_process(tasks, 80, partial(extract_volume_info, input_root=input_root), with_proc_num=True)
    results = [x[0] for x in outputs]
    errors = [x[1] for x in outputs]
    results = reduce(lambda x, y: x + y, results)
    results = pd.DataFrame(results)
    results['oct_path'] = results['oct_path'].map(lambda x: x.replace('\uf022', ' '))
    ensure_file('/home/octusr2/projects/data_fast/proceeded/volume_info/volume_info.csv')
    results.to_csv('/home/octusr2/projects/data_fast/proceeded/volume_info/volume_info.csv', index=False)
    errors = reduce(lambda x, y: x + y, errors)
    pkl_dump(errors, 'data/errors_volume_info.pkl')


def extract_cp(tasks, input_root, output_root, info, size):
    info = info.copy()
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
                cp = cut_cp_from_imgs(imgs, volume_size=volume_size, flip=eye=='OS', radius=1.4, threshold=100, size=(size, size))
                ensure_file(output_path)
                Image.fromarray(cp).save(output_path)
            except Exception as err:
                errors.append((task, err))
                # print(f'Error with file {str(task)}: {err}')
    return errors


def extract_cp_mp(tasks, size):
    input_root = Path('/home/octusr2/projects/data_fast/proceeded/volume_targz')
    output_root = Path(f'/home/octusr2/projects/data_fast/proceeded/cp_projection/{size}')
    info = pd.read_csv('/home/octusr2/projects/data_fast/proceeded/volume_info/volume_info.csv')
    info['oct_path'] = info['oct_path'].map(lambda x: str(Path(x).with_suffix('')))
    info = info.drop_duplicates(subset='oct_path')
    info = info.set_index('oct_path')
    
    errors = run_multi_process(tasks, 80, partial(extract_cp, input_root=input_root, output_root=output_root, info=info, size=size), with_proc_num=True)
    errors = reduce(lambda x, y: x + y, errors)
    pkl_dump(errors, 'data/errors_cp.pkl')





def calculate_position(arr):
	#拼接数组函数,计算映射函数
    array_num = list(itertools.chain.from_iterable(arr))
    array_num = np.array(array_num)
    key = np.unique(array_num)
    result = {}
    for k in key:
        index=np.argwhere(arr == k)#返回所有符合条件的索引值，更多的方法和类参数需要看
        v=[tuple(index[i]) for i in range(0,len(index))]
        result[k] = v
    result = {key:val for key, val in result.items() if  len(val)<9}
    valid_point=[]
    for key,value in result.items():
        valid_point+=value   
    return result,valid_point


def concat_images(images_path:list):
    img_list=[Image.open(fn) for fn in images_path]
    width,height=img_list[0].size
    result=Image.new(img_list[0].mode,(width,height*len(img_list)))
    
    for i,im in enumerate(images_path):
        result.paste(im,box=(i*width,0))
    return result


def get_specified_image_path(centre_img_path,num=3):#更加鲁棒的路径获取函数，之前错的地方是没有考虑到_7和_50
    #生成数字时候不能直接盲目的取-1，所以将函数修改的更好了
    res=[]
    index=int(Path(centre_img_path).name.split('_')[1][:-4])
    begin=index-(num-1)//2
    end=index+(num-1)//2
    while(begin<=end):
        res.append(str(Path(centre_img_path).parent)+'/'+Path(centre_img_path).stem.split('_')[0]+'_'+str(begin)+'.jpg')
        begin+=1
    return res

def concat_images(images_path:list):
    img_list=[Image.open(fn).convert('RGB') for fn in images_path]
    width,height=img_list[0].size
    result=Image.new(img_list[0].mode,(width*len(img_list),height))
    
    for i,im in enumerate(img_list):
        result.paste(im,box=(i*width,0,(i+1)*width,height))
    return result

def reverse_dict(countNums:dict):
    res={}
    for k,v in countNums.items():
        for tmp in v:
           res[str(tmp).replace(' ','')]=int(k)
           #print(tmp)
    return res