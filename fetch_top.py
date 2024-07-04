from ginniro.util import fetch_top
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', dest='source_path', type=str,
                        help='source_path')
parser.add_argument('--target_path', dest='target_path', type=str, default='[TARGET_PATH]', 
                        help='target_path')
parser.add_argument('--fname', dest='fname', type=str, default='agt_9_performance_records.json',
                        help='fname')
parser.add_argument('--start', dest='start', type=int, default=30, help='start')
parser.add_argument('--end', dest='end', type=int, default=274, help='end')

args = parser.parse_args()
params = vars(args)

params['target_path'] = params['source_path'] + '_top10'

source_path = params['source_path']
save_path = params['target_path']
fname = params['fname']
start = params['start']
end = params['end']


res = fetch_top(source_path, fname, start=start, end=end, k=20)

if not os.path.exists(save_path):
    os.makedirs(save_path)

for idx in range(len(res)):
    fn = res[idx][0]
    srcfile = os.path.join(source_path, fn, fname)
    if not os.path.exists(os.path.join(save_path, str(idx+1))):
        os.makedirs(os.path.join(save_path, str(idx+1)))
    dstfile = os.path.join(save_path, str(idx+1), fname)
    shutil.copyfile(srcfile, dstfile) 
