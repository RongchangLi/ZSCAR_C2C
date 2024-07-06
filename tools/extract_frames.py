import os
import threading

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( "--video_root",help="path of downloaded videos",type=str,)
parser.add_argument("--frame_root", help="path of extracted frames", type=str)
parser.add_argument("--n_threads", help="threads to process", type=int, default=16)



def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, cfg):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(cfg.video_root, video,
                                                                                             cfg.frame_root, video[:-5])
    os.system(cmd)


def target(video_list,cfg):
    for video in video_list:
        os.makedirs(os.path.join(cfg.frame_root, video[:-5]))
        extract(video,cfg)


if __name__ == '__main__':
    cfg = parser.parse_args()
    if not os.path.exists(cfg.video_root):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(cfg.frame_root):
        os.makedirs(cfg.frame_root)

    video_list = os.listdir(cfg.video_root)
    splits = list(split(video_list, cfg.num_threads))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()