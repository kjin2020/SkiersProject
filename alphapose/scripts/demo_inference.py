"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import torch
from tqdm import tqdm
import natsort

from detector.apis import get_detector # Done
# Tracker (ReID weights, cython_bbox) is imported lazily in run_inference when POSE_TRACK is on.
from alphapose.alphapose.models import builder # Done
from alphapose.alphapose.utils.config import update_config # Done
from alphapose.alphapose.utils.detector import DetectionLoader # Done
from alphapose.alphapose.utils.file_detector import FileDetectionLoader # Done
from alphapose.alphapose.utils.transforms import flip, flip_heatmap # Done
from alphapose.alphapose.utils.vis import getTime # Done
from alphapose.alphapose.utils.webcam_detector import WebCamDetectionLoader # Done
from alphapose.alphapose.utils.writer import DataWriter # Done

# """----------------------------- Demo options -----------------------------"""
# parser = argparse.ArgumentParser(description='AlphaPose Demo')
# parser.add_argument('--cfg', type=str, required=True,
#                     help='experiment configure file name')
# parser.add_argument('--checkpoint', type=str, required=True,
#                     help='checkpoint file name')
# parser.add_argument('--sp', default=False, action='store_true',
#                     help='Use single process for pytorch')
# parser.add_argument('--detector', dest='detector',
#                     help='detector name', default="yolo")
# parser.add_argument('--detfile', dest='detfile',
#                     help='detection result file', default="")
# parser.add_argument('--indir', dest='inputpath',
#                     help='image-directory', default="")
# parser.add_argument('--list', dest='inputlist',
#                     help='image-list', default="")
# parser.add_argument('--image', dest='inputimg',
#                     help='image-name', default="")
# parser.add_argument('--outdir', dest='outputpath',
#                     help='output-directory', default="data/alphapose_output/")
# parser.add_argument('--save_img', default=False, action='store_true',
#                     help='save result as image')
# parser.add_argument('--vis', default=False, action='store_true',
#                     help='visualize image')
# parser.add_argument('--showbox', default=False, action='store_true',
#                     help='visualize human bbox')
# parser.add_argument('--profile', default=False, action='store_true',
#                     help='add speed profiling at screen output')
# parser.add_argument('--format', type=str,
#                     help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
# parser.add_argument('--min_box_area', type=int, default=0,
#                     help='min box area to filter out')
# parser.add_argument('--detbatch', type=int, default=5,
#                     help='detection batch size PER GPU')
# parser.add_argument('--posebatch', type=int, default=64,
#                     help='pose estimation maximum batch size PER GPU')
# parser.add_argument('--eval', dest='eval', default=False, action='store_true',
#                     help='save the result json as coco format, using image index(int) instead of image name(str)')
# parser.add_argument('--gpus', type=str, dest='gpus', default="0",
#                     help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
# parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
#                     help='the length of result buffer, where reducing it will lower requirement of cpu memory')
# parser.add_argument('--flip', default=False, action='store_true',
#                     help='enable flip testing')
# parser.add_argument('--debug', default=False, action='store_true',
#                     help='print detail information')
# """----------------------------- Video options -----------------------------"""
# parser.add_argument('--video', dest='video',
#                     help='video-name', default="")
# parser.add_argument('--webcam', dest='webcam', type=int,
#                     help='webcam number', default=-1)
# parser.add_argument('--save_video', dest='save_video',
#                     help='whether to save rendered video', default=False, action='store_true')
# parser.add_argument('--vis_fast', dest='vis_fast',
#                     help='use fast rendering', action='store_true', default=False)
# """----------------------------- Tracking options -----------------------------"""
# parser.add_argument('--pose_flow', dest='pose_flow',
#                     help='track humans in video with PoseFlow', action='store_true', default=False)
# parser.add_argument('--pose_track', dest='pose_track',
#                     help='track humans in video with reid', action='store_true', default=False)

# args = parser.parse_args()
# cfg = update_config(args.cfg)

# if platform.system() == 'Windows':
#     args.sp = True

# args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
# args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
# args.detbatch = args.detbatch * len(args.gpus)
# args.posebatch = args.posebatch * len(args.gpus)
# args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

# if not args.sp:
#     torch.multiprocessing.set_start_method('forkserver', force=True)
#     torch.multiprocessing.set_sharing_strategy('file_system')


def check_input(args):
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    # for detection results
    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]

        return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info(args):
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


# if __name__ == "__main__":
#     mode, input_source = check_input()

#     if not os.path.exists(args.outputpath):
#         os.makedirs(args.outputpath)

#     # Load detection loader
#     if mode == 'webcam':
#         det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
#         det_worker = det_loader.start()
#     elif mode == 'detfile':
#         det_loader = FileDetectionLoader(input_source, cfg, args)
#         det_worker = det_loader.start()
#     else:
#         det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
#         det_worker = det_loader.start()

#     # Load pose model
#     pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

#     print('Loading pose model from %s...' % (args.checkpoint,))
#     pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
#     pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
#     if args.pose_track:
#         tracker = Tracker(tcfg, args)
#     if len(args.gpus) > 1:
#         pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
#     else:
#         pose_model.to(args.device)
#     pose_model.eval()

#     runtime_profile = {
#         'dt': [],
#         'pt': [],
#         'pn': []
#     }

#     # Init data writer
#     queueSize = 2 if mode == 'webcam' else args.qsize
#     if args.save_video and mode != 'image':
#         from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
#         if mode == 'video':
#             video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
#         else:
#             video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
#         video_save_opt.update(det_loader.videoinfo)
#         writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize, video_name=os.path.basename(input_source)).start()
#     else:
#         writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize, video_name=os.path.basename(input_source)).start()

#     if mode == 'webcam':
#         print('Starting webcam demo, press Ctrl + C to terminate...')
#         sys.stdout.flush()
#         im_names_desc = tqdm(loop())
#     else:
#         data_len = det_loader.length
#         im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

#     batchSize = args.posebatch
#     if args.flip:
#         batchSize = int(batchSize / 2)
#     try:
#         for i in im_names_desc:
#             start_time = getTime()
#             with torch.no_grad():
#                 (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
#                 if orig_img is None:
#                     break
#                 if boxes is None or boxes.nelement() == 0:
#                     writer.save(None, None, None, None, None, orig_img, im_name)
#                     continue
#                 if args.profile:
#                     ckpt_time, det_time = getTime(start_time)
#                     runtime_profile['dt'].append(det_time)
#                 # Pose Estimation
#                 inps = inps.to(args.device)
#                 datalen = inps.size(0)
#                 leftover = 0
#                 if (datalen) % batchSize:
#                     leftover = 1
#                 num_batches = datalen // batchSize + leftover
#                 hm = []
#                 for j in range(num_batches):
#                     inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
#                     if args.flip:
#                         inps_j = torch.cat((inps_j, flip(inps_j)))
#                     hm_j = pose_model(inps_j)
#                     if args.flip:
#                         hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
#                         hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
#                     hm.append(hm_j)
#                 hm = torch.cat(hm)
#                 if args.profile:
#                     ckpt_time, pose_time = getTime(ckpt_time)
#                     runtime_profile['pt'].append(pose_time)
#                 if args.pose_track:
#                     boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
#                 hm = hm.cpu()
#                 writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
#                 if args.profile:
#                     ckpt_time, post_time = getTime(ckpt_time)
#                     runtime_profile['pn'].append(post_time)

#             if args.profile:
#                 # TQDM
#                 im_names_desc.set_description(
#                     'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
#                         dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
#                 )
#         print_finish_info()
#         while(writer.running()):
#             time.sleep(1)
#             print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
#         writer.stop()
#         det_loader.stop()
#     except Exception as e:
#         print(repr(e))
#         print('An error as above occurs when processing the images, please check it')
#         pass
#     except KeyboardInterrupt:
#         print_finish_info()
#         # Thread won't be killed when press Ctrl+C
#         if args.sp:
#             det_loader.terminate()
#             while(writer.running()):
#                 time.sleep(1)
#                 print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
#             writer.stop()
#         else:
#             # subprocesses are killed, manually clear queues

#             det_loader.terminate()
#             writer.terminate()
#             writer.clear_queues()
#             det_loader.clear_queues()

def run_inference(args_cfg):

    class Args:
        cfg = args_cfg.ALPHA_ARGS.CFG_PATH
        checkpoint = args_cfg.ALPHA_ARGS.WEIGHTS_PATH
        video = args_cfg.VIDEO_PATH
        outputpath = args_cfg.ALPHA_ARGS.get("OUTPUT_PATH", '') or "data/alphapose_output/"
        save_video = args_cfg.ALPHA_ARGS.SAVE_VIDEO
        vis_fast = args_cfg.ALPHA_ARGS.VIS_FAST
        sp = True  # Force single-process on Windows
        detector = 'yolo'
        gpus = args_cfg.ALPHA_ARGS.GPUS
        pose_track = args_cfg.ALPHA_ARGS.POSE_TRACK
        pose_flow = False
        min_box_area = 0
        detbatch = 5
        posebatch = 64
        webcam = -1
        qsize = 1024
        flip = False
        profile = False
        save_img = False
        vis = False
        showbox = args_cfg.ALPHA_ARGS.SHOWBOX
        format = args_cfg.ALPHA_ARGS.get("FORMAT", '') or "coco"
        eval = False
        save_json = args_cfg.ALPHA_ARGS.SAVE_JSON
        yolo_weight_path = args_cfg.ALPHA_ARGS.YOLO_WEIGHT_PATH
    
    args = Args()

    cfg = update_config(args.cfg)

    if platform.system() == 'Windows':
        args.sp = True

    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.detbatch = args.detbatch * len(args.gpus)
    args.posebatch = args.posebatch * len(args.gpus)
    args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

    if not args.sp:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    mode, input_source = check_input(args)

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        from trackers.tracker_api import Tracker
        from trackers.tracker_cfg import cfg as tcfg
        from trackers import track
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    if args.save_video and mode != 'image':
        from alphapose.alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
        if mode == 'video':
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize, video_name=os.path.basename(input_source)).start()
    else:
        writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize, video_name=os.path.basename(input_source)).start()

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info(args)
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        det_loader.stop()

        return writer.get_output_path(), writer.results()

    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info(args)
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()


