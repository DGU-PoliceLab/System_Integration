from argparse import ArgumentParser
from _Utils.logger import get_logger

LOGGER = get_logger(name = '[VARIABLE]')

def get_arg(category = None, arg= None):
    if category == 'root':
        args = get_root_args()
    elif category == 'sort':
        args = get_sort_args()
    else:
        raise RuntimeError(f'Category{category} is not supported, (Supported category: {category})')
    args = vars(args)
    try:
        t = args[arg]
        return t
    except:
        LOGGER.warning(f"'{arg}' does not exist in argument category {category}, category {category} : {args}")

    

def get_root_args():
    parser = ArgumentParser()
    parser.add_argument('--test', type=str, default='test',help='test')
    parser.add_argument('--nas_path', type=str, default= "/System_Integration/_Output/NAS", help='NAS path'),
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)') # 480 -> 1080 수정완료.
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0,1], help='filter by class: --class 0, or --6class 0 2 3') # 0몸통, 1얼굴, 같이 쓰려면 [0,1]
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--save_snapshot', action='store_true', default=True, help='save snapshots for each detected object')
    parser.add_argument('--snapshot_dir', type=str, default=f'/home/mhncity/data/person/27/', help='directory for saving snapshots')
    parser.add_argument('--video_writer', action='store_true', default=True, help='save video')
    parser.add_argument('--video_file', type=str, default='rtsp', help='local video file')
    parser.add_argument('--max_person_num', type=int, default=4, help='maximum number of people')
    # Tracking Args (BoTSORT)
    parser.add_argument("--track_high_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.2, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=150, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=True, action="store_true",
                        help="fuse score and iou for association")
    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")
    # ReID (Fast-ReID)
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    # RTMO
    parser.add_argument(
        '--rtmo-config', 
        type=str, 
        default="_PoseEstimation/mmlab/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py", 
        help='mmpose rtmo checkpoint')
    parser.add_argument(
        '--rtmo_checkpoint', 
        type=str, 
        default="_PoseEstimation/mmlab/mmpose/checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth", 
        help='mmpose rtmo checkpoint')
    # Event Delay
    parser.add_argument('--event_delay', type=int, default=0, help='Event insert delay time')

    args = parser.parse_args()
    args.jde = False
    args.ablation = False
    return args

def get_sort_args():
    parser = ArgumentParser("BoT-SORT")
    parser.add_argument("--demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    # Tracking Args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")
    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")
    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default="_Tracker/BoTSORT/fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default="_Tracker/BoTSORT/fast_reid/checkpoints/MOT17/mot17_sbs_S50.pth", type=str,help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    sort_args = parser.parse_args()
    return sort_args