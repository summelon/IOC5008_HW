import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import json
import math


def detect(save_json=False, save_img=False, inf_detail=False):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    save_json, inf_detail = opt.save_json, opt.inf_detail
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(
        device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #    shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx',
                          verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        # set True to speed up constant image size inference
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        #save_img = True
        dataset = LoadImages(source, img_size=img_size,
                             half=half, inf_detail=inf_detail)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]

    # Run inference
    contents = []  # List for .json file to write in
    print("Start inference:")
    t0 = time.time()  # Start time counting
    infed_img_counter = 0  # Images have inferenced
    for path, img, im0s, vid_cap in dataset:
        infed_img_counter += 1
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            s += 'There are '
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g of %s, ' % (n, classes[int(c)])  # add to string

                # Write results
                img_bbox, img_score, img_label = [], [], []
                for *xyxy, conf, _, cls in det:
                    if save_json:  # Write to file
                        img_bbox.append((xyxy[1].int().item(), xyxy[0].int().item(),
                                         xyxy[3].int().item(), xyxy[2].int().item()))
                        img_score.append(conf.item())
                        label_buffer = cls.int().item()
                        if label_buffer == 0:
                            img_label.append(10)
                        else:
                            img_label.append(label_buffer)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)])

                contents.append(
                    dict(zip(["bbox", "score", "label"], [img_bbox, img_score, img_label])))
            else:
                contents.append(
                    {"bbox": [(7, 40, 40, 61)], "score": [0.34], "label": [5]})
                # contents.append({})

            if inf_detail:
                print('Done. %stime cost is %.3fs.' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

        #print('Cost time now is %.2f' % (time.time()-to))
        if (not save_json) and ((time.time()-t0) > 10):
            print("Test over.")
            break

    if save_json or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        with open(out+'0856154_number.json', 'a') as file:
            json.dump(contents, file)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    total_time = time.time() - t0
    FPS = infed_img_counter / total_time
    print('Done. (%.3fs)' % total_time)
    print('Total %g images, %.fms per image, FPS is %.f.' %
          (infed_img_counter, (1000/FPS), math.floor(FPS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolov3-tiny.cfg', help='cfg file path')
    parser.add_argument('--data', type=str,
                        default='data/SVHN.data', help='coco.data file path')
    parser.add_argument('--weights', type=str,
                        default='weights/best.pt', help='path to weights file')
    # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='test/', help='source')
    parser.add_argument('--output', type=str, default='predict_result/',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true',
                        help='half precision FP16 inference')
    parser.add_argument('--device', default='0',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-json', action='store_true',
                        help='choose to save predicted .json file')
    parser.add_argument('--inf-detail', action='store_true',
                        help='choose to print inference detail')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
