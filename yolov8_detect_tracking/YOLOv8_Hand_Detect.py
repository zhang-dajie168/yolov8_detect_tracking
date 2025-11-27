#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API

from time import time
import argparse
import logging 
import os
from pathlib import Path

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Hand Detection on RDK')
    parser.add_argument('--model-path', type=str, default='ptq_models/yolov8x_detect_bayese_640x640_nv12_modified.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../../resource/datasets/COCO2017/assets/bus.jpg', 
                       help='Path to Load Test Image.')
    parser.add_argument('--video-path', type=str, default='', 
                       help='Path to input video file for detection.')
    parser.add_argument('--camera-id', type=int, default=-1, 
                       help='Camera ID for real-time detection (default: -1 for no camera)')
    parser.add_argument('--img-save-path', type=str, default='py_result.jpg', 
                       help='Path to save output image.')
    parser.add_argument('--video-save-path', type=str, default='output_video.mp4', 
                       help='Path to save output video.')
    parser.add_argument('--classes-num', type=int, default=3, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--show-display', action='store_true', 
                       help='Show real-time display (only for video/camera mode)')
    opt = parser.parse_args()
    logger.info(opt)

    # 实例化模型
    model = YOLOv8_Detect(opt.model_path, opt.score_thres, opt.nms_thres, opt.classes_num, opt.reg)
    
    # 检测模式判断
    if opt.video_path:
        # 视频文件检测模式
        process_video(model, opt.video_path, opt.video_save_path, opt.show_display)
    elif opt.camera_id >= 0:
        # 摄像头实时检测模式
        process_camera(model, opt.camera_id, opt.video_save_path, opt.show_display)
    else:
        # 单张图片检测模式
        process_image(model, opt.test_img, opt.img_save_path)


def process_image(model, image_path, save_path):
    """处理单张图片"""
    logger.info(f"Processing image: {image_path}")
    
    # 读图
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    # 准备输入数据
    input_tensor = model.bgr2nv12(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    results = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    for class_id, score, x1, y1, x2, y2 in results:
        print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)
    
    # 保存结果
    cv2.imwrite(save_path, img)
    logger.info("\033[1;32m" + f"Image saved in path: \"./{save_path}\"" + "\033[0m")


def process_video(model, video_path, save_path, show_display=False):
    """处理视频文件"""
    logger.info(f"Processing video: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video info: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 检测当前帧
        input_tensor = model.bgr2nv12(frame)
        outputs = model.c2numpy(model.forward(input_tensor))
        results = model.postProcess(outputs)
        
        # 绘制检测结果
        for class_id, score, x1, y1, x2, y2 in results:
            draw_detection(frame, (x1, y1, x2, y2), score, class_id)
        
        # 显示处理进度
        if frame_count % 30 == 0:
            elapsed_time = time() - start_time
            fps_processed = frame_count / elapsed_time
            logger.info(f"Processed {frame_count}/{total_frames} frames, FPS: {fps_processed:.2f}")
        
        # 写入输出视频
        out.write(frame)
        
        # 实时显示
        if show_display:
            cv2.imshow('YOLOv8 Hand Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 释放资源
    cap.release()
    out.release()
    if show_display:
        cv2.destroyAllWindows()
    
    total_time = time() - start_time
    logger.info(f"Video processing completed: {frame_count} frames in {total_time:.2f}s, Average FPS: {frame_count/total_time:.2f}")
    logger.info(f"Output video saved: {save_path}")


def process_camera(model, camera_id, save_path, show_display=True):
    """处理摄像头实时视频流"""
    logger.info(f"Starting camera detection: Camera ID {camera_id}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera: {camera_id}")
        return
    
    # 获取摄像头属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Camera info: {width}x{height}, FPS: {fps}")
    
    # 创建视频写入器（如果指定了保存路径）
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from camera")
                break
                
            frame_count += 1
            
            # 检测当前帧
            input_tensor = model.bgr2nv12(frame)
            outputs = model.c2numpy(model.forward(input_tensor))
            results = model.postProcess(outputs)
            
            # 绘制检测结果
            for class_id, score, x1, y1, x2, y2 in results:
                draw_detection(frame, (x1, y1, x2, y2), score, class_id)
            
            # 显示FPS
            elapsed_time = time() - start_time
            current_fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 写入输出视频
            if save_path:
                out.write(frame)
            
            # 实时显示
            if show_display:
                cv2.imshow('YOLOv8 Hand Detection - Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        logger.info("Camera detection interrupted by user")
    
    # 释放资源
    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()
    
    total_time = time() - start_time
    logger.info(f"Camera detection completed: {frame_count} frames in {total_time:.2f}s, Average FPS: {frame_count/total_time:.2f}")
    if save_path:
        logger.info(f"Output video saved: {save_path}")


class BaseModel:
    def __init__(self, model_file: str) -> None:
        # 加载BPU的bin模型
        try:
            self.quantize_model = dnn.load(model_file)
            logger.info(f"Yolo模型加载成功: {model_file}")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_file))
            logger.error(e)
            exit(1)

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"Model input size: {self.model_input_height}x{self.model_input_weight}")

    def letterbox_resize(self, img: np.ndarray) -> np.ndarray:
        """使用letterbox方式调整图像大小"""
        img_h, img_w = img.shape[0:2]
        
        # 计算缩放比例
        scale = min(1.0 * self.model_input_height / img_h, 1.0 * self.model_input_weight / img_w)
        if scale <= 0:
            raise ValueError("Invalid scale factor.")
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # 计算填充
        x_shift = (self.model_input_weight - new_w) // 2
        y_shift = (self.model_input_height - new_h) // 2
        x_other = self.model_input_weight - new_w - x_shift
        y_other = self.model_input_height - new_h - y_shift
        
        # 调整大小并填充
        resized_img = cv2.resize(img, (new_w, new_h))
        resized_img = cv2.copyMakeBorder(resized_img, y_shift, y_other, x_shift, x_other, 
                                       cv2.BORDER_CONSTANT, value=[127, 127, 127])
        
        # 保存缩放和偏移参数用于后处理
        self.x_scale = scale
        self.y_scale = scale
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.orig_img_h = img_h
        self.orig_img_w = img_w
        
        # logger.debug(f"Letterbox resize: scale={scale:.2f}, shift=({x_shift}, {y_shift})")
        return resized_img

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to the NV12 format using letterbox resize."""
        begin_time = time()
        
        # 使用letterbox调整大小
        resized_img = self.letterbox_resize(bgr_img)
        
        height, width = resized_img.shape[0], resized_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        
        # logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12

    def forward(self, input_tensor: np.array) -> list[dnn.pyDNNTensor]:
        begin_time = time()
        outputs = self.quantize_model[0].forward(input_tensor)
        # logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time()
        numpy_outputs = [dnnTensor.buffer for dnnTensor in outputs]
        # logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return numpy_outputs


class YOLOv8_Detect(BaseModel):
    def __init__(self, model_file: str, conf_thres: float, nms_thres: float, classes_num: int, reg: int):
        super().__init__(model_file)
        
        # 打印模型输入输出信息
        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        # 将反量化系数准备好
        self.s_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")

        # DFL求期望的系数
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 阈值参数
        self.SCORE_THRESHOLD = conf_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        self.REG = reg
        self.CLASSES_NUM = classes_num
        
        # logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        # logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)
        # logger.info(f"REG = {self.REG}, CLASSES_NUM = {self.CLASSES_NUM}")

    def postProcess(self, outputs: list[np.ndarray]) -> list:
        begin_time = time()
        
        # reshape
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        m_clses = outputs[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[3].reshape(-1, self.REG * 4)
        l_clses = outputs[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[5].reshape(-1, self.REG * 4)

        # classify: 利用numpy向量化操作完成阈值筛选
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # xyxy 2 xyhw
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        xyhw2 = np.hstack([dbboxes[:,0:2], hw])

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            if np.sum(id_indices) == 0:
                continue
                
            indices = cv2.dnn.NMSBoxes(xyhw2[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
                
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                # 还原到原始图像坐标
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)

                # 边界检查
                x1 = max(0, min(x1, self.orig_img_w))
                x2 = max(0, min(x2, self.orig_img_w))
                y1 = max(0, min(y1, self.orig_img_h))
                y2 = max(0, min(y2, self.orig_img_h))

                results.append((i, scores[id_indices][indic], x1, y1, x2, y2))

        # logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return results


coco_names = ["person", "ok", "stop"]  # 您的三个类别

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, bbox, score, class_id) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()
    
    
	# 单张图片检测
	#python3 YOLOv8_Hand_Detect.py --model-path yolov8n_3class_bayese_640x640_nv12_modified.bin --test-img ./image/e178ddea-11df-4e5a-a0ae-5bf008aa5f18.jpg

	# 视频文件检测
	#python3 YOLOv8_Hand_Detect.py --model-path yolov8n_3class_bayese_640x640_nv12_modified.bin --video-path /home/sunrise/yolov8-hand-detect/1.webm --video-save-path ./output.mp4 --show-display

	# 摄像头实时检测
	#python3 YOLOv8_Hand_Detect.py --model-path yolov8n_3class_bayese_640x640_nv12_modified.bin --camera-id 0 --video-save-path ./camera_output.mp4 --show-display
    
    
