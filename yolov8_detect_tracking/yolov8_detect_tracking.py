#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point32
from geometry_msgs.msg import PolygonStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import copy

# 导入RDK YOLO模型和BOTSORT跟踪器
from .YOLOv8_Hand_Detect import YOLOv8_Detect
from .BOTSort_rdk import BOTSORT, OSNetReID


class TrackedTarget:
    """跟踪目标信息类 - 使用__slots__优化内存"""
    __slots__ = ('track_id', 'bbox', 'feature', 'height_pixels', 'first_seen_time', 
                 'last_seen_time', 'last_update_time', 'lost_frames', 'is_recovered', 
                 'is_switched', 'original_track_id', 'recovery_time', 'update_paused')
    
    def __init__(self, track_id: int, bbox: List[float], feature: np.ndarray, 
                 height_pixels: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.height_pixels = height_pixels
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
        self.is_switched = False
        self.original_track_id = track_id
        self.recovery_time = None
        self.update_paused = False
    
    def update(self, bbox: List[float], feature: np.ndarray, height_pixels: float, timestamp: float):
        """更新目标信息"""
        self.bbox = bbox
        self.feature = feature
        self.height_pixels = height_pixels
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
    
    def mark_lost(self):
        """标记目标丢失"""
        self.lost_frames += 1
        self.update_paused = True
        self.recovery_time = None
    
    def mark_recovered(self, timestamp: float):
        """标记目标找回 - 取消冷却期，立即恢复正常更新"""
        self.is_recovered = True
        self.lost_frames = 0
        self.recovery_time = None  # 不再设置恢复时间
        self.update_paused = False  # 立即结束暂停状态
    
    def switch_to_new_id(self, new_track_id: int):
        """切换到新的跟踪ID"""
        self.is_switched = True
        self.track_id = new_track_id

class Yolov8HandTrackNode(Node):
    def __init__(self):
        super().__init__('yolov8_hand_track_node')

        # 声明参数
        self._declare_parameters()
        
        # 获取参数
        self._get_parameters()
        
        self.min_process_interval = 1.0 / self.max_processing_fps
        self.last_process_time = time.time()

        # 初始化模型和组件
        self._initialize_components()
        
        # 初始化变量
        self._initialize_variables()
        
        self.get_logger().info("YOLOv8 Hand Track Node initialized with ReID recovery (Optimized)")
        self.print_parameters()

    def _declare_parameters(self):
        """声明所有参数"""
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('ok_confirm_frames', 3)
        self.declare_parameter('tracking_protection_time', 5.0)
        self.declare_parameter('reid_similarity_threshold', 0.8)
        self.declare_parameter('height_change_threshold', 0.15)
        self.declare_parameter('lost_timeout_threshold', 10.0)
        self.declare_parameter('reid_model_path', 'osnet_64x128_nv12.bin')
        self.declare_parameter('roi_threshold', 0.5)  # 降低IoU阈值
        
    def _get_parameters(self):
        """获取参数值"""
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.ok_confirm_frames = self.get_parameter('ok_confirm_frames').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.height_change_threshold = self.get_parameter('height_change_threshold').value
        self.lost_timeout_threshold = self.get_parameter('lost_timeout_threshold').value
        self.roi_threshold = self.get_parameter('roi_threshold').value

    def print_parameters(self):
        """打印参数信息"""
        self.get_logger().info("===== 参数配置信息 =====")
        self.get_logger().info(f"置信度阈值: {self.conf_threshold}")
        self.get_logger().info(f"最大处理帧率: {self.max_processing_fps}FPS")
        self.get_logger().info(f"OK手势确认帧数: {self.ok_confirm_frames}")
        self.get_logger().info(f"跟踪保护时间: {self.tracking_protection_time}s")
        self.get_logger().info(f"ReID相似度阈值: {self.reid_similarity_threshold}")
        self.get_logger().info(f"高度变化阈值: {self.height_change_threshold}")
        self.get_logger().info(f"丢失超时阈值: {self.lost_timeout_threshold}s")
        self.get_logger().info(f"ROI重叠阈值: {self.roi_threshold}")
        self.get_logger().info("=========================")

    def _initialize_components(self):
        """初始化模型和跟踪器"""
        model_path = self.get_parameter('model_path').value
        reid_model_path = self.get_parameter('reid_model_path').value
        
        # 加载YOLOv8 hand detect模型
        self.model = YOLOv8_Detect(model_path, self.conf_threshold, 0.45, 3, 16)
        
        # 初始化ReID编码器
        self.reid_encoder = None
        try:
            self.reid_encoder = OSNetReID(reid_model_path)
            self.get_logger().info(f"ReID模型加载成功: {reid_model_path}")
        except Exception as e:
            self.get_logger().error(f"ReID模型加载失败: {e}")
        
        # 初始化BOTSORT跟踪器
        tracker_args = {
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 10,
            'match_thresh': 0.68,
            'fuse_score': False,
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.7,
            'with_reid': False,
            'reid_model_path': reid_model_path
        }
        self.tracker = BOTSORT(tracker_args)
        
        # 初始化CV bridge
        self.bridge = CvBridge()
        
        # 创建订阅和发布
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.detect_pose_pub = self.create_publisher(Image, 'tracks', 10)
        self.keypoint_tracks_pub = self.create_publisher(PolygonStamped, '/keypoint_tracks', 10)

    def _initialize_variables(self):
        """初始化变量"""
        # 跟踪相关变量
        self.tracked_persons: Dict[int, Dict] = {}

        # 手势检测历史
        self.ok_gesture_history: Dict[int, deque] = {}
        self.stop_gesture_history: Dict[int, deque] = {}
        
        # 当前正在跟踪的目标ID
        self.current_tracking_id = None
        
        # 跟踪目标信息存储
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        
        # 目标丢失时间记录
        self.target_lost_time: Optional[float] = None
        
        # 类别名称映射
        self.class_names = {0: "person", 1: "ok", 2: "stop"}

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集区域
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        # 计算交集面积
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # 计算并集面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def find_person_for_gesture(self, gesture_box, person_boxes):
        """为手势找到对应的人体边界框 - 使用中心点匹配"""
        gx1, gy1, gx2, gy2 = gesture_box
        gesture_center_x = (gx1 + gx2) / 2
        gesture_center_y = (gy1 + gy2) / 2
        
        best_person_box = None
        best_person_id = None
        min_distance = float('inf')
        
        for person_id, person_box in person_boxes.items():
            px1, py1, px2, py2 = person_box
            
            # 检查手势中心点是否在人体框内
            if (px1 <= gesture_center_x <= px2 and 
                py1 <= gesture_center_y <= py2):
                
                # 计算中心点到人体框中心的距离
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2
                distance = ((gesture_center_x - person_center_x) ** 2 + 
                        (gesture_center_y - person_center_y) ** 2) ** 0.5
                
                self.get_logger().info(f"手势中心在ID {person_id} 框内，距离: {distance:.1f}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_person_box = person_box
                    best_person_id = person_id
        
        # 如果找到包含手势中心点的人体框，直接返回
        if best_person_id is not None:
            return best_person_id, best_person_box, 1.0  # 重叠比例为1.0
        
        # 如果没有找到，使用原来的IoU方法作为备选
        return self.find_person_for_gesture_fallback(gesture_box, person_boxes)

    def find_person_for_gesture_fallback(self, gesture_box, person_boxes):
        """备选方法：使用重叠比例"""
        best_overlap = 0
        best_person_box = None
        best_person_id = None
        
        gx1, gy1, gx2, gy2 = gesture_box
        gesture_area = (gx2 - gx1) * (gy2 - gy1)
        
        for person_id, person_box in person_boxes.items():
            px1, py1, px2, py2 = person_box
            
            # 计算交集面积
            inter_x1 = max(gx1, px1)
            inter_y1 = max(gy1, py1)
            inter_x2 = min(gx2, px2)
            inter_y2 = min(gy2, py2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            overlap = inter_area / gesture_area if gesture_area > 0 else 0
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_person_box = person_box
                best_person_id = person_id
        
        return best_person_id, best_person_box, best_overlap

    def extract_feature_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """从边界框提取特征 - 优化内存分配"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # 边界检查
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        try:
            if self.reid_encoder is not None:
                return self.reid_encoder.extract_feature(crop)
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)
        
    def save_tracked_target(self, track_id: int, bbox: List[float], image: np.ndarray, timestamp: float):
        """保存跟踪目标信息 - 每帧更新所有特征信息"""
        if track_id not in self.tracked_targets:
            # 新目标：提取特征
            feature = self.extract_feature_from_bbox(image, bbox)
            height_pixels = bbox[3] - bbox[1]
            self.tracked_targets[track_id] = TrackedTarget(track_id, bbox, feature, height_pixels, timestamp)
            return
        
        target = self.tracked_targets[track_id]
        
        # 如果更新被暂停（仅用于其他暂停情况），跳过特征更新
        if target.update_paused:
            target.bbox = bbox
            target.height_pixels = bbox[3] - bbox[1]
            target.last_seen_time = timestamp
            target.lost_frames = 0
            return
        
        # 每帧都更新特征（高精度模式）
        feature = self.extract_feature_from_bbox(image, bbox)
        height_pixels = bbox[3] - bbox[1]
        target.update(bbox, feature, height_pixels, timestamp)

    def try_recover_lost_target(self, current_tracks: List[Dict], image: np.ndarray, timestamp: float) -> Optional[int]:
        """立即尝试找回丢失的跟踪目标 - 增加高度筛选"""
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        
        target = self.tracked_targets[self.current_tracking_id]
        
        # 立即记录丢失时间
        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，开始立即ReID匹配找回")
        
        # 准备候选目标
        candidate_tracks = []
        for track in current_tracks:
            track_id = track['track_id']
            is_currently_tracked = (
                track_id in self.tracked_persons and 
                self.tracked_persons[track_id]['is_tracking'] and
                track_id != self.current_tracking_id
            )
            
            if not is_currently_tracked:
                candidate_tracks.append(track)
        
        if not candidate_tracks:
            return None
        
        # 获取丢失目标的高度信息
        target_height_pixels = target.height_pixels
        self.get_logger().info(f"目标 {self.current_tracking_id} 丢失时高度: {target_height_pixels:.1f}px")
        
        # ReID匹配
        best_match_id = None
        best_similarity = 0.0
        
        for track in candidate_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            # 计算当前候选目标的高度
            x1, y1, x2, y2 = bbox
            candidate_height_pixels = y2 - y1
            
            # 高度变化筛选
            height_ratio = candidate_height_pixels / target_height_pixels
            height_change = abs(1.0 - height_ratio)
            
            # 如果高度变化超过阈值，跳过该候选目标
            if height_change > self.height_change_threshold:
                self.get_logger().warning(f"候选目标 ID:{track_id} 高度变化 {height_change:.3f} 超过阈值 {self.height_change_threshold}, 跳过匹配")
                continue
            
            candidate_feature = self.extract_feature_from_bbox(image, bbox)
            
            if candidate_feature is not None and np.any(candidate_feature):
                similarity = np.dot(target.feature, candidate_feature) / (
                    np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
                )
                
                # 记录匹配信息（包括高度信息）
                self.get_logger().info(f"候选目标 ID:{track_id} ReID相似度: {similarity:.3f}, 高度变化: {height_change:.3f}")
                
                if similarity >= self.reid_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track_id
        
        if best_match_id is not None:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} ReID找回成功! 匹配ID: {best_match_id}, 相似度: {best_similarity:.3f}"
            )
            
            target_bbox = next(t['bbox'] for t in candidate_tracks if t['track_id'] == best_match_id)
            target.mark_recovered(timestamp)
            
            # 重要修改：找回后立即结束暂停状态，恢复正常特征更新
            target.update_paused = False
            target.recovery_time = None
            
            # 立即保存目标信息（每帧更新特征）
            self.save_tracked_target(self.current_tracking_id, target_bbox, image, timestamp)
            
            recovered_id = best_match_id
            
            if best_match_id != self.current_tracking_id:
                if best_match_id in self.tracked_targets:
                    self.tracked_targets[best_match_id].switch_to_new_id(best_match_id)
                    self.tracked_targets[best_match_id].original_track_id = self.current_tracking_id
                    self.tracked_targets[best_match_id].mark_recovered(timestamp)
                    # 同样结束新目标的暂停状态
                    self.tracked_targets[best_match_id].update_paused = False
                    self.tracked_targets[best_match_id].recovery_time = None
            
            self.target_lost_time = None
            
            if recovered_id in self.tracked_persons:
                self.tracked_persons[recovered_id]['is_tracking'] = True
                self.tracked_persons[recovered_id]['tracking_start_time'] = timestamp
                self.tracked_persons[recovered_id]['last_seen_time'] = timestamp
            
            return recovered_id
        
        self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败: 无匹配目标达到阈值")
        return None

    def _verify_target_with_reid(self, target: TrackedTarget, track: Dict, image: np.ndarray, timestamp: float) -> Optional[int]:
        """使用ReID验证目标身份 - 增加高度筛选"""
        track_id = track['track_id']
        bbox = track['bbox']
        
        # 计算当前目标的高度
        x1, y1, x2, y2 = bbox
        candidate_height_pixels = y2 - y1
        target_height_pixels = target.height_pixels
        
        # 高度变化筛选
        height_ratio = candidate_height_pixels / target_height_pixels
        height_change = abs(1.0 - height_ratio)
        
        # 如果高度变化超过阈值，直接返回失败
        if height_change > self.height_change_threshold:
            self.get_logger().warning(f"验证目标 ID:{track_id} 高度变化 {height_change:.3f} 超过阈值 {self.height_change_threshold}, 验证失败")
            return None
        
        candidate_feature = self.extract_feature_from_bbox(image, bbox)
        
        if candidate_feature is not None and np.any(candidate_feature):
            similarity = np.dot(target.feature, candidate_feature) / (
                np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
            )
            
            if similarity >= self.reid_similarity_threshold:
                self.get_logger().info(f"ReID验证成功: ID {track_id}, 相似度: {similarity:.3f}, 高度变化: {height_change:.3f}")
                
                target.mark_recovered(timestamp)
                # 重要修改：验证成功后立即结束暂停状态
                target.update_paused = False
                target.recovery_time = None
                
                self.save_tracked_target(target.track_id, bbox, image, timestamp)
                self.target_lost_time = None
                
                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id]['is_tracking'] = True
                    self.tracked_persons[track_id]['last_seen_time'] = timestamp
                    if track_id == target.track_id:
                        pass
                    else:
                        self.tracked_persons[track_id]['tracking_start_time'] = timestamp
                
                return track_id
            else:
                self.get_logger().warning(f"ReID验证失败: ID {track_id}, 相似度: {similarity:.3f}")

        return None

    def image_callback(self, msg):
        """图像回调 - 优化性能"""
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
        
        self.last_process_time = current_time

        try:
            # 图像转换
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # YOLO推理
            input_tensor = self.model.bgr2nv12(cv_image)
            outputs = self.model.c2numpy(self.model.forward(input_tensor))
            
            # 后处理
            results = self.model.postProcess(outputs)

            # 分离检测结果
            person_detections = []
            ok_gestures = []
            stop_gestures = []
            
            for class_id, score, x1, y1, x2, y2 in results:
                if class_id == 0:  # person
                    person_detections.append([x1, y1, x2-x1, y2-y1, score, 0])
                elif class_id == 1:  # ok手势
                    ok_gestures.append((x1, y1, x2, y2, score))
                    self.get_logger().info(f"检测到OK手势: ({x1}, {y1}, {x2}, {y2}), 置信度: {score:.2f}")
                elif class_id == 2:  # stop手势
                    stop_gestures.append((x1, y1, x2, y2, score))
                    self.get_logger().info(f"检测到STOP手势: ({x1}, {y1}, {x2}, {y2}), 置信度: {score:.2f}")

            # 跟踪person检测结果
            tracking_results = self.tracker.update(person_detections, cv_image)

            # 处理跟踪结果
            tracks = []
            person_boxes = {}  # 存储person的边界框，用于手势匹配
            
            for result in tracking_results:
                x, y, w, h, track_id, score, cls, _, _ = result  # 去掉关键点相关参数
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                track_data = {
                    'track_id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(score),
                }
                tracks.append(track_data)
                person_boxes[track_id] = (x1, y1, x2, y2)

            # 更新跟踪状态
            self._update_tracking_state(tracks, cv_image, current_time, ok_gestures, stop_gestures, person_boxes)
            
            # 清理长时间未出现的跟踪目标
            self._cleanup_old_tracks(current_time, set(track['track_id'] for track in tracks))

            # 可视化并发布结果
            self._publish_results(cv_image, tracks, msg.header, ok_gestures, stop_gestures)
                   

        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def _update_tracking_state(self, tracks: List[Dict], cv_image: np.ndarray, current_time: float, 
                             ok_gestures: List, stop_gestures: List, person_boxes: Dict):
        """更新跟踪状态 - 提取为独立方法"""
        current_track_ids = set()
        
        for track in tracks:
            track_id = track['track_id']
            current_track_ids.add(track_id)
            
            if track_id not in self.tracked_persons:
                self._initialize_new_track(track_id, current_time)
            else:
                self.tracked_persons[track_id]['last_seen_time'] = current_time

            # 如果当前正在跟踪这个目标，保存特征
            if self.current_tracking_id == track_id:
                self.save_tracked_target(track_id, track['bbox'], cv_image, current_time)

        # 处理手势控制逻辑
        self._process_gesture_control(ok_gestures, stop_gestures, person_boxes, cv_image, current_time)

        # 处理丢失目标
        self._handle_lost_targets(current_track_ids, tracks, cv_image, current_time)

    def _initialize_new_track(self, track_id: int, current_time: float):
        """初始化新跟踪目标"""
        self.tracked_persons[track_id] = {
            'is_tracking': False,
            'tracking_start_time': 0.0,
            'last_ok_time': 0.0,
            'first_seen_time': current_time,
            'last_seen_time': current_time
        }
        # 确保手势历史记录被正确初始化
        if track_id not in self.ok_gesture_history:
            self.ok_gesture_history[track_id] = deque(maxlen=self.ok_confirm_frames)
        if track_id not in self.stop_gesture_history:
            self.stop_gesture_history[track_id] = deque(maxlen=self.ok_confirm_frames)

    def _process_gesture_control(self, ok_gestures: List, stop_gestures: List, 
                               person_boxes: Dict, cv_image: np.ndarray, current_time: float):
        """处理手势控制逻辑 - 修复版本"""
        # 处理ok手势检测
        if ok_gestures:
            self.get_logger().info(f"开始处理 {len(ok_gestures)} 个OK手势")
            
        for ok_gesture in ok_gestures:
            x1, y1, x2, y2, score = ok_gesture
            ok_box = (x1, y1, x2, y2)
            
            # 找到与ok手势重叠度最高的人体
            person_id, person_box, iou = self.find_person_for_gesture(ok_box, person_boxes)
            
            if person_id is not None:
                self.get_logger().info(f"OK手势与ID {person_id} 的IoU: {iou:.3f} (阈值: {self.roi_threshold})")
                
                if iou >= self.roi_threshold:
                    # 确保手势历史记录存在
                    if person_id not in self.ok_gesture_history:
                        self.ok_gesture_history[person_id] = deque(maxlen=self.ok_confirm_frames)
                    
                    # 添加手势检测记录
                    self.ok_gesture_history[person_id].append(True)
                    current_count = len(self.ok_gesture_history[person_id])
                    ok_confirmed = current_count >= self.ok_confirm_frames
                    
                    self.get_logger().info(f"ID {person_id} OK手势历史: {current_count}/{self.ok_confirm_frames}")
                    
                    if ok_confirmed:
                        self._handle_ok_gesture(person_id, person_box, cv_image, current_time)
                else:
                    self.get_logger().warning(f"OK手势与ID {person_id} 的IoU {iou:.3f} 低于阈值 {self.roi_threshold}")
            else:
                self.get_logger().warning("未找到与OK手势匹配的人员")

        # 处理stop手势检测
        for stop_gesture in stop_gestures:
            x1, y1, x2, y2, score = stop_gesture
            stop_box = (x1, y1, x2, y2)
            
            # 找到与stop手势重叠度最高的人体
            person_id, person_box, iou = self.find_person_for_gesture(stop_box, person_boxes)
            
            if person_id is not None and iou >= self.roi_threshold:
                # 确保手势历史记录存在
                if person_id not in self.stop_gesture_history:
                    self.stop_gesture_history[person_id] = deque(maxlen=self.ok_confirm_frames)
                
                # 添加手势检测记录
                self.stop_gesture_history[person_id].append(True)
                current_count = len(self.stop_gesture_history[person_id])
                stop_confirmed = current_count >= self.ok_confirm_frames
                
                self.get_logger().info(f"ID {person_id} STOP手势历史: {current_count}/{self.ok_confirm_frames}")
                
                if stop_confirmed:
                    self._handle_stop_gesture(person_id, current_time)

    def _handle_ok_gesture(self, person_id: int, person_box: Tuple, cv_image: np.ndarray, current_time: float):
        """处理ok手势确认"""
        person = self.tracked_persons[person_id]
        in_cooldown_period = (current_time - person['last_ok_time'] < self.tracking_protection_time)
        
        if not in_cooldown_period:
            self.current_tracking_id = person_id
            person['is_tracking'] = True
            person['tracking_start_time'] = current_time
            person['last_ok_time'] = current_time
            
            # 清空手势历史
            if person_id in self.ok_gesture_history:
                self.ok_gesture_history[person_id].clear()
            
            self.save_tracked_target(person_id, list(person_box), cv_image, current_time)
            self.target_lost_time = None
            self.get_logger().info(f"🎯 开始跟踪 ID: {person_id} (OK手势确认)")

    def _handle_stop_gesture(self, person_id: int, current_time: float):
        """处理stop手势确认"""
        if self.current_tracking_id == person_id:
            person = self.tracked_persons[person_id]
            in_protection_period = (current_time - person['tracking_start_time'] < self.tracking_protection_time)
            
            if not in_protection_period:
                person['is_tracking'] = False
                person['last_ok_time'] = current_time
                
                # 清空手势历史
                if person_id in self.stop_gesture_history:
                    self.stop_gesture_history[person_id].clear()
                
                self.current_tracking_id = None
                self.target_lost_time = None
                self.get_logger().info(f"🛑 停止跟踪 ID: {person_id}")

    def _handle_lost_targets(self, current_track_ids: set, tracks: List[Dict], 
                            cv_image: np.ndarray, current_time: float):
        """处理丢失目标 - 添加自动找回功能"""
        if self.current_tracking_id is not None and self.current_tracking_id not in current_track_ids:
            if self.current_tracking_id in self.tracked_targets:
                self.tracked_targets[self.current_tracking_id].mark_lost()
                
                if self.current_tracking_id in self.tracked_persons:
                    self.tracked_persons[self.current_tracking_id]['is_tracking'] = False
                
                # 确保丢失时间被正确设置
                if self.target_lost_time is None:
                    self.target_lost_time = current_time
                    self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失，立即启动ReID匹配找回")
                
                # 检查是否超时
                time_since_lost = current_time - self.target_lost_time
                
                if time_since_lost > self.lost_timeout_threshold:
                    self.get_logger().warning(
                        f"目标 {self.current_tracking_id} 丢失超过 {self.lost_timeout_threshold} 秒，停止跟踪并清除目标信息，需要重新OK手势选择跟踪目标"
                    )
                    self._clear_tracking_target()
                    return
                
                recovered_id = self.try_recover_lost_target(tracks, cv_image, current_time)
                if recovered_id is not None:
                    # 重要：在设置新ID前，清理原目标的跟踪状态
                    old_tracking_id = self.current_tracking_id
                    self.current_tracking_id = recovered_id
                    
                    # 确保新目标被正确标记为跟踪状态
                    if recovered_id in self.tracked_persons:
                        self.tracked_persons[recovered_id]['is_tracking'] = True
                        self.tracked_persons[recovered_id]['tracking_start_time'] = current_time
                        self.tracked_persons[recovered_id]['last_seen_time'] = current_time
                    
                    # 重置丢失时间
                    self.target_lost_time = None
                    
                    # 保存新目标的特征信息
                    if recovered_id in [t['track_id'] for t in tracks]:
                        track = next(t for t in tracks if t['track_id'] == recovered_id)
                        self.save_tracked_target(recovered_id, track['bbox'], cv_image, current_time)
                    
                    self.get_logger().info(f"ReID找回成功，从ID {old_tracking_id} 切换到新ID: {recovered_id}")
                else:
                    self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败，保持丢失状态")
                    
                self.get_logger().info(f"======================================================")
        
        elif self.current_tracking_id is not None and self.current_tracking_id in current_track_ids:
            if self.target_lost_time is not None:
                track = next(t for t in tracks if t['track_id'] == self.current_tracking_id)
                verified_id = self._verify_target_with_reid(
                    self.tracked_targets[self.current_tracking_id], track, cv_image, current_time
                )
                
                if verified_id is not None:
                    # ReID验证成功，重置丢失时间
                    self.target_lost_time = None
                    self.get_logger().info(f"目标 {self.current_tracking_id}重新出现，ReID验证成功，继续跟踪")
                    
                    if verified_id in self.tracked_persons:
                        self.tracked_persons[verified_id]['is_tracking'] = True
                        self.tracked_persons[verified_id]['last_seen_time'] = current_time
                else:
                    # ReID验证失败，但目标重新出现 - 关键修改：继续累积丢失时间
                    time_since_lost = current_time - self.target_lost_time

                    # 检查是否超时
                    if time_since_lost > self.lost_timeout_threshold:
                        self.get_logger().warning(
                            f"目标 {self.current_tracking_id} ReID验证失败超过 {self.lost_timeout_threshold} 秒，停止跟踪并清除目标信息，需要重新OK手势选择跟踪目标"
                        )
                        self._clear_tracking_target()
                        return
                    
                self.get_logger().info(f"======================================================")

    def _clear_tracking_target(self):
        """清除当前跟踪目标的所有信息"""
        if self.current_tracking_id is not None:
            target_id = self.current_tracking_id          
            # 清除所有相关存储
            if target_id in self.tracked_persons:
                del self.tracked_persons[target_id]
            if target_id in self.ok_gesture_history:
                del self.ok_gesture_history[target_id]
            if target_id in self.stop_gesture_history:
                del self.stop_gesture_history[target_id]
            if target_id in self.tracked_targets:
                del self.tracked_targets[target_id]
      
        # 重置跟踪状态
        self.current_tracking_id = None
        self.target_lost_time = None
        
    def _cleanup_old_tracks(self, current_time: float, current_track_ids: set):
        """清理长时间未出现的跟踪目标"""
        max_track_age = 5.0
        
        for track_id in list(self.tracked_persons.keys()):
            if track_id == self.current_tracking_id:
                continue
                
            if track_id not in current_track_ids:
                last_seen = self.tracked_persons[track_id]['last_seen_time']
                if current_time - last_seen > max_track_age:
                    self._remove_track(track_id)

    def _remove_track(self, track_id: int):
        """移除跟踪目标"""
        if track_id in self.tracked_persons:
            del self.tracked_persons[track_id]
        if track_id in self.ok_gesture_history:
            del self.ok_gesture_history[track_id]
        if track_id in self.stop_gesture_history:
            del self.stop_gesture_history[track_id]
        if track_id in self.tracked_targets:
            target = self.tracked_targets[track_id]
            if target.is_switched and target.original_track_id not in self.tracked_targets:
                original_id = target.original_track_id
                self.tracked_targets[original_id] = copy.deepcopy(target)
                self.tracked_targets[original_id].track_id = original_id
                self.tracked_targets[original_id].is_switched = False
            
            del self.tracked_targets[track_id]
        
    def _publish_results(self, image: np.ndarray, tracks: List[Dict], header, ok_gestures, stop_gestures):
        """发布结果 - 合并可视化"""
        self.visualize_results(image, tracks, ok_gestures, stop_gestures)
        self.publish_tracked_keypoints(tracks, header)

        # 只在有订阅者时才进行可视化发布
        if self.detect_pose_pub.get_subscription_count() > 0:
            detect_pose_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            detect_pose_msg.header = header
            self.detect_pose_pub.publish(detect_pose_msg)

    def visualize_results(self, image: np.ndarray, tracks: List[Dict], ok_gestures, stop_gestures):
        """简化版可视化跟踪结果 - 优化绘制性能"""
        display_image = image.copy()
        
        # 绘制手势检测结果
        for ok_gesture in ok_gestures:
            x1, y1, x2, y2, score = ok_gesture
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"OK: {score:.2f}"
            cv2.putText(display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for stop_gesture in stop_gestures:
            x1, y1, x2, y2, score = stop_gesture
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"STOP: {score:.2f}"
            cv2.putText(display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['conf']

            # 确定跟踪状态和颜色
            is_tracking = (track_id == self.current_tracking_id and 
                        track_id in self.tracked_persons and 
                        self.tracked_persons[track_id]['is_tracking'])
            
            color = (255, 0, 0) if is_tracking else (0, 255, 0)
            thickness = 3 if is_tracking else 2

            # 绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            label = f"ID:{track_id} {confidence:.2f}"
            if is_tracking:
                label = f"TRACKING {label}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 更新图像
        image[:] = display_image

    def publish_tracked_keypoints(self, tracks: List[Dict], header):
        """发布边界框和肩部关键点坐标和置信度 - 简化优化版"""
        current_tracking_id = self.current_tracking_id
        
        # 查找当前跟踪的目标
        tracking_target = None
        for track in tracks:
            track_id = track['track_id']
            if (current_tracking_id is not None and track_id == current_tracking_id) or \
            (track_id in self.tracked_persons and self.tracked_persons[track_id]['is_tracking']):
                tracking_target = track
                break
        
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = "camera_link"
        
        if tracking_target:
            # 发布正常跟踪状态
            track_id = tracking_target['track_id']
            x1, y1, x2, y2 = tracking_target['bbox']
            
            # 构建消息点：状态信息 + 边界框
            points = [
                Point32(x=float(track_id), y=1.0, z=2.0),  # 状态点
                Point32(x=float(x1), y=float(y1), z=0.0),   # 边界框左上
                Point32(x=float(x2), y=float(y2), z=0.0),   # 边界框右下
            ]
            
            polygon_msg.polygon.points = points
            # self.get_logger().info(f"📤 发布跟踪信息: ID {track_id}")
            
        elif current_tracking_id is not None:
            # 发布目标丢失状态
            points = [
                Point32(x=float(current_tracking_id), y=0.0, z=0.0),  # 状态点：y=0表示丢失
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
            ]
            polygon_msg.polygon.points = points
            self.get_logger().info(f"📤 发布目标丢失状态: ID {current_tracking_id}")
        
        else:
            # 无跟踪目标状态
            points = [Point32(x=0.0, y=0.0, z=0.0) for _ in range(3)]  # 3个零值点
            polygon_msg.polygon.points = points
        
        # 发布消息
        self.keypoint_tracks_pub.publish(polygon_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = Yolov8HandTrackNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"Node initialization failed: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()