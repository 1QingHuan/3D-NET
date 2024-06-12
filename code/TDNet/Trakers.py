import numpy as np
from numpy.linalg import inv
from filterpy.kalman import KalmanFilter

np.random.seed(0)

class KalmanBoxTracker(object):
  """
  这个类表示观察到的单个跟踪对象的内部状态，以边界框的形式表示。
  """
  count = 0  # 用于跟踪创建的对象数量

  def __init__(self, bbox):
    """
    使用初始边界框初始化一个跟踪器。
    """
    # 定义常速度模型
    self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 创建一个7维状态，4维测量的卡尔曼滤波器
    dt = 1/30  # 时间步长
    # 状态转移矩阵
    self.kf.F = np.array([[1, 0, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]])

    # 测量矩阵
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]])

    # 测量噪声协方差矩阵
    self.kf.R = np.array([[0.1, 0, 0, 0],
                          [0, 0.1, 0, 0],
                          [0, 0, 40, 0],
                          [0, 0, 0, 40]])

    # 初始状态协方差矩阵
    self.kf.P = np.eye(7) * 10

    # 过程噪声协方差矩阵
    self.kf.Q = np.array([[1, 0, 0, 0, 0, 0, 0],  # x
                          [0, 1, 0, 0, 0, 0, 0],  # y
                          [0, 0, 1, 0, 0, 0, 0],  # s
                          [0, 0, 0, 1, 0, 0, 0],  # r
                          [0, 0, 0, 0, 1, 0, 0],  # vx
                          [0, 0, 0, 0, 0, 1, 0],  # vw
                          [0, 0, 0, 0, 0, 0, 1]]) * 0.1  # vs

    # 将初始边界框转换为状态向量
    self.kf.x[:4] = convert_bbox_to_z(bbox)

    # 其他属性初始化
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox):
    """
    使用观察到的边界框更新状态向量。
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    推进状态向量，并返回预测的边界框估计值。
    """
    if (self.kf.x[6] + self.kf.x[2]) <= 0:
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if self.time_since_update > 0:
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    返回当前的边界框估计值。
    """
    return convert_x_to_bbox(self.kf.x)

class mKalmanFilter:
    def __init__(self, x, y, vx, vy, ax, ay, dt=1/9):
        hd2t = (dt**2) * 0.5  # 计算半个时间步长的平方

        # 初始化状态向量，包括位置、速度和加速度
        self.X = np.array([[float(x)], [float(y)], [float(vx)], [float(vy)], [float(ax)], [float(ay)]])

        # 初始化状态协方差矩阵，假设初始不确定性较大
        self.P = np.eye(self.X.shape[0]) * 100

        # 定义状态转移矩阵，基于常加速度模型
        self.F = np.array([[1, 0, dt, 0, hd2t, 0],
                           [0, 1, 0, dt, 0, hd2t],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]], np.float32)

        # 定义过程噪声协方差矩阵，控制模型的不确定性
        self.Q = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0.1, 0, 0, 0],
                           [0, 0, 0, 0.1, 0, 0],
                           [0, 0, 0, 0, 0.5, 0],
                           [0, 0, 0, 0, 0, 0.5]], np.float32) * 0.01

        # 初始化观测向量
        self.Z = np.zeros((6, 1), np.float32)

        # 定义观测矩阵，只观测位置
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], np.float32)

        # 定义观测噪声协方差矩阵
        self.R = np.eye(2) * 6

    def predict(self):
        # 预测下一状态
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X

    def correct(self, Z):
        # 根据观测值更新状态
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)  # 计算卡尔曼增益
        self.X += K @ (Z - self.H @ self.X)  # 更新状态向量
        self.P = self.P - K @ self.H @ self.P  # 更新状态协方差矩阵
        return self.X
    
class KF_3D:
    def __init__(self, x, y, vx, vy):
        # 初始化状态向量，包括位置和速度
        self.X = np.array([[float(x)], [float(y)], [float(vx)], [float(vy)]])

        # 初始化状态协方差矩阵，假设初始不确定性较小
        self.P = np.eye(self.X.shape[0]) * 10

        # 定义状态转移矩阵，基于常速度模型
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], np.float32)

        # 定义过程噪声协方差矩阵，控制模型的不确定性
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0.9, 0],
                           [0, 0, 0, 0.9]], np.float32) * 0.1

        # 初始化观测向量
        self.Z = np.zeros((4, 1), np.float32)

        # 定义观测矩阵，只观测位置
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], np.float32)

        # 定义观测噪声协方差矩阵
        self.R = np.eye(2) * 100

    def predict(self):
        # 预测下一状态
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X

    def correct(self, Z):
        # 根据观测值更新状态
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)  # 计算卡尔曼增益
        self.X += K @ (Z - self.H @ self.X)  # 更新状态向量
        self.P = self.P - K @ self.H @ self.P  # 更新状态协方差矩阵
        return self.X

class KalmanTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox, R= 500., P=10., Q=0.01):
      """
      Initialises a tracker using initial bounding box.
      """
      #define constant velocity model
      self.kf = KalmanFilter(dim_x=4, dim_z=2) 
      self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
      self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

      self.kf.R[0:,0:] *= R
      self.kf.P[2:,2:] *= P #give high uncertainty to the unobservable initial velocities
      self.kf.P *= 10.
      self.kf.Q[-1,-1] *= Q
      self.kf.Q[2:,2:] *= Q

      self.kf.x[:2] = bbox
      self.time_since_update = 0
      self.id = KalmanTracker.count
      KalmanTracker.count += 1
      self.history = []
      self.hits = 0
      self.hit_streak = 0
      self.age = 0

    def update(self,bbox):
      """
      Updates the state vector with observed bbox.
      """
      self.time_since_update = 0
      self.history = []
      self.hits += 1
      self.hit_streak += 1
      self.kf.update(bbox)

    def predict(self):
      """
      Advances the state vector and returns the predicted bounding box estimate.
      """
      if((self.kf.x[2]+self.kf.x[0])<=0):
        self.kf.x[2] *= 0.0
      self.kf.predict()
      self.age += 1
      if(self.time_since_update>0):
        self.hit_streak = 0
      self.time_since_update += 1
      self.history.append(self.kf.x)
      return self.history[-1]

    def get_state(self):
      """
      Returns the current bounding box estimate.
      """
      return self.kf.x

class SORT(object):
  """
      SORT: A Simple, Online and Realtime Tracker
      Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.
      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.
      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>.
  """
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 12))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t][0].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]][0].update(dets[m[0], :5])
      self.trackers[m[1]][1] = dets[m[0], 5:]
      

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:5])
        self.trackers.append([trk, dets[i,5:]])
        
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk[0].get_state()[0]
        d2 = trk[1]
        if (trk[0].time_since_update < 1) and (trk[0].hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk[0].id+1],d2)).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk[0].time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)

    return np.empty((0,12))

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  
  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):                                                                                                   
  """                                                                                                                      
  From SORT: Computes IUO between two bboxes in the form [l,t,w,h]                                                         
  """                                                                                                                      
  bb_gt = np.expand_dims(bb_gt, 0)                                                                                         
  bb_test = np.expand_dims(bb_test, 1)                                                                                     
                                                                                                                           
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])                                                                         
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])                                                                         
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])                                                                         
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])                                                                         
  w = np.maximum(0., xx2 - xx1)                                                                                            
  h = np.maximum(0., yy2 - yy1)                                                                                            
  wh = w * h                                                                                                               
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
