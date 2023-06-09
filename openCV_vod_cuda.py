import cv2

# 打开视频文件
video_path = 'media/vod.mp4'  # 替换为实际的视频路径
cap = cv2.VideoCapture(video_path)
faceCascade = cv2.cuda_CascadeClassifier('haarcascade_frontalface_alt2.xml')

# 创建 CUDA 上下文
cv2.cuda.setDevice(0)  # 设置要使用的 CUDA 设备的索引
ctx = cv2.cuda.DeviceInfo()

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将帧数据上传到 CUDA 上下文
    frame_gpu = cv2.cuda_GpuMat()
    frame_gpu.upload(frame)
    
    # 在 CUDA 上进行显示
    re =  cv2.cuda.resize(frame_gpu, (852, 480))  # 调整帧的大小
    gr = cv2.cuda.cvtColor(re, cv2.COLOR_BGR2GRAY)  # 将帧转换为灰度图像
    faceRect = faceCascade.detectMultiScale(gr, 1.1, 2)
    result = faceRect.download() 

    for (x, y, w, h) in result:
        cv2.rectangle(re, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 将处理后的帧从 CUDA 转回 CPU
    #result_frame = frame_gpu.download()
    result_gr = re.download()
    
    # 显示帧
    cv2.imshow('CUDA Video Playback gr', result_gr)
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象和关闭窗口
cap.release()
cv2.destroyAllWindows()
