import torch
import cv2
import os
import json

yolov5_repo_path = "yolov5"

# 커스텀 모델 가중치(weights) 파일의 로컬 경로
custom_model_path = "/home/intel/embedded/d020rs-yolov5-rock-paper-scissors-Mumbed/src/yolov5/runs/train/rps_yolov5m_xpu4/weights/best1.pt"

# 1. XPU (Intel GPU) 설정: 사용 가능하면 XPU, 아니면 CPU 사용
device = 'xpu' if torch.xpu.is_available() else 'cpu'
print(f"Using device: {device}")

model = torch.hub.load(yolov5_repo_path, 'custom', path=custom_model_path, source='local')
model.to(device)

# Video capture
cap = cv2.VideoCapture(0)
input_size = 640

while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # break the loop on error
    if not ret:
        break

    # TODO: 카메라 입력의 크기(frame_h, frame_w)와 모델의 입력 크기(input_h, input_w) 구하기
    frame_h, frame_w, _ = frame.shape

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # TODO: 추론 전 입력 크기 보정 (640x640)
    resized_frame = cv2.resize(rgb_frame, (input_size, input_size))
    results = model(resized_frame)

    # Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # 인식결과를 표시하기 위한 좌표를 얻음
        x1, y1, x2, y2, conf, cls_idx = obj
        
        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        label = f"{model.names[int(cls_idx)]} {conf:.2f}"
        
        # TODO: 출력 바운딩박스 크기 조절
        orig_x1 = int(x1 * (frame_w / input_size))
        orig_y1 = int(y1 * (frame_h / input_size))
        orig_x2 = int(x2 * (frame_w / input_size))
        orig_y2 = int(y2 * (frame_h / input_size))

        # OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (orig_x1, orig_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # 화면 표시
    cv2.imshow("YOLOv5", frame)

    # 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()