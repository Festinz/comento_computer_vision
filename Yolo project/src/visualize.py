"""
Week3: 모델 평가 및 결과 시각화
업무 안내서 [수행 방법] 4번 기반

실행: python visualize.py
"""

import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================
# 4-1. 모델 평가 방법
# ============================================

# 학습된 모델 로드
model = YOLO("runs/train/exp/weights/best.pt")

# 모델 평가 결과
metrics = model.val()
print(metrics)

# 이 코드 실행 시:
# 모델의 정확도(AP, Recall, Precision) 출력

# ============================================
# 4-2. Matplotlib을 활용한 성능 평가 시각화
# ============================================

# Precision, Recall 그래프 출력
plt.plot(metrics['precision'], label="Precision")
plt.plot(metrics['recall'], label="Recall")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.title("Model Performance")
plt.savefig("../results/model_performance.png", dpi=150, bbox_inches='tight')
plt.show()

print("시각화 결과 저장 완료: ../results/model_performance.png")
