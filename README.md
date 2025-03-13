# Space Adaptive CommonScenes

이 프로젝트는 CommonScenes 모델을 확장하여 현실 공간 데이터를 활용한 Space Adaptive 기능을 추가한 버전입니다.

## 개요

Space Adaptive CommonScenes는 기존 CommonScenes 모델에 현실 공간 데이터를 활용하여 가상 공간 생성 시 현실 공간의 레이아웃 특성을 반영할 수 있는 기능을 추가했습니다.
이를 통해 "(가상, FRONT) 나이트 스탠드를 (현실, 스캔 데이터) 침대 왼쪽에" 같은 형태의 조건부 생성이 가능합니다.

## 주요 기능

- **현실 공간 데이터 활용**: 스캔된 현실 공간의 레이아웃 정보를 모델에 통합
- **하이브리드 생성 방식**: 현실 공간과 가상 공간의 특성을 혼합하여 새로운 공간 생성
- **조건부 객체 배치**: 현실 공간의 객체를 기준으로 가상 객체 배치 가능

## 사용 방법

### 학습

현실 공간 데이터를 활용하여 모델을 학습하려면 다음 명령어를 사용합니다:

```bash
python scripts/train_3dfront_with_real.py --exp experiments/space_adaptive --real_space_data spacedata/real_space_sample.json --use_real_space True --real_space_weight 0.3
```

### 평가

현실 공간 데이터를 활용하여 모델을 평가하려면 다음 명령어를 사용합니다:

```bash
python scripts/eval_3dfront_with_real.py --exp experiments/space_adaptive --epoch 45 --real_space_data spacedata/real_space_sample.json --use_real_space True --real_space_weight 0.3
```

## 데이터 형식

현실 공간 데이터는 다음과 같은 형식으로 제공되어야 합니다:

```json
{
  "scans": [
    {
      "scan": "RealSpace-ID",
      "objects": {
        "1": "object_class_1",
        "2": "object_class_2",
        ...
      },
      "relationships": [
        [subject_id, object_id, relationship_id, "relationship_name"],
        ...
      ]
    }
  ],
  "RealSpace-ID": {
    "1": {
      "param7": [width, height, depth, x, y, z, rotation],
      "8points": [...],
      "scale": [1, 1, 1],
      "model_path": null
    },
    ...
    "scene_center": [x, y, z]
  }
}
```

## 구현 세부 사항

- **임베딩 생성**: 현실 공간 데이터를 VAE 인코더를 통해 잠재 공간으로 임베딩
- **조건화 메커니즘**: 가상 공간 생성 시 현실 공간 임베딩으로 조건화
- **하이브리드 손실 함수**: 가상 공간과 현실 공간 특성을 모두 고려한 손실 함수