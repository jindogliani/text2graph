from __future__ import print_function
import sys
sys.path.append("..")
sys.path.append(".")
import torch
import numpy as np
import json
import os
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph

# 텐서를 기본 Python 자료형으로 변환하는 함수
def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(item) for item in obj]
    else:
        return obj

# 데이터셋 로드
dataset = ThreedFrontDatasetSceneGraph(
    root="/mnt/dataset/FRONT/",
    split='val_scans',
    shuffle_objs=True,
    use_SDF=False,
    use_scene_rels=True,
    with_changes=False,
    with_feats=False,
    with_CLIP=True,
    large=True,
    seed=False,
    room_type='all',
    recompute_clip=False)

# 첫 번째 항목 가져오기
a = dataset[0]
print(f"Scan ID: {a['scan_id']}")
print(f"객체 개수: {len(a['encoder']['objs'])}")

# JSON으로 변환할 수 있는 형태로 변환
json_data = tensor_to_list(a)

# 저장 경로 확인 및 생성
os.makedirs("spacedata", exist_ok=True)
output_path = "spacedata/sg-front_sample.json"

# JSON 파일로 저장
with open(output_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"데이터가 {output_path}에 저장되었습니다.") 