import json
import os
import re

# 파일 경로 설정
input_file_lounge = 'Lounge_relationships_preprocessed.txt'
lounge_bounding_boxes_file = 'Lounge_bounding_boxes.json'
input_file_studio = 'Studio_relationships_preprocessed.txt'
studio_bounding_boxes_file = 'Studio_bounding_boxes.json'

output_file = 'relationships_spacedata.json'
flat_output_file = 'relationships_spacedata_flat.json'
relationships_file = '../SG-FRONT/relationships.txt'

# 현재 스크립트 경로 기준으로 상대 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path_lounge = os.path.join(current_dir, input_file_lounge)
bounding_boxes_path_lounge = os.path.join(current_dir, lounge_bounding_boxes_file)

output_path = os.path.join(current_dir, output_file)
flat_output_path = os.path.join(current_dir, flat_output_file)
relationships_path = os.path.join(current_dir, relationships_file)

# 관계 유형 로드
relationship_types = []
with open(relationships_path, 'r') as f:
    relationship_types = [line.strip() for line in f.readlines()]

# 관계 유형을 인덱스로 매핑 (1부터 시작)
relationship_to_idx = {rel: idx+1 for idx, rel in enumerate(relationship_types)}
print(relationship_to_idx)

# 바운딩 박스 정보 로드
with open(bounding_boxes_path_lounge, 'r') as f:
    bounding_boxes = json.load(f)

# 객체 정보 추출 (Lounge_bounding_boxes.json에서)
objects = {}
for scan_name, scan_data in bounding_boxes.items():
    for obj_id, obj_info in scan_data.items():
        # "object" 필드에서 객체 유형 추출
        # 예: "15_dining_table: 1" -> "15_dining_table"
        object_type = obj_info["object"].split(":")[0].strip()
        # 숫자와 언더스코어 제거하여 깔끔한 객체 이름만 남기기
        clean_object_type = re.sub(r'^\d+_', '', object_type)
        objects[obj_id] = clean_object_type
print(objects)

# 관계 데이터 로드 및 파싱
relationships = []
with open(input_path_lounge, 'r') as f:
    for line in f:
        line = line.strip()
        
        # 개선된 파싱 로직: 공백을 포함한 관계 유형 처리
        # 형식: "1 front 4", "1 same super category as 3" 등
        # 패턴: 숫자 [관계] 숫자
        pattern = r"^(\d+)\s+(.*?)\s+(\d+)$"
        match = re.match(pattern, line)
        
        if match:
            subj_id, rel_type, obj_id = match.groups()
            
            # 관계 유형을 인덱스로 변환
            rel_idx = relationship_to_idx.get(rel_type)
            if rel_idx:
                # [주체ID, 대상ID, 관계타입ID, 관계설명]
                relationships.append([int(subj_id), int(obj_id), rel_idx, rel_type])
            else:
                print(f"경고: 알 수 없는 관계 유형 '{rel_type}'")

# 결과 JSON 생성
result = {
    "scans": [
        {
            "scan": "Lounge",
            "objects": objects,
            "relationships": relationships
        }
    ]
}

# 일반 형식으로 JSON 파일 저장 (인덴트 있음)
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

# 평평한 형식의 JSON 파일 생성 (relationships 리스트를 인덴트 없이 처리)
# 이를 위해 수동으로 JSON 형식을 구성
with open(flat_output_path, 'w') as f:
    # 파일 시작 부분 작성
    f.write('{\n  "scans": [\n    {\n')
    f.write('      "scan": "Lounge",\n')
    
    # 객체 정보 작성
    f.write('      "objects": {\n')
    obj_entries = []
    for obj_id, obj_name in objects.items():
        obj_entries.append(f'        "{obj_id}": "{obj_name}"')
    f.write(',\n'.join(obj_entries))
    f.write('\n      },\n')
    
    # relationships 리스트 시작
    f.write('      "relationships": [')
    
    # 각 관계를 한 줄에 작성
    rel_entries = []
    for rel in relationships:
        rel_json = json.dumps(rel)  # 리스트를 JSON 문자열로 변환
        rel_entries.append(rel_json)
    
    # 관계 항목들을 쉼표로 구분하여 한 줄씩 작성
    f.write('\n        ')
    f.write(',\n        '.join(rel_entries))
    
    # relationships 리스트 종료 및 파일 종료
    f.write('\n      ]\n    }\n  ]\n}')

print(f"파일이 성공적으로 생성되었습니다: {output_path}")
print(f"인덴트 없는 relationships 버전이 생성되었습니다: {flat_output_path}")
print(f"총 {len(relationships)}개의 관계가 처리되었습니다.") 