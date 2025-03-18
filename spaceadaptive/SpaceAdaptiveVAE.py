import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import DBSCAN
import sys
sys.path.append('../')
try:
    from model.VAE_spaceadaptive import VAE
except ImportError:
    print("VAE 모듈을 불러올 수 없습니다. 테스트 모드로 실행합니다.")
    VAE = None

class SpaceAdaptiveVAE:
    """
    현실 공간과 가상 씬 데이터 간의 상호작용을 관리하는 클래스
    
    이 클래스는 다음 기능을 제공합니다:
    1. 현실 공간 임베딩 기반 유사 가상 씬 식별
    2. 현실 공간의 구조적 특성 분석
    3. 유사 가상 씬들의 조각을 조합한 하이브리드 씬 생성
    4. 하이브리드 씬 기반 추론 및 평가
    """
    
    def __init__(self, vae_model=VAE, config=None):
        """
        SpaceAdaptiveVAE 클래스 초기화
        
        Args:
            vae_model: 기존의 VAE_spaceadaptive 모델 인스턴스
            config: 부가 설정
        """
        self.vae = VAE
        self.config = config or {}
        self.real_space_embedding = None
        self.similar_scenes = None
        self.hybrid_scene = None
        self.space_data = None
        self.space_id = "Lounge"
        self.room_type = "all"
        self.space_data_path = "/home/commonscenes/spaceadaptive/spacedata/spacedata.json"
        
    def load_real_space_data(self, space_data_path):
        with open(space_data_path, 'r') as f:
            space_data = json.load(f)
        
        self.space_data = space_data
        # VAE 모델에도 데이터 설정
        self.vae.space_data = space_data
        print(f"현실 공간 데이터 로드 완료: {len(space_data)} 공간")
        return space_data
        
    def space_data_processing(self, space_data, room_type='all', space_id='Lounge'):
        """
        현실 공간 데이터를 SG-FRONT 형식으로 처리하여 텐서로 변환
        
        Args:
            room_type: 방 유형 (bedroom, living, all 등)
            space_data: 처리할 공간 데이터
            space_id: 처리할 특정 공간 ID (예: 'Lounge', 'Studio')
            
        Returns:
            tuple: (objs_tensor, boxes_tensor, triples_tensor) 형태의 텐서 튜플
        """
        # 1. classes_*.txt 파일에서 클래스 목록 로드
        classes_file = f'/mnt/dataset/FRONT/classes_{room_type}.txt'
        object_idx_to_name = []
        object_name_to_idx = {}
        
        with open(classes_file, "r") as f:
            for idx, line in enumerate(f.readlines()):
                class_name = line.strip()
                object_idx_to_name.append(class_name)
                object_name_to_idx[class_name] = idx
        
        # 2. 대상 스캔 찾기
        target_scan = None
        for scan in space_data["scans"]:
            if scan["scan"] == space_id:
                target_scan = scan
                break
        
        if target_scan is None:
            print(f"경고: {space_id} ID를 가진 스캔을 찾을 수 없습니다.")
            if len(space_data["scans"]) > 0:
                print(f"대신 첫 번째 스캔({space_data['scans'][0]['scan']})을 사용합니다.")
                target_scan = space_data["scans"][0]
            else:
                raise ValueError("처리할 스캔 데이터가 없습니다.")
        
        # 객체 클래스, 바운딩 박스, 관계 추출
        objs = []
        boxes = []
        obj_id_to_idx = {}
        
        # 3. 객체 정보 처리
        for idx, (obj_id, obj_class) in enumerate(target_scan["objects"].items()):
            obj_id_to_idx[obj_id] = idx
            
            # 클래스 이름이 매핑에 있는지 확인
            if obj_class in object_name_to_idx:
                class_idx = object_name_to_idx[obj_class]
                objs.append(class_idx)
                
                # 바운딩 박스 처리
                if "bbox" in target_scan and obj_id in target_scan["bbox"]:
                    box = target_scan["bbox"][obj_id]["param7"]
                    # SG-FRONT 데이터셋도 바운딩박스를 param7으로 하는가? => 그렇다.
                    # scene center가 다 달라서 바운딩박스의 절대좌표가 다 다를 텐데 이를 어떻게 처리해야 하는가? => 데이터셋 로드할 때 정규화
                    # => 현실공간 데이터는 이미 scene center가 (0,0,0)이라 정규화 필요 없음.
                    # 회전 각도 양자화 (24개의 이산 각도 값으로 변환)
                    bins = np.linspace(np.deg2rad(-180), np.deg2rad(180), 24)
                    angle_index = np.digitize(box[6], bins)
                    box[6] = angle_index
                    
                    boxes.append(box)
                else:
                    boxes.append([0, 0, 0, 0, 0, 0, 0])
            else:
                print(f"경고: {obj_class} 클래스가 매핑에 없습니다: {classes_file}")
        
        # 4. scene 객체 추가
        scene_class_idx = object_name_to_idx["_scene_"]
        objs.append(scene_class_idx)
        boxes.append([-1, -1, -1, -1, -1, -1, -1])
        
        # 5. 관계 정보 처리
        triples = []
        
        # 관계 유형(predicate) 매핑 로드
        rel_file = '/mnt/dataset/FRONT/relationships.txt'
        pred_idx_to_name = ['in\n']  # 0번 관계는 'in'
        
        with open(rel_file, "r") as f:
            pred_idx_to_name += f.readlines()
        
        pred_name_to_idx = {name.strip(): idx for idx, name in enumerate(pred_idx_to_name)}
        
        # 관계 정보 처리
        for rel in target_scan["relationships"]:
            subj_id = str(rel[0])
            obj_id = str(rel[1])
            rel_name = rel[3]  # 관계 이름(텍스트)
            
            if rel_name in pred_name_to_idx:
                rel_idx = pred_name_to_idx[rel_name]
                
                if subj_id in obj_id_to_idx and obj_id in obj_id_to_idx:
                    subj_idx = obj_id_to_idx[subj_id]
                    obj_idx = obj_id_to_idx[obj_id]
                    triples.append([subj_idx, rel_idx, obj_idx])
            else:
                print(f"경고: {rel_name} 관계가 매핑에 없습니다")
        
        # 6. scene과의 관계 추가
        scene_idx = len(objs) - 1
        for i in range(len(objs) - 1):
            triples.append([i, 0, scene_idx])  # 0은 'in' 관계
        
        # 텐서 변환
        objs_tensor = torch.tensor(objs, dtype=torch.long)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float)
        triples_tensor = torch.tensor(triples, dtype=torch.long)
        
        return objs_tensor, boxes_tensor, triples_tensor
    
    def encode_real_space(self, room_type='all', space_id=None):
        """
        현실 공간 데이터를 VAE 모델로 인코딩하여 임베딩 생성
        
        Args:
            room_type: 방 유형 ('bedroom', 'living', 'all' 등)
            space_id: 특정 공간 ID (None이면 모든 공간 처리)
            
        Returns:
            embeddings: 현실 공간 임베딩 딕셔너리 {space_id: embedding}
        """
        # 공간 데이터가 로드되지 않았다면 오류 발생
        if self.space_data is None:
            print("현실 공간 데이터가 로드되지 않았습니다. load_real_space_data()를 먼저 호출하세요.")
            return None
        
        # 모델이 초기화되지 않았다면 오류 발생
        if not isinstance(self.vae, nn.Module):
            print("VAE 모델이 초기화되지 않았습니다.")
            return None
            
        # 처리할 공간 ID 결정
        space_ids = []
        if space_id is not None:
            # 특정 공간 ID만 처리
            space_ids = [space_id]
        else:
            # 모든 공간 처리
            space_ids = list(self.space_data["scans"].keys())
            
        # 각 공간별 임베딩 생성
        embeddings = {}
        for current_space_id in space_ids:
            print(f"현실 공간 인코딩 중: {current_space_id}")
            
            # 데이터 처리 및 텐서 변환 (room_type은 필요에 따라 설정)
            room_type = 'all'  # 또는 공간별 room_type 매핑 사용
            objs_tensor, boxes_tensor, triples_tensor = self.space_data_processing(
                space_data=self.space_data,
                room_type=room_type, 
                space_id=current_space_id
            )
            
            # VAE 모델에 텐서를 전달하여 인코딩
            # VAE_spaceadaptive.py 모델 내 encode_real_space_tensors() 참고
            embedding = self.vae.encode_real_space_tensors(objs_tensor, boxes_tensor, triples_tensor)
            embeddings[current_space_id] = embedding
        
        # 임베딩을 VAE 모델의 멤버 변수에만 저장
        self.vae.real_space_embeddings = embeddings
        
        # SpaceAdaptiveVAE 클래스에서는 참조만 유지
        self.real_space_embedding = self.vae.real_space_embeddings
        
        print(f"현실 공간 인코딩 완료")
        return self.vae.real_space_embeddings
    
    def set_cuda(self):
        """
        모델과 임베딩을 CUDA 장치로 이동시키는 메서드
        """
        # VAE 모델 CUDA 설정
        self.vae.set_cuda()
        
        # 임베딩은 VAE 모델 내에서 관리되므로 여기서는 추가 작업 불필요
        # real_space_embedding은 vae.real_space_embeddings의 참조임
    
    def train_with_real_space(self, space_data_path, train_dataloader, epochs=10, room_type='all', space_id=None):
        """
        현실 공간 데이터를 활용하여 VAE 모델 학습
        
        Args:
            space_data_path: 현실 공간 데이터 파일 경로
            train_dataloader: 학습용 데이터로더
            epochs: 학습 에폭 수
            room_type: 방 유형 ('bedroom', 'living', 'all' 등)
            space_id: 특정 공간 ID (None이면 모든 공간 처리)
            
        Returns:
            real_space_embedding: 학습 후 현실 공간 임베딩
        """
        # 1. 모델이 아직 현실 공간 데이터를 로드하지 않았다면 로드
        if self.space_data is None:
            self.load_real_space_data(space_data_path)
        
        # 2. 현실 공간 임베딩 생성 (아직 생성되지 않은 경우)
        if self.vae.real_space_embeddings is None:
            self.encode_real_space(room_type, space_id)
        
        # 3. 일반적인 학습 과정은 train_spaceadaptive.py에서 처리
        # 여기서는 현실 공간 임베딩 값만 반환
        
        return self.vae.real_space_embeddings
    
    def encode_virtual_scene(self, scene_data):
        """
        가상 씬 데이터를 인코딩하여 임베딩 벡터 생성
        identify_similar_virtual_scenes()에서 가상씬 임베딩과 현실공간 임베딩을 비교하기 위함.
        encode_real_space()이랑 거의 유사한 기능 제공
        
        Args:
            scene_data: 가상 씬 데이터 (객체, 관계 등)
            
        Returns:
            scene_embedding: 가상 씬의 임베딩 벡터
        """
        objs = scene_data['objs']
        boxes = scene_data['boxes']
        triples = scene_data['triples']
        
        # 이미 텐서인지 확인하고 변환
        objs_tensor = objs if isinstance(objs, torch.Tensor) else torch.tensor(objs, dtype=torch.long)
        boxes_tensor = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float)
        triples_tensor = triples if isinstance(triples, torch.Tensor) else torch.tensor(triples, dtype=torch.long)
        
        # 한 번에 CUDA로 이동
        objs_tensor = objs_tensor.cuda()
        boxes_tensor = boxes_tensor.cuda()
        triples_tensor = triples_tensor.cuda()
        
        # 모델 타입에 따른 인코딩
        if self.vae.type_ == 'v1_box' or self.vae.type_ == 'v2_box':
            mu, _ = self.vae.vae_box.encoder(objs_tensor, triples_tensor, boxes_tensor, None)
        elif self.vae.type_ == 'v1_full':
            mu, _ = self.vae.vae.encoder(objs_tensor, triples_tensor, boxes_tensor, None)
        elif self.vae.type_ == 'v2_full':
            if self.vae.vae_v2.clip:
                dummy_text_feats = torch.zeros((len(objs), 512)).cuda()
                dummy_rel_feats = torch.zeros((len(triples), 512)).cuda()
                mu, _ = self.vae.vae_v2.encoder(objs_tensor, triples_tensor, boxes_tensor, 
                                              None, dummy_text_feats, dummy_rel_feats)
            else:
                mu, _ = self.vae.vae_v2.encoder(objs_tensor, triples_tensor, boxes_tensor, None)
                
        return mu
    
    def calculate_embedding_similarity(self, embedding1, embedding2):
        """
        두 임베딩 벡터 간의 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩 벡터
            embedding2: 두 번째 임베딩 벡터
            
        Returns:
            similarity: 두 임베딩 간의 유사도 (0~1 범위)
        """
        # 코사인 유사도 계산
        cos_sim = F.cosine_similarity(embedding1, embedding2, dim=0)
        
        # 유클리드 거리 계산 (정규화된 형태로)
        euclidean_dist = torch.norm(embedding1 - embedding2, p=2)
        max_dist = torch.sqrt(torch.tensor(embedding1.shape[0] * 2.0).float().cuda())  # 최대 가능한 거리
        euclidean_sim = 1.0 - (euclidean_dist / max_dist)
        
        # 두 유사도의 가중 평균
        similarity = 0.7 * cos_sim + 0.3 * euclidean_sim # TODO 가중치 값 정해지지 않음. 이렇게 임의로 설정 가능?
        
        return similarity.item()
    
    def identify_similar_virtual_scenes(self, virtual_scenes_dataset, top_k=10):
        """
        현실 공간과 유사한 가상 씬들을 식별
        
        Args:
            virtual_scenes_dataset: 가상 씬 데이터셋
            top_k: 반환할 상위 유사 씬 개수
            
        Returns:
            similar_scenes: (씬, 유사도) 튜플의 리스트
        """
        if self.real_space_embedding is None:
            raise ValueError("현실 공간 임베딩이 없습니다. train_with_real_space를 먼저 호출하세요.")
        similar_scenes = []
        
        # 각 가상 씬에 대해 유사도 계산
        for scene_id, scene_data in virtual_scenes_dataset.items():
            scene_embedding = self.encode_virtual_scene(scene_data)
            
            # 현실 공간 임베딩과의 유사도 계산
            # 현실 공간이 여러 개면 가장 유사한 것의 유사도 사용
            if isinstance(self.real_space_embedding, dict):
                max_similarity = 0
                for space_id, space_emb in self.real_space_embedding.items():
                    similarity = self.calculate_embedding_similarity(space_emb, scene_embedding)
                    max_similarity = max(max_similarity, similarity)
                similarity = max_similarity
            else:
                similarity = self.calculate_embedding_similarity(self.real_space_embedding[self.space_id], scene_embedding)
                
            similar_scenes.append((scene_id, scene_data, similarity))
        
        # 유사도 기준 정렬 및 상위 k개 반환
        similar_scenes.sort(key=lambda x: x[2], reverse=True)
        self.similar_scenes = similar_scenes[:top_k]
        
        return self.similar_scenes
    
    def analyze_space_regions(self, space_data):
        """
        공간을 의미 있는 영역으로 분할
        
        Args:
            space_data: 분석할 공간 데이터
            
        Returns:
            regions: 식별된 영역들의 딕셔너리
        """
        # 1. 객체 위치 데이터 추출
        obj_positions = []
        obj_info = []
        
        for obj_id, obj in space_data["objects"].items():
            if "bbox" in obj:
                # 바운딩 박스의 중심 위치 계산
                center_x = obj["bbox"][0]
                center_y = obj["bbox"][1]
                center_z = obj["bbox"][2]
                obj_positions.append([center_x, center_y, center_z])
                obj_info.append({
                    "id": obj_id,
                    "class": obj["class"],
                    "bbox": obj["bbox"]
                })
        
        if not obj_positions:
            return {"entire_space": space_data}
        
        # 2. 밀도 기반 클러스터링으로 영역 식별
        positions_array = np.array(obj_positions)
        clustering = DBSCAN(eps=1.5, min_samples=2).fit(positions_array)
        labels = clustering.labels_
        
        # 3. 클러스터별로 객체 그룹화
        regions = defaultdict(lambda: {"objects": {}, "relationships": []})
        
        for i, label in enumerate(labels):
            region_name = f"region_{label}" if label >= 0 else "outliers"
            obj_id = obj_info[i]["id"]
            regions[region_name]["objects"][obj_id] = obj_info[i]
        
        # 4. 관계 정보 분배
        for rel in space_data.get("relationships", []):
            subj_id = str(rel["subject"])
            obj_id = str(rel["object"])
            
            # 두 객체가 같은 영역에 있는 관계만 해당 영역에 할당
            for region_name, region_data in regions.items():
                if subj_id in region_data["objects"] and obj_id in region_data["objects"]:
                    regions[region_name]["relationships"].append(rel)
        
        return dict(regions)
    
    def split_scene_into_fragments(self, scene_data):
        """
        가상 씬을 작은 조각(프래그먼트)으로 분할
        
        Args:
            scene_data: 분할할 가상 씬 데이터
            
        Returns:
            fragments: 씬 조각들의 리스트
        """
        # 공간 영역 분석 함수를 활용하여 씬을 조각으로 분할
        regions = self.analyze_space_regions(scene_data)
        
        # 각 영역을 독립적인 조각으로 변환
        fragments = []
        for region_name, region_data in regions.items():
            if len(region_data["objects"]) >= 2:  # 최소 2개 이상의 객체가 있는 영역만 고려
                fragments.append(region_data)
        
        return fragments
    
    def find_best_matching_fragment(self, target_region, similar_scenes):
        """
        대상 영역과 가장 잘 맞는 가상 씬 조각 탐색
        
        Args:
            target_region: 맞출 대상 영역 데이터
            similar_scenes: 유사 가상 씬 리스트
            
        Returns:
            best_fragment: 가장 적합한 씬 조각
        """
        best_fragment = None
        best_similarity = 0
        
        # 각 유사 씬에서 조각 추출 및 유사도 계산
        for _, scene_data, overall_similarity in similar_scenes:
            fragments = self.split_scene_into_fragments(scene_data)
            
            for fragment in fragments:
                # 대상 영역과 조각 간의 유사도 계산
                region_embedding = self.encode_virtual_scene(target_region)
                fragment_embedding = self.encode_virtual_scene(fragment)
                
                fragment_similarity = self.calculate_embedding_similarity(region_embedding, fragment_embedding)
                
                # 기존 씬의 전체 유사도도 고려하여 가중치 부여
                weighted_similarity = 0.7 * fragment_similarity + 0.3 * overall_similarity
                
                if weighted_similarity > best_similarity:
                    best_similarity = weighted_similarity
                    best_fragment = fragment
        
        return best_fragment
    
    def integrate_fragment(self, hybrid_scene, fragment, target_region):
        """
        가상 씬 조각을 하이브리드 씬에 통합
        
        Args:
            hybrid_scene: 현재 구성 중인 하이브리드 씬
            fragment: 통합할 씬 조각
            target_region: 대상 영역 정보
            
        Returns:
            updated_hybrid_scene: 업데이트된 하이브리드 씬
        """
        if fragment is None:
            return hybrid_scene
        
        # 하이브리드 씬이 비어있으면 초기화
        if not hybrid_scene:
            hybrid_scene = {"objects": {}, "relationships": []}
        
        # 객체 ID 오프셋 계산 (기존 ID와 충돌 방지)
        max_id = 0
        for obj_id in hybrid_scene["objects"]:
            try:
                max_id = max(max_id, int(obj_id))
            except ValueError:
                pass
        id_offset = max_id + 1
        
        # ID 매핑 테이블 생성
        id_mapping = {}
        
        # 1. 객체 통합
        for obj_id, obj_info in fragment["objects"].items():
            new_id = str(id_offset + int(obj_id))
            id_mapping[obj_id] = new_id
            
            # 객체 위치 조정 (대상 영역 중심 기준)
            adjusted_obj = obj_info.copy()
            if "bbox" in obj_info:
                # 대상 영역의 중심 계산
                target_centers = [obj["bbox"][:3] for obj in target_region["objects"].values()]
                if target_centers:
                    target_center = np.mean(target_centers, axis=0)
                    
                    # 조각의 중심 계산
                    fragment_centers = [obj["bbox"][:3] for obj in fragment["objects"].values()]
                    fragment_center = np.mean(fragment_centers, axis=0)
                    
                    # 위치 조정 (이동)
                    offset = target_center - fragment_center
                    bbox = np.array(adjusted_obj["bbox"])
                    bbox[:3] += offset
                    adjusted_obj["bbox"] = bbox.tolist()
            
            hybrid_scene["objects"][new_id] = adjusted_obj
        
        # 2. 관계 통합
        for rel in fragment["relationships"]:
            if str(rel["subject"]) in id_mapping and str(rel["object"]) in id_mapping:
                new_rel = rel.copy()
                new_rel["subject"] = id_mapping[str(rel["subject"])]
                new_rel["object"] = id_mapping[str(rel["object"])]
                hybrid_scene["relationships"].append(new_rel)
        
        return hybrid_scene
    
    def ensure_scene_consistency(self, hybrid_scene):
        """
        하이브리드 씬의 일관성 확보
        
        Args:
            hybrid_scene: 일관성을 확보할 하이브리드 씬
            
        Returns:
            consistent_scene: 일관성이 확보된 씬
        """
        # 1. 객체 간 충돌 확인 및 해결
        obj_boxes = []
        for obj_id, obj_info in hybrid_scene["objects"].items():
            if "bbox" in obj_info:
                bbox = obj_info["bbox"]
                obj_boxes.append((obj_id, bbox))
        
        # 충돌 확인 및 조정
        for i in range(len(obj_boxes)):
            for j in range(i + 1, len(obj_boxes)):
                id1, box1 = obj_boxes[i]
                id2, box2 = obj_boxes[j]
                
                # 간단한 AABB 충돌 테스트
                if self.check_box_collision(box1, box2):
                    # 충돌 해결 (간단한 이동 전략)
                    self.resolve_collision(hybrid_scene["objects"][id1], hybrid_scene["objects"][id2])
        
        # 2. 관계 일관성 확인 및 조정
        valid_relationships = []
        for rel in hybrid_scene["relationships"]:
            subj_id = str(rel["subject"])
            obj_id = str(rel["object"])
            
            if subj_id in hybrid_scene["objects"] and obj_id in hybrid_scene["objects"]:
                # 관계의 유효성 확인 및 조정
                valid_relationships.append(rel)
        
        hybrid_scene["relationships"] = valid_relationships
        
        return hybrid_scene
    
    def check_box_collision(self, box1, box2):
        """
        두 바운딩 박스 간의 충돌 확인
        
        Args:
            box1: 첫 번째 바운딩 박스 [x, y, z, dx, dy, dz]
            box2: 두 번째 바운딩 박스 [x, y, z, dx, dy, dz]
            
        Returns:
            collision: 충돌 여부 (True/False)
        """
        # 간단한 AABB 충돌 테스트
        box1_min = np.array(box1[:3]) - np.array(box1[3:6]) / 2
        box1_max = np.array(box1[:3]) + np.array(box1[3:6]) / 2
        
        box2_min = np.array(box2[:3]) - np.array(box2[3:6]) / 2
        box2_max = np.array(box2[:3]) + np.array(box2[3:6]) / 2
        
        # 모든 축에서 겹치는지 확인
        overlap_x = box1_min[0] <= box2_max[0] and box2_min[0] <= box1_max[0]
        overlap_y = box1_min[1] <= box2_max[1] and box2_min[1] <= box1_max[1]
        overlap_z = box1_min[2] <= box2_max[2] and box2_min[2] <= box1_max[2]
        
        return overlap_x and overlap_y and overlap_z
    
    def resolve_collision(self, obj1, obj2):
        """
        두 객체 간의 충돌 해결
        
        Args:
            obj1: 첫 번째 객체 정보
            obj2: 두 번째 객체 정보
        """
        # 간단한 이동 전략 (두 번째 객체를 약간 이동)
        if "bbox" in obj2:
            bbox = obj2["bbox"]
            
            # y축(높이)으로 약간 이동
            bbox[1] += 0.5
            
            # x-z 평면에서 약간 이동
            direction = np.random.rand(2) * 2 - 1  # -1~1 사이의 랜덤 방향
            direction = direction / np.linalg.norm(direction)  # 정규화
            
            bbox[0] += direction[0] * 0.3
            bbox[2] += direction[1] * 0.3
            
            obj2["bbox"] = bbox
    
    def generate_hybrid_scene_from_similar(self, real_space_data=None):
        """
        식별된 유사 가상 씬들을 조합하여 하이브리드 씬 생성
        
        Args:
            real_space_data: 실제 공간 데이터 (None이면 저장된 데이터 사용)
            
        Returns:
            hybrid_scene: 생성된 현실-가상 합성 씬
        """
        if self.similar_scenes is None:
            raise ValueError("유사 가상 씬이 식별되지 않았습니다. identify_similar_virtual_scenes를 먼저 호출하세요.")
        
        if real_space_data is None:
            real_space_data = self.vae.real_space_data
            
        if real_space_data is None:
            raise ValueError("현실 공간 데이터가 없습니다.")
        
        # 1. 공간을 영역으로 분할
        real_space_regions = self.analyze_space_regions(real_space_data)
        
        # 2. 하이브리드 씬 초기화
        hybrid_scene = {"objects": {}, "relationships": []}
        
        # 3. 각 영역에 가장 잘 맞는 가상 씬 조각 선택 및 통합
        for region_name, region_data in real_space_regions.items():
            best_fragment = self.find_best_matching_fragment(region_data, self.similar_scenes)
            if best_fragment:
                hybrid_scene = self.integrate_fragment(hybrid_scene, best_fragment, region_data)
        
        # 4. 씬 일관성 확보
        hybrid_scene = self.ensure_scene_consistency(hybrid_scene)
        self.hybrid_scene = hybrid_scene
        
        return hybrid_scene
    
    def save_hybrid_scene(self, output_path):
        """
        생성된 하이브리드 씬을 파일로 저장
        
        Args:
            output_path: 저장할 파일 경로
        """
        if self.hybrid_scene is None:
            raise ValueError("하이브리드 씬이 생성되지 않았습니다.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.hybrid_scene, f, indent=2)
        
        print(f"하이브리드 씬이 {output_path}에 저장되었습니다.")
    
    def load_hybrid_scene(self, input_path):
        """
        저장된 하이브리드 씬 파일 로드
        
        Args:
            input_path: 로드할 파일 경로
            
        Returns:
            hybrid_scene: 로드된 하이브리드 씬
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {input_path}")
        
        with open(input_path, 'r') as f:
            self.hybrid_scene = json.load(f)
        
        print(f"하이브리드 씬이 {input_path}에서 로드되었습니다.")
        return self.hybrid_scene
    
    def inference_with_hybrid_scene(self, input_condition=None):
        """
        하이브리드 씬을 바탕으로 조건에 맞는 새로운 씬 생성
        
        Args:
            input_condition: 생성 조건 (예: 추가할 객체 등)
            
        Returns:
            generated_scene: 생성된 씬
        """
        if self.hybrid_scene is None:
            raise ValueError("하이브리드 씬이 없습니다. generate_hybrid_scene_from_similar 또는 load_hybrid_scene을 먼저 호출하세요.")
        
        # TODO: 구체적인 추론 로직 구현 => 나중에
        # 여기서는 하이브리드 씬을 기반으로 조건에 맞게 변형하거나 샘플링하는 로직이 필요
        
        return self.hybrid_scene

if __name__ == "__main__":
    print("SpaceAdaptiveVAE 테스트 실행")
    
    # 테스트 모드로 SpaceAdaptiveVAE 인스턴스 생성
    space_adaptive_vae = SpaceAdaptiveVAE(vae_model=None)
    space_data = space_adaptive_vae.load_real_space_data("/home/commonscenes/spaceadaptive/spacedata/spacedata.json")
    objs, boxes, triples = space_adaptive_vae.space_data_processing(space_data=space_data, room_type='all', space_id='Lounge')

    print(f"처리 결과: {len(objs)} 객체, {len(triples)} 관계")
    print(f"객체 클래스: {objs}")
    print(f"관계 트리플: {triples}")