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
    print("VAE 모듈을 불러올 수 없습니다!! 테스트 모드로 실행합니다.")
    VAE = None
from model.losses import bce_loss
from helpers.util import bool_flag, _CustomDataParallel


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
        # 전달받은 vae_model을 그대로 사용
        self.vae = vae_model  # VAE에서 vae_model로 변경
        self.config = config or {}
        
        # VAE 모델 확인
        if self.vae is not None:
            if isinstance(self.vae, type):
                print(f"VAE 모델 클래스가 로드되었습니다: {self.vae.__name__}")
            else:
                print(f"VAE 모델 인스턴스가 로드되었습니다: {type(self.vae).__name__}")
        else:
            print("경고: VAE 모델이 로드되지 않았습니다.")
            
        self.real_space_embedding = None
        self.similar_scenes = None
        self.hybrid_scene = None
        self.space_data_preprocessed = None
        self.space_data = None
        self.space_id = "Lounge"
        self.room_type = "all"
        self.space_data_path = "/home/commonscenes/spaceadaptive/spacedata/spacedata.json"
        
    def load_real_space_data(self, space_data_path): # 검수완료
        with open(space_data_path, 'r') as f:
            space_data = json.load(f)
        
        self.space_data_preprocessed = space_data
        space_data_preprocessed = space_data
        # VAE 모델에도 데이터 설정
        # self.vae.space_data = space_data
        print(f"현실 공간 데이터 로드 완료: {len(space_data)} 공간")
        return space_data_preprocessed
        
    def space_data_processing(self, space_data_preprocessed, room_type='all', space_id='Lounge'): # 검수완료
        """
        현실 공간 데이터를 SG-FRONT 형식으로 처리하여 텐서로 변환
        
        Args:
            room_type: 방 유형 (bedroom, living, all 등)
            space_data: 처리할 공간 데이터
            space_id: 처리할 특정 공간 ID (예: 'Lounge', 'Studio')
            
        Returns:
            tuple: (objs_tensor, boxes_tensor, triples_tensor) 형태의 텐서 튜플
        """
        # self.space_data가 None인 경우 초기화
        if self.space_data is None:
            self.space_data = {}
        if space_id not in self.space_data:
            self.space_data[space_id] = {}

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
        for scan in space_data_preprocessed["scans"]:
            if scan["scan"] == space_id:
                target_scan = scan
                break
        
        if target_scan is None:
            print(f"경고: {space_id} ID를 가진 스캔을 찾을 수 없습니다.")
            if len(space_data_preprocessed["scans"]) > 0:
                print(f"대신 첫 번째 스캔({space_data_preprocessed['scans'][0]['scan']})을 사용합니다.")
                target_scan = space_data_preprocessed["scans"][0]
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
        objs_tensor = objs if isinstance(objs, torch.Tensor) else torch.tensor(objs, dtype=torch.long)
        boxes_tensor = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float)
        triples_tensor = triples if isinstance(triples, torch.Tensor) else torch.tensor(triples, dtype=torch.long)
        
        # train_spaceadaptive.py와 동일한 데이터 구조로 저장
        self.space_data[space_id] = {
            'objs': objs_tensor,
            'boxes': boxes_tensor,
            'triples': triples_tensor
        }
        
        # VAE 모델에도 데이터 설정
        self.vae.space_data = self.space_data
        space_data = self.space_data

        return space_data, objs_tensor, boxes_tensor, triples_tensor
    
    def encode_real_space(self, room_type='all', space_id=None): # 검수완료
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
            space_ids = list(self.space_data_preprocessed["scans"].keys())
            
        # 각 공간별 임베딩 생성
        embeddings = {}
        for current_space_id in space_ids:
            print(f"현실 공간 인코딩 중: {current_space_id}")
            
            # 데이터 처리 및 텐서 변환 (room_type은 필요에 따라 설정)
            room_type = 'all'  # 또는 공간별 room_type 매핑 사용
            space_data, objs_tensor, boxes_tensor, triples_tensor = self.space_data_processing(
                space_data_preprocessed=self.space_data_preprocessed,
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
        # self.vae.set_cuda()
        
        # 임베딩은 VAE 모델 내에서 관리되므로 여기서는 추가 작업 불필요
        # real_space_embedding은 vae.real_space_embeddings의 참조임
    
    def train_with_real_space(self, space_data_path='/home/commonscenes/spaceadaptive/spacedata/spacedata.json', room_type='all', space_id=None): # 검수완료
        """
        현실 공간 데이터를 로드하고 전처리한 후 VAE 모델에 임베딩 생성
        한번의 함수 호출로 load_real_space_data, space_data_processing, encode_real_space 작업 수행
        Args:
            space_data_path: 현실 공간 데이터 파일 경로
            room_type: 방 유형 ('bedroom', 'living', 'all' 등)
            space_id: 특정 공간 ID (None이면 모든 공간 처리)
        Returns:
            tuple: (space_data, objs_tensor, boxes_tensor, triples_tensor, real_space_embedding)
        """

        # 1. 모델이 아직 현실 공간 데이터를 로드하지 않았다면 로드
        if self.space_data is None:
            self.space_data_preprocessed = self.load_real_space_data(space_data_path)
            space_data, objs_tensor, boxes_tensor, triples_tensor = self.space_data_processing(self.space_data_preprocessed, room_type, space_id)
        else:
            # 이미 데이터가 로드되어 있는 경우
            space_data = self.space_data
            if space_id in self.space_data:
                objs_tensor = self.space_data[space_id]['objs']
                boxes_tensor = self.space_data[space_id]['boxes']
                triples_tensor = self.space_data[space_id]['triples']
            else:
                # space_id가 없는 경우 처리
                space_data, objs_tensor, boxes_tensor, triples_tensor = self.space_data_processing(self.space_data_preprocessed, room_type, space_id)
        
        # 2. 현실 공간 임베딩 생성 (아직 생성되지 않은 경우)
        if not hasattr(self.vae, 'real_space_embeddings') or self.vae.real_space_embeddings is None:
            real_space_embedding = self.encode_real_space(room_type, space_id)
        else:
            real_space_embedding = self.vae.real_space_embeddings
        
        # CUDA 설정
        self.set_cuda()
        
        return space_data, objs_tensor, boxes_tensor, triples_tensor, real_space_embedding
    
    def encode_virtual_scene(self, scene_data): # 검수완료
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

        boxes_coords = boxes_tensor[:, :6]  # 처음 6개 파라미터 (좌표/크기)
        angles = None
        if boxes_tensor.size(1) > 6:
            angles = boxes_tensor[:, 6].long() - 1  # 7번째 파라미터 (각도)
            # 각도 범위 보정 (0-23 사이로)
            angles = torch.where(angles > 0, angles, torch.zeros_like(angles))
            angles = torch.where(angles < 24, angles, torch.zeros_like(angles))
        
        # CLIP 특성이 필요한 경우 더미 데이터 생성
        dummy_text_feats = torch.zeros((len(objs_tensor), 512)).cuda()
        dummy_rel_feats = torch.zeros((len(triples_tensor), 512)).cuda()
        
        # 모델 타입에 따른 인코딩
        if  self.vae.type_ == 'v2_box':
            mu, _ = self.vae.vae_box.encoder(objs_tensor, triples_tensor, boxes_coords, None, dummy_text_feats, dummy_rel_feats, angles)
        elif self.vae.type_ == 'v2_full':
            mu, _ = self.vae.vae_v2.encoder(objs_tensor, triples_tensor, boxes_coords, None, dummy_text_feats, dummy_rel_feats, angles)
                
        return mu
    
    def calculate_embedding_similarity(self, virtual_embedding, real_embedding): # 검수완료
        """
        두 임베딩 벡터 간의 유사도 계산
        Args:
            vritual_embedding: 가상 씬 임베딩 벡터
            real_embedding: 현실 공간 임베딩 벡터
        Returns:
            similarity: 두 임베딩 간의 유사도 (0~1 범위)
        """
        virtual_emb = virtual_embedding.detach().clone()
        real_emb = real_embedding.detach().clone()
   
        virtual_emb_mean = virtual_emb.mean(dim=0)  # [객체수, 64] -> [64]
        virtual_emb_mean = virtual_emb_mean.cuda()
        real_emb_mean = real_emb.mean(dim=0)  # [객체수, 64] -> [64]
        real_emb_mean = real_emb_mean.cuda()
        
        # virtual_dim = virtual_emb_mean.shape[0]  # 64차원 또는 그 외
        # real_dim = real_emb_mean.shape[0]        # 64차원 또는 그 외
        # output_dim = 64
        # self.virtual_adapter = nn.Linear(virtual_dim, output_dim).cuda()
        # self.real_space_adapter = nn.Linear(real_dim, output_dim).cuda()
        # virtual_embedding_adapted = self.virtual_adapter(virtual_emb_mean)
        # real_embedding_adapted = self.real_space_adapter(real_emb_mean)

        # 코사인 유사도 계산
        cos_sim = F.cosine_similarity(virtual_emb_mean, real_emb_mean, dim=0)
        
        # 유클리드 거리 계산 (정규화된 형태로)
        euclidean_dist = torch.norm(virtual_emb_mean - real_emb_mean, p=2)
        max_dist = torch.sqrt(torch.tensor(real_emb_mean.shape[0] * 2.0).float().cuda())  # 최대 가능한 거리
        euclidean_sim = 1.0 - (euclidean_dist / max_dist)
        
        # 두 유사도의 가중 평균
        # 코사인 유사도: 벡터의 방향(의미)을 중요시하며, 객체 배치의 의미적 유사성을 잘 포착합니다
        # 유클리드 유사도: 절대적 거리를 고려하여 객체 크기나 실제 공간 거리도 반영합니다
        similarity = 0.7 * cos_sim + 0.3 * euclidean_sim # 가중치 값 그때그때 조절.
        
        return similarity.item()
    
    def identify_similar_virtual_scenes(self, virtual_scenes_dataset, top_k=200): # 검수완료
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
                    similarity = self.calculate_embedding_similarity(scene_embedding, space_emb)
                    max_similarity = max(max_similarity, similarity)
                similarity = max_similarity
            else:
                similarity = self.calculate_embedding_similarity(scene_embedding, self.real_space_embedding[self.space_id])
                
            similar_scenes.append((scene_id, scene_data, scene_embedding, similarity))
        
        # 유사도 기준 정렬 및 상위 k개 반환
        similar_scenes.sort(key=lambda x: x[3], reverse=True) # 유사도 기준 정렬 맞지..?
        self.similar_scenes = similar_scenes[:top_k]
        
        return self.similar_scenes
    
    # Not in Use.
    def analyze_space_regions(self, space_id, space_data): # 검수완료
        """
        공간을 의미 있는 영역으로 분할
        Args:
            space_id: 분석할 공간 ID
            space_data: 분석할 공간 데이터
        Returns:
            regions: 식별된 영역들의 딕셔너리
        """
        # 1. 객체 위치 데이터 추출
        obj_positions = []
        obj_info = []
        
        # 현실 공간의 data_processing()이 끝나 있어야 함.
        for i in range(len(space_data[space_id]["objs"])):
            # boxes 텐서에서 중심 위치 추출 (x, y, z)
            center_x = space_data[space_id]["boxes"][i][3]
            center_y = space_data[space_id]["boxes"][i][4]
            center_z = space_data[space_id]["boxes"][i][5]
            obj_positions.append([center_x, center_y, center_z])
            obj_info.append({
                "id": i,
                "class": space_data[space_id]["objs"][i].item(),  # 텐서에서 값 추출
                "boxes": space_data[space_id]["boxes"][i]
            })
        
        if not obj_positions:
            return {"entire_space": space_data[space_id]}
        
        # 2. 밀도 기반 클러스터링으로 영역 식별
        positions_array = np.array(obj_positions)
        # TODO 클러스터링 알고리즘을 바꾸거나 파라미터 조정 필요. 현재는 DBSCAN 사용. 클러스터링 할 때 scene이랑 floor 뺄 방법 고민 필요.
        # TODO 클러스터링 말고 그냥 객체 하나하나 별로 고민할 방법 필요.
        clustering = DBSCAN(eps=1.0, min_samples=2).fit(positions_array)
        labels = clustering.labels_
        # labels = array([0, 0, 1, 1, -1, 0, 2, 2, 1, -1])
        
        # 3. 클러스터별로 객체 그룹화
        regions = {}
        for label in set(labels): # {0, 1, 2, -1}
            region_name = f"region_{label}" if label >= 0 else "outliers"
            regions[region_name] = {
                "objs": [],      # 객체 인덱스 리스트
                "obj_classes": [],  # 객체 클래스 리스트
                "boxes": [],     # 바운딩 박스 리스트
                "triples": []    # 관계 리스트
            }
        
        # 객체를 해당 영역에 할당
        # labels 인덱싱이랑 space_data[space_id]["objs"], obj_info 인덱싱이랑 같음.
        for i, label in enumerate(labels):
            region_name = f"region_{label}" if label >= 0 else "outliers"
            regions[region_name]["objs"].append(i)  # 객체 인덱스 추가
            regions[region_name]["obj_classes"].append(obj_info[i]["class"])  # 객체 클래스 추가
            regions[region_name]["boxes"].append(obj_info[i]["boxes"])  # 박스 정보 추가
        
        # 4. 관계 정보 분배
        # triples 텐서는 [subject_idx, relation_type, object_idx] 형태
        for triple_idx in range(space_data[space_id]["triples"].shape[0]):
            triple = space_data[space_id]["triples"][triple_idx]
            subj_id = triple[0].item()  # 주체 객체 인덱스
            rel_type = triple[1].item()  # 관계 유형
            obj_id = triple[2].item()    # 대상 객체 인덱스
            
            # 두 객체가 같은 영역에 있는 관계만 해당 영역에 할당
            for region_name, region_data in regions.items():
                if subj_id in region_data["objs"] and obj_id in region_data["objs"]:
                    # 해당 영역에 관계 추가
                    regions[region_name]["triples"].append([subj_id, rel_type, obj_id])
        
        # 각 영역의 텐서 데이터 변환
        for region_name, region_data in regions.items():
            # 리스트를 텐서로 변환
            regions[region_name]["objs"] = torch.tensor(region_data["obj_classes"])
            regions[region_name]["boxes"] = torch.stack(region_data["boxes"]) if region_data["boxes"] else torch.tensor([])
            regions[region_name]["triples"] = torch.tensor(region_data["triples"]) if region_data["triples"] else torch.tensor([])
        
        return regions
    
    # Not in Use.
    def split_scene_into_fragments(self, scene_id, scene_data): # 검수 완료
        """
        가상 씬을 작은 조각으로 분할
        Args:
            scene_data: 분할할 가상 씬 데이터
        Returns:
            fragments: 씬 조각들의 리스트
        """
        # 공간 영역 분석 함수를 활용하여 씬을 조각으로 분할
        regions = self.analyze_space_regions(scene_id, scene_data)
        
        # 각 영역을 독립적인 조각으로 변환
        fragments = []
        for region_name, region_data in regions.items():
            if len(region_data["objs"]) >= 2:  # 최소 2개 이상의 객체가 있는 영역만 고려
                fragments.append(region_data)
        
        return fragments
    
    # Not in Use.
    def find_best_matching_fragment(self, target_region, similar_scenes): # 검수 완료
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
        for scene_id, scene_data, overall_similarity in similar_scenes: # (scene_id, scene_data, similarity)
            fragments = self.split_scene_into_fragments(scene_id, scene_data)
            
            for fragment in fragments:
                # 대상 영역과 조각 간의 유사도 계산
                region_embedding = self.encode_virtual_scene(target_region) # target_region도 {"region_0": region_data(scene_data)}에서 region_data만 갖고 옴.
                fragment_embedding = self.encode_virtual_scene(fragment) # fragment는 region_data여서 바로 들어가도 될 듯.
                fragment_similarity = self.calculate_embedding_similarity(region_embedding, fragment_embedding)
                
                # 기존 씬의 전체 유사도도 고려하여 가중치 부여
                weighted_similarity = 0.7 * fragment_similarity + 0.3 * overall_similarity #TODO 가중치 조정 필요
                
                if weighted_similarity > best_similarity:
                    best_similarity = weighted_similarity
                    best_fragment = fragment
        
        return best_fragment
    
    # Not in Use.
    def integrate_fragment(self, hybrid_scene, fragment, target_region):
        """
        가상 씬 조각을 하이브리드 씬에 통합
        Args:
            hybrid_scene: 현재 구성 중인 하이브리드 씬 {"objs":, "boxes":, "triples"}
            fragment: 통합할 씬 조각 {"objs":, "boxes":, "triples"}
            target_region: 대상 영역 정보 {"objs":, "boxes":, "triples"}
        Returns:
            updated_hybrid_scene: 업데이트된 하이브리드 씬
        """
        if fragment is None:
            return hybrid_scene
        
        # 하이브리드 씬이 비어있으면 초기화
        if not hybrid_scene or 'objs' not in hybrid_scene:
            hybrid_scene = {"objs": torch.tensor([]), "boxes": torch.tensor([]), "triples": torch.tensor([])}
        
        # 프래그먼트 데이터
        fragment_objs = fragment["objs"]
        fragment_boxes = fragment["boxes"]
        fragment_triples = fragment["triples"]
        
        # 타겟 영역 데이터
        target_boxes = target_region["boxes"]
        
        # 현재 객체 수 (ID 오프셋으로 사용)
        current_obj_count = len(hybrid_scene["objs"]) if "objs" in hybrid_scene and len(hybrid_scene["objs"]) > 0 else 0
        
        # 위치 조정을 위한 중심점 계산
        if len(target_boxes) > 0 and len(fragment_boxes) > 0:
            # 타겟 영역 중심 계산 (박스의 중심점 좌표 사용)
            target_centers = torch.stack([box[3:6] for box in target_boxes]) if len(target_boxes) > 0 else None
            target_center = torch.mean(target_centers, dim=0) if target_centers is not None else None
            
            # 프래그먼트 중심 계산
            fragment_centers = torch.stack([box[3:6] for box in fragment_boxes]) if len(fragment_boxes) > 0 else None
            fragment_center = torch.mean(fragment_centers, dim=0) if fragment_centers is not None else None
            
            # 이동 오프셋 계산
            if target_center is not None and fragment_center is not None:
                offset = target_center - fragment_center
                
                # 박스 위치 조정 (복사본 생성)
                adjusted_boxes = fragment_boxes.clone()
                adjusted_boxes[:, 3:6] += offset  # 중심점 조정
            else:
                adjusted_boxes = fragment_boxes
        else:
            adjusted_boxes = fragment_boxes
        
        # 1. 객체 및 박스 통합
        if len(fragment_objs) > 0:
            if current_obj_count == 0:
                hybrid_scene["objs"] = fragment_objs
                hybrid_scene["boxes"] = adjusted_boxes
            else:
                hybrid_scene["objs"] = torch.cat([hybrid_scene["objs"], fragment_objs])
                hybrid_scene["boxes"] = torch.cat([hybrid_scene["boxes"], adjusted_boxes])
        
        # 2. 관계 통합 (ID 조정)
        if len(fragment_triples) > 0:
            # 관계 텐서 복사 및 주체/객체 ID 조정
            adjusted_triples = fragment_triples.clone()
            adjusted_triples[:, 0] += current_obj_count  # 주체 ID 조정
            adjusted_triples[:, 2] += current_obj_count  # 객체 ID 조정
            
            if current_obj_count == 0 or 'triples' not in hybrid_scene or len(hybrid_scene["triples"]) == 0:
                hybrid_scene["triples"] = adjusted_triples
            else:
                hybrid_scene["triples"] = torch.cat([hybrid_scene["triples"], adjusted_triples])
        
        # 디버깅 정보
        print(f"프래그먼트 통합 완료:")
        print(f"- 통합된 객체 수: {len(fragment_objs)}")
        print(f"- 통합된 관계 수: {len(fragment_triples)}")
        print(f"- 총 객체 수: {len(hybrid_scene['objs'])}")
        print(f"- 총 관계 수: {len(hybrid_scene['triples'])}")
        
        return hybrid_scene
        
    # Not in Use.
    def generate_hybrid_scene_from_similar(self, space_data=None, similar_scenes=None): 
        """
        식별된 유사 가상 씬들을 조합하여 하이브리드 씬 생성
        Args:
            real_space_data: 실제 공간 데이터 (None이면 저장된 데이터 사용)
        Returns:
            hybrid_scene: 생성된 현실-가상 합성 씬
        """
        if similar_scenes is None:
            similar_scenes = self.similar_scenes
        if self.similar_scenes is None:
            raise ValueError("유사 가상 씬이 식별되지 않았습니다. identify_similar_virtual_scenes를 먼저 호출하세요.")
        if space_data is None:
            space_data = self.space_data
        if space_data is None:
            raise ValueError("현실 공간 데이터가 없습니다.")
        
        # 1. 공간을 영역으로 분할
        # TODO 구조 변경 필요... 영역 분할 방법 변경 필요. 지금은 클러스터링으로 해버리는데 제대로 영역 분할이 안됨.
        real_space_regions = self.analyze_space_regions(self.space_id, space_data)
        
        # 2. 하이브리드 씬 초기화 (텐서 기반 구조로 초기화)
        hybrid_scene = {"objs": torch.tensor([]), "boxes": torch.tensor([]), "triples": torch.tensor([])}
        
        # 3. 각 영역에 가장 잘 맞는 가상 씬 조각 선택 및 통합
        for region_name, region_data in real_space_regions.items():
            # region_name이 'outliers'인 경우 건너뛰기 (노이즈 영역)
            if region_name == 'outliers':
                continue
                
            best_fragment = self.find_best_matching_fragment(region_data, similar_scenes)
            if best_fragment:
                hybrid_scene = self.integrate_fragment(hybrid_scene, best_fragment, region_data)
        
        # 4. 씬 일관성 확보
        # hybrid_scene = self.ensure_scene_consistency(hybrid_scene)
        self.hybrid_scene = hybrid_scene
        
        return hybrid_scene
    
    # SpaceAdaptive: 현실 공간 손실 계산
    # Not in Use.
    def calculate_real_space_loss_v2(self, z, real_space_embedding=None):
        """
        현실 공간 임베딩과 생성된 잠재 벡터 간의 손실을 계산하는 메서드
        Args:
            z (torch.Tensor): 생성된 잠재 벡터
            real_space_embedding (torch.Tensor): 현실 공간 임베딩
        Returns:
            torch.Tensor: 계산된 손실값
        """
        if real_space_embedding is None:
            real_space_embedding = self.real_space_embedding[self.space_id]

        # 모든 텐서의 복사본 생성 및 그래디언트 분리
        z_detached = z.detach().clone() if isinstance(z, torch.Tensor) else z
        real_space_emb_detached = real_space_embedding.detach().clone() if isinstance(real_space_embedding, torch.Tensor) else real_space_embedding
        
        # candidates_during_training 초기화 확인 및 생성
        if self.space_id not in self.space_data or "candidates_during_training" not in self.space_data[self.space_id]:
            # 현실 객체 수만큼 빈 리스트 생성
            num_real_objects = len(real_space_embedding)
            self.space_data[self.space_id]["candidates_during_training"] = [[] for _ in range(num_real_objects)]

        # 중간 계산은 그래디언트 추적 없이 수행
        with torch.no_grad():
            for i, real_obj_emd in enumerate(real_space_emb_detached):
                similarities = F.cosine_similarity(real_obj_emd.unsqueeze(0), z_detached, dim=1)
                matched_idx = similarities.argmax().item()
                similarity_value = similarities[matched_idx].item()

                candidate = {
                "emd": z_detached[matched_idx].clone(),  # 임베딩 저장
                "similarity": similarity_value,  # 유사도 저장
                }

                self.space_data[self.space_id]["candidates_during_training"][i].append(candidate)
                # 유사도 기준 내림차순 정렬
                self.space_data[self.space_id]["candidates_during_training"][i].sort(key=lambda x: x["similarity"], reverse=True)
            
                # 리스트 크기 제한 (최대 20개)
                max_candidates = 20
                if len(self.space_data[self.space_id]["candidates_during_training"][i]) > max_candidates:
                    self.space_data[self.space_id]["candidates_during_training"][i] = \
                        self.space_data[self.space_id]["candidates_during_training"][i][:max_candidates]
            
            cumulative_z = []
            for i, candidates in enumerate(self.space_data[self.space_id]["candidates_during_training"]):
                cumulative_z.append(candidates[0]["emd"])
            
            cumulative_z_tensor = torch.stack(cumulative_z)

            # MSE 손실 계산
            latent_loss = F.mse_loss(cumulative_z_tensor, real_space_emb_detached, reduction='mean')
            
            return latent_loss
    
    # SpaceAdaptive: 현실 공간 손실 계산
    def calculate_real_space_loss_v3(self, z, real_space_embedding=None):
        """현실 공간 임베딩과 생성된 잠재 벡터 간의 손실 계산"""
        if real_space_embedding is None:
            real_space_embedding = self.real_space_embedding[self.space_id]
        
        # 배치 크기와 현실 객체 수
        batch_size = z.size(0)
        num_real_objects = real_space_embedding.size(0)
        
        # 중요: 그래디언트 차단 없이 손실 계산
        # i*j 텐서를 만들어서 현실공간과 가상씬의 모든 조합에 대해서 전역적인 유사도 계산
        similarity_matrix = torch.zeros(num_real_objects, batch_size, device=z.device)
        
        for i in range(num_real_objects):
            for j in range(batch_size):
                # 코사인 유사도 계산 
                similarity_matrix[i, j] = F.cosine_similarity(
                    real_space_embedding[i].unsqueeze(0),
                    z[j].unsqueeze(0),
                    dim=1
                )
        
        # 최적 매칭 찾기
        matched_pairs = []
        used_gen_indices = set()
        
        for i in range(num_real_objects):
            best_similarity = -float('inf')
            best_gen_idx = -1
            
            for j in range(batch_size):
                if j not in used_gen_indices and similarity_matrix[i, j] > best_similarity:
                    best_similarity = similarity_matrix[i, j]
                    best_gen_idx = j
            
            if best_gen_idx != -1:
                matched_pairs.append((i, best_gen_idx))
                used_gen_indices.add(best_gen_idx)
        
        if not matched_pairs:
            return torch.tensor(0.0, device=z.device)
        
        # 매칭된 객체끼리 손실 계산
        real_indices = [i for i, _ in matched_pairs]
        gen_indices = [j for _, j in matched_pairs]
        
        real_matched = real_space_embedding[real_indices]
        gen_matched = z[gen_indices]
        
        # 두 종류의 손실 조합
        mse_loss = F.mse_loss(gen_matched, real_matched)
        cos_loss = 1.0 - F.cosine_similarity(gen_matched, real_matched, dim=1).mean()
        
        # 조합된 손실
        loss = 0.7 * mse_loss + 0.3 * cos_loss
        
        # 디버깅 출력 (선택사항)
        # if hasattr(self, 'step_counter') and self.step_counter % 50 == 0:
        #     print(f"Space Loss: {loss.item():.4f} (MSE: {mse_loss.item():.4f}, Cos: {cos_loss.item():.4f})")
        
        # 후보 저장은 별도 함수로 분리 (학습에 영향 없음)
        # self._store_candidates(z.detach(), real_space_embedding, similarity_matrix)
        
        return loss

    def generate_hybrid_scene_from_similar_v2(self, space_data=None, similar_scenes=None, space_id=None, real_space_embedding=None):
        """개선된 하이브리드 씬 생성 함수 - 객체 단위 접근"""
        if similar_scenes is None:
            similar_scenes = self.similar_scenes
        if space_data is None:
            space_data = self.space_data
        if space_id is None:
            space_id = self.space_id
        if real_space_embedding is None:
            real_space_embedding = self.real_space_embedding
            real_space_embed = real_space_embedding[space_id]
        
        # 1. 현실 공간의 각 객체마다 유사 객체 후보 탐색
        real_objects = []
        for i, obj_class in enumerate(space_data[space_id]["objs"]):
            # scene이나 floor 객체는 제외 (ID: 0, 35)
            # if obj_class.item() in [0, 35]:
            #     continue
                
            real_objects.append({
                "id": i,
                "class": obj_class.item(),
                "box": space_data[space_id]["boxes"][i],
                "embedding": real_space_embed[i],  # 객체별 임베딩
                "similar_candidates": []  # 유사 가상 객체 후보 리스트
            })
        
        # 2. 각 객체별로 유사 후보 탐색 
        for real_obj in real_objects:
            for scene_id, scene_data, scene_embedding, _ in similar_scenes:
                for j, virtual_obj_class in enumerate(scene_data["objs"]):
                    # 동일한 클래스의 객체만 고려
                    if virtual_obj_class.item() == real_obj["class"]:
                        # 임베딩 유사도 계산
                        similarity = self.calculate_embedding_similarity(
                        scene_embedding[j].unsqueeze(0),  # 단일 객체 임베딩으로 변환
                        real_obj["embedding"].unsqueeze(0)  # 단일 객체 임베딩으로 변환
                        )
                        
                        real_obj["similar_candidates"].append({
                        "scene_id": scene_id,
                        "obj_id": j,
                        "similarity": similarity,
                        "box": scene_data["boxes"][j],
                        "embedding": scene_embedding[j]
                        })
            
            # 유사도 기준 정렬
            real_obj["similar_candidates"].sort(key=lambda x: x["similarity"], reverse=True)
        
        # 3. 최적의 후보 선택하여 하이브리드 씬 구성
        hybrid_scene = self.construct_hybrid_scene_from_candidates(real_objects)
        
        self.hybrid_scene = hybrid_scene
        return hybrid_scene

    def construct_hybrid_scene_from_candidates(self, real_objects):
        """후보 객체들로부터 하이브리드 씬 구성"""
        # 초기화
        hybrid_objs = []
        hybrid_boxes = []
        hybrid_triples = []
        
        # 각 실제 객체마다 가장 유사한 가상 객체 선택
        for real_obj in real_objects:
            if real_obj["similar_candidates"]:
                # 가장 유사한 후보 선택
                best_candidate = real_obj["similar_candidates"][0]
                
                hybrid_objs.append(real_obj["class"])
                
                # 실제 객체 위치와 가상 객체 형태를 결합한 바운딩 박스
                hybrid_box = best_candidate["box"].clone()
                # 위치는 실제 객체의 위치 사용
                hybrid_box[3:6] = real_obj["box"][3:6]
                hybrid_boxes.append(hybrid_box)
            else:
                # 유사 후보가 없으면 실제 객체 그대로 사용
                hybrid_objs.append(real_obj["class"])
                hybrid_boxes.append(real_obj["box"])
        
        # 관계 구성 (단순 구현)
        # 더 복잡한 관계 추론이 필요하면 추가 로직 개발 필요
        
        # 텐서 변환
        hybrid_scene = {
            "objs": torch.tensor(hybrid_objs),
            "boxes": torch.stack(hybrid_boxes) if hybrid_boxes else torch.tensor([]),
            "triples": torch.tensor(hybrid_triples) if hybrid_triples else torch.tensor([])
        }
        
        return hybrid_scene

    # Not in Use.
    def ensure_scene_consistency(self, hybrid_scene):
        """
        하이브리드 씬의 일관성 확보
        Args:
            hybrid_scene: 일관성을 확보할 하이브리드 씬 {objs, boxes, triples}
        Returns:
            consistent_scene: 일관성이 확보된 씬
        """
        
        if not hybrid_scene or 'objs' not in hybrid_scene:
            return hybrid_scene
            
        # 유효한 객체가 2개 미만이면 일관성 검사 생략
        if len(hybrid_scene["objs"]) < 2:
            return hybrid_scene
            
        # 1. 객체 간 충돌 확인 및 해결
        num_objs = len(hybrid_scene["objs"])
        boxes = hybrid_scene["boxes"]
        
        # 충돌 확인 및 조정
        collision_count = 0
        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                box1 = boxes[i]
                box2 = boxes[j]
                
                # 바닥(floor)이나 씬 자체는 충돌 검사에서 제외
                if hybrid_scene["objs"][i].item() in [0, 35] or hybrid_scene["objs"][j].item() in [0, 35]:
                    continue
                
                # 충돌 감지 (바운딩 박스 중심점과 크기 정보 기반)
                if self.check_box_collision(box1, box2):
                    # 충돌 해결
                    boxes[i], boxes[j] = self.resolve_collision(box1, box2)
                    collision_count += 1
        
        if collision_count > 0:
            print(f"총 {collision_count}개의 객체 충돌 해결됨")
        
        # 2. 관계 일관성 확인 및 조정
        # 유효하지 않은 객체 ID를 참조하는 관계 제거
        if len(hybrid_scene["triples"]) > 0:
            valid_triples_mask = (hybrid_scene["triples"][:, 0] < num_objs) & (hybrid_scene["triples"][:, 2] < num_objs)
            valid_triples = hybrid_scene["triples"][valid_triples_mask]
            
            if len(valid_triples) < len(hybrid_scene["triples"]):
                print(f"{len(hybrid_scene['triples']) - len(valid_triples)}개의 유효하지 않은 관계 제거됨")
                hybrid_scene["triples"] = valid_triples
        
        return hybrid_scene
    # Not in Use.
    def check_box_collision(self, box1, box2): 
        """
        두 바운딩 박스 간의 충돌 확인
        Args:
            box1: 첫 번째 바운딩 박스 텐서 [sx, sy, sz, cx, cy, cz, angle]
            box2: 두 번째 바운딩 박스 텐서 [sx, sy, sz, cx, cy, cz, angle]
        Returns:
            collision: 충돌 여부 (True/False)
        """
        # 중심점과 크기 추출
        center1 = box1[3:6]  # cx, cy, cz
        center2 = box2[3:6]  # cx, cy, cz
        
        size1 = box1[0:3]  # sx, sy, sz
        size2 = box2[0:3]  # sx, sy, sz
        
        # 각 축에 대한 거리 계산
        distance = torch.abs(center1 - center2)
        
        # 충돌 검사를 위한 최소 거리 계산 (두 박스의 절반 크기 합)
        min_distance = (size1 + size2) / 2.0
        
        # 모든 축에서 거리가 최소 거리보다 작다면 충돌
        collision = torch.all(distance < min_distance)
        
        return collision.item()
    # Not in Use.
    def resolve_collision(self, box1, box2):
        """
        두 바운딩 박스 간의 충돌 해결
        Args:
            box1: 첫 번째 바운딩 박스 텐서 [sx, sy, sz, cx, cy, cz, angle]
            box2: 두 번째 바운딩 박스 텐서 [sx, sy, sz, cx, cy, cz, angle]
        Returns:
            (box1, box2): 충돌이 해결된 두 바운딩 박스
        """
        import torch
        
        # 두 박스의 중심점
        center1 = box1[3:6]  # cx, cy, cz
        center2 = box2[3:6]  # cx, cy, cz
        # 중심점 간의 방향 벡터
        direction = center2 - center1
        
        # 방향 벡터 정규화 (0으로 나누기 방지)
        if torch.all(direction == 0):
            # 두 중심점이 완전히 겹치는 경우, 임의의 방향 설정
            direction = torch.tensor([1.0, 0.0, 0.0], device=box1.device)
        else:
            direction = direction / torch.norm(direction)
        
        # 두 박스의 크기
        size1 = box1[0:3]  # sx, sy, sz
        size2 = box2[0:3]  # sx, sy, sz
        
        # 충돌 해결을 위한 이동 거리 계산
        overlap = (size1 + size2) / 2.0 - torch.abs(center2 - center1)
        max_overlap_idx = torch.argmax(overlap)
        move_distance = overlap[max_overlap_idx] * 1.1  # 10% 추가 거리
        
        # 박스 이동 (두 박스를 서로 반대 방향으로 이동)
        adjusted_box1 = box1.clone()
        adjusted_box2 = box2.clone()
        
        adjusted_box1[3:6] = center1 - direction * move_distance / 2.0
        adjusted_box2[3:6] = center2 + direction * move_distance / 2.0
        
        return adjusted_box1, adjusted_box2
    
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

    # from model.VAE_spaceadaptive import VAE
    # VAE = VAE(type='v2_full')
    space_adaptive_vae = SpaceAdaptiveVAE(vae_model=VAE)
    
    space_data_preprocessed = space_adaptive_vae.load_real_space_data("/home/commonscenes/spaceadaptive/spacedata/spacedata.json")
    space_data, objs, boxes, triples = space_adaptive_vae.space_data_processing(space_data_preprocessed=space_data_preprocessed, room_type='all', space_id='Lounge')
    # space_data, objs, boxes, triples, real_space_embedding = space_adaptive_vae.train_with_real_space(space_data_path="/home/commonscenes/spaceadaptive/spacedata/spacedata.json", room_type='all', space_id='Lounge')
    real_space_regions = space_adaptive_vae.analyze_space_regions(space_id='Lounge', space_data=space_data)
    # 클러스터링이 되기는 하는데 2개만 생기고 나머지 객체가 전부다 아웃라이어가 되어버리네...
    
    # print(space_data)
    '''
    {'Lounge': {'objs': tensor([14,  2, 31,  2,  5, 12, 29,  3,  3, 35,  0]), 'boxes': tensor([[ 1.0000e+00,  1.1991e+00,  1.8504e+00,  1.2800e+00, -5.4659e-02,
         -7.3731e-02,  1.2000e+01],
        [ 5.2918e-01,  2.2284e+00,  9.6057e-01, -3.0798e+00,  4.7320e-01,
          5.8112e-01,  1.2000e+01],
        [ 2.4411e+00,  8.8599e-01,  1.4663e+00, -7.4000e-01, -2.2885e-01,
          2.8200e+00,  1.8000e+01],
        [ 4.6071e-01,  1.7600e+00,  9.3456e-01,  3.2400e+00,  3.0631e-01,
          6.4300e-01,  2.4000e+01],
        [ 9.4635e-01,  1.2330e+00,  9.5529e-01, -5.3011e-02,  5.5067e-03,
         -2.1411e+00,  2.4000e+01],
        [ 9.5067e-01,  8.9378e-01,  1.8182e+00, -1.0436e+00, -1.1823e-01,
         -2.5364e+00,  1.2000e+01],
        [ 1.0000e+00,  8.7200e-01,  2.4578e+00, -2.6360e+00, -1.0634e-01,
         -2.2496e+00,  2.4000e+01],
        [ 6.3734e-01,  1.9979e+00,  1.0000e+00,  1.2993e+00,  3.9340e-01,
         -1.5200e+00,  2.4000e+01],
        [ 4.6416e-01,  2.3129e+00,  1.2526e+00, -2.9530e+00,  5.7089e-01,
          2.7130e+00,  1.8000e+01],
        [ 7.2600e+00,  2.2204e-16,  8.4485e+00,  1.5140e-01, -6.7000e-01,
         -6.4500e-02,  1.2000e+01],
        [-1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00,
         -1.0000e+00, -1.0000e+00]]), 'triples': tensor([[ 0,  3,  3],
        [ 0,  2,  7],
        [ 0, 14,  2],
        [ 0,  5,  4],
        [ 1, 14,  3],
        [ 1,  1,  8],
        [ 1,  2,  6],
        [ 1,  5,  5],
        [ 1, 11,  8],
        [ 2,  5,  0],
        [ 2,  5,  1],
        [ 2,  2,  8],
        [ 3, 14,  1],
        [ 3,  3,  0],
        [ 3,  5,  7],
        [ 4,  3,  5],
        [ 4,  4,  7],
        [ 4,  5,  0],
        [ 4,  3,  6],
        [ 4,  9,  5],
        [ 5,  3,  4],
        [ 5,  4,  6],
        [ 5,  5,  6],
        [ 5,  5,  0],
        [ 5,  8,  4],
        [ 6,  3,  5],
        [ 6,  5,  4],
        [ 6,  3,  4],
        [ 6,  1,  1],
        [ 7,  1,  0],
        [ 7,  4,  4],
        [ 7,  5,  3],
        [ 7, 14,  8],
        [ 7,  9,  8],
        [ 8, 10,  7],
        [ 8,  2,  1],
        [ 8,  3,  2],
        [ 8, 10,  1],
        [ 0,  7,  9],
        [ 1,  7,  9],
        [ 2,  7,  9],
        [ 3,  7,  9],
        [ 4,  7,  9],
        [ 5,  7,  9],
        [ 6,  7,  9],
        [ 7,  7,  9],
        [ 8,  7,  9],
        [ 0,  0, 10],
        [ 1,  0, 10],
        [ 2,  0, 10],
        [ 3,  0, 10],
        [ 4,  0, 10],
        [ 5,  0, 10],
        [ 6,  0, 10],
        [ 7,  0, 10],
        [ 8,  0, 10],
        [ 9,  0, 10]])}}
    '''

    print(real_space_regions)
    '''
    {'region_0': {'objs': tensor([14, 35]), 'obj_classes': [14, 35], 'boxes': tensor([[ 1.0000e+00,  1.1991e+00,  1.8504e+00,  1.2800e+00, -5.4659e-02,
         -7.3731e-02,  1.2000e+01],
        [ 7.2600e+00,  2.2204e-16,  8.4485e+00,  1.5140e-01, -6.7000e-01,
         -6.4500e-02,  1.2000e+01]]), 'triples': tensor([[0, 7, 9]])}, 'region_1': {'objs': tensor([ 5, 12]), 'obj_classes': [5, 12], 'boxes': tensor([[ 9.4635e-01,  1.2330e+00,  9.5529e-01, -5.3011e-02,  5.5067e-03,
         -2.1411e+00,  2.4000e+01],
        [ 9.5067e-01,  8.9378e-01,  1.8182e+00, -1.0436e+00, -1.1823e-01,
         -2.5364e+00,  1.2000e+01]]), 'triples': tensor([[4, 3, 5],
        [4, 9, 5],
        [5, 3, 4],
        [5, 8, 4]])}, 'outliers': {'objs': tensor([ 2, 31,  2, 29,  3,  3,  0]), 'obj_classes': [2, 31, 2, 29, 3, 3, 0], 'boxes': tensor([[ 0.5292,  2.2284,  0.9606, -3.0798,  0.4732,  0.5811, 12.0000],
        [ 2.4411,  0.8860,  1.4663, -0.7400, -0.2289,  2.8200, 18.0000],
        [ 0.4607,  1.7600,  0.9346,  3.2400,  0.3063,  0.6430, 24.0000],
        [ 1.0000,  0.8720,  2.4578, -2.6360, -0.1063, -2.2496, 24.0000],
        [ 0.6373,  1.9979,  1.0000,  1.2993,  0.3934, -1.5200, 24.0000],
        [ 0.4642,  2.3129,  1.2526, -2.9530,  0.5709,  2.7130, 18.0000],
        [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000]]), 'triples': tensor([[ 1, 14,  3],
        [ 1,  1,  8],
        [ 1,  2,  6],
        [ 1, 11,  8],
        [ 2,  5,  1],
        [ 2,  2,  8],
        [ 3, 14,  1],
        [ 3,  5,  7],
        [ 6,  1,  1],
        [ 7,  5,  3],
        [ 7, 14,  8],
        [ 7,  9,  8],
        [ 8, 10,  7],
        [ 8,  2,  1],
        [ 8,  3,  2],
        [ 8, 10,  1],
        [ 1,  0, 10],
        [ 2,  0, 10],
        [ 3,  0, 10],
        [ 6,  0, 10],
        [ 7,  0, 10],
        [ 8,  0, 10]])}}
    '''