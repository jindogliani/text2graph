from __future__ import print_function
import sys
sys.path.append("..")
sys.path.append(".")
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
from tqdm import tqdm
import json
from helpers.psutil import FreeMemLinux
from helpers.util import normalize_box_params
from omegaconf import OmegaConf
import clip
import random
import pickle
import trimesh
import h5py
import open3d as o3d

changed_relationships_dict = {
        'left': 'right',
        'right': 'left',
        'front': 'behind',
        'behind': 'front',
        'bigger than': 'smaller than',
        'smaller than': 'bigger than',
        'taller than': 'shorter than',
        'shorter than': 'taller than',
        'close by': 'close by',
        'same style as': 'same style as',
        'same super category as': 'same super category as',
        'same material as': 'same material as',
        "symmetrical to": "symmetrical to",
        "standing on":"standing on",
        "above":"above"
    }

def load_ckpt(ckpt): #체크포인트 불러오는 함수
    map_fn = lambda storage, loc: storage
    if type(ckpt) == str:
        state_dict = torch.load(ckpt, map_location=map_fn)
    else:
        state_dict = ckpt
    return state_dict

class ThreedFrontDatasetSceneGraph(data.Dataset):
    def __init__(self, root, root_3dfront='', split='train', shuffle_objs=False, pass_scan_id=False, use_SDF=False,
                 use_scene_rels=False, data_len=None, with_changes=True, scale_func='diag', eval=False,
                 eval_type='addition', with_feats=False, with_CLIP=False,
                 seed=True, large=False, recompute_feats=False, recompute_clip=False,
                 room_type='bedroom'):

        self.room_type = room_type
        self.seed = seed
        self.with_feats = with_feats
        self.with_CLIP = with_CLIP
        self.cond_model = None
        self.large = large #확장된 혹은 축소된 클래스 세트로 세밀한 혹은 축약된 표현
        self.recompute_feats = recompute_feats
        self.recompute_clip = recompute_clip

        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        self.scale_func = scale_func
        self.with_changes = with_changes
        self.use_SDF = use_SDF
        self.sdf_res = 64
        self.root = root
        # list of class categories
        self.catfile = os.path.join(self.root,'classes_{}.txt'.format(self.room_type))
        self.cat = {}
        self.scans = []
        self.obj_paths = []
        self.data_len = data_len
        self.use_scene_rels = use_scene_rels

        self.fm = FreeMemLinux('GB')
        self.vocab = {}
        
        #vocab 사전 안에 'object_idx_to_name' key에 객체 이름 리스트 저장
        with open(os.path.join(self.root, 'classes_{}.txt'.format(self.room_type)), "r") as f:
            self.vocab['object_idx_to_name'] = f.readlines()
        # with open(os.path.join(self.root, 'classes_all.txt'), "r") as f:
        #     self.vocab['object_idx_to_name'] = f.readlines()
        
        #vocab 사전 안에 'pred_idx_to_name' key에 관계 이름 리스트 저장.
        with open(os.path.join(self.root, 'relationships.txt'), "r") as f:
            self.vocab['pred_idx_to_name'] = ['in\n']
            self.vocab['pred_idx_to_name']+=f.readlines()

        # list of relationship categories

        self.relationships = self.read_relationships(os.path.join(self.root, 'relationships.txt'))
        self.relationships_dict = dict(zip(self.relationships,range(1,len(self.relationships)+1)))
        self.relationships_dict_r = dict(zip(self.relationships_dict.values(), self.relationships_dict.keys()))
        # 사전형 및 리버스 사전형 생성 키값 1-관계갯수.

        if split == 'train_scans': # training set
            self.rel_json_file = os.path.join(self.root, 'relationships_{}_trainval.json'.format(self.room_type))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_{}_trainval.json'.format(self.room_type))
            self.box_normalized_stats = os.path.join(self.root, 'boxes_centered_stats_{}_trainval.txt'.format(self.room_type))
        else: # test set
            self.rel_json_file = os.path.join(self.root, 'relationships_{}_test.json'.format(self.room_type))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_{}_test.json'.format(self.room_type))
            self.box_normalized_stats = os.path.join(self.root, 'boxes_centered_stats_{}_test.txt'.format(self.room_type))


        self.relationship_json, self.objs_json, self.tight_boxes_json = \
                self.read_relationship_json(self.rel_json_file, self.box_json_file)

        for scene, infos in self.tight_boxes_json.items():
            for id, info in infos.items():
                if 'model_path' in info:
                    if info['model_path']:
                        info['model_path'] = root + info['model_path'][36:]
        # 3D-FUTURE-model/eddaa6f4-a280-426c-887d-447889d51461/raw_model.obj 이렇게 읽어옴.
        # root = '/home/commonscenes/FRONT/' 로 설정할 것. #TODO

        self.padding = 0.2
        self.eval = eval
        self.pass_scan_id = pass_scan_id
        self.shuffle_objs = shuffle_objs
        
        #CLIP 모델로 객체/관계 피처 임베딩 생성할 때 ~FRONT/visualization 으로 저장
        self.root_3dfront = root_3dfront
        if self.root_3dfront == '':
            self.root_3dfront = os.path.join(self.root, 'visualization')
            if not os.path.exists(self.root_3dfront):
                os.makedirs(self.root_3dfront)

        # not self.large 일때 객체들 카테고리 단순화 하기 위해 
        self.mapping_full2simple = json.load(open(os.path.join(self.root, "mapping.json"), "r"))

        with open(self.catfile, 'r') as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category
        # print(self.cat) #HJS floor
        # {'_scene_': '_scene_', 'armchair': 'armchair', 'bookshelf': 'bookshelf', 'cabinet': 'cabinet', 'ceiling_lamp': 'ceiling_lamp', 'chair': 'chair', 'chaise_longue_sofa': 'chaise_longue_sofa', 'children_cabinet': 'children_cabinet', 'chinese_chair': 'chinese_chair', 'coffee_table': 'coffee_table', 'console_table': 'console_table', 'corner_side_table': 'corner_side_table', 'desk': 'desk', 'dining_chair': 'dining_chair', 'dining_table': 'dining_table', 'double_bed': 'double_bed', 'dressing_chair': 'dressing_chair', 'dressing_table': 'dressing_table', 'kids_bed': 'kids_bed', 'l_shaped_sofa': 'l_shaped_sofa', 'lazy_sofa': 'lazy_sofa', 'lounge_chair': 'lounge_chair', 'loveseat_sofa': 'loveseat_sofa', 'multi_seat_sofa': 'multi_seat_sofa', 'nightstand': 'nightstand', 'pendant_lamp': 'pendant_lamp', 'round_end_table': 'round_end_table', 'shelf': 'shelf', 'single_bed': 'single_bed', 'sofa': 'sofa', 'stool': 'stool', 'table': 'table', 'tv_stand': 'tv_stand', 'wardrobe': 'wardrobe', 'wine_cabinet': 'wine_cabinet', 'floor': 'floor'}
        
        # self.classes = dict(zip(sorted(self.cat), range(len(self.cat)))) #Floor를 e와 g사이에 ID로 배정. 근데 뒤에 인덱스가 꼬임.
        self.classes = {key: idx for idx, key in enumerate(self.cat.keys())} #HJS floor 맨마지막 클래스 ID로 할당
        self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
        # print(self.classes)
        # {'_scene_': 0, 'armchair': 1, 'bookshelf': 2, 'cabinet': 3, 'ceiling_lamp': 4, 'chair': 5, 'chaise_longue_sofa': 6, 'children_cabinet': 7, 'chinese_chair': 8, 'coffee_table': 9, 'console_table': 10, 'corner_side_table': 11, 'desk': 12, 'dining_chair': 13, 'dining_table': 14, 'double_bed': 15, 'dressing_chair': 16, 'dressing_table': 17, 'floor': 18, 'kids_bed': 19, 'l_shaped_sofa': 20, 'lazy_sofa': 21, 'lounge_chair': 22, 'loveseat_sofa': 23, 'multi_seat_sofa': 24, 'nightstand': 25, 'pendant_lamp': 26, 'round_end_table': 27, 'shelf': 28, 'single_bed': 29, 'sofa': 30, 'stool': 31, 'table': 32, 'tv_stand': 33, 'wardrobe': 34, 'wine_cabinet': 35}

        #리스트 메서드 활용하려고 객체 리스트화
        points_classes = list(self.classes.keys())
        points_classes.remove('_scene_')

        # points_classes = ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'chaise_longue_sofa', 'children_cabinet', 'chinese_chair',
        #                       'coffee_table', 'console_table', 'corner_side_table', 'desk', 'dining_chair', 'dining_table', 'double_bed', 'dressing_chair',
        #                       'dressing_table', 'kids_bed', 'l_shaped_sofa', 'lazy_sofa', 'lounge_chair', 'loveseat_sofa', 'multi_seat_sofa', 'nightstand', 'pendant_lamp',
        #                       'round_end_table', 'shelf', 'single_bed', 'sofa', 'stool', 'table', 'tv_stand', 'wardrobe', 'wine_cabinet']

        # Why we do this here? It is because that we want to make each category evenly sampled during the diffusion
        # training. You can see the objects are classified/mapped into coarse categories, which will cause the number
        # of objects in each category are very different from each other. For example, Chairs are the most and lamps
        # are the fewest among all the objects. So when we sample a batch in more fine-grained classes, the problem
        # can be alleviated.
        
        # 세분화를 단순화 해주기 이전에 복구용으로 하나 만들어놓은 것
        self.vocab['object_idx_to_name_grained'] = self.vocab['object_idx_to_name']

        # 세분화된(fine-grained) 클래스 분류를 통해 동일한 확률로 샘플링. 객체 별 데이터셋의 수량 불균형 문제를 해결해줌.
        if not self.large:
            self.fine_grained_classes = dict(zip(sorted([voc.strip('\n') for voc in self.vocab['object_idx_to_name']]),range(len(self.vocab['object_idx_to_name']))))
            self.vocab['object_idx_to_name'] = [self.mapping_full2simple[voc.strip('\n')]+'\n' for voc in self.vocab['object_idx_to_name']]
            self.classes = dict(zip(sorted(list(set([voc.strip('\n') for voc in self.vocab['object_idx_to_name']]))),
                                        range(len(list(set(self.vocab['object_idx_to_name']))))))
            self.classes_r = dict(zip(self.classes.values(), self.classes.keys()))
            points_classes = list(set([self.mapping_full2simple[class_] for class_ in points_classes]))
        # set()은 중복된 항목 삭제
        # self.vocab['object_idx_to_name']는 이제 단순화(coarse-grained)된 정보만 갖고 있음.
        # self.vocab['object_idx_to_name']를 통해서 균일한 샘플링이 가능하다. lamp나 chair나 리스트에는 딱 하나만 있어서

        points_classes_idx = [self.classes[pc] for pc in points_classes]
        self.point_classes_idx = points_classes_idx + [0]
        # 왜 다시 0번 인덱스 (scene의 인덱스)를 추가하는 거지? #TODO

        self.sorted_cat_list = sorted(self.cat)
        self.files = {}
        
        #eval_type 종류: addition, removal, relationship 
        self.eval_type = eval_type

        # check if all shape features exist. If not they get generated here (once)
        if with_feats: 
            print('Assume you downloaded the DeepSDF codes and SDFs. If not, please download in README.md')
            # for index in tqdm(range(len(self))):
            #     self.__getitem__(index)
            self.recompute_feats = False #재연산이 False이면 SDF 생성을 위한 feature 연산이 필요없음.

        # check if all clip features exist. If not they get generated here (once)
        if self.with_CLIP:
            self.cond_model, preprocess = clip.load("ViT-B/32", device='cuda') #CLIP 모델 소환
            self.cond_model_cpu, preprocess_cpu = clip.load("ViT-B/32", device='cpu')
            print('loading CLIP')
            print(
                'Checking for missing clip feats. This can be slow the first time.')
            for index in tqdm(range(len(self))):
                self.__getitem__(index)
            # self.__getitem__(0) #HJS 디버깅용 데이터셋 확인용
            self.recompute_clip = False
        #len(self)는 __len__()에 의해 결정됨. 현재는 len(self.scans)라서 __getitem__()의 인덱스로 들어가도 문제 없음
        

    def read_relationship_json(self, json_file, box_json_file): #DONE
        """ Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)

        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for scan in data['scans']:

                relationships = []
                for relationship in scan["relationships"]:
                    relationship[2] -= 1 # relationships.txt에 있는 관계 인덱스 ID와 python의 인덱스 ID를 맞추기 위해 사용
                    relationships.append(relationship)

                # for every scan in rel json, we append the scan id
                rel[scan["scan"]] = relationships
                self.scans.append(scan["scan"])

                objects = {}
                boxes = {}
                for k, v in scan["objects"].items():
                    # if not self.large:
                    #     objects[int(k)] = self.mapping_full2simple[v]
                    # else:
                    objects[int(k)] = v

                    try:
                        boxes[int(k)] = {}
                        boxes[int(k)]['param7'] = box_data[scan["scan"]][k]["param7"] #pos_x,y,z rot_x,y,z angle(theta)
                        boxes[int(k)]['param7'][6] = boxes[int(k)]['param7'][6] #angle, theta
                        boxes[int(k)]['scale'] = box_data[scan["scan"]][k]["scale"] # size of bounding box
                    except Exception as e:
                        # probably box was not saved because there were 0 points in the instance!
                        print(e)
                    try:
                        boxes[int(k)]['model_path']  = box_data[scan["scan"]][k]["model_path"]
                    except Exception as e:
                        print(e)
                        continue
                boxes["scene_center"] = box_data[scan["scan"]]["scene_center"] #씬 가운데 값
                objs[scan["scan"]] = objects #"scan": "LivingDiningRoom-27430" => 이름을 키값으로 객체 정보 저장
                tight_boxes[scan["scan"]] = boxes #"LivingDiningRoom-27430" => 이름을 키값으로 바운딩박스 정보 저장
        return rel, objs, tight_boxes

    def read_relationships(self, read_file): #DONE
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships

    # 3D 포인트 클라우드 데이터를 정규화
    # 중심점에서 제일 먼 거리를 기준으로 normalization. 가장 거리가 먼 값은 1.
    def norm_points(self, p):
        centroid = np.mean(p, axis=0)
        m = np.max(np.sqrt(np.sum(p ** 2, axis=1)))
        p = (p - centroid) / float(m)
        return p

    def get_key(self, dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return None

    def __getitem__(self, index):
        scan_id = self.scans[index]

        # instance2label, the whole instance ids in this scene e.g. {1: 'floor', 2: 'wall', 3: 'picture', 4: 'picture'}
        instance2label = self.objs_json[scan_id]
        keys = list(instance2label.keys())

        if self.shuffle_objs:
            random.shuffle(keys)

        feats_in = None
        clip_feats_ins = None
        clip_feats_rel = None

        # If true, expected paths to saved clip features will be set here
        # root~/visulization에 CLIP 피처 저장 
        if self.with_CLIP:
            self.clip_feats_path = os.path.join(self.root_3dfront, scan_id,
                                                'CLIP_{}.pkl'.format(scan_id))
            if not self.large:
                self.clip_feats_path = os.path.join(self.root_3dfront, scan_id,
                                                        'CLIP_small_{}.pkl'.format(scan_id))
            if not os.path.exists(os.path.join(self.root_3dfront, scan_id)):
                os.makedirs(os.path.join(self.root_3dfront, scan_id))
            if self.recompute_clip:
                self.clip_feats_path += 'tmp'

        feats_path = self.root + "/DEEPSDF_reconstruction/Codes/" # for Graph-to-3D

        # Load points for debug
        # 재연산이 True일 때 혹은 DeepSDF 경로가 없을 때
        if self.with_feats and (not os.path.exists(feats_path) or self.recompute_feats):
            if scan_id in self.files: # Caching
                (points_list, points_norm_list, instances_list) = self.files[scan_id]
            else:
                points_list=np.array([]).reshape(-1,3)
                points_norm_list = np.array([]).reshape(-1, 3)
                instances_list=np.array([]).reshape(-1,1)
                for key_, value_ in self.tight_boxes_json[scan_id].items():
                    if isinstance(key_,int):
                        path = self.tight_boxes_json[scan_id][key_]["model_path"]
                        # object points
                        if path is not None:
                            raw_mesh = trimesh.load(path)
                            position = self.tight_boxes_json[scan_id][key_]["param7"][3:6]
                            theta = self.tight_boxes_json[scan_id][key_]["param7"][-1]
                            R = np.zeros((3, 3))
                            R[0, 0] = np.cos(theta)
                            R[0, 2] = -np.sin(theta)
                            R[2, 0] = np.sin(theta)
                            R[2, 2] = np.cos(theta)
                            R[1, 1] = 1.
                            points = raw_mesh.copy().vertices
                            point_norm = self.norm_points(points) #normliazed in each individual boxes
                            points = points.dot(R) + position # not centered yet
                        # floor points
                        else:
                            position = self.tight_boxes_json[scan_id][key_]["param7"][3:6]
                            l,w = self.tight_boxes_json[scan_id][key_]["param7"][0], self.tight_boxes_json[scan_id][key_]["param7"][2]
                            x = l * np.random.random(1000)+ position[0] - l/2
                            z = w * np.random.random(1000)+ position[2] - w/2
                            y = np.repeat(0,1000)
                            points = np.vstack((x,y,z)).transpose()
                            point_norm = self.norm_points(points)
                        points_list = np.concatenate((points_list, points), axis=0)
                        points_norm_list = np.concatenate((points_norm_list, point_norm), axis=0)
                        instances = np.repeat(key_, points.shape[0]).reshape(-1, 1)
                        instances_list = np.concatenate((instances_list, instances), axis=0)

                if self.fm.user_free > 5:
                    self.files[scan_id] = (points_list, points_norm_list, instances_list)

            print("shifting points")
            points_list = points_list - np.array(self.tight_boxes_json[scan_id]['scene_center']) # centered in the scene

        instance2mask = {}
        instance2mask[0] = 0

        cat_ids = []
        cat_ids_grained = []
        tight_boxes = []

        counter = 0

        instances_order = []
        selected_shapes = []
        obj_sdf_list = []

        # key: 1 of 1: 'floor' instance_id              keys: whole instance ids
        for key in keys:
            # get objects from the selected list of classes of 3dssg
            scene_instance_id = key
            scene_instance_class = instance2label[key]
            if not self.large:
                scene_class_id_grained = self.fine_grained_classes[scene_instance_class]
                scene_instance_class = self.mapping_full2simple[scene_instance_class]
                scene_class_id = self.classes[scene_instance_class] #인스턴스 이름으로 class id찾아내서 저장

            else:
                scene_class_id = self.classes[scene_instance_class] # class id in the entire dataset ids
            instance2mask[scene_instance_id] = counter + 1
            counter += 1

            # mask to cat:
            if (scene_class_id >= 0) and (scene_instance_id > 0):
                selected_shapes.append(True)
                cat_ids.append(scene_class_id)
                if not self.large:
                    cat_ids_grained.append(scene_class_id_grained)
                else:
                    cat_ids_grained.append(scene_class_id)
                bbox = np.array(self.tight_boxes_json[scan_id][key]['param7'].copy())
                bbox[3:6] -= np.array(self.tight_boxes_json[scan_id]['scene_center'])

                instances_order.append(key)
                bins = np.linspace(np.deg2rad(-180), np.deg2rad(180), 24)
                angle = np.digitize(bbox[6], bins)
                bbox = normalize_box_params(bbox,file=self.box_normalized_stats)
                bbox[6] = angle

                tight_boxes.append(bbox)
            
            if self.use_SDF:
                if self.tight_boxes_json[scan_id][key]["model_path"] is None:
                    obj_sdf_list.append(torch.zeros((1, self.sdf_res, self.sdf_res, self.sdf_res))) # floor
                else:
                    sdf_path = os.path.join(self.tight_boxes_json[scan_id][key]["model_path"].replace('3D-FUTURE-model', "3D-FUTURE-SDF").rsplit('/', 1)[0], 'ori_sample_grid.h5')
                    h5_f = h5py.File(sdf_path, 'r')
                    obj_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
                    sdf = torch.Tensor(obj_sdf).view(1, self.sdf_res, self.sdf_res, self.sdf_res)
                    sdf = torch.clamp(sdf, min=-0.2, max=0.2)
                    obj_sdf_list.append(sdf)
            else:
                obj_sdf_list = None

        if self.with_CLIP:
            # If precomputed features exist, we simply load them
            if os.path.exists(self.clip_feats_path):
                clip_feats_dic = pickle.load(open(self.clip_feats_path, 'rb'))

                clip_feats_ins = clip_feats_dic['instance_feats']
                clip_feats_order = np.asarray(clip_feats_dic['instance_order'])
                ordered_feats = []
                for inst in instances_order:
                    clip_feats_in_instance = inst == clip_feats_order
                    ordered_feats.append(clip_feats_ins[:-1][clip_feats_in_instance])
                ordered_feats.append(clip_feats_ins[-1][np.newaxis,:]) # should be room's feature
                clip_feats_ins = list(np.concatenate(ordered_feats, axis=0))
                clip_feats_rel = clip_feats_dic['rel_feats']

        if self.with_feats:
            # If precomputed features exist, we simply load them
            latents = []
            #for key_, value_ in self.tight_boxes_json[scan_id].items():
            for key_ in instances_order: # get the objects in order
                if isinstance(key_, int):
                    path = self.tight_boxes_json[scan_id][key_]["model_path"]
                    if path is None:
                        latent_code = np.zeros([1, 256]) #for the floor, latent_code.shape[1]=256
                        #print("why is it none?")
                    else:
                        model_id = path.split('/')[-2]
                        latent_code_path = feats_path + model_id + "/sdf.pth"
                        latent_code = torch.load(latent_code_path, map_location="cpu")[0]
                        latent_code = latent_code.detach().numpy()
                    latents.append(latent_code)
            latents.append(np.zeros([1, 256])) # for the room shape
            feats_in = list(np.concatenate(latents, axis=0))

        triples = []
        words = []
        rel_json = self.relationship_json[scan_id]
        for r in rel_json: # create relationship triplets from data
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():
                subject = instance2mask[r[0]] - 1
                object = instance2mask[r[1]] - 1
                predicate = r[2] + 1
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
                    if not self.large:
                        words.append(self.mapping_full2simple[instance2label[r[0]]] + ' ' + r[3] + ' ' + self.mapping_full2simple[instance2label[r[1]]])
                    else:
                        words.append(instance2label[r[0]]+' '+r[3]+' '+instance2label[r[1]]) # TODO check
            else:
                continue

        if self.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            scene_idx = len(cat_ids)
            for i, ob in enumerate(cat_ids):
                triples.append([i, 0, scene_idx]) 
                words.append(self.get_key(self.classes, ob) + ' ' + 'in' + ' ' + 'room')
            cat_ids.append(0) # TODO check
            cat_ids_grained.append(0)
            # dummy scene box
            tight_boxes.append([-1, -1, -1, -1, -1, -1, -1])
            if self.use_SDF:
                obj_sdf_list.append(torch.zeros((1, self.sdf_res, self.sdf_res, self.sdf_res))) # _scene_

        # if features are requested but the files don't exist, we run all loaded pointclouds through clip
        # to compute them and then save them for future usage
        if self.with_CLIP and (not os.path.exists(self.clip_feats_path) or clip_feats_ins is None) and self.cond_model is not None:
            num_cat = len(cat_ids)
            feats_rel = {}
            obj_cat = []
            with torch.no_grad():
                for i in range(num_cat - 1):
                    obj_cat.append(self.get_key(self.classes, cat_ids[i]))
                obj_cat.append('room') # TODO check
                #CLIP을 사용하겠다고 했는데 피처 파일이 없는 경우 => 객체 피처 생성
                text_obj = clip.tokenize(obj_cat).to('cuda')
                #encode_text() => 텍스트를 벡터 임베딩으로 변환하는 CLIP 함수
                feats_ins = self.cond_model.encode_text(text_obj).detach().cpu().numpy()
                #관계 피처 생성
                text_rel = clip.tokenize(words).to('cuda')
                rel = self.cond_model.encode_text(text_rel).detach().cpu().numpy()
                for i in range(len(words)):
                    feats_rel[words[i]] = rel[i]

            clip_feats_in = {}
            clip_feats_in['instance_feats'] = feats_ins
            clip_feats_in['instance_order'] = instances_order
            clip_feats_in['rel_feats'] = feats_rel
            path = os.path.join(self.clip_feats_path)
            if self.recompute_clip:
                path = path[:-3]

            pickle.dump(clip_feats_in, open(path, 'wb'))
            clip_feats_ins = list(clip_feats_in['instance_feats'])
            clip_feats_rel = clip_feats_in['rel_feats']

        output = {}
        # prepare outputs
        #다 숫자리스트로 저장되고 이후에 넘파이로 변환.
        output['encoder'] = {}
        output['encoder']['objs'] = cat_ids
        output['encoder']['objs_grained'] = cat_ids_grained # not needed for encoder
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = tight_boxes
        output['encoder']['words'] = words

        if self.with_CLIP:
            output['encoder']['text_feats'] = clip_feats_ins
            clip_feats_rel_new = []
            if clip_feats_rel != None:
                for word in words:
                    clip_feats_rel_new.append(clip_feats_rel[word])
                output['encoder']['rel_feats'] = clip_feats_rel_new

        if self.with_feats:
            output['encoder']['feats'] = feats_in

        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:
            if not self.eval: #트레이닝일 때. 평가가 아닐 때.
                if self.with_changes: #랜점으로 조작해서 데이터 변형
                    output['manipulate']['type'] = ['relationship', 'addition', 'none'][
                        np.random.randint(3)]  # removal is trivial - so only addition and rel change
                else:
                    output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                    else:
                        output['manipulate']['type'] = 'none'
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'])
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        output['manipulate']['type'] = 'none'
            else: #평가 단계일 때.
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        return -1
        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64)) # this is changed
        output['encoder']['objs_grained'] = torch.from_numpy(np.array(output['encoder']['objs_grained'], dtype=np.int64)) # this doesn't matter
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['encoder']['text_feats'] = torch.from_numpy(np.array(output['encoder']['text_feats'], dtype=np.float32)) # this is changed
            output['encoder']['rel_feats'] = torch.from_numpy(np.array(output['encoder']['rel_feats'], dtype=np.float32))
        if self.with_feats:
            output['encoder']['feats'] = torch.from_numpy(np.array(output['encoder']['feats'], dtype=np.float32))

        # these two should have the same amount.
        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))
        output['decoder']['objs_grained'] = torch.from_numpy(np.array(output['decoder']['objs_grained'], dtype=np.int64))

        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64)) # this is changed
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.with_CLIP:
            output['decoder']['text_feats'] = torch.from_numpy(np.array(output['decoder']['text_feats'], dtype=np.float32))
            output['decoder']['rel_feats'] = torch.from_numpy(np.array(output['decoder']['rel_feats'], dtype=np.float32)) # this is changed
        if self.with_feats:
            output['decoder']['feats'] = torch.from_numpy(np.array(output['decoder']['feats'], dtype=np.float32))
        if self.use_SDF:
            output['decoder']['sdfs'] = torch.stack(obj_sdf_list,dim=0)

        output['scan_id'] = scan_id
        output['instance_id'] = instances_order

        return output

    def remove_node_and_relationship(self, graph):
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        excluded = [self.classes['floor']]

        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)

        graph['objs'].pop(node_id)
        if self.with_feats:
            graph['feats'].pop(node_id)
        if self.with_CLIP:
            graph['text_feats'].pop(node_id)

        graph['boxes'].pop(node_id)

        to_rm = []
        for i,x in reversed(list(enumerate(graph['triples']))):
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)
                if self.with_CLIP:
                    graph['rel_feats'].pop(i)
                    graph['words'].pop(i)

        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        return node_id

    def modify_relship(self, graph, interpretable=False):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        '''15 same material as' '14 same super category as' '13 same style as' '12 symmetrical to' '11 shorter than' '10 taller than' '9 smaller than'
         '8 bigger than' '7 standing on' '6 above' '5 close by' '4 behind' '3 front' '2 right' '1 left'
         '0: none'''
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [1, 2, 3, 4, 8, 9, 10, 11]
        rel_dict = {1: 'left', 2: 'right', 3: 'front', 4: 'behind', 8: 'bigger than', 9: 'smaller than', 10: 'taller than', 11: 'shorter than'}
        did_change = False
        trials = 0
        excluded = [self.classes['floor']]
        eval_excluded = [self.classes['floor']]

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            trials += 1

            if pred == 0:
                continue
            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            if interpretable:
                if graph['objs'][obj] in eval_excluded or graph['objs'][sub] in eval_excluded: # don't use the floor
                    continue
                if pred not in interpretable_rels:
                    continue
                else:
                    new_pred = self.relationships_dict[changed_relationships_dict[self.relationships_dict_r[pred]]]
            else:

                #close by, above, standing on, same material......
                if self.relationships_dict_r[pred] == changed_relationships_dict[self.relationships_dict_r[pred]]:
                    new_pred = np.random.randint(1, 12)
                    #didnt change
                    if new_pred == pred:
                        continue
                # left, right, front, behind, bigger, smaller.....
                else:
                    new_pred = self.relationships_dict[changed_relationships_dict[self.relationships_dict_r[pred]]]


            graph['words'][idx] = graph['words'][idx].replace(self.relationships_dict_r[graph['triples'][idx][1]],self.relationships_dict_r[new_pred])
            graph['changed_id'] = idx
            graph['triples'][idx][1] = new_pred

            did_change = True
        return idx, (sub, obj), did_change

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            return len(self.scans)

    # __getitem__()에서 나온 데이터셋을 배치로 묶어서 모델이 학습할 수 있도록 변환
    def collate_fn_vaegan(self, batch, use_points=False):
        """
        Collate function to be used when wrapping a RIODatasetSceneGraph in a
        DataLoader. Returns a dictionary
        """

        out = {}

        out['scene_points'] = []
        out['scan_id'] = []
        out['instance_id'] = []

        out['missing_nodes'] = []
        out['missing_nodes_decoder'] = []
        out['manipulated_nodes'] = []
        global_node_id = 0
        global_dec_id = 0
        for i in range(len(batch)):
            if batch[i] == -1:
                return -1
            # notice only works with single batches
            out['scan_id'].append(batch[i]['scan_id'])
            out['instance_id'].append(batch[i]['instance_id'])

            if batch[i]['manipulate']['type'] == 'addition':
                out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added'])
                out['missing_nodes_decoder'].append(global_dec_id + batch[i]['manipulate']['added'])
            elif batch[i]['manipulate']['type'] == 'relationship':
                rel, (sub, obj) = batch[i]['manipulate']['relship']
                out['manipulated_nodes'].append(global_dec_id + sub)
                out['manipulated_nodes'].append(global_dec_id + obj)

            if 'scene' in batch[i]:
                out['scene_points'].append(batch[i]['scene'])

            global_node_id += len(batch[i]['encoder']['objs'])
            global_dec_id += len(batch[i]['decoder']['objs'])

        for key in ['encoder', 'decoder']:
            all_objs, all_boxes, all_triples = [], [], []
            all_objs_grained = []
            all_obj_to_scene, all_triple_to_scene = [], []
            all_points = []
            all_sdfs = []
            all_feats = []
            all_text_feats = []
            all_rel_feats = []

            obj_offset = 0

            for i in range(len(batch)):
                if batch[i] == -1:
                    print('this should not happen')
                    continue
                (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

                if 'points' in batch[i][key]:
                    all_points.append(batch[i][key]['points'])
                if 'sdfs' in batch[i][key]:
                    all_sdfs.append(batch[i][key]['sdfs'])
                if 'feats' in batch[i][key]:
                    all_feats.append(batch[i][key]['feats'])
                if 'text_feats' in batch[i][key]:
                    all_text_feats.append(batch[i][key]['text_feats'])
                if 'rel_feats' in batch[i][key]:
                    if 'changed_id' in batch[i][key]:
                        idx = batch[i][key]['changed_id']
                        if self.with_CLIP:
                            text_rel = clip.tokenize(batch[i][key]['words'][idx]).to('cpu')
                            rel = self.cond_model_cpu.encode_text(text_rel).detach().numpy()
                            batch[i][key]['rel_feats'][idx] = torch.from_numpy(np.squeeze(rel))


                    all_rel_feats.append(batch[i][key]['rel_feats'])

                num_objs, num_triples = objs.size(0), triples.size(0)

                all_objs.append(batch[i][key]['objs'])
                all_objs_grained.append(batch[i][key]['objs_grained'])
                all_boxes.append(boxes)

                if triples.dim() > 1:
                    triples = triples.clone()
                    triples[:, 0] += obj_offset
                    triples[:, 2] += obj_offset

                    all_triples.append(triples)
                    all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

                all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

                obj_offset += num_objs

            all_objs = torch.cat(all_objs)
            all_objs_grained = torch.cat(all_objs_grained)
            all_boxes = torch.cat(all_boxes)

            all_obj_to_scene = torch.cat(all_obj_to_scene)

            if len(all_triples) > 0:
                all_triples = torch.cat(all_triples)
                all_triple_to_scene = torch.cat(all_triple_to_scene)
            else:
                return -1

            outputs = {'objs': all_objs,
                       'objs_grained': all_objs_grained,
                       'tripltes': all_triples,
                       'boxes': all_boxes,
                       'obj_to_scene': all_obj_to_scene,
                       'triple_to_scene': all_triple_to_scene}

            if len(all_sdfs) > 0:
                outputs['sdfs'] = torch.cat(all_sdfs)
            if len(all_points) > 0:
                all_points = torch.cat(all_points)
                outputs['points'] = all_points

            if len(all_feats) > 0:
                all_feats = torch.cat(all_feats)
                outputs['feats'] = all_feats
            if len(all_text_feats) > 0:
                all_text_feats = torch.cat(all_text_feats)
                outputs['text_feats'] = all_text_feats
            if len(all_rel_feats) > 0:
                all_rel_feats = torch.cat(all_rel_feats)
                outputs['rel_feats'] = all_rel_feats
            out[key] = outputs

        return out


    def collate_fn_vaegan_points(self,batch):
        """ Wrapper of the function collate_fn_vaegan to make it also return points
        """
        # 단순히 포인트 클라우드가 있을 때 self.collate_fn_vaegan()를 실행하기 위한 랩핑 함수 
        return self.collate_fn_vaegan(batch, use_points=True)


if __name__ == "__main__":
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
    a = dataset[0]
    
    print("Scan ID:", a['scan_id'])
    print("객체 개수:", len(a['encoder']['objs']))
    print("관계 개수:", len(a['encoder']['triples']))
    print("객체 ID 목록:", a['encoder']['objs']) #classes_bedroom.txt에서 객체 ID값을 갖고옴.
    # print("관계 리스트:", a['encoder']['triples'])
    print("바운딩 박스:", a['encoder']['boxes'].shape)
    
    if 'text_feats' in a['encoder']:
        print("CLIP 객체 특징 벡터 크기:", a['encoder']['text_feats'].shape)
    if 'rel_feats' in a['encoder']:
        print("CLIP 관계 특징 벡터 크기:", a['encoder']['rel_feats'].shape)

    for x in ['encoder', 'decoder']:
        en_obj = a[x]['objs'].cpu().numpy().astype(np.int32)
        en_triples = a[x]['triples'].cpu().numpy().astype(np.int32)
        #instance
        sub = en_triples[:,0]
        obj = en_triples[:,2]
        #cat
        instance_ids = np.array(sorted(list(set(sub.tolist() + obj.tolist())))) #0-n
        cat_ids = en_obj[instance_ids]
        texts = [dataset.classes_r[cat_id] for cat_id in cat_ids]
        objs = dict(zip(instance_ids.tolist(),texts))
        objs = {str(key): value for key, value in objs.items()}
        txt_list = []
        for rel in en_triples[:,1]:
            if rel == 0:
                txt = 'in'
                txt_list.append(txt)
                continue
            txt = dataset.relationships_dict_r[rel]
            txt_list.append(txt)
        txt_list = np.array(txt_list)
        rel_list = np.vstack((sub,obj,en_triples[:,1],txt_list)).transpose()

    # print("객체 정보 (ID → 이름 변환):")
    # for obj_id, obj_name in objs.items():
    #     print(f"  ID {obj_id}: {obj_name}")

    # for i in range(len(rel_list)):
    #     sub, obj, rel_id, rel_name = rel_list[i]
    #     print(f"  {objs[str(sub)]} --({rel_name})--> {objs[str(obj)]}")