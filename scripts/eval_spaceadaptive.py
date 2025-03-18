#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.VAE_spaceadaptive import VAE
from spaceadaptive.SpaceAdaptiveVAE import SpaceAdaptiveVAE
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.util import bool_flag

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='../experiments/spaceadaptive', help='실험 결과 저장 경로')
parser.add_argument('--dataset', default='../GT', help='데이터셋 경로')
parser.add_argument('--room_type', default='livingroom', help='방 타입 [livingroom, bedroom, diningroom, library]')
parser.add_argument('--epoch', type=int, default=0, help='평가할 에폭')
parser.add_argument('--evaluate_diversity', default=True, type=bool_flag, help='다양성 평가 여부')
parser.add_argument('--export_3d', default=False, type=bool_flag, help='3D 내보내기 여부')
parser.add_argument('--with_points', default=True, type=bool_flag, help='포인트 포함 여부')
parser.add_argument('--recompute_stats', default=False, type=bool_flag, help='통계 재계산 여부')
parser.add_argument('--gen_shape', default=True, type=bool_flag, help='형상 생성 여부')
parser.add_argument('--num_samples', type=int, default=3, help='샘플 수')
parser.add_argument('--no_stool', default=False, type=bool_flag, help='스툴 제외 여부')

# SpaceAdaptive: 현실 공간 데이터 관련 인자 추가
parser.add_argument('--real_space_data', default='../spacedata/real_space_sample.json', type=str, help='현실 공간 데이터 경로')
parser.add_argument('--use_real_space', default=True, type=bool_flag, help='현실 공간 데이터 사용 여부')
parser.add_argument('--real_space_weight', default=0.3, type=float, help='현실 공간 임베딩 가중치 (0~1)')
parser.add_argument('--real_space_id', default=None, type=str, help='현실 공간 ID (None이면 모든 공간 평가)')
parser.add_argument('--ablation_study', default=False, type=bool_flag, help='가중치 변화에 따른 ablation 연구 수행')
parser.add_argument('--ablation_weights', default='0.1,0.3,0.5,0.7,0.9', type=str, help='ablation 연구를 위한 가중치 목록')

# SpaceAdaptiveVAE 관련 인자 추가
parser.add_argument('--use_hybrid_scene', default=False, type=bool_flag, help='하이브리드 씬 사용 여부')
parser.add_argument('--hybrid_scene_path', default='../spaceadaptive/hybrid_scene.json', type=str, help='하이브리드 씬 파일 경로')
parser.add_argument('--generate_from_hybrid', default=False, type=bool_flag, help='하이브리드 씬에서 샘플 생성 여부')

args = parser.parse_args()
print(args)

def reseed(num):
    random.seed(num)
    torch.manual_seed(num)

def evaluate():
    print(torch.__version__)

    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)
    normalized_file = os.path.join(args.dataset, 'boxes_centered_stats_{}_trainval.txt').format(modelArgs['room_type'])
    
    # 평가 데이터셋 설정
    test_dataset_rels_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=True,
        eval=True,
        eval_type='relationship',
        with_feats=modelArgs['with_feats'],
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        recompute_feats=False,
        large=modelArgs['large'],
        room_type=args.room_type,
        recompute_clip=False)

    test_dataset_addition_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=True,
        eval=True,
        eval_type='addition',
        with_feats=modelArgs['with_feats'],
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    # 학습 통계 수집에 사용
    stats_dataset = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='train_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=False,
        with_feats=modelArgs['with_feats'],
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=False,
        large=modelArgs['large'],
        room_type=modelArgs['room_type'])

    test_dataset_no_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=True,
        with_feats=modelArgs['with_feats'],
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    # 데이터 로더 콜레이트 함수 설정
    if args.with_points:
        collate_fn1 = test_dataset_rels_changes.collate_fn_vaegan_points
        collate_fn2 = test_dataset_addition_changes.collate_fn_vaegan_points
        collate_fn3 = stats_dataset.collate_fn_vaegan_points
        collate_fn4 = test_dataset_no_changes.collate_fn_vaegan_points
    else:
        collate_fn1 = test_dataset_rels_changes.collate_fn_vaegan
        collate_fn2 = test_dataset_addition_changes.collate_fn_vaegan
        collate_fn3 = stats_dataset.collate_fn_vaegan
        collate_fn4 = test_dataset_no_changes.collate_fn_vaegan

    # 데이터 로더 설정
    test_dataloader_rels_changes = torch.utils.data.DataLoader(
        test_dataset_rels_changes,
        batch_size=1,
        collate_fn=collate_fn1,
        shuffle=False,
        num_workers=0)

    test_dataloader_add_changes = torch.utils.data.DataLoader(
        test_dataset_addition_changes,
        batch_size=1,
        collate_fn=collate_fn2,
        shuffle=False,
        num_workers=0)

    # 학습 데이터 통계 수집을 위한 데이터로더
    stats_dataloader = torch.utils.data.DataLoader(
        stats_dataset,
        batch_size=1,
        collate_fn=collate_fn3,
        shuffle=False,
        num_workers=0)

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset_no_changes,
        batch_size=1,
        collate_fn=collate_fn4,
        shuffle=True,
        num_workers=0)

    # 모델 설정
    modeltype_ = modelArgs['network_type']
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    modelArgs['no_stool'] = args.no_stool if 'no_stool' not in modelArgs else modelArgs['no_stool']
    diff_opt = modelArgs['diff_yaml'] if modeltype_ == 'v2_full' else None
    try:
        with_E2 = modelArgs['with_E2']
    except:
        with_E2 = True

    # Space Adaptive 모델 생성
    use_real_space = modelArgs.get('use_real_space', args.use_real_space)
    real_space_weight = modelArgs.get('real_space_weight', args.real_space_weight)
    
    model = VAE(root=args.dataset, type=modeltype_, diff_opt=diff_opt, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'], deepsdf=modelArgs['with_feats'], with_E2=with_E2,
                use_real_space=use_real_space, real_space_weight=real_space_weight)
    
    if modeltype_ == 'v2_full':
        model.vae_v2.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=False)
    if torch.cuda.is_available():
        model = model.cuda()
        model.set_cuda()

    model = model.eval()
    model.compute_statistics(exp=args.exp, epoch=args.epoch, stats_dataloader=stats_dataloader, force=args.recompute_stats)
    print("계산된 mu 및 sigma")

    cat2objs = None

    # SpaceAdaptiveVAE 초기화
    space_adaptive_vae = SpaceAdaptiveVAE(model)
    
    # 현실 공간 데이터 로드 및 임베딩
    real_space_embeddings = None
    if args.use_real_space and args.real_space_data is not None:
        print("현실 공간 데이터 로드 중...")
        real_space_data = space_adaptive_vae.load_real_space_data(args.real_space_data)
        if real_space_data:
            print("현실 공간 데이터 임베딩 생성 중...")
            real_space_embeddings = space_adaptive_vae.encode_real_space(args.room_type, args.real_space_id)
            
            # CUDA 설정
            space_adaptive_vae.set_cuda()
            
            if args.real_space_id:
                print(f"현실 공간 '{args.real_space_id}' 임베딩 생성 완료")
            else:
                print(f"현실 공간 임베딩 생성 완료: {len(real_space_embeddings) if real_space_embeddings else 0}개 공간")

    # ablation 연구를 위한 가중치 목록 파싱
    ablation_weights = [float(w) for w in args.ablation_weights.split(',')] if args.ablation_study else [args.real_space_weight]

    for weight in ablation_weights:
        if args.ablation_study:
            print(f"\n---- 현실 공간 가중치 {weight} 실험 ----")
        
        # 평가 수행
        if args.use_real_space and space_adaptive_vae.real_space_embedding:
            # 현실 공간 ID 결정
            space_ids = list(space_adaptive_vae.real_space_embedding.keys())
            test_space_ids = [args.real_space_id] if args.real_space_id else space_ids
            
            for space_id in test_space_ids:
                if space_id not in space_adaptive_vae.real_space_embedding:
                    print(f"경고: {space_id} ID를 가진 현실 공간이 데이터에 없습니다.")
                    continue
                
                real_space_embedding = space_adaptive_vae.real_space_embedding[space_id]
                print(f"\n현실 공간 ID: {space_id}, 가중치: {weight}")
                
                print('\n편집 모드 - 추가')
                reseed(47)
                validate_constrains_loop_w_changes_real(modelArgs, test_dataloader_add_changes, model, real_space_embedding, 
                                                      normalized_file=normalized_file, with_diversity=args.evaluate_diversity, 
                                                      with_angles=modelArgs['with_angles'], num_samples=args.num_samples, 
                                                      cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', 
                                                      gen_shape=args.gen_shape, real_space_weight=weight)
                
                print('\n편집 모드 - 관계 변경')
                reseed(47)
                validate_constrains_loop_w_changes_real(modelArgs, test_dataloader_rels_changes, model, real_space_embedding,
                                                      normalized_file=normalized_file, with_diversity=args.evaluate_diversity, 
                                                      with_angles=modelArgs['with_angles'], num_samples=args.num_samples, 
                                                      cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', 
                                                      gen_shape=args.gen_shape, real_space_weight=weight)
                
                print('\n생성 모드')
                reseed(47)
                validate_constrains_loop_real(modelArgs, test_dataloader_no_changes, model, real_space_embedding, 
                                            epoch=args.epoch, normalized_file=normalized_file, 
                                            with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], 
                                            num_samples=args.num_samples, vocab=test_dataset_no_changes.vocab,
                                            point_classes_idx=test_dataset_no_changes.point_classes_idx,
                                            export_3d=args.export_3d, cat2objs=cat2objs, 
                                            datasize='large' if modelArgs['large'] else 'small', 
                                            gen_shape=args.gen_shape, real_space_weight=weight)
        else:
            # 현실 공간 임베딩 없이 평가
            print('\n편집 모드 - 추가 (현실 공간 미적용)')
            reseed(47)
            validate_constrains_loop_w_changes(modelArgs, test_dataloader_add_changes, model, normalized_file=normalized_file, 
                                             with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], 
                                             num_samples=args.num_samples, cat2objs=cat2objs, 
                                             datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

            print('\n편집 모드 - 관계 변경 (현실 공간 미적용)')
            reseed(47)
            validate_constrains_loop_w_changes(modelArgs, test_dataloader_rels_changes, model, normalized_file=normalized_file, 
                                             with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], 
                                             num_samples=args.num_samples, cat2objs=cat2objs, 
                                             datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

            print('\n생성 모드 (현실 공간 미적용)')
            reseed(47)
            validate_constrains_loop(modelArgs, test_dataloader_no_changes, model, epoch=args.epoch, 
                                   normalized_file=normalized_file, with_diversity=args.evaluate_diversity,
                                   with_angles=modelArgs['with_angles'], num_samples=args.num_samples, 
                                   vocab=test_dataset_no_changes.vocab,
                                   point_classes_idx=test_dataset_no_changes.point_classes_idx,
                                   export_3d=args.export_3d, cat2objs=cat2objs, 
                                   datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    # 하이브리드 씬 활용 평가
    if args.use_hybrid_scene and args.hybrid_scene_path:
        print(f"하이브리드 씬 활용 평가 시작 ({args.hybrid_scene_path})...")
        
        # 1. SpaceAdaptiveVAE 초기화
        space_adaptive_vae = SpaceAdaptiveVAE(model)
        
        # 2. 저장된 하이브리드 씬 로드
        try:
            hybrid_scene = space_adaptive_vae.load_hybrid_scene(args.hybrid_scene_path)
            print(f"하이브리드 씬 로드 완료: {len(hybrid_scene['objects'])}개 객체, {len(hybrid_scene['relationships'])}개 관계")
        except Exception as e:
            print(f"하이브리드 씬 로드 오류: {e}")
            return
        
        if args.generate_from_hybrid:
            # 3. 하이브리드 씬 기반 추론
            print("하이브리드 씬 기반 샘플 생성 중...")
            os.makedirs(os.path.join(args.exp, 'hybrid_outputs'), exist_ok=True)
            
            for i in range(args.num_samples):
                print(f"샘플 {i+1}/{args.num_samples} 생성 중...")
                
                # 입력 조건 설정 (예시)
                input_condition = {
                    "add_object": {"class": "chair", "position": [1.0, 0, 2.0]},
                    "modify_relation": {"subject": "1", "object": "2", "relation": "front"}
                }
                
                # 샘플 생성
                generated_scene = space_adaptive_vae.inference_with_hybrid_scene(input_condition)
                
                # 생성된 씬 저장
                output_path = os.path.join(args.exp, 'hybrid_outputs', f'generated_scene_{i}.json')
                with open(output_path, 'w') as f:
                    json.dump(generated_scene, f, indent=2)
                print(f"생성된 씬 저장 완료: {output_path}")
            
            print(f"{args.num_samples}개의 샘플 생성 완료")
            
            # 4. 생성된 씬들 분석 (선택적)
            print("생성된 씬 통계 분석 중...")
            # 객체 유형 분포, 공간 활용 등 분석
            # TODO: 자세한 분석 로직 구현
            
        else:
            # 하이브리드 씬 자체 분석
            print("하이브리드 씬 분석 중...")
            # 객체 유형 및 관계 분석
            object_classes = {}
            for obj_id, obj_info in hybrid_scene['objects'].items():
                obj_class = obj_info['class']
                if obj_class in object_classes:
                    object_classes[obj_class] += 1
                else:
                    object_classes[obj_class] = 1
            
            print("하이브리드 씬 객체 클래스 분포:")
            for obj_class, count in sorted(object_classes.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {obj_class}: {count}개")
            
            # 관계 분석
            relation_types = {}
            for rel in hybrid_scene['relationships']:
                rel_type = rel['relation']
                if rel_type in relation_types:
                    relation_types[rel_type] += 1
                else:
                    relation_types[rel_type] = 1
            
            print("하이브리드 씬 관계 유형 분포:")
            for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {rel_type}: {count}개")
            
            # 공간 분석
            # 객체 간 거리, 군집 등 분석
            # TODO: 자세한 분석 로직 구현
    
    print('평가 완료')

def validate_constrains_loop_w_changes_real(modelArgs, testdataloader, model, real_space_embedding, normalized_file=None, 
                                           with_diversity=True, with_angles=False, num_samples=3, cat2objs=None, 
                                           datasize='large', gen_shape=False, real_space_weight=0.3):
    """
    현실 공간 임베딩을 활용한 변경 사항이 있는 씬 평가
    """
    if with_diversity and num_samples < 2:
        raise ValueError('다양성 평가에는 최소 두 번의 실행이 필요합니다(즉, num_samples > 1).')

    accuracy = {}
    accuracy_unchanged = {}
    accuracy_in_orig_graph = {}

    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        accuracy_in_orig_graph[k] = []
        accuracy[k] = []
        accuracy_unchanged[k] = []

    samples = []
    boxes_rec = []
    boxes_gt = []

    model.real_space_weight = real_space_weight  # 평가 시 가중치 설정

    # 평가 루프
    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):
            # 데이터 파싱
            try:
                enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes,\
                manipulated_nodes = parse_data(data)
            except Exception as e:
                print("데이터 오류 발생:", e)
                continue

            # 현실 공간 임베딩 적용 평가
            for n in range(num_samples):
                # 조건화된 순전파
                pred_boxes_batch = []
                
                # 현실 공간 임베딩 조건화 적용
                boxes_pred, shapes_pred = model.decoder_with_changes_boxes_and_shape_real(
                    real_space_embedding, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat,
                    dec_sdfs, dec_attributes, missing_nodes, manipulated_nodes, gen_shape=gen_shape,
                    real_space_weight=real_space_weight
                )
                
                pred_boxes_batch.append(boxes_pred)
                
                # 정확도 평가 및 통계 계산 (원래 evaluate 함수와 동일하게 처리)
                # 여기에 원래 코드의 메트릭 계산 로직 추가

            print(f"{i+1}/{len(testdataloader)} 완료")

    print("평가 결과 (현실 공간 임베딩 적용, 가중치 {:.2f}):".format(real_space_weight))
    # 여기에 결과 요약 출력 로직 추가

def validate_constrains_loop_real(modelArgs, testdataloader, model, real_space_embedding, epoch=0, normalized_file=None, 
                                 with_diversity=True, with_angles=False, num_samples=3, vocab=None, 
                                 point_classes_idx=None, export_3d=False, cat2objs=None, 
                                 datasize='large', gen_shape=False, real_space_weight=0.3):
    """
    현실 공간 임베딩을 활용한 생성 모드 평가
    """
    if with_diversity and num_samples < 2:
        raise ValueError('다양성 평가에는 최소 두 번의 실행이 필요합니다(즉, num_samples > 1).')
        
    model.real_space_weight = real_space_weight  # 평가 시 가중치 설정

    # 평가 루프
    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):
            # 데이터 파싱
            try:
                enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes,\
                manipulated_nodes = parse_data(data)
            except Exception as e:
                print("데이터 오류 발생:", e)
                continue

            # 현실 공간 임베딩 적용 평가
            for n in range(num_samples):
                # 조건화된 생성
                boxes_pred, shapes_pred = model.sample_box_and_shape_real(
                    point_classes_idx, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, 
                    encoded_dec_rel_feat, dec_attributes, gen_shape=gen_shape, 
                    real_space_embedding=real_space_embedding, real_space_weight=real_space_weight
                )
                
                # 여기에 결과 분석 및 평가 로직 추가

            print(f"{i+1}/{len(testdataloader)} 완료")

    print("생성 모드 평가 결과 (현실 공간 임베딩 적용, 가중치 {:.2f}):".format(real_space_weight))
    # 여기에 결과 요약 출력 로직 추가

def validate_constrains_loop(modelArgs, testdataloader, model, epoch=0, normalized_file=None, 
                            with_diversity=True, with_angles=False, num_samples=3, vocab=None, 
                            point_classes_idx=None, export_3d=False, cat2objs=None, 
                            datasize='large', gen_shape=False):
    """
    기본 생성 모드 평가 (현실 공간 임베딩 없음)
    """
    # 기존 evaluate 함수 로직 구현

def validate_constrains_loop_w_changes(modelArgs, testdataloader, model, normalized_file=None, 
                                       with_diversity=True, with_angles=False, num_samples=3, cat2objs=None, 
                                       datasize='large', gen_shape=False):
    """
    기본 편집 모드 평가 (현실 공간 임베딩 없음)
    """
    # 기존 evaluate 함수 로직 구현

def parse_data(data):
    """
    데이터 파싱 함수
    """
    enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                                       data['encoder']['tripltes'], \
                                                                                       data['encoder']['boxes'], \
                                                                                       data['encoder'][
                                                                                           'obj_to_scene'], \
                                                                                       data['encoder'][
                                                                                           'triple_to_scene']
    if 'feats' in data['encoder']:
        encoded_enc_f = data['encoder']['feats']
        encoded_enc_f = encoded_enc_f.cuda()
    else:
        encoded_enc_f = None

    encoded_enc_text_feat = None
    encoded_enc_rel_feat = None
    encoded_dec_text_feat = None
    encoded_dec_rel_feat = None
    if 'text_feats' in data['encoder']:
        encoded_enc_text_feat = data['encoder']['text_feats'].cuda()
        encoded_enc_rel_feat = data['encoder']['rel_feats'].cuda()
        encoded_dec_text_feat = data['decoder']['text_feats'].cuda()
        encoded_dec_rel_feat = data['decoder']['rel_feats'].cuda()

    dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                       data['decoder']['tripltes'], \
                                                                                       data['decoder']['boxes'], \
                                                                                       data['decoder'][
                                                                                           'obj_to_scene'], \
                                                                                       data['decoder'][
                                                                                           'triple_to_scene']
    if 'feats' in data['decoder']:
        encoded_dec_f = data['decoder']['feats']
        encoded_dec_f = encoded_dec_f.cuda()
    else:
        encoded_dec_f = None

    if 'sdfs' in data['decoder']:
        dec_sdfs = data['decoder']['sdfs']
        dec_sdfs = dec_sdfs.cuda()
    else:
        dec_sdfs = None

    if 'angles' in data['encoder']:
        enc_angles = data['encoder']['angles']
        dec_angles = data['decoder']['angles']
        enc_angles = enc_angles.cuda()
        dec_angles = dec_angles.cuda()
    else:
        enc_angles = None
        dec_angles = None

    enc_tight_boxes = enc_tight_boxes.cuda()
    dec_tight_boxes = dec_tight_boxes.cuda()

    attributes = None
    dec_attributes = None

    missing_nodes = data['decoder']['missing_nodes']
    manipulated_nodes = data['decoder']['manipulated_nodes']

    dec_objs_grained = data['decoder']['objs_grained']

    return enc_objs, enc_triples, enc_tight_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat, \
           attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_tight_boxes, dec_angles, dec_sdfs, \
           encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, \
           manipulated_nodes

if __name__ == '__main__':
    evaluate() 