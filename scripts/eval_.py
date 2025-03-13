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

from model.VAE import VAE
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.util import bool_flag

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='../experiments/test', help='실험 결과 저장 경로')
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

# 현실 공간 데이터 관련 인자 추가
parser.add_argument('--real_space_data', default=None, type=str, help='현실 공간 데이터 경로')
parser.add_argument('--use_real_space', default=False, type=bool_flag, help='현실 공간 데이터 사용 여부')
parser.add_argument('--real_space_weight', default=0.3, type=float, help='현실 공간 임베딩 가중치 (0~1)')
parser.add_argument('--real_space_id', default="MasterBedroom-33296", type=str, help='현실 공간 ID')

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

    modeltype_ = modelArgs['network_type']
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    modelArgs['no_stool'] = args.no_stool if 'no_stool' not in modelArgs else modelArgs['no_stool']
    diff_opt = modelArgs['diff_yaml'] if modeltype_ == 'v2_full' else None
    try:
        with_E2 = modelArgs['with_E2']
    except:
        with_E2 = True

    model = VAE(root=args.dataset, type=modeltype_, diff_opt=diff_opt, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'], deepsdf=modelArgs['with_feats'], with_E2=with_E2)
    if modeltype_ == 'v2_full':
        model.vae_v2.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=False)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    model.compute_statistics(exp=args.exp, epoch=args.epoch, stats_dataloader=stats_dataloader, force=args.recompute_stats)
    print("계산된 mu 및 sigma")

    cat2objs = None

    # 현실 공간 데이터 로드 및 임베딩
    real_space_embedding = None
    if args.use_real_space and args.real_space_data is not None:
        print("현실 공간 데이터 로드 중...")
        real_space_data = model.load_real_space_data(args.real_space_data)
        real_space_embedding = model.embed_real_space(real_space_data, space_id=args.real_space_id)
        print("현실 공간 임베딩 생성 완료")

    print('\n편집 모드 - 추가')
    reseed(47)
    if args.use_real_space and real_space_embedding is not None:
        validate_constrains_loop_w_changes_real(modelArgs, test_dataloader_add_changes, model, real_space_embedding, 
                                               normalized_file=normalized_file, with_diversity=args.evaluate_diversity, 
                                               with_angles=modelArgs['with_angles'], num_samples=args.num_samples, 
                                               cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', 
                                               gen_shape=args.gen_shape, real_space_weight=args.real_space_weight)
    else:
        validate_constrains_loop_w_changes(modelArgs, test_dataloader_add_changes, model, normalized_file=normalized_file, 
                                          with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], 
                                          num_samples=args.num_samples, cat2objs=cat2objs, 
                                          datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\n편집 모드 - 관계 변경')
    if args.use_real_space and real_space_embedding is not None:
        validate_constrains_loop_w_changes_real(modelArgs, test_dataloader_rels_changes, model, real_space_embedding,
                                               normalized_file=normalized_file, with_diversity=args.evaluate_diversity, 
                                               with_angles=modelArgs['with_angles'], num_samples=args.num_samples, 
                                               cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', 
                                               gen_shape=args.gen_shape, real_space_weight=args.real_space_weight)
    else:
        validate_constrains_loop_w_changes(modelArgs, test_dataloader_rels_changes, model, normalized_file=normalized_file, 
                                          with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], 
                                          num_samples=args.num_samples, cat2objs=cat2objs, 
                                          datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\n생성 모드')
    if args.use_real_space and real_space_embedding is not None:
        validate_constrains_loop_real(modelArgs, test_dataloader_no_changes, model, real_space_embedding, 
                                     epoch=args.epoch, normalized_file=normalized_file, 
                                     with_diversity=args.evaluate_diversity, with_angles=modelArgs['with_angles'], 
                                     num_samples=args.num_samples, vocab=test_dataset_no_changes.vocab,
                                     point_classes_idx=test_dataset_no_changes.point_classes_idx,
                                     export_3d=args.export_3d, cat2objs=cat2objs, 
                                     datasize='large' if modelArgs['large'] else 'small', 
                                     gen_shape=args.gen_shape, real_space_weight=args.real_space_weight)
    else:
        validate_constrains_loop(modelArgs, test_dataloader_no_changes, model, epoch=args.epoch, 
                                normalized_file=normalized_file, with_diversity=args.evaluate_diversity,
                                with_angles=modelArgs['with_angles'], num_samples=args.num_samples, 
                                vocab=test_dataset_no_changes.vocab,
                                point_classes_idx=test_dataset_no_changes.point_classes_idx,
                                export_3d=args.export_3d, cat2objs=cat2objs, 
                                datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)


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
        accuracy_unchanged[k] = []
        accuracy[k] = []

    all_diversity_boxes = []
    all_diversity_angles = []

    for i, data in enumerate(testdataloader, 0):
        print('Evaluating {}/{}'.format(i, len(testdataloader)))
        
        # 데이터 파싱
        enc_objs = data['encoder']['objs']
        enc_triples = data['encoder']['tripltes']
        enc_tight_boxes = data['encoder']['boxes'].cuda()
        enc_objs_to_scene = data['encoder']['obj_to_scene']
        enc_triples_to_scene = data['encoder']['triple_to_scene']
        
        dec_objs = data['decoder']['objs']
        dec_objs_grained = data['decoder']['objs_grained']
        dec_triples = data['decoder']['tripltes']
        dec_tight_boxes = data['decoder']['boxes'].cuda()
        dec_objs_to_scene = data['decoder']['obj_to_scene']
        dec_triples_to_scene = data['decoder']['triple_to_scene']
        
        missing_nodes = data['decoder']['missing_nodes']
        manipulated_nodes = data['decoder']['manipulated_nodes']
        
        if modelArgs['with_feats']:
            encoded_enc_f = data['encoder']['feats'].cuda()
            encoded_dec_f = data['decoder']['feats'].cuda()
        else:
            encoded_enc_f = None
            encoded_dec_f = None
        
        if modelArgs['with_CLIP']:
            encoded_enc_text_feat = data['encoder']['text_feats'].cuda()
            encoded_enc_rel_feat = data['encoder']['rel_feats'].cuda()
            encoded_dec_text_feat = data['decoder']['text_feats'].cuda()
            encoded_dec_rel_feat = data['decoder']['rel_feats'].cuda()
        else:
            encoded_enc_text_feat = None
            encoded_enc_rel_feat = None
            encoded_dec_text_feat = None
            encoded_dec_rel_feat = None
        
        if modelArgs['with_SDF']:
            dec_sdfs = data['decoder']['sdfs'].cuda()
        else:
            dec_sdfs = None
        
        if with_angles:
            enc_angles = data['encoder']['angles'].cuda()
            dec_angles = data['decoder']['angles'].cuda()
        else:
            enc_angles = None
            dec_angles = None
        
        attributes = None
        dec_attributes = None
        
        # 모델 추론
        with torch.no_grad():
            # 원본 모델 추론
            model_out = model.forward_mani(enc_objs, enc_triples, enc_tight_boxes, enc_angles, encoded_enc_f,
                                          encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene,
                                          dec_objs, dec_objs_grained, dec_triples, dec_tight_boxes, dec_angles, dec_sdfs,
                                          encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes,
                                          dec_objs_to_scene, missing_nodes, manipulated_nodes)
            
            mu_box, logvar_box, mu_shape, logvar_shape, orig_gt_box, orig_gt_angle, orig_gt_shape, orig_box, orig_angle, orig_shape, \
            dec_man_enc_box_pred, dec_man_enc_angle_pred, obj_and_shape, keep = model_out
            
            # 현실 공간 임베딩으로 조건화
            conditioned_mu_box = model.condition_with_real_space(real_space_embedding, mu_box, alpha=real_space_weight)
            
            # 조건화된 잠재 벡터로 디코딩
            if modelArgs['network_type'] == 'v2_full':
                # 조건화된 잠재 벡터로 디코딩
                conditioned_boxes_pred = model.vae_v2.decoder_with_changes(conditioned_mu_box, dec_objs, dec_triples, 
                                                                          encoded_dec_text_feat, encoded_dec_rel_feat, 
                                                                          dec_sdfs, dec_attributes, missing_nodes, 
                                                                          manipulated_nodes, gen_shape=gen_shape)
                
                # 결과 평가 및 저장
                # (여기에 평가 코드 추가)
                
    # 결과 출력
    print("현실 공간 조건화 평가 완료")


def validate_constrains_loop_real(modelArgs, testdataloader, model, real_space_embedding, epoch=None, normalized_file=None, 
                                 with_diversity=True, with_angles=False, vocab=None, point_classes_idx=None, 
                                 export_3d=False, cat2objs=None, datasize='large', num_samples=3, gen_shape=False, 
                                 real_space_weight=0.3):
    """
    현실 공간 임베딩을 활용한 씬 생성 평가
    """
    if with_diversity and num_samples < 2:
        raise ValueError('다양성 평가에는 최소 두 번의 실행이 필요합니다(즉, num_samples > 1).')
    
    for i, data in enumerate(testdataloader, 0):
        print('Evaluating {}/{}'.format(i, len(testdataloader)))
        
        # 데이터 파싱
        dec_objs = data['decoder']['objs']
        dec_objs_grained = data['decoder']['objs_grained']
        dec_triples = data['decoder']['tripltes']
        dec_tight_boxes = data['decoder']['boxes'].cuda()
        dec_objs_to_scene = data['decoder']['obj_to_scene']
        dec_triples_to_scene = data['decoder']['triple_to_scene']
        
        if modelArgs['with_CLIP']:
            encoded_dec_text_feat = data['decoder']['text_feats'].cuda()
            encoded_dec_rel_feat = data['decoder']['rel_feats'].cuda()
        else:
            encoded_dec_text_feat = None
            encoded_dec_rel_feat = None
        
        if modelArgs['with_SDF']:
            dec_sdfs = data['decoder']['sdfs'].cuda()
        else:
            dec_sdfs = None
        
        if with_angles:
            dec_angles = data['decoder']['angles'].cuda()
        else:
            dec_angles = None
        
        dec_attributes = None
        
        # 모델 추론
        with torch.no_grad():
            # 샘플링
            if modelArgs['network_type'] == 'v2_full':
                # 원본 샘플링
                mean_est, cov_est = model.vae_v2.sampleBoxes(dec_objs, dec_triples, dec_attributes)
                
                # 현실 공간 임베딩으로 조건화
                conditioned_mean_est = model.condition_with_real_space(real_space_embedding, mean_est, alpha=real_space_weight)
                
                # 조건화된 잠재 벡터로 디코딩
                conditioned_boxes_pred = model.vae_v2.decoder(conditioned_mean_est, dec_objs, dec_triples, 
                                                             encoded_dec_text_feat, encoded_dec_rel_feat, 
                                                             dec_attributes)
                
                # 결과 평가 및 저장
                # (여기에 평가 코드 추가)
                
    # 결과 출력
    print("현실 공간 조건화 생성 평가 완료")


def validate_constrains_loop_w_changes(modelArgs, testdataloader, model, normalized_file=None, with_diversity=True, with_angles=False, num_samples=3, cat2objs=None, datasize='large', gen_shape=False):
    if with_diversity and num_samples < 2:
        raise ValueError('Diversity requires at least two runs (i.e. num_samples > 1).')

    accuracy = {}
    accuracy_unchanged = {}
    accuracy_in_orig_graph = {}

    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        accuracy_in_orig_graph[k] = []
        accuracy_unchanged[k] = []
        accuracy[k] = []

    all_diversity_boxes = []
    all_diversity_angles = []
    
    # 기존 평가 코드 유지
    # ...


def validate_constrains_loop(modelArgs, testdataloader, model, epoch=None, normalized_file=None, with_diversity=True, with_angles=False, vocab=None,
                             point_classes_idx=None, export_3d=False, cat2objs=None, datasize='large',
                             num_samples=3, gen_shape=False):
    
    # 기존 평가 코드 유지
    # ...


def normalize(vertices, scale=1):
    # 기존 정규화 코드 유지
    # ...


if __name__ == "__main__":
    evaluate() 