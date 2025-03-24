#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append('../')
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from model.VAE_spaceadaptive import VAE
from spaceadaptive.SpaceAdaptiveVAE import SpaceAdaptiveVAE
from model.discriminators import BoxDiscriminator, ShapeAuxillary
from model.losses import bce_loss
from helpers.util import bool_flag, _CustomDataParallel

from model.losses import calculate_model_losses

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='../experiments/space_adaptive_lounge', help='실험 결과 저장 경로')
parser.add_argument('--logf', default='logs', help='로그 저장 경로')
parser.add_argument('--outf', default='checkpoint', help='모델 저장 경로')
parser.add_argument('--dataset', default='/mnt/dataset/FRONT', help='데이터셋 경로')
parser.add_argument('--room_type', default='all', help='방 타입 [livingroom, bedroom, diningroom, library]')
parser.add_argument('--workers', type=int, default=8, help='데이터 로딩 워커 수')
parser.add_argument('--batchSize', type=int, default=8, help='배치 크기')
parser.add_argument('--nepoch', type=int, default=500, help='최대 에폭 수')
parser.add_argument('--weight_D_box', type=float, default=0.1, help='박스 판별자 가중치')
parser.add_argument('--auxlr', type=float, default=0.0001, help='보조 네트워크 학습률')
parser.add_argument('--shuffle_objs', default=False, type=bool_flag, help='객체 셔플 여부')
parser.add_argument('--use_scene_rels', default=True, type=bool_flag, help='씬 관계 사용 여부')
parser.add_argument('--with_changes', default=True, type=bool_flag, help='변경 사항 포함 여부')
parser.add_argument('--with_feats', default=True, type=bool_flag, help='특성 포함 여부')
parser.add_argument('--with_SDF', default=False, type=bool_flag, help='SDF 포함 여부')
parser.add_argument('--with_CLIP', default=False, type=bool_flag, help='CLIP 특성 포함 여부')
parser.add_argument('--large', default=True, type=bool_flag, help='대형 데이터셋 사용 여부')
parser.add_argument('--loadmodel', default=False, type=bool_flag, help='모델 로드 여부')
parser.add_argument('--loadepoch', default=0, type=int, help='로드할 에폭')
parser.add_argument('--pooling', default='avg', choices=['avg', 'max'], help='풀링 방식')
parser.add_argument('--with_angles', default=False, type=bool_flag, help='각도 포함 여부')
parser.add_argument('--num_box_params', default=6, type=int, help='박스 파라미터 수')
parser.add_argument('--residual', default=True, type=bool_flag, help='잔차 연결 사용 여부')
parser.add_argument('--with_E2', default=True, type=bool_flag)
parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='v2_box', choices=['v2_box', 'v2_full', 'v1_box', 'v1_full'], type=str)
parser.add_argument('--diff_yaml', default='../config/v2_full.yaml', type=str, help='diffusion 네트워크 설정 [cross_attn/concat]')
parser.add_argument('--vis_num', type=int, default=8, help='학습 중 시각화 수')

# SpaceAdaptive 모델 특화 인자
# 현실 공간 데이터 관련 인자 추가
parser.add_argument('--space_data_path', default='../spaceadaptive/spacedata/spacedata.json', type=str, help='현실 공간 데이터 경로')
parser.add_argument('--use_real_space', default=True, type=bool_flag, help='현실 공간 데이터 사용 여부')
parser.add_argument('--real_space_id', default='Lounge', type=str, help='특정 현실 공간 ID (None이면 랜덤 선택)')

parser.add_argument('--real_space_weight', default=0.5, type=float, help='현실 공간 임베딩 가중치 (0~1)')
parser.add_argument('--real_space_loss_weight', default=0.5, type=float, help='현실 공간 손실 가중치')
parser.add_argument('--real_space_condition_freq', default=0.7, type=float, help='현실 공간 조건화 빈도 (0~1)')
parser.add_argument('--snapshot', default=20, type=int, help='모델 저장 주기 (에폭 단위)')

# SpaceAdaptiveVAE 관련 추가 인자
parser.add_argument('--training_done', default=False, type=bool_flag, help='학습 단계 건너뛰기 여부')
parser.add_argument('--use_space_adaptive', default=True, type=bool_flag, help='SpaceAdaptiveVAE 사용 여부')
parser.add_argument('--hybrid_scene_output', default='../spaceadaptive/hybrid_scene.json', type=str, help='생성된 하이브리드 씬 저장 경로')
parser.add_argument('--top_k_scenes', default=100, type=int, help='선택할 상위 유사 씬 개수')

# python train_spaceadaptive.py --room_type all --dataset /mnt/dataset/FRONT --residual True --network_type v2_box --with_SDF False --with_CLIP False --batchSize 8 --workers 8 --nepoch 1000 --large True --training_done True --loadmodel True --loadepoch 360

# Lounge 3/22 학습 완료
# python train_spaceadaptive.py --room_type all --dataset /mnt/dataset/FRONT --residual True --network_type v2_full --with_SDF False --with_CLIP False --batchSize 8 --workers 8 --nepoch 1000 --large True --training_done False
# Lounge 3/23 하이브리드 그래프 생성
# python train_spaceadaptive.py --exp ../experiments/space_adaptive --network_type v2_box --training_done True --loadmodel True --loadepoch 360
# no_adaptive 3/23 학습
# python train_spaceadaptive.py --training_done False --exp ../experiments/no_adaptive --network_type v2_box --use_real_space False

# Studio 3/24 학습
# python train_spaceadaptive.py --training_done False --exp ../experiments/space_adaptive_studio --network_type v2_box --use_real_space True

args = parser.parse_args()
print(args)

def parse_data(data):
    enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                                      data['encoder']['tripltes'], \
                                                                                      data['encoder']['boxes'], \
                                                                                      data['encoder'][
                                                                                          'obj_to_scene'], \
                                                                                      data['encoder'][
                                                                                          'triple_to_scene']
    if args.with_feats:
        encoded_enc_f = data['encoder']['feats'] #사전 인코딩된 latent point features를 가져옴.
        encoded_enc_f = encoded_enc_f.cuda() #텐서를 GPU로 이동

    dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                      data['decoder']['tripltes'], \
                                                                                      data['decoder']['boxes'], \
                                                                                      data['decoder']['obj_to_scene'], \
                                                                                      data['decoder']['triple_to_scene']
    dec_objs_grained = data['decoder']['objs_grained']
    dec_sdfs = None
    if 'sdfs' in data['decoder']:
        dec_sdfs = data['decoder']['sdfs']
    if 'feats' in data['decoder']:
        encoded_dec_f = data['decoder']['feats']
        encoded_dec_f = encoded_dec_f.cuda()
    
    encoded_enc_text_feat = None
    encoded_enc_rel_feat = None
    encoded_dec_text_feat = None
    encoded_dec_rel_feat = None
    if args.with_CLIP:
        encoded_enc_text_feat = data['encoder']['text_feats'].cuda() #텐서를 GPU로 이동
        encoded_enc_rel_feat = data['encoder']['rel_feats'].cuda()
        encoded_dec_text_feat = data['decoder']['text_feats'].cuda()
        encoded_dec_rel_feat = data['decoder']['rel_feats'].cuda()
    else:
        # V2_BOX/V2_FULL 모델용 더미 텐서 생성
        if args.network_type in ['v2_box', 'v2_full']:
            # 객체 수에 맞는 더미 텐서 생성
            encoded_enc_text_feat = torch.zeros((len(enc_objs), 512)).cuda()
            encoded_enc_rel_feat = torch.zeros((len(enc_triples), 512)).cuda()
            encoded_dec_text_feat = torch.zeros((len(dec_objs), 512)).cuda()
            encoded_dec_rel_feat = torch.zeros((len(dec_triples), 512)).cuda()

    # changed nodes
    missing_nodes = data['missing_nodes']
    manipulated_nodes = data['manipulated_nodes']

    enc_objs, enc_triples, enc_tight_boxes = enc_objs.cuda(), enc_triples.cuda(), enc_tight_boxes.cuda()
    dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
    dec_objs_grained = dec_objs_grained.cuda()

    enc_scene_nodes = enc_objs == 0
    dec_scene_nodes = dec_objs == 0
    if not args.with_feats:
        with torch.no_grad():
            encoded_enc_f = None  
            encoded_dec_f = None 

    # set all scene (dummy) node encodings to zero
    try:
        encoded_enc_f[enc_scene_nodes] = torch.zeros(
            [torch.sum(enc_scene_nodes), encoded_enc_f.shape[1]]).float().cuda()
        encoded_dec_f[dec_scene_nodes] = torch.zeros(
            [torch.sum(dec_scene_nodes), encoded_dec_f.shape[1]]).float().cuda()
    except:
        if args.network_type == 'v1_box':
            encoded_enc_f = None
            encoded_dec_f = None

    if args.num_box_params == 7:
        # all parameters, including angle, procesed by the box_net
        enc_boxes = enc_tight_boxes
        dec_boxes = dec_tight_boxes
    elif args.num_box_params == 6:
        # no angle. this will be learned separately if with_angle is true
        enc_boxes = enc_tight_boxes[:, :6]
        dec_boxes = dec_tight_boxes[:, :6]
    else:
        raise NotImplementedError

    # limit the angle bin range from 0 to 24
    # enc_angles의 값이 0 ~ 23 범위 안에 있도록 보정됨
    enc_angles = enc_tight_boxes[:, 6].long() - 1
    enc_angles = torch.where(enc_angles > 0, enc_angles, torch.zeros_like(enc_angles))
    enc_angles = torch.where(enc_angles < 24, enc_angles, torch.zeros_like(enc_angles))
    dec_angles = dec_tight_boxes[:, 6].long() - 1
    dec_angles = torch.where(dec_angles > 0, dec_angles, torch.zeros_like(dec_angles))
    dec_angles = torch.where(dec_angles < 24, dec_angles, torch.zeros_like(dec_angles))

    attributes = None

    return enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat, \
           attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs, \
           encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes

def train():
    """ 제공된 인자를 기반으로 네트워크 학습
    """
    args.manualSeed = random.randint(1, 10000)  # 선택적으로 시드 고정 7494
    print("Random Seed: ", args.manualSeed)

    print(torch.__version__)

    random.seed(args.manualSeed) # Python 기본 random 모듈의 시드 설정
    torch.manual_seed(args.manualSeed) # PyTorch의 시드 설정. 랜덤성을 통제. PyTorch 내부의 텐서 초기화, 데이터 샘플링, 모델 학습 중 발생하는 랜덤성을 고정.

    # 학습용 씬 그래프 데이터셋 인스턴스화
    dataset = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='train_scans',
        shuffle_objs=args.shuffle_objs,
        use_SDF=args.with_SDF,
        use_scene_rels=args.use_scene_rels,
        with_changes=args.with_changes,
        with_feats=args.with_feats,
        with_CLIP=args.with_CLIP,
        large=args.large,
        seed=False,
        room_type=args.room_type,
        recompute_feats=False,
        recompute_clip=False)

    collate_fn = dataset.collate_fn_vaegan_points
    # 데이터셋에서 데이터 로더 인스턴스화
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=int(args.workers))

    # 객체 클래스 및 관계 클래스 수
    num_classes = len(dataset.classes)
    num_relationships = len(dataset.relationships) + 1

    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    
    # Space Adaptive VAE 모델 인스턴스화
    model = VAE(root=args.dataset, type=args.network_type, diff_opt=args.diff_yaml, vocab=dataset.vocab,
                replace_latent=args.replace_latent, with_changes=args.with_changes, residual=args.residual,
                gconv_pooling=args.pooling, with_angles=args.with_angles, num_box_params=args.num_box_params,
                deepsdf=args.with_feats, clip=args.with_CLIP, with_E2=args.with_E2,
                use_real_space=args.use_real_space, real_space_weight=args.real_space_weight)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loadmodel:
        model.load_networks(exp=args.exp, epoch=args.loadepoch, restart_optim=False)

    # SpaceAdaptive 현실 공간 로드 및 임베딩
    # SpaceAdaptiveVAE 초기화 (초기화된 model 인스턴스 전달)
    space_adaptive_vae = SpaceAdaptiveVAE(model)
    
    # 현실 공간 데이터 로드 및 임베딩
    real_space_embedding = None
    space_data = None
    if args.use_real_space and args.space_data_path is not None:
        print("현실 공간 데이터 로드 중...")
        # 3. 현실 공간 데이터 로드 및 임베딩 (아직 로드하지 않은 경우)
        if space_adaptive_vae.space_data is None:
            # train_with_real_space() 함수 호출로 통합 처리
            space_data, _, _, _, real_space_embedding = space_adaptive_vae.train_with_real_space(
                space_data_path=args.space_data_path,
                room_type=args.room_type,
                space_id=args.real_space_id
            )
            # print(space_data)
            print(f"현실 공간 데이터 처리 완료: {len(space_data) if space_data else 0}개 공간")
            print(real_space_embedding)

    # 판별자 모델 설정
    if args.weight_D_box > 0:
        boxD = BoxDiscriminator(6, num_relationships, num_classes)
        optimizerDbox = optim.Adam(filter(lambda p: p.requires_grad, boxD.parameters()), lr=args.auxlr,
                                   betas=(0.9, 0.999))
        boxD.cuda()
        boxD = boxD.train()

    # 형상 보조 분류기 설정
    shapeClassifier = ShapeAuxillary(256, len(dataset.cat))
    shapeClassifier = shapeClassifier.cuda()
    shapeClassifier.train()
    optimizerShapeAux = optim.Adam(filter(lambda p: p.requires_grad, shapeClassifier.parameters()), lr=args.auxlr,
                                   betas=(0.9, 0.999))
    
    # tensorboard 작성기 초기화
    writer = SummaryWriter(args.exp + "/" + args.logf)

    # v1 및 v2_box 모델용 옵티마이저. v2_full인 경우 자체 옵티마이저 사용.
    if args.network_type != 'v2_full':
        params = filter(lambda p: p.requires_grad, list(model.parameters()))
        optimizer_bl = optim.Adam(params, lr=args.auxlr)
        optimizer_bl.step()

    print("---- 모델 및 데이터셋 구축 완료 ----")
    
    if not os.path.exists(args.exp + "/" + args.outf):
        os.makedirs(args.exp + "/" + args.outf)

    # 설정 저장
    with open(os.path.join(args.exp, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("모든 매개변수를 다음 위치에 저장:")
    print(os.path.join(args.exp, 'args.json'))
    
    torch.autograd.set_detect_anomaly(True)
    counter = model.counter if model.counter else 0

    start_epoch = model.epoch if model.epoch else 0
    if not args.training_done:
        print("---- 학습 루프 시작! ----")
        iter_start_time = time.time()
        print(f"iter_start_time: {iter_start_time}")
        for epoch in range(start_epoch, args.nepoch):
            print('Epoch: {}/{}'.format(epoch, args.nepoch))
            for i, data in enumerate(dataloader, 0):
                # 데이터 파싱
                try:
                    enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                    attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                    encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, dec_objs_to_scene, missing_nodes,\
                    manipulated_nodes = parse_data(data)
                except Exception as e:
                    print("데이터 오류 발생:", e)
                    continue

                # SpaceAdaptive 현실 공간 조건화 처리
                # 현실 공간 임베딩 랜덤 선택 (조건화에 사용)
                real_space_id = None
                real_space_embedding = None
                
                # 현실 공간 조건화 여부 결정 (확률 기반)
                use_real_space_this_batch = args.use_real_space and space_adaptive_vae.real_space_embedding and random.random() < args.real_space_condition_freq
                
                if use_real_space_this_batch:
                    if args.real_space_id and args.real_space_id in space_adaptive_vae.real_space_embedding:
                        real_space_id = args.real_space_id
                    else:
                        # 랜덤하게 현실 공간 ID 선택
                        real_space_id = random.choice(list(space_adaptive_vae.real_space_embedding.keys()))
                    real_space_embedding = space_adaptive_vae.real_space_embedding[real_space_id]
                    # real_space_embedding => 사전이 아니라 텐서 형식임.

                # 네트워크 순전파
                if real_space_embedding is not None:
                    model_out = model.forward_mani(
                        enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f,
                        encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene,
                        dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,
                        encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes,
                        dec_objs_to_scene, missing_nodes, manipulated_nodes, real_space_id=real_space_id
                    )
                else:
                    model_out = model.forward_mani(
                        enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f,
                        encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene,
                        dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,
                        encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes,
                        dec_objs_to_scene, missing_nodes, manipulated_nodes
                    )

                mu_box, logvar_box, mu_shape, logvar_shape, orig_gt_box, orig_gt_angle, orig_gt_shape, orig_box, orig_angle, orig_shape, \
                dec_man_enc_box_pred, dec_man_enc_angle_pred, obj_and_shape, keep = model_out

                vae_loss_box, vae_losses_box = calculate_model_losses(args,
                                                                    orig_gt_box,
                                                                    orig_box,
                                                                    name='box', withangles=args.with_angles,
                                                                    angles_pred=orig_angle,
                                                                    mu=mu_box, logvar=logvar_box, angles=orig_gt_angle,
                                                                    KL_weight=0.1, writer=writer, counter=counter)

                ## From Graph-to-3D
                # initiate the loss
                boxGloss = 0
                loss_genShape = 0
                loss_genShapeFake = 0
                loss_shape_fake_g = 0
                new_shape_loss, new_shape_losses = 0, 0
                
                if args.network_type == 'v2_full':
                    vae_loss_shape, vae_losses_shape = 0, 0
                    new_shape_loss, new_shape_losses = model.vae_v2.Diff.loss_df, model.vae_v2.Diff.loss_dict
                    model.vae_v2.Diff.update_loss()
                else:
                    # set shape loss to 0 if we are only predicting layout
                    vae_loss_shape, vae_losses_shape = 0, 0

                if args.with_changes:
                    oriented_gt_boxes = torch.cat([dec_boxes], dim=1)
                    boxes_pred_in = keep * oriented_gt_boxes + (1 - keep) * dec_man_enc_box_pred

                    if args.weight_D_box == 0:
                        # Generator loss
                        boxGloss = 0
                        # Discriminator loss
                        gamma = 0.1
                        boxDloss_real = 0
                        boxDloss_fake = 0
                        reg_loss = 0
                    else:
                        logits, _ = boxD(dec_objs, dec_triples, boxes_pred_in, keep)
                        logits_fake, reg_fake = boxD(dec_objs, dec_triples, boxes_pred_in.detach(), keep, with_grad=True,
                                                    is_real=False)
                        logits_real, reg_real = boxD(dec_objs, dec_triples, oriented_gt_boxes, with_grad=True, is_real=True)
                        # Generator loss
                        boxGloss = bce_loss(logits, torch.ones_like(logits))
                        # Discriminator loss
                        gamma = 0.1
                        boxDloss_real = bce_loss(logits_real, torch.ones_like(logits_real))
                        boxDloss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
                        # Regularization by gradient penalty
                        reg_loss = torch.mean(reg_real + reg_fake)

                    # gradient penalty
                    boxDloss = boxDloss_fake + boxDloss_real + (gamma / 2.0) * reg_loss
                    optimizerDbox.zero_grad()
                    boxDloss.backward()

                # 현실 공간 손실 계산 (현실 공간 임베딩을 사용한 경우)
                vae_loss_realspace = 0.0
            
                # print("space_data[real_space_id]['boxes']")
                # print(space_data[real_space_id]['boxes'])
                # print("orig_gt_box")
                # print(orig_gt_box)

                if use_real_space_this_batch and real_space_embedding is not None:
                    # clone() 사용하여 in-place 연산 방지
                    real_space_emb_clone = real_space_embedding.clone() if isinstance(real_space_embedding, torch.Tensor) else real_space_embedding
                    boxes_clone = None
                    if space_data[real_space_id]['boxes'] is not None:
                        boxes_clone = space_data[real_space_id]['boxes'].clone() if isinstance(space_data[real_space_id]['boxes'], torch.Tensor) else space_data[real_space_id]['boxes']
                    mu_box_clone = mu_box.clone() if isinstance(mu_box, torch.Tensor) else mu_box
                    orig_box_clone = None
                    if orig_box is not None:
                        orig_box_clone = orig_box.clone() if isinstance(orig_box, torch.Tensor) else orig_box
                    vae_loss_realspace = model.calculate_real_space_loss(mu_box_clone, real_space_emb_clone, orig_box_clone, boxes_clone)

                # 기본 손실 계산
                loss = vae_loss_box + vae_loss_shape + 0.1 * loss_genShape + 100 * new_shape_loss
                if args.with_changes:
                    loss += args.weight_D_box * boxGloss  # + b_loss
                    
                # 현실 공간 손실 추가 (사용하는 경우)
                if use_real_space_this_batch and vae_loss_realspace > 0:
                    # 손실이 0보다 큰 경우에만 가중치를 적용하여 추가
                    # vae_loss_realspace를 직접 더하는 대신 분리된 버전을 생성하여 사용
                    real_space_loss_weighted = args.real_space_loss_weight * vae_loss_realspace.detach().clone()
                    loss = loss + real_space_loss_weighted

                # # 디버깅 정보 출력
                # if torch.is_tensor(mu_box):
                #     print(f"[디버깅] mu_box 크기: {mu_box.shape}, requires_grad: {mu_box.requires_grad}")
                # if use_real_space_this_batch and torch.is_tensor(real_space_emb_clone):
                #     print(f"[디버깅] real_space_emb_clone 크기: {real_space_emb_clone.shape}")
                # if use_real_space_this_batch and torch.is_tensor(mu_box_clone):
                #     print(f"[디버깅] mu_box_clone 크기: {mu_box_clone.shape}, requires_grad: {mu_box_clone.requires_grad}")
                # print(f"[디버깅] vae_loss_realspace: {vae_loss_realspace}")

                # optimize
                loss.backward(retain_graph=True)

                # Cap the occasional super mutant gradient spikes
                # Do now a gradient step and plot the losses
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                if args.network_type == 'v2_full':
                    torch.nn.utils.clip_grad_norm_(model.vae_v2.Diff.df_module.parameters(), 5.0)
                    for group in model.vae_v2.optimizerFULL.param_groups:
                        for p in group['params']:
                            if p.grad is not None and p.requires_grad and torch.isnan(p.grad).any():
                                print('NaN grad in step {}.'.format(counter))
                                p.grad[torch.isnan(p.grad)] = 0
                else:
                    for group in optimizer_bl.param_groups:
                        for p in group['params']:
                            if p.grad is not None and p.requires_grad and torch.isnan(p.grad).any():
                                print('NaN grad in step {}.'.format(counter))
                                p.grad[torch.isnan(p.grad)] = 0

                if args.with_changes:
                    if args.network_type == 'v1_full':
                        optimizerShapeAux.step()
                    optimizerDbox.step()

                if args.network_type == 'v2_full':
                    model.vae_v2.optimizerFULL.step()
                else:
                    optimizer_bl.step()

                # 로깅 및 모델 저장
                iter_net_time = time.time()
                eta = ((iter_net_time - iter_start_time) / (i + 1)) * len(dataloader) - (
                        iter_net_time - iter_start_time)

                counter += 1
                if counter % 100 == 0:
                    message = "loss at {}, (ETA: {:.2f}h): box {:.4f}\trealspace {:.4f}\tshape {:.4f}\t".format(
                        counter, eta, vae_loss_box, vae_loss_realspace, vae_loss_shape)
                    if args.network_type == 'v2_full':
                        loss_diff = model.vae_v2.Diff.get_current_errors()
                        for k, v in loss_diff.items():
                            message += '%s: %.6f ' % (k, v)
                    print(message)

                writer.add_scalar('Train_Loss_BBox', vae_loss_box, counter)
                writer.add_scalar('Train_Loss_RealSpace', vae_loss_realspace, counter)
                writer.add_scalar('Train_Loss_Shape', vae_loss_shape, counter)

                if args.network_type == 'v2_full':
                    t = (time.time() - iter_start_time) / args.batchSize
                    loss_diff = model.vae_v2.Diff.get_current_errors()
                    model.vae_v2.visualizer.print_current_errors(writer, counter, loss_diff, t)
                    if counter % 1000 == 0:
                        # DDIM 샘플링은 Diffusion 모델에서 생성 과정을 가속화하는 샘플링 기법
                        model.vae_v2.Diff.gen_shape_after_foward(num_obj=args.vis_num)
                        model.vae_v2.visualizer.display_current_results(writer, model.vae_v2.Diff.get_current_visuals(
                            dataset.classes_r, obj_and_shape[0].detach().cpu().numpy(), num_obj=args.vis_num),
                                                                        counter, phase='train')

                    current_lr = model.vae_v2.update_learning_rate()
                    writer.add_scalar("learning_rate", current_lr, counter)

            # 에폭 완료 후 모델 저장
            if epoch % args.snapshot == 0:
                print(f"Epoch {epoch} 모델 저장 중...")
                model.save(args.exp, args.outf, epoch, counter)
        
    # 학습 종료 후 SpaceAdaptiveVAE 처리
    if args.training_done:
        if args.use_space_adaptive and args.use_real_space and args.space_data_path:
            print("SpaceAdaptiveVAE 처리 시작...")
            
            # 1. 가상 씬 데이터셋 로드
            print(f"가상 씬 데이터셋 로드 중")
            virtual_scenes = {}
            for i in range(len(dataset)):
                data = dataset[i]
                scene_id = data['scan_id']
                scene_data = {
                    'objs': data['encoder']['objs'],  # 객체 ID 매핑
                    'boxes': data['encoder']['boxes'],  # 바운딩 박스 정보
                    'triples': data['encoder']['triples']  # 관계 목록
                }
                virtual_scenes[scene_id] = scene_data
            
            # 3. 현실 공간 데이터 로드 및 4. 현실 공간 임베딩 (아직도 없는 경우)
            if space_data is None and real_space_embedding is None:
                space_data, _, _, _, real_space_embedding = space_adaptive_vae.train_with_real_space(
                    space_data_path=args.space_data_path,
                    room_type=args.room_type,
                    space_id=args.real_space_id
                )
            
            # 5. 유사 가상 씬 식별
            print("현실 공간과 유사한 가상 씬 식별 중...")
            similar_scenes = space_adaptive_vae.identify_similar_virtual_scenes(virtual_scenes, top_k=args.top_k_scenes)
            print(f"현실 공간과 가장 유사한 {len(similar_scenes)}개 가상 씬 식별 완료")
            
            # 6. 하이브리드 씬 생성
            # TODO 수정 필요
            print("하이브리드 씬 생성 중...")
            hybrid_scene = space_adaptive_vae.generate_hybrid_scene_from_similar()
            print("하이브리드 씬 생성 완료")
            
            # 7. 생성된 하이브리드 씬 저장
            hybrid_scene_path = os.path.join(args.exp, f'hybrid_scene_epoch_{epoch}.json')
            space_adaptive_vae.save_hybrid_scene(hybrid_scene_path)
            print(f"하이브리드 씬 저장 완료: {hybrid_scene_path}")
            
            # 8. 최신 하이브리드 씬 심볼릭 링크 생성
            latest_hybrid_scene_path = args.hybrid_scene_output
            os.makedirs(os.path.dirname(latest_hybrid_scene_path), exist_ok=True)
            if os.path.exists(latest_hybrid_scene_path):
                os.remove(latest_hybrid_scene_path)
            try:
                os.symlink(hybrid_scene_path, latest_hybrid_scene_path)
                print(f"최신 하이브리드 씬 링크 생성: {latest_hybrid_scene_path}")
            except:
                # 심볼릭 링크 생성 실패 시 복사
                import shutil
                shutil.copy2(hybrid_scene_path, latest_hybrid_scene_path)
                print(f"최신 하이브리드 씬 복사: {latest_hybrid_scene_path}")

    print('Training completed!')
    writer.close()

if __name__ == '__main__':
    train() 