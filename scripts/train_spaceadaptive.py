#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import glob
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
from model.discriminator import BoxDiscriminator, ShapeAuxillary
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.metrics import calculate_model_losses
from helpers.util import bool_flag, bce_loss

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='../experiments/spaceadaptive', help='실험 결과 저장 경로')
parser.add_argument('--logf', default='logs', help='로그 저장 경로')
parser.add_argument('--outf', default='models', help='모델 저장 경로')
parser.add_argument('--dataset', default='../GT', help='데이터셋 경로')
parser.add_argument('--room_type', default='livingroom', help='방 타입 [livingroom, bedroom, diningroom, library]')
parser.add_argument('--workers', type=int, help='데이터 로딩 워커 수', default=6)
parser.add_argument('--batchSize', type=int, default=32, help='배치 크기')
parser.add_argument('--nepoch', type=int, default=10000, help='최대 에폭 수')
parser.add_argument('--weight_D_box', type=float, default=0.1, help='박스 판별자 가중치')
parser.add_argument('--auxlr', type=float, default=0.0001, help='보조 네트워크 학습률')
parser.add_argument('--shuffle_objs', default=False, type=bool_flag, help='객체 셔플 여부')
parser.add_argument('--use_scene_rels', default=True, type=bool_flag, help='씬 관계 사용 여부')
parser.add_argument('--with_changes', default=True, type=bool_flag, help='변경 사항 포함 여부')
parser.add_argument('--with_feats', default=True, type=bool_flag, help='특성 포함 여부')
parser.add_argument('--with_SDF', default=True, type=bool_flag, help='SDF 포함 여부')
parser.add_argument('--with_CLIP', default=True, type=bool_flag, help='CLIP 특성 포함 여부')
parser.add_argument('--large', default=False, type=bool_flag, help='대형 데이터셋 사용 여부')
parser.add_argument('--loadmodel', default=False, type=bool_flag, help='모델 로드 여부')
parser.add_argument('--loadepoch', default=0, type=int, help='로드할 에폭')
parser.add_argument('--pooling', default='avg', choices=['avg', 'max'], help='풀링 방식')
parser.add_argument('--with_angles', default=False, type=bool_flag, help='각도 포함 여부')
parser.add_argument('--num_box_params', default=6, type=int, help='박스 파라미터 수')
parser.add_argument('--residual', default=True, type=bool_flag, help='잔차 연결 사용 여부')
parser.add_argument('--with_E2', default=True, type=bool_flag)
parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='v2_full', choices=['v2_box', 'v2_full', 'v1_box', 'v1_full'], type=str)
parser.add_argument('--diff_yaml', default='../config/v2_full.yaml', type=str, help='diffusion 네트워크 설정 [cross_attn/concat]')
parser.add_argument('--vis_num', type=int, default=8, help='학습 중 시각화 수')

# SpaceAdaptive 모델 특화 인자
# 현실 공간 데이터 관련 인자 추가
parser.add_argument('--space_data_path', default='../spaceadaptive/spacedata/spacedata.json', type=str, help='현실 공간 데이터 경로')
parser.add_argument('--use_real_space', default=True, type=bool_flag, help='현실 공간 데이터 사용 여부')
parser.add_argument('--real_space_id', default='Lounge', type=str, help='특정 현실 공간 ID (None이면 랜덤 선택)')

parser.add_argument('--real_space_weight', default=0.3, type=float, help='현실 공간 임베딩 가중치 (0~1)')
parser.add_argument('--real_space_loss_weight', default=0.5, type=float, help='현실 공간 손실 가중치')
parser.add_argument('--real_space_condition_freq', default=0.7, type=float, help='현실 공간 조건화 빈도 (0~1)')

# SpaceAdaptiveVAE 관련 추가 인자
parser.add_argument('--use_space_adaptive', default=False, type=bool_flag, help='SpaceAdaptiveVAE 사용 여부')
parser.add_argument('--virtual_scenes_dir', default='../datasample/SG-FRONT', type=str, help='가상 씬 데이터셋 경로')
parser.add_argument('--hybrid_scene_output', default='../spaceadaptive/hybrid_scene.json', type=str, help='생성된 하이브리드 씬 저장 경로')
parser.add_argument('--top_k_scenes', default=20, type=int, help='선택할 상위 유사 씬 개수')

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
    else:
        encoded_enc_f = None

    encoded_enc_text_feat = None
    encoded_enc_rel_feat = None
    encoded_dec_text_feat = None
    encoded_dec_rel_feat = None
    if args.with_CLIP:
        encoded_enc_text_feat = data['encoder']['text_feats'].cuda() #텐서를 GPU로 이동
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
    if args.with_feats:
        encoded_dec_f = data['decoder']['feats']
        encoded_dec_f = encoded_dec_f.cuda()
    else:
        encoded_dec_f = None

    if args.with_SDF:
        dec_sdfs = data['decoder']['sdfs']
        dec_sdfs = dec_sdfs.cuda()
    else:
        dec_sdfs = None

    if args.with_angles:
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
        model.set_cuda()

    if args.loadmodel:
        model.load_networks(exp=args.exp, epoch=args.loadepoch, restart_optim=False)

    # SpaceAdaptive 현실 공간 로드 및 임베딩
    # SpaceAdaptiveVAE 초기화
    space_adaptive_vae = SpaceAdaptiveVAE(model)
    
    # 현실 공간 데이터 로드 및 임베딩
    real_space_embeddings = None
    if args.use_real_space and args.space_data_path is not None:
        print("현실 공간 데이터 로드 중...")
        real_space_data = space_adaptive_vae.load_real_space_data(args.space_data_path)
        if real_space_data:
            print("현실 공간 데이터 임베딩 생성 중...")
            real_space_embeddings = space_adaptive_vae.encode_real_space()
            print(f"현실 공간 임베딩 생성 완료: {len(real_space_embeddings) if real_space_embeddings else 0}개 공간")
            
            # CUDA 설정
            space_adaptive_vae.set_cuda()

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

    print("---- 학습 루프 시작! ----")
    iter_start_time = time.time()
    start_epoch = model.epoch if model.epoch else 0
    for epoch in range(start_epoch, args.nepoch):
        print('Epoch: {}/{}'.format(epoch, args.nepoch))
        for i, data in enumerate(dataloader, 0):
            # 데이터 파싱
            try:
                enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes,\
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

            # 네트워크 순전파
            if real_space_embedding is not None:
                model_out = model.forward_mani(
                    enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f,
                    encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene,
                    dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,
                    encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes,
                    dec_objs_to_scene, missing_nodes, manipulated_nodes, real_space_id=real_space_id
                )
            else:
                model_out = model.forward_mani(
                    enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f,
                    encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene,
                    dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,
                    encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes,
                    dec_objs_to_scene, missing_nodes, manipulated_nodes
                )

            # 모델 출력 파싱
            if args.network_type == 'v2_full':
                z_obj_vecs, z_rel_vecs, z_obj_vecs_enc, z_obj_sigmas, z_rel_vecs_enc, z_rel_sigmas, \
                z_diff, boxes_pred, diff_loss, feat_shape, mu_shape, logvar_shape, z_samples, box_data, \
                codes_pred, diff_out = model_out
            else:
                z_obj_vecs, z_rel_vecs, z_obj_vecs_enc, z_obj_sigmas, z_rel_vecs_enc, z_rel_sigmas, \
                z_diff, boxes_pred, diff_loss, feat_shape, mu_shape, logvar_shape, z_samples, box_data, \
                codes_pred = model_out

            # 손실 계산
            total_kl_loss, mask_ratio, triplet_ratio, \
            z_obj_kl_loss, z_rel_kl_loss, z_shape_kl_loss = calculate_model_losses(
                z_obj_vecs, z_rel_vecs, z_obj_vecs_enc, z_obj_sigmas, z_rel_vecs_enc, z_rel_sigmas,
                mu_shape, logvar_shape, enc_objs_to_scene, dec_objs_to_scene)

            # 현실 공간 손실 계산 (현실 공간 임베딩을 사용한 경우)
            real_space_loss = 0.0
            if use_real_space_this_batch and real_space_embedding is not None:
                # 생성된 잠재 벡터와 현실 공간 임베딩 간의 손실 계산
                real_space_boxes_pred = None  # 실제 현실 공간의 바운딩 박스 (있다면)
                real_space_loss = model.real_space_loss(z_diff, real_space_embedding, boxes_pred, real_space_boxes_pred)
                writer.add_scalar('Train/real_space_loss', real_space_loss.item(), counter)

            # 모델별 최적화 진행
            if args.network_type == 'v2_full':
                G_loss, D_loss, losses = model.vae_v2.optimize_params(
                    total_kl_loss, diff_loss, feat_shape, real_space_loss if use_real_space_this_batch else None,
                    args.real_space_loss_weight if use_real_space_this_batch else 0.0,
                    diff_out=diff_out, z=z_samples, loss_debug=False)
                
                writer.add_scalar('Train/KL_loss', total_kl_loss.item(), counter)
                writer.add_scalar('Train/diff_loss', diff_loss.item(), counter)
                writer.add_scalar('Train/G_loss', G_loss.item(), counter)
                writer.add_scalar('Train/D_loss', D_loss.item(), counter)
                
                # 추가 손실 기록
                for k, v in losses.items():
                    writer.add_scalar('Train/{}'.format(k), v, counter)
            else:
                # v1_box, v2_box, v1_full 모델용 최적화
                optimizer_bl.zero_grad()
                
                # 기본 손실
                loss = total_kl_loss + diff_loss
                
                # 현실 공간 손실 추가 (사용하는 경우)
                if use_real_space_this_batch and real_space_loss > 0:
                    loss += args.real_space_loss_weight * real_space_loss
                
                loss.backward()
                optimizer_bl.step()
                
                writer.add_scalar('Train/total_loss', loss.item(), counter)
                writer.add_scalar('Train/KL_loss', total_kl_loss.item(), counter)
                writer.add_scalar('Train/diff_loss', diff_loss.item(), counter)
                if use_real_space_this_batch:
                    writer.add_scalar('Train/real_space_loss', real_space_loss.item(), counter)

            # 로깅 및 모델 저장
            iter_net_time = time.time()
            eta = ((iter_net_time - iter_start_time) / (i + 1)) * len(dataloader) - (
                    iter_net_time - iter_start_time)
            
            if i % 25 == 0:
                print('[{}/{}][{}/{}] KL: {:.4f}, Diff: {:.4f}, REAL: {:.4f}, Mask: {:.2f}, triple: {:.2f}, ETA: {:.2f}h'.format(
                    epoch, args.nepoch, i, len(dataloader),
                    total_kl_loss.item(), diff_loss.item(), 
                    real_space_loss.item() if use_real_space_this_batch else 0.0,
                    mask_ratio, triplet_ratio, eta / 3600.0))

            counter += 1

        # 에폭 완료 후 모델 저장
        if epoch % args.snapshot == 0:
            model.save(args.exp, args.outf, epoch, counter)
        
        # 학습 종료 후 SpaceAdaptiveVAE 처리
        if args.use_space_adaptive and args.use_real_space and args.space_data_path and epoch % args.snapshot == 0:
            print("SpaceAdaptiveVAE 처리 시작...")
            
            # 1. 가상 씬 데이터셋 로드
            print(f"가상 씬 데이터셋 로드 중 ({args.virtual_scenes_dir})...")
            virtual_scenes = {}
            scene_files = glob.glob(os.path.join(args.virtual_scenes_dir, "*.json"))
            for scene_file in scene_files:
                scene_id = os.path.basename(scene_file).split('.')[0]
                with open(scene_file, 'r') as f:
                    virtual_scenes[scene_id] = json.load(f)
            print(f"{len(virtual_scenes)}개의 가상 씬 데이터 로드 완료")
            
            # 3. 현실 공간 데이터 로드 (아직 로드하지 않은 경우)
            if space_adaptive_vae.real_space_data is None:
                space_adaptive_vae.load_real_space_data(args.space_data_path)
            
            # 4. 현실 공간 임베딩 생성
            print("현실 공간 임베딩 생성 중...")
            real_space_embedding = space_adaptive_vae.train_with_real_space(args.space_data_path, dataloader)
            
            # 5. 유사 가상 씬 식별
            print("현실 공간과 유사한 가상 씬 식별 중...")
            similar_scenes = space_adaptive_vae.identify_similar_virtual_scenes(virtual_scenes, top_k=args.top_k_scenes)
            print(f"현실 공간과 가장 유사한 {len(similar_scenes)}개 가상 씬 식별 완료")
            
            # 6. 하이브리드 씬 생성
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

if __name__ == '__main__':
    train() 