from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
import time

sys.path.append('../')
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from model.VAE import VAE
from model.discriminators import BoxDiscriminator, ShapeAuxillary
from model.losses import bce_loss
from helpers.util import bool_flag, _CustomDataParallel

from model.losses import calculate_model_losses

import torch.nn.functional as F
import json

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# standard hyperparameters, batch size, learning rate, etc
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--auxlr', type=float, help='auxiliary learning rate, not for v2_full', default=0.0001)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
# 데이터셋을 한 바퀴(전체) 학습하는 횟수. 데이터셋이 1000개 샘플로 구성되어 있다면, 1 Epoch 동안 모델은 모든 1000개 샘플을 한 번 학습.
# Epoch이 너무 작으면 → 모델이 충분히 학습되지 않음 (Underfitting). # Epoch이 너무 크면 → 모델이 과적합(Overfitting)될 가능성이 있음.
# Validation Loss가 줄어들지 않는 지점에서 멈추는 게 좋음

# paths and filenames
parser.add_argument('--outf', type=str, default='checkpoint', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', required=False, type=str, default="/mnt/dataset/FRONT",
                    help="dataset path")
parser.add_argument('--logf', default='logs', help='folder to save tensorboard logs')
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name') #학습된 모델이 저장되는 경로
parser.add_argument('--room_type', default='bedroom', help='room type [bedroom, livingroom, diningroom, library, all]')

# GCN parameters
parser.add_argument('--residual', type=bool_flag, default=False, help="residual in GCN")
parser.add_argument('--pooling', type=str, default='avg', help="pooling method in GCN")

# dataset related
parser.add_argument('--large', default=False, type=bool_flag,
                    help='large set of class labels. Use mapping.json when false')
parser.add_argument('--use_scene_rels', type=bool_flag, default=True, help="connect all nodes to a root scene node")

parser.add_argument('--use_E2', type=bool_flag, default=True, help="if use relation encoder in the diffusion branch")
parser.add_argument('--with_SDF', type=bool_flag, default=False)  # TODO
#TODO
parser.add_argument('--with_feats', type=bool_flag, default=False,
                    help="if true reads latent point features instead of pointsets."
                         "If not existing, they get generated at the beginning.")  #TODO
parser.add_argument('--with_CLIP', type=bool_flag, default=True,
                    help="if use CLIP features. Set true for the full version")
parser.add_argument('--shuffle_objs', type=bool_flag, default=True, help="shuffle objs of a scene")
parser.add_argument('--use_canonical', default=True, type=bool_flag)  # TODO
parser.add_argument('--with_angles', default=True, type=bool_flag)
parser.add_argument('--num_box_params', default=6, type=int, help="number of the dimension of the bbox. [6,7]")

# training and architecture related
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--weight_D_box', default=0.1, type=float, help="Box Discriminator")
parser.add_argument('--with_changes', default=True, type=bool_flag)

parser.add_argument('--loadmodel', default=False, type=bool_flag)
parser.add_argument('--loadepoch', default=90, type=int, help='only valid when loadmodel is true')

parser.add_argument('--with_E2', default=True, type=bool_flag)
parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='v2_full', choices=['v2_box', 'v2_full', 'v1_box', 'v1_full'], type=str)
parser.add_argument('--diff_yaml', default='../config/v2_full.yaml', type=str,
                    help='config of the diffusion network [cross_attn/concat]')

parser.add_argument('--vis_num', type=int, default=8, help='for visualization in the training')

args = parser.parse_args()
print(args)
# python train_3dfront.py --room_type livingroom --dataset /mnt/dataset/FRONT --residual True --network_type v2_full --with_SDF True --with_CLIP True --batchSize 8 --workers 8 --loadmodel True --loadepoch 315 --nepoch 2051 --large False

# python train_3dfront.py --room_type livingroom --dataset /mnt/dataset/FRONT --residual True 
# --network_type v2_full --with_SDF True --with_CLIP True 
# v1_box: Graph-to-Box, v1_full: Graph-to-3D (DeepSDF version) v2_box:layout branch of CommonScenes, v2_full: CommonScenes
# --batchSize 8 --workers 8 --loadmodel False --nepoch 10000 --large False

# CLIP 없이도 가능한지 테스트용
# python train_3dfront.py --room_type livingroom --dataset /mnt/dataset/FRONT --residual True --network_type v2_full --with_SDF True --with_CLIP False --batchSize 8 --workers 8 --exp ../experiments/test

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
                                                                                      data['decoder']['obj_to_scene'], \
                                                                                      data['decoder']['triple_to_scene']
    dec_objs_grained = data['decoder']['objs_grained']
    dec_sdfs = None
    if 'sdfs' in data['decoder']:
        dec_sdfs = data['decoder']['sdfs']
    if 'feats' in data['decoder']:
        encoded_dec_f = data['decoder']['feats']
        encoded_dec_f = encoded_dec_f.cuda()

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
            encoded_enc_f = None  # TODO #HJS
            encoded_dec_f = None  # TODO

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
    """ Train the network based on the provided argparse parameters
    """
    args.manualSeed = random.randint(1, 10000)  # optionally fix seed 7494
    print("Random Seed: ", args.manualSeed)

    print(torch.__version__)

    random.seed(args.manualSeed) # Python 기본 random 모듈의 시드 설정
    torch.manual_seed(args.manualSeed) # PyTorch의 시드 설정. 랜덤성을 통제. PyTorch 내부의 텐서 초기화, 데이터 샘플링, 모델 학습 중 발생하는 랜덤성을 고정.

    # instantiate scene graph dataset for training
    dataset = ThreedFrontDatasetSceneGraph( #dataset/threedfront_dataset.py에서 갖고 옴
        root=args.dataset, # /mnt/dataset/FRONT
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
    # instantiate data loader from dataset # 모델 학습에 맞게 배치(batch) 단위로 로드
    # 학습 전에 배치사이즈 결정 및 결정에 따라 데이터셋 개수 로드
    # 배치를 collate_fn으로 딕셔너리 형태로 저장 # DataLoader에서 배치 인자를 자동으로 전달
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize, # 한 번의 학습에 사용할 샘플 개수
        collate_fn=collate_fn, # 복잡한 데이터 구조를 가져서 사용자 정의가 필요함.
        shuffle=True,
        num_workers=int(args.workers))

    # number of object classes and relationship classes
    num_classes = len(dataset.classes)
    num_relationships = len(dataset.relationships) + 1

    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    # instantiate the model
    # residual=args.residual => distribution_before가 True이면 GCN 이전에 분포를 적용하고, False이면 GCN 이후에 적용
    # 잠재 공간의 확률 분포(일반적으로 가우시안 분포) # 분포에서 샘플링함으로써 같은 입력에 대해서도 약간씩 다른 출력을 생성
    # KL 발산(Kullback-Leibler divergence)을 통해 잠재 분포가 표준 정규 분포에 가까워지도록 유도
    model = VAE(root=args.dataset, type=args.network_type, diff_opt=args.diff_yaml, vocab=dataset.vocab,
                replace_latent=args.replace_latent, with_changes=args.with_changes, residual=args.residual,
                gconv_pooling=args.pooling, with_angles=args.with_angles, num_box_params=args.num_box_params,
                deepsdf=args.with_feats, clip=args.with_CLIP, with_E2=args.with_E2)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loadmodel: # 학습된 모델을 불러오는 경우
        model.load_networks(exp=args.exp, epoch=args.loadepoch, restart_optim=False)
        # model/VAE 안에 load_networks() 함수 호출
        # exp=args.exp => 학습된 모델이 저장되는 경로
        # checkpoint: 학습 중간 결과를 저장하는 모델 파일 (.pth)

    ## From Graph-to-3D
    # instantiate a relationship discriminator that considers the boxes and the semantic labels
    # if the loss weight is larger than zero
    # also create an optimizer for it
    if args.weight_D_box > 0: #box discriminator weight
        boxD = BoxDiscriminator(6, num_relationships, num_classes)
        optimizerDbox = optim.Adam(filter(lambda p: p.requires_grad, boxD.parameters()), lr=args.auxlr,
                                   betas=(0.9, 0.999))
        boxD.cuda()
        boxD = boxD.train()

    ## From Graph-to-3D
    # instantiate auxiliary discriminator for shape and a respective optimizer
    shapeClassifier = ShapeAuxillary(256, len(dataset.cat)) # 모델 호출 시 자동으로 forward() 실행됨
    shapeClassifier = shapeClassifier.cuda()
    shapeClassifier.train()
    optimizerShapeAux = optim.Adam(filter(lambda p: p.requires_grad, shapeClassifier.parameters()), lr=args.auxlr,
                                   betas=(0.9, 0.999))
    # parameters(): PyTorch에서 모든 nn.Module 클래스(모든 신경망 모델의 기본 클래스)에 내장된 메소드 # 모델의 모든 학습 가능한 파라미터(가중치와 편향)를 반환
    # filter(): 주어진 함수를 만족하는 요소만 추출
    # lambda p: p.requires_grad: 모든 파라미터에 대해 requires_grad가 True인 것만 추출 # requires_grad: 파라미터가 학습 가능한지 여부를 결정
    # initialize tensorboard writer
    writer = SummaryWriter(args.exp + "/" + args.logf)

    # optimizer for model v1 and v2_box. if it is v2_full, use its own optimizer.
    if args.network_type != 'v2_full':
        params = filter(lambda p: p.requires_grad, list(model.parameters()))
        optimizer_bl = optim.Adam(params, lr=args.auxlr) # auxlr: 학습률(Learning Rate) 얼마나 빠르게 학습할지를 결정
        optimizer_bl.step() # optimizer_bl: VAE 모델의 가중치를 업데이트하는 역할
        # 옵티마이저는 계산된 Loss의 기울기(gradient)를 사용하여 모델의 파라미터를 업데이트

    # v1_box: Shape와 Bounding Box의 잠재 공간(Latent Space)이 분리
    # v1_full: shared - Shape와 Bounding Box가 공유된 잠재 공간(Latent Space)
    # v2_box
    # v2_full: shared - Shape와 Bounding Box가 공유된 잠재 공간(Latent Space)

    print("---- Model and Dataset built ----")

    if not os.path.exists(args.exp + "/" + args.outf):
        os.makedirs(args.exp + "/" + args.outf)

    # save parameters so that we can read them later on evaluation
    # json 형태로 arguments parameters들을 저장함.
    with open(os.path.join(args.exp, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Saving all parameters under:")
    print(os.path.join(args.exp, 'args.json'))
    
    torch.autograd.set_detect_anomaly(True)
    # 모델의 파라미터에 대한 기울기(gradient)를 계산하고 업데이트하는 과정에서 발생하는 문제를 자동으로 감지하고 디버깅하는 기능
    counter = model.counter if model.counter else 0
    # 모델의 학습 횟수를 카운트하는 변수

    print("---- Starting training loop! ----")
    iter_start_time = time.time()
    start_epoch = model.epoch if model.epoch else 0
    for epoch in range(start_epoch, args.nepoch):
        print('Epoch: {}/{}'.format(epoch, args.nepoch))
        for i, data in enumerate(dataloader, 0):

            # parse the data to the network
            try:
                enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, dec_objs_to_scene, missing_nodes,\
                manipulated_nodes = parse_data(data)
            except Exception as e:
                print('Exception', str(e))
                continue

            if args.network_type != 'v2_full':
                optimizer_bl.zero_grad() # zero_grad()는 옵티마이저에 연결된 모든 파라미터의 기울기를 0으로 초기화
            else:
                model.vae_v2.optimizerFULL.zero_grad()
            # v2 also uses this
            optimizerShapeAux.zero_grad()

            model = model.train()

            if args.weight_D_box > 0:
                optimizerDbox.zero_grad()

            model_out = model.forward_mani(enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f,
                                           encoded_enc_text_feat, encoded_enc_rel_feat, attributes,
                                           enc_objs_to_scene,
                                           dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,
                                           encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes,
                                           dec_objs_to_scene,
                                           missing_nodes, manipulated_nodes)

            mu_box, logvar_box, mu_shape, logvar_shape, orig_gt_box, orig_gt_angle, orig_gt_shape, orig_box, orig_angle, orig_shape, \
            dec_man_enc_box_pred, dec_man_enc_angle_pred, obj_and_shape, keep = model_out

            ## From Graph-to-3D
            # initiate the loss
            boxGloss = 0
            loss_genShape = 0
            loss_genShapeFake = 0
            loss_shape_fake_g = 0
            new_shape_loss, new_shape_losses = 0, 0

            if args.network_type == 'v1_full':
                shape_logits_fake_d, probs_fake_d = shapeClassifier(obj_and_shape[1].detach())
                shape_logits_fake_g, probs_fake_g = shapeClassifier(obj_and_shape[1])
                shape_logits_real, probs_real = shapeClassifier(encoded_dec_f.detach())

                # auxiliary loss. can the discriminator predict the correct class for the generated shape?
                loss_shape_real = torch.nn.functional.cross_entropy(shape_logits_real, obj_and_shape[0])
                loss_shape_fake_d = torch.nn.functional.cross_entropy(shape_logits_fake_d, obj_and_shape[0])
                loss_shape_fake_g = torch.nn.functional.cross_entropy(shape_logits_fake_g, obj_and_shape[0])
                # standard discriminator loss
                loss_genShapeFake = bce_loss(probs_fake_g, torch.ones_like(probs_fake_g))
                loss_dShapereal = bce_loss(probs_real, torch.ones_like(probs_real))
                loss_dShapefake = bce_loss(probs_fake_d, torch.zeros_like(probs_fake_d))

                loss_dShape = loss_dShapefake + loss_dShapereal + loss_shape_real + loss_shape_fake_d
                loss_genShape = loss_genShapeFake + loss_shape_fake_g
                loss_dShape.backward()

            vae_loss_box, vae_losses_box = calculate_model_losses(args,
                                                                  orig_gt_box,
                                                                  orig_box,
                                                                  name='box', withangles=args.with_angles,
                                                                  angles_pred=orig_angle,
                                                                  mu=mu_box, logvar=logvar_box, angles=orig_gt_angle,
                                                                  KL_weight=0.1, writer=writer, counter=counter)
            if args.network_type == 'v1_full':
                vae_loss_shape, vae_losses_shape = calculate_model_losses(args,
                                                                          orig_gt_shape,
                                                                          orig_shape,
                                                                          name='shape', withangles=False,
                                                                          mu=mu_shape, logvar=logvar_shape,
                                                                          KL_weight=0.1, writer=writer, counter=counter)
            elif args.network_type == 'v2_full':
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

            loss = vae_loss_box + vae_loss_shape + 0.1 * loss_genShape + 100 * new_shape_loss
            if args.with_changes:
                loss += args.weight_D_box * boxGloss  # + b_loss

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

            counter += 1
            if counter % 100 == 0:
                message = "loss at {}: box {:.4f}\tshape {:.4f}\tdiscr RealFake {:.4f}\t discr Classifcation {:.4f}\t".format(
                    counter, vae_loss_box, vae_loss_shape, loss_genShapeFake,
                    loss_shape_fake_g)
                if args.network_type == 'v2_full':
                    loss_diff = model.vae_v2.Diff.get_current_errors()
                    for k, v in loss_diff.items():
                        message += '%s: %.6f ' % (k, v)
                print(message)

            writer.add_scalar('Train_Loss_BBox', vae_loss_box, counter)
            writer.add_scalar('Train_Loss_Shape', vae_loss_shape, counter)
            writer.add_scalar('Train_Loss_loss_genShapeFake', loss_genShapeFake, counter)
            writer.add_scalar('Train_Loss_loss_shape_fake_g', loss_shape_fake_g, counter)

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

        if epoch % 20 == 0:
            model.save(args.exp, args.outf, epoch, counter=counter)
            print('saved model_{}'.format(epoch))

    writer.close()


if __name__ == "__main__":
    train()


# python train_3dfront.py --room_type livingroom --dataset /mnt/dataset/FRONT --residual True --network_type v2_full --with_SDF True --with_CLIP True --batchSize 8 --workers 8 --loadmodel True --loadepoch 480 --nepoch 2041 --large False