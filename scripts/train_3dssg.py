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
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.VAE import VAE
from model.discriminators import BoxDiscriminator, ShapeAuxillary
from dataset.dataset import RIODatasetSceneGraph
from model.losses import calculate_model_losses
from model.losses import bce_loss
from helpers.util import bool_flag, _CustomDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='../experiments/3dssgtest', help='Path to save experiment results')
parser.add_argument('--logf', default='logs', help='Path to save logs')
parser.add_argument('--outf', default='checkpoint', help='Path to save models')
parser.add_argument('--dataset', default='/home/commonscenes/datasample/3DSSG/3DSSG', help='Dataset path')
parser.add_argument('--workers', type=int, help='Number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=32, help='Batch size')
parser.add_argument('--nepoch', type=int, default=10000, help='Maximum number of epochs')
parser.add_argument('--weight_D_box', type=float, default=0.1, help='Box discriminator weight')
parser.add_argument('--auxlr', type=float, default=0.0001, help='Auxiliary network learning rate')
parser.add_argument('--shuffle_objs', default=False, type=bool_flag, help='Whether to shuffle objects')
parser.add_argument('--use_scene_rels', default=True, type=bool_flag, help='Whether to use scene relationships')
parser.add_argument('--with_changes', default=True, type=bool_flag, help='Whether to include changes')
parser.add_argument('--with_feats', default=True, type=bool_flag, help='Whether to include features')
parser.add_argument('--with_SDF', default=False, type=bool_flag, help='Whether to include SDF')
parser.add_argument('--with_CLIP', default=False, type=bool_flag, help='Whether to include CLIP features')
parser.add_argument('--large', default=False, type=bool_flag, help='Whether to use large dataset')
parser.add_argument('--loadmodel', default=False, type=bool_flag, help='Whether to load model')
parser.add_argument('--loadepoch', default=0, type=int, help='Epoch to load')
parser.add_argument('--pooling', default='avg', choices=['avg', 'max'], help='Pooling method')
parser.add_argument('--with_angles', default=False, type=bool_flag, help='Whether to include angles')
parser.add_argument('--num_box_params', default=6, type=int, help='Number of box parameters')
parser.add_argument('--residual', default=True, type=bool_flag, help='Whether to use residual connections')
parser.add_argument('--with_E2', default=True, type=bool_flag)
parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='v1_full', choices=['v2_box', 'v2_full', 'v1_box', 'v1_full'], type=str)
parser.add_argument('--diff_yaml', default='../config/v2_full.yaml', type=str, help='Diffusion network config')
parser.add_argument('--vis_num', type=int, default=8, help='Number of visualizations during training')

# python train_3dssg.py --exp ../experiments/3dssgtest --dataset /home/commonscenes/datasample/3DSSG/3DSSG --residual True --network_type v1_full --with_feats False --batchSize 8 --workers 8 --nepoch 1000

# Real space data related arguments
parser.add_argument('--real_space_data', default=None, type=str, help='Path to real space data')
parser.add_argument('--use_real_space', default=False, type=bool_flag, help='Whether to use real space data')
parser.add_argument('--real_space_weight', default=0.3, type=float, help='Real space embedding weight (0~1)')
parser.add_argument('--real_space_loss_weight', default=0.5, type=float, help='Real space loss weight')
parser.add_argument('--real_space_id', default='MasterBedroom-33296', type=str, help='Real space embedding identifier')

args = parser.parse_args()
print(args)

def parse_data(data):
    enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                                   data['encoder']['tripltes'], \
                                                                                   data['encoder']['boxes'], \
                                                                                   data['encoder']['obj_to_scene'], \
                                                                                   data['encoder']['triple_to_scene']
    
    # 기본 텐서들 GPU로 이동
    enc_objs = enc_objs.cuda()
    enc_triples = enc_triples.cuda()
    enc_tight_boxes = enc_tight_boxes.cuda()
    enc_objs_to_scene = enc_objs_to_scene.cuda()
    enc_triples_to_scene = enc_triples_to_scene.cuda()
    
    if args.with_feats:
        encoded_enc_f = data['encoder']['feats']
        encoded_enc_f = encoded_enc_f.cuda()
    else:
        encoded_enc_f = None

    encoded_enc_text_feat = None
    encoded_enc_rel_feat = None
    encoded_dec_text_feat = None
    encoded_dec_rel_feat = None
    if args.with_CLIP:
        encoded_enc_text_feat = data['encoder']['text_feats'].cuda()
        encoded_enc_rel_feat = data['encoder']['rel_feats'].cuda()
        encoded_dec_text_feat = data['decoder']['text_feats'].cuda()
        encoded_dec_rel_feat = data['decoder']['rel_feats'].cuda()

    dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                   data['decoder']['tripltes'], \
                                                                                   data['decoder']['boxes'], \
                                                                                   data['decoder']['obj_to_scene'], \
                                                                                   data['decoder']['triple_to_scene']
    
    # 디코더 텐서들 GPU로 이동
    dec_objs = dec_objs.cuda()
    dec_triples = dec_triples.cuda()
    dec_tight_boxes = dec_tight_boxes.cuda()
    dec_objs_to_scene = dec_objs_to_scene.cuda()
    dec_triples_to_scene = dec_triples_to_scene.cuda()
    
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

    # missing_nodes와 manipulated_nodes가 없는 경우를 처리
    try:
        missing_nodes = data['decoder']['missing_nodes']
        if isinstance(missing_nodes, torch.Tensor):
            missing_nodes = missing_nodes.cuda()
    except (KeyError, TypeError):
        missing_nodes = []  # 빈 리스트는 그대로 사용
        print("Warning: 'missing_nodes' not found in data")
    
    try:
        manipulated_nodes = data['decoder']['manipulated_nodes']
        if isinstance(manipulated_nodes, torch.Tensor):
            manipulated_nodes = manipulated_nodes.cuda()
    except (KeyError, TypeError):
        manipulated_nodes = []  # 빈 리스트는 그대로 사용
        print("Warning: 'manipulated_nodes' not found in data")

    attributes = None
    dec_attributes = None

    return enc_objs, enc_triples, enc_tight_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat, \
           attributes, enc_objs_to_scene, dec_objs, None, dec_triples, dec_tight_boxes, dec_angles, dec_sdfs, \
           encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, \
           manipulated_nodes

def train():
    args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)

    print(torch.__version__)

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # 데이터셋 초기화
    dataset = RIODatasetSceneGraph(
        root=args.dataset,
        root_3rscan="/mnt/dataset/3RScan/data/3RScan",
        label_file='labels.instances.align.annotated.v2.ply',
        npoints=1024,
        split='train_scans',  # 'test'에서 'train_scans'로 변경
        shuffle_objs=args.shuffle_objs,
        use_scene_rels=args.use_scene_rels,
        with_changes=args.with_changes,
        with_feats=args.with_feats,
        with_CLIP=args.with_CLIP,
        with_BERT=False,
        large=args.large,
        seed=False,
        use_splits=True,
        use_canonical=True
    )

    collate_fn = dataset.collate_fn_vaegan_points
    # collate_fn_vaegan_points는 이미 import되어 있음
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=int(args.workers))

    num_classes = len(dataset.classes)
    num_relationships = len(dataset.relationships) + 1

    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    
    model = VAE(root=args.dataset, type=args.network_type, diff_opt=args.diff_yaml, vocab=dataset.vocab,
                replace_latent=args.replace_latent, with_changes=args.with_changes, residual=args.residual,
                gconv_pooling=args.pooling, with_angles=args.with_angles, num_box_params=args.num_box_params,
                deepsdf=args.with_feats, clip=args.with_CLIP, with_E2=args.with_E2)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loadmodel:
        model.load_networks(exp=args.exp, epoch=args.loadepoch, restart_optim=False)

    if args.weight_D_box > 0:
        boxD = BoxDiscriminator(6, num_relationships, num_classes)
        optimizerDbox = optim.Adam(filter(lambda p: p.requires_grad, boxD.parameters()), lr=args.auxlr,
                                   betas=(0.9, 0.999))
        boxD.cuda()
        boxD = boxD.train()

    shapeClassifier = ShapeAuxillary(256, len(dataset.cat))
    shapeClassifier = shapeClassifier.cuda()
    shapeClassifier.train()
    optimizerShapeAux = optim.Adam(filter(lambda p: p.requires_grad, shapeClassifier.parameters()), lr=args.auxlr,
                                   betas=(0.9, 0.999))
    
    writer = SummaryWriter(args.exp + "/" + args.logf)

    if args.network_type != 'v2_full':
        params = filter(lambda p: p.requires_grad, list(model.parameters()))
        optimizer_bl = optim.Adam(params, lr=args.auxlr)
        optimizer_bl.step()

    print("---- 모델 및 데이터셋 구축 완료 ----")

    if not os.path.exists(args.exp + "/" + args.outf):
        os.makedirs(args.exp + "/" + args.outf)

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
            try:
                enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_f, encoded_enc_text_feat, encoded_enc_rel_feat,\
                attributes, enc_objs_to_scene, dec_objs, dec_objs_grained, dec_triples, dec_boxes, dec_angles, dec_sdfs,\
                encoded_dec_f, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, dec_objs_to_scene, missing_nodes,\
                manipulated_nodes = parse_data(data)
            except Exception as e:
                print('Exception', str(e))
                continue

            if args.network_type != 'v2_full':
                optimizer_bl.zero_grad()
            else:
                model.vae_v2.optimizerFULL.zero_grad()
            
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

            boxGloss = 0
            loss_genShape = 0
            loss_genShapeFake = 0
            loss_shape_fake_g = 0
            new_shape_loss, new_shape_losses = 0, 0

            if args.network_type == 'v1_full':
                shape_logits_fake_d, probs_fake_d = shapeClassifier(obj_and_shape[1].detach())
                shape_logits_fake_g, probs_fake_g = shapeClassifier(obj_and_shape[1])
                shape_logits_real, probs_real = shapeClassifier(encoded_dec_f.detach())

                loss_shape_real = torch.nn.functional.cross_entropy(shape_logits_real, obj_and_shape[0])
                loss_shape_fake_d = torch.nn.functional.cross_entropy(shape_logits_fake_d, obj_and_shape[0])
                loss_shape_fake_g = torch.nn.functional.cross_entropy(shape_logits_fake_g, obj_and_shape[0])
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
                vae_loss_shape, vae_losses_shape = 0, 0

            if args.with_changes:
                oriented_gt_boxes = torch.cat([dec_boxes], dim=1)
                boxes_pred_in = keep * oriented_gt_boxes + (1 - keep) * dec_man_enc_box_pred

                if args.weight_D_box == 0:
                    boxGloss = 0
                    gamma = 0.1
                    boxDloss_real = 0
                    boxDloss_fake = 0
                    reg_loss = 0
                else:
                    logits, _ = boxD(dec_objs, dec_triples, boxes_pred_in, keep)
                    logits_fake, reg_fake = boxD(dec_objs, dec_triples, boxes_pred_in.detach(), keep, with_grad=True,
                                                 is_real=False)
                    logits_real, reg_real = boxD(dec_objs, dec_triples, oriented_gt_boxes, with_grad=True, is_real=True)
                    boxGloss = bce_loss(logits, torch.ones_like(logits))
                    gamma = 0.1
                    boxDloss_real = bce_loss(logits_real, torch.ones_like(logits_real))
                    boxDloss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
                    reg_loss = torch.mean(reg_real + reg_fake)

                boxDloss = boxDloss_fake + boxDloss_real + (gamma / 2.0) * reg_loss
                optimizerDbox.zero_grad()
                boxDloss.backward()

            loss = vae_loss_box + vae_loss_shape + 0.1 * loss_genShape + 100 * new_shape_loss
            
            if args.with_changes:
                loss += args.weight_D_box * boxGloss

            loss.backward(retain_graph=True)

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