import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import os
import glob
import numpy as np

import trimesh
from termcolor import colored
from model.VAEGAN_V1BOX import Sg2ScVAEModel as v1_box
from model.VAEGAN_V1FULL import Sg2ScVAEModel as v1_full
from model.VAEGAN_V2BOX import Sg2ScVAEModel as v2_box
from model.VAEGAN_V2FULL import Sg2ScVAEModel as v2_full

class VAE(nn.Module):
    # diff_opt는 "diffusion options"의 약자
    def __init__(self, root="../GT",type='v1_box', diff_opt = '../config/v2_full.yaml', vocab=None, replace_latent=False, with_changes=True, distribution_before=True,
                 residual=False, gconv_pooling='avg', with_angles=False, num_box_params=6, lr_full=None, deepsdf=False, clip=True, with_E2=True,
                 use_real_space=False, real_space_weight=0.3):
        super().__init__()
        assert type in ['v1_box', 'v1_full', 'v2_box', 'v2_full'], '{} is not included'.format(type)
        # v1_box: Graph-to-Box, v1_full: Graph-to-3D (DeepSDF version)
        # v2_box:layout branch of CommonScenes, v2_full: CommonScenes

        self.type_ = type
        self.vocab = vocab
        self.with_angles = with_angles
        self.epoch = 0
        self.v1full_database = os.path.join(root, "DEEPSDF_reconstruction")
        
        # SpaceAdaptive Model
        # 현실 공간 관련 파라미터 추가
        self.use_real_space = use_real_space # Lounge, Studio 씬그래프 정보 모두 포함.
        self.real_space_weight = real_space_weight
        self.real_space_embeddings = None
        self.space_data = None

        if self.type_ == 'v1_box':
            assert replace_latent is not None
            self.vae_box = v1_box(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                               input_dim=num_box_params, replace_latent=replace_latent, use_angles=with_angles,
                               residual=residual, gconv_pooling=gconv_pooling, gconv_num_layers=5)
        elif self.type_ == 'v1_full': # Graph-to-3D (DeepSDF version) # 장면 그래프 → VAE 인코더 → 잠재 벡터 → VAE 디코더 → DeepSDF 잠재 코드 → DeepSDF 디코더 → 3D 형상
            self.classes_ = sorted(list(set(self.vocab['object_idx_to_name'])))
            self.v1code_base = os.path.join(self.v1full_database, 'Codes')
            self.v1mesh_base = os.path.join(self.v1full_database, 'Meshes')
            # self.code_dict_path = os.path.join(self.v1full_database, 'deepsdf_code.json')
            id_names = os.listdir(self.v1code_base)
            self.code_dict = {}
            for id_name in id_names:
                latent_code = torch.load(os.path.join(self.v1code_base, id_name, 'sdf.pth'), map_location="cpu")[0]
                latent_code = latent_code.detach().numpy() # DeepSDF 네트워크에서 학습된 잠재 코드 저장
                self.code_dict[id_name] = latent_code[0]
            # assert: 특정 조건이 참인지 확인하고, 만약 조건이 거짓이면 프로그램을 중단
            # distribution_before가 True이면 GCN 이전에 분포를 적용하고, False이면 GCN 이후에 적용
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.vae = v1_full(vocab, embedding_dim=128, decoder_cat=True, mlp_normalization="batch",
                              gconv_num_layers=5, gconv_num_shared_layer=5, with_changes=with_changes, use_angles=with_angles,
                              distribution_before=distribution_before, replace_latent=replace_latent,
                              num_box_params=num_box_params, residual=residual, shape_input_dim=256 if deepsdf else 128)
        elif self.type_ == 'v2_box':
            assert replace_latent is not None
            self.vae_box = v2_box(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                               input_dim=num_box_params, replace_latent=replace_latent, use_angles=with_angles,
                               residual=residual, gconv_pooling=gconv_pooling, gconv_num_layers=5)
        elif self.type_ == 'v2_full':
            self.diff_opt = diff_opt
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.vae_v2 = v2_full(vocab, self.diff_opt, diffusion_bs=16, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                              gconv_num_layers=5, use_angles=with_angles, distribution_before=distribution_before, use_E2=with_E2, replace_latent=replace_latent,
                              num_box_params=num_box_params, residual=residual, clip=clip)
            self.vae_v2.optimizer_ini()
        self.counter = 0

    def set_cuda(self):
        """
        모델과 임베딩을 CUDA 장치로 이동시키는 메서드
        """
        if self.type_ == 'v2_full':
            self.vae_v2.set_cuda()
        elif self.type_ == 'v1_full':
            if hasattr(self, 'vae') and self.vae is not None:
                self.vae.set_cuda()
        elif self.type_ == 'v1_box' or self.type_ == 'v2_box':
            if hasattr(self, 'vae_box') and self.vae_box is not None:
                self.vae_box.set_cuda()
        
        # 현실 공간 임베딩이 있다면 CUDA로 이동
        if hasattr(self, 'real_space_embeddings') and self.real_space_embeddings is not None:
            self.real_space_embeddings = {k: v.cuda() for k, v in self.real_space_embeddings.items()}


    # SpaceAdaptive: 텐서 데이터 인코딩
    def encode_real_space_tensors(self, objs_tensor, boxes_tensor, triples_tensor):
        """
        이미 처리된 텐서 데이터를 인코딩하여 임베딩 벡터 생성
        
        Args:
            objs_tensor (torch.Tensor): 객체 클래스 텐서
            boxes_tensor (torch.Tensor): 바운딩 박스 텐서
            triples_tensor (torch.Tensor): 관계 텐서
            
        Returns:
            torch.Tensor: 인코딩된 임베딩 벡터
        """
        # 텐서를 GPU로 이동
        objs_tensor = objs_tensor.cuda()
        boxes_tensor = boxes_tensor.cuda()
        triples_tensor = triples_tensor.cuda()
        
        # 모델 타입에 따른 인코딩
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            mu, logvar = self.vae_box.encoder(objs_tensor, triples_tensor, boxes_tensor, None)
            embedding = mu  # 평균 벡터만 사용
        elif self.type_ == 'v1_full':
            mu, logvar = self.vae.encoder(objs_tensor, triples_tensor, boxes_tensor, None)
            embedding = mu
        elif self.type_ == 'v2_full':
            # CLIP 특성이 필요한 경우 더미 데이터 생성
            if self.vae_v2.clip:
                dummy_text_feats = torch.zeros((len(objs_tensor), 512)).cuda()
                dummy_rel_feats = torch.zeros((len(triples_tensor), 512)).cuda()
                mu, logvar = self.vae_v2.encoder(objs_tensor, triples_tensor, boxes_tensor, 
                                              None, dummy_text_feats, dummy_rel_feats)
            else:
                mu, logvar = self.vae_v2.encoder(objs_tensor, triples_tensor, boxes_tensor, None)
            embedding = mu
        return embedding

    def condition_with_real_space(self, real_space_embedding, z, alpha=None):
        """
        가상 공간 생성 시 현실 공간 임베딩으로 조건화하는 메서드
        
        Args:
            real_space_embedding (torch.Tensor): 현실 공간 임베딩
            z (torch.Tensor): 가상 공간 잠재 벡터
            alpha (float, optional): 현실 공간과 가상 공간의 혼합 비율 (0~1). None이면 self.real_space_weight 사용
        
        Returns:
            torch.Tensor: 조건화된 잠재 벡터
        """
        if alpha is None:
            alpha = self.real_space_weight
        
        # 현실 공간 임베딩과 가상 공간 잠재 벡터를 혼합
        conditioned_z = alpha * real_space_embedding + (1 - alpha) * z
        
        return conditioned_z
    
    def real_space_loss(self, z, real_space_embedding, boxes_pred, real_space_boxes_pred):
        """
        현실 공간 임베딩과 생성된 잠재 벡터 간의 손실을 계산하는 메서드
        
        Args:
            z (torch.Tensor): 생성된 잠재 벡터
            real_space_embedding (torch.Tensor): 현실 공간 임베딩
            boxes_pred (torch.Tensor): 생성된 바운딩 박스
            real_space_boxes_pred (torch.Tensor): 현실 공간 바운딩 박스
            
        Returns:
            torch.Tensor: 계산된 손실값
        """
        # 1. 잠재 공간에서의 손실 - 현실 공간 임베딩과 생성된 잠재 벡터 간의 MSE 손실
        latent_loss = F.mse_loss(z, real_space_embedding, reduction='mean')
        
        # 2. 바운딩 박스 공간에서의 손실 - 현실 공간 바운딩 박스와 생성된 바운딩 박스 간의 손실
        # 충돌 방지를 위한 손실 (겹치는 영역에 페널티)
        box_loss = 0.0
        if boxes_pred is not None and real_space_boxes_pred is not None:
            # 바운딩 박스 간 겹침 계산
            box_loss = self.calculate_box_overlap_loss(boxes_pred, real_space_boxes_pred)
        
        # 가중치 적용 및 손실 결합
        weighted_loss = (self.real_space_weight * latent_loss) + ((1 - self.real_space_weight) * box_loss)
        
        return weighted_loss
    
    def calculate_box_overlap_loss(self, boxes1, boxes2):
        """
        두 바운딩 박스 세트 간의 겹침을 계산하는 메서드
        
        Args:
            boxes1 (torch.Tensor): 첫 번째 바운딩 박스 세트 [N, 6] (x, y, z, dx, dy, dz)
            boxes2 (torch.Tensor): 두 번째 바운딩 박스 세트 [M, 6] (x, y, z, dx, dy, dz)
            
        Returns:
            torch.Tensor: 겹침에 대한 손실값
        """
        # 바운딩 박스 형식: [x, y, z, dx, dy, dz]
        # 각 바운딩 박스의 최소/최대 좌표 계산
        boxes1_min = boxes1[:, :3] - boxes1[:, 3:] / 2  # [N, 3]
        boxes1_max = boxes1[:, :3] + boxes1[:, 3:] / 2  # [N, 3]
        
        boxes2_min = boxes2[:, :3] - boxes2[:, 3:] / 2  # [M, 3]
        boxes2_max = boxes2[:, :3] + boxes2[:, 3:] / 2  # [M, 3]
        
        # 모든 박스 쌍에 대한 겹침 계산
        loss = 0.0
        for i in range(boxes1.shape[0]):
            for j in range(boxes2.shape[0]):
                # 두 박스의 겹치는 영역 계산
                overlap_min = torch.max(boxes1_min[i], boxes2_min[j])  # [3]
                overlap_max = torch.min(boxes1_max[i], boxes2_max[j])  # [3]
                
                # 겹치는 영역이 있는지 확인
                is_overlap = torch.all(overlap_min < overlap_max)
                
                if is_overlap:
                    # 겹치는 영역의 부피 계산
                    overlap_size = overlap_max - overlap_min  # [3]
                    overlap_volume = torch.prod(overlap_size)
                    
                    # 손실에 추가
                    loss += overlap_volume
        
        return loss if loss > 0 else torch.tensor(0.0).to(boxes1.device)

    def forward_mani(self, enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene, dec_objs, dec_objs_grained,
                     dec_triples, dec_boxes, dec_angles, dec_sdfs, dec_shapes, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes, real_space_id=None):

        # 현실 공간 임베딩 적용 (사용 설정된 경우)
        use_real_space_embedding = self.use_real_space and real_space_id is not None and self.real_space_embeddings is not None

        if self.type_ == 'v1_full':
            # mu, logvar: 잠재 공간의 분포 매개변수
            # boxes, angles, obj_and_shape: 모든 객체(변경된 객체 포함)의 예측값
            # orig_gt_boxes, orig_gt_angles, orig_gt_shapes: 원본 데이터
            mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, obj_and_shape, keep = \
                self.vae.forward(enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, attributes, enc_objs_to_scene,
                                 dec_objs, dec_triples, dec_boxes, dec_angles, dec_shapes, dec_attributes, dec_objs_to_scene,
                                 missing_nodes, manipulated_nodes)
            
            # 현실 공간 임베딩 적용
            if use_real_space_embedding:
                real_space_emb = self.real_space_embeddings[real_space_id]
                # 현실 공간 손실 계산 및 적용은 train() 함수에서 처리

            return mu, logvar, mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, obj_and_shape, keep

        elif self.type_ == 'v1_box':
            mu_boxes, logvar_boxes, orig_gt_boxes, orig_gt_angles, orig_boxes, orig_angles, boxes, angles, keep = self.vae_box.forward(
                enc_objs, enc_triples, enc_boxes, attributes, encoded_enc_text_feat, encoded_enc_rel_feat, enc_objs_to_scene, dec_objs, dec_triples, dec_boxes,
                dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, enc_angles, dec_angles)
            
            # 현실 공간 임베딩 적용
            if use_real_space_embedding:
                real_space_emb = self.real_space_embeddings[real_space_id]
                # 현실 공간 손실 계산 및 적용은 train() 함수에서 처리

            return mu_boxes, logvar_boxes, None, None, orig_gt_boxes, orig_gt_angles, None, orig_boxes, orig_angles, None, boxes, angles, None, keep

        elif self.type_ == 'v2_box':
            mu_boxes, logvar_boxes, orig_gt_boxes, orig_gt_angles, orig_boxes, orig_angles, boxes, angles, keep = self.vae_box.forward(
                enc_objs, enc_triples, enc_boxes, encoded_enc_text_feat, encoded_enc_rel_feat, attributes, enc_objs_to_scene, dec_objs, dec_triples, dec_boxes,
                encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, enc_angles, dec_angles)
            
            # 현실 공간 임베딩 적용
            if use_real_space_embedding:
                real_space_emb = self.real_space_embeddings[real_space_id]
                # 현실 공간 손실 계산 및 적용은 train() 함수에서 처리

            return mu_boxes, logvar_boxes, None, None, orig_gt_boxes, orig_gt_angles, None, orig_boxes, orig_angles, None, boxes, angles, None, keep
        
        elif self.type_ == 'v2_full':
            # 기존 forward 호출
            mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, boxes, angles, obj_and_shape, keep = \
                self.vae_v2.forward(enc_objs, enc_triples, enc_boxes, enc_text_feat=encoded_enc_text_feat, enc_rel_feat=encoded_enc_rel_feat, 
                                   attributes=attributes, enc_objs_to_scene=enc_objs_to_scene, dec_objs=dec_objs, dec_objs_grained=dec_objs_grained,
                                   dec_triples=dec_triples, dec=dec_boxes, dec_text_feat=encoded_dec_text_feat, dec_rel_feat=encoded_dec_rel_feat, 
                                   dec_attributes=dec_attributes, dec_objs_to_scene=dec_objs_to_scene, missing_nodes=missing_nodes, 
                                   manipulated_nodes=manipulated_nodes, dec_sdfs=dec_sdfs, enc_angles=enc_angles, dec_angles=dec_angles)
            
            # 현실 공간 임베딩 적용
            if use_real_space_embedding:
                real_space_emb = self.real_space_embeddings[real_space_id]
                # 현실 공간 손실 계산 및 적용은 train() 함수에서 처리
            
            return mu, logvar, mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, obj_and_shape, keep

    def load_networks(self, exp, epoch, strict=True, restart_optim=False):
        if self.type_ == 'v1_box':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                strict=strict
            )
        elif self.type_ == 'v1_full':
            print()
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch))).state_dict()
            # epoch=10이면, 체크포인트 파일은 'model10.pth'를 불러오게 됨
            self.vae.load_state_dict(
                ckpt,
                strict=strict
                # strict=True이면 모델을 불러올 때 모든 파라미터가 정확히 일치해야만 로드
            )
        elif self.type_ == 'v2_box':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                strict=strict
            )
        elif self.type_ == 'v2_full':
            from omegaconf import OmegaConf
            diff_cfg = OmegaConf.load(self.diff_opt)
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)))
            diff_state_dict = {}
            diff_state_dict['vqvae'] = ckpt.pop('vqvae')
            diff_state_dict['df'] = ckpt.pop('df')
            diff_state_dict['opt'] = ckpt.pop('opt')
            try:
                self.epoch = ckpt.pop('epoch')
                self.counter = ckpt.pop('counter')
            except:
                print('no epoch or counter record.')
            self.vae_v2.load_state_dict(
                ckpt,
                strict=strict
            )
            print(colored('[*] v2_box successfully restored from: %s' % os.path.join(exp, 'checkpoint',
                                                                                        'model{}.pth'.format(epoch)),
                          'blue'))
            self.vae_v2.Diff.vqvae.load_state_dict(diff_state_dict['vqvae'])
            self.vae_v2.Diff.df.load_state_dict(diff_state_dict['df'])

            # restart_optim=False → 이전에 학습된 옵티마이저 상태를 불러와서 학습을 이어서 진행 가능.
            if not restart_optim:
                import torch.optim as optim
                self.vae_v2.optimizerFULL.load_state_dict(diff_state_dict['opt'])
                # self.vae_v2.scheduler = optim.lr_scheduler.StepLR(self.vae_v2.optimizerFULL, 10000, 0.9)
                self.vae_v2.scheduler = optim.lr_scheduler.LambdaLR(self.vae_v2.optimizerFULL, lr_lambda=self.vae_v2.lr_lambda,
                                                        last_epoch=int(self.counter - 1))

            # for multi-gpu (deprecated)
            if diff_cfg.hyper.distributed:
                self.vae_v2.Diff.make_distributed(diff_cfg)
                self.vae_v2.Diff.df_module = self.vae_v2.Diff.df.module
                self.vae_v2.Diff.vqvae_module = self.vae_v2.Diff.vqvae.module
            else:
                self.vae_v2.Diff.df_module = self.vae_v2.Diff.df
                self.vae_v2.Diff.vqvae_module = self.vae_v2.Diff.vqvae
            print(colored('[*] v2_shape successfully restored from: %s' % os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch)), 'blue'))

    # 통계 정보 계산 또는 로드 # 평가에서 잠재 공간의 통계적 특성을 분석하여 모델이 학습한 분포를 이해
    # mean_est_box: 잠재 공간의 평균 벡터 cov_est_box: 잠재 공간의 공분산 행렬
    def compute_statistics(self, exp, epoch, stats_dataloader, force=False): # eval, inference 과정에서 사용
        box_stats_f = os.path.join(exp, 'checkpoint', 'model_stats_box_{}.pkl'.format(epoch))
        stats_f = os.path.join(exp, 'checkpoint', 'model_stats_{}.pkl'.format(epoch))
        if self.type_ == 'v1_box':
            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))

        elif self.type_ == 'v1_full':
            if os.path.exists(stats_f) and not force:
                stats = pickle.load(open(stats_f, 'rb'))
                self.mean_est, self.cov_est = stats[0], stats[1]
            else:
                self.mean_est, self.cov_est = self.vae.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est, self.cov_est], open(stats_f, 'wb'))
        elif self.type_ == 'v2_box':
            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))
        elif self.type_ == 'v2_full':
            if os.path.exists(stats_f) and not force:
                stats = pickle.load(open(stats_f, 'rb'))
                self.mean_est, self.cov_est = stats[0], stats[1]
            else:
                self.mean_est, self.cov_est = self.vae_v2.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est, self.cov_est], open(stats_f, 'wb'))

    # 이미 학습된 디코더를 사용하여 잠재 백터로부터 3D 객체 생성
    def decoder_with_changes_boxes_and_shape(self, z_box, z_shape, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, box_data=None, gen_shape=False):
        if self.type_ == 'v1_full':
            boxes, feats, keep = self.vae.decoder_with_changes(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            points, _ = self.decode_g2sv1(objs, feats, box_data, retrieval=True)
        elif self.type_ == 'v1_box' or self.type_ == 'v2_box':
            boxes, keep = self.decoder_with_changes_boxes(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes)
            points = None
        elif self.type_ == 'v2_full':
            boxes, sdfs, keep = self.vae_v2.decoder_with_changes(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                                               manipulated_nodes, gen_shape=gen_shape)
            return boxes, sdfs, keep

        return boxes, points, keep

    def decoder_with_changes_boxes(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            return self.vae_box.decoder_with_changes(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes)
        if self.type_ == 'v1_full':
            return None, None

    def decoder_with_changes_shape(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes, atlas):
        if self.type_ == 'v1_full':
            return None, None
    
    def decoder_boxes_and_shape(self, z_box, z_shape, objs, triples, attributes, atlas=None):
        angles = None
        if self.type_ == 'v1_full':
            boxes, angles, feats = self.vae.decoder(z_box, objs, triples, attributes)
            points = atlas.forward_inference_from_latent_space(feats, atlas.get_grid()) if atlas is not None else feats
        elif self.type_ == 'v1_box':
            boxes, angles = self.decoder_boxes(z_box, objs, triples, attributes)
            points = None
        return boxes, angles, points

    def decoder_boxes(self, z, objs, triples, attributes):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            if self.with_angles:
                return self.vae_box.decoder(z, objs, triples, attributes)
            else:
                return self.vae_box.decoder(z, objs, triples, attributes), None
        elif self.type_ == 'v1_full':
            return None, None

    def decoder_with_additions_boxes_and_shape(self, z_box, z_shape, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                               manipulated_nodes, gen_shape=False):
        if self.type_ == 'v1_full':
            outs, keep = self.vae.decoder_with_additions(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            return outs[:2], None, outs[2], keep
        elif self.type_ == 'v1_box' or self.type_ == 'v2_box':
            boxes, keep = self.decoder_with_additions_boxs(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes,
                                                                manipulated_nodes)
            return boxes, None, keep
        elif self.type_ == 'v2_full':
            boxes, sdfs, keep = self.vae_v2.decoder_with_additions(z_box, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes,
                                                         manipulated_nodes, gen_shape=gen_shape)
            return boxes, sdfs, keep
        else:
            print("error, no this type")

    def decoder_with_additions_boxs(self, z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes, manipulated_nodes):
        boxes, angles, keep = None, None, None
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            boxes, keep = self.vae_box.decoder_with_additions(z, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, attributes, missing_nodes,
                                                            manipulated_nodes, (self.mean_est_box, self.cov_est_box))

        elif self.type_ == 'v1_full':
            return  None, None, None
        return boxes, angles, keep

    # 이미 학습된 인코더를 사용하여 잠재 벡터 계산 평가/추론 단계에서 호출됨
    def encode_box_and_shape(self, objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, feats, boxes, angles=None, attributes=None):
        if not self.with_angles:
            angles = None
        if self.type_ == 'v1_box' or self.type_ == 'v2_box' or self.type_ == 'v2_full':
            return self.encode_box(objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, boxes, angles, attributes), (None, None)
        elif self.type_ == 'v1_full':
            with torch.no_grad():
                z, log_var = self.vae.encoder(objs, triples, boxes, feats, attributes, angles)
                return (z, log_var), (z, log_var)

    def encode_shape(self, objs, triples, feats, attributes=None):
        if self.type_ == 'v1_full':
            return None, None

    def encode_box(self, objs, triples, encoded_enc_text_feat, encoded_enc_rel_feat, boxes, angles=None, attributes=None):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            z, log_var = self.vae_box.encoder(objs, triples, boxes, attributes, encoded_enc_text_feat, encoded_enc_rel_feat, angles)
        elif self.type_ == 'v2_full':
            z, log_var = self.vae_v2.encoder(objs, triples, boxes, attributes, encoded_enc_text_feat,
                                              encoded_enc_rel_feat, angles)
        elif self.type_ == 'v1_full':
            return None, None
        return z, log_var

    def sample_box_and_shape(self, point_classes_idx, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None, gen_shape=False):
        if self.type_ == 'v1_full':
            return self.vae.sample_3dfront(point_classes_idx, self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)
        elif self.type_ == 'v2_full':
            return self.vae_v2.sample(point_classes_idx, self.mean_est, self.cov_est, dec_objs, dec_triplets, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat,
                                   attributes, gen_shape=gen_shape)
        boxes = self.sample_box(dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
        shapes = self.sample_shape(point_classes_idx, dec_objs, dec_triplets, attributes)
        return boxes, shapes

    def get_closest_vec(self, class_name, shape_vec, box_data):
        import numpy as np

        obj_ids = list(box_data[class_name].keys())
        # names = list(self.code_dict.keys())
        codes = np.vstack([self.code_dict[obj_id] for obj_id in obj_ids])
        mses = np.sum((codes - shape_vec.detach().cpu().numpy()) ** 2, axis=-1)
        id_min = np.argmin(mses)
        return obj_ids[id_min], codes[id_min]

    def decode_g2sv1(self, cats, shape_vecs, box_data, retrieval=False):
        if retrieval:
            vec_list = []
            mesh_list= []
            for (cat, shape_vec) in zip(cats, shape_vecs):
                class_name = self.classes_[cat].strip('\n')
                if class_name == 'floor' or class_name == '_scene_':
                    continue
                name_, vec_ = self.get_closest_vec(class_name, shape_vec, box_data)
                vec_list.append(vec_)
                obj = trimesh.load(os.path.join(self.v1mesh_base,name_,'sdf.ply'))
                mesh_list.append(obj)

        return mesh_list, vec_list

    def sample_box(self, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            return self.vae_box.sampleBoxes(self.mean_est_box, self.cov_est_box, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)
        elif self.type_ == 'v1_full':
            return self.vae.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)[0]
        elif self.type_ == 'v2_full':
            return self.vae_v2.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes)[0]

    def sample_shape(self, point_classes_idx, dec_objs, dec_triplets, attributes=None):
        if self.type_ == 'v1_full':
            return self.vae.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)[1]

    def sample_box_from_latent(self, z, dec_objs, dec_triplets, encoded_dec_text_feat, encoded_dec_rel_feat, attributes=None):
        """
        주어진 잠재 벡터에서 박스 샘플링
        
        Args:
            z: 잠재 벡터
            dec_objs, dec_triplets, ...: 디코딩 파라미터
            
        Returns:
            boxes: 생성된 바운딩 박스
        """
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            return self.vae_box.decoder(z, dec_objs, dec_triplets, attributes, encoded_dec_text_feat, encoded_dec_rel_feat)[0]
        elif self.type_ == 'v1_full':
            return self.vae.decoder(z, dec_objs, dec_triplets, attributes)[0]
        elif self.type_ == 'v2_full':
            return self.vae_v2.decoder(z, dec_objs, dec_triplets, attributes, encoded_dec_text_feat, encoded_dec_rel_feat)[0]
    
    # SpaceAdaptive Model
    def decoder_with_changes_boxes_and_shape_real(self, real_space_embedding, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, gen_shape=False, real_space_weight=0.3):
        """
        현실 공간 임베딩을 활용한 디코딩
        
        Args:
            real_space_embedding: 현실 공간 임베딩
            objs, triples, ...: 일반적인 디코딩 파라미터들
            real_space_weight: 현실 공간 임베딩 가중치
            
        Returns:
            boxes, shapes: 생성된 바운딩 박스와 형상
        """
        # 현실 공간 임베딩과 랜덤 잠재 벡터 생성
        batch_size = 1  # 기본값
        z_dim = real_space_embedding.shape[-1]
        z_random = torch.randn(batch_size, z_dim).cuda()
        
        # 현실 공간 임베딩과 랜덤 벡터 결합
        z_hybrid = self.condition_with_real_space(real_space_embedding, z_random, alpha=real_space_weight)
        
        # 결합된 잠재 벡터로 디코딩
        if self.type_ == 'v2_full':
            boxes, shapes, _ = self.vae_v2.decoder_with_changes(z_hybrid, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, gen_shape=gen_shape)
            return boxes, shapes
        else:
            # 다른 모델 타입에 대한 처리
            boxes, shapes, _ = self.decoder_with_changes_boxes_and_shape(z_hybrid, None, objs, triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_sdfs, attributes, missing_nodes, manipulated_nodes, gen_shape=gen_shape)
            return boxes, shapes

    # SpaceAdaptive Model
    def sample_box_and_shape_real(self, point_classes_idx, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, gen_shape=False, real_space_embedding=None, real_space_weight=0.3):
        """
        현실 공간 임베딩을 활용한 샘플링
        
        Args:
            point_classes_idx, dec_objs, ...: 일반적인 샘플링 파라미터들
            real_space_embedding: 현실 공간 임베딩
            real_space_weight: 현실 공간 임베딩 가중치
            
        Returns:
            boxes, shapes: 생성된 바운딩 박스와 형상
        """
        if real_space_embedding is None:
            # 일반적인 샘플링
            return self.sample_box_and_shape(point_classes_idx, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, gen_shape)
        
        # 현실 공간 임베딩과 랜덤 잠재 벡터 결합
        batch_size = 1  # 기본값
        z_dim = real_space_embedding.shape[-1]
        z_random = torch.randn(batch_size, z_dim).cuda()
        z_hybrid = self.condition_with_real_space(real_space_embedding, z_random, alpha=real_space_weight)
        
        # 결합된 잠재 벡터로 샘플링
        if self.type_ == 'v2_full':
            return self.vae_v2.sample_from_latent(point_classes_idx, z_hybrid, dec_objs, dec_triples, dec_sdfs, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes, gen_shape=gen_shape)
        elif self.type_ == 'v1_full':
            return self.vae.sample_from_latent(point_classes_idx, z_hybrid, dec_objs, dec_triples, dec_attributes)
        else:
            # v1_box, v2_box 처리
            boxes = self.sample_box_from_latent(z_hybrid, dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, dec_attributes)
            shapes = None
            return boxes, shapes

    def save(self, exp, outf, epoch, counter=None):
        if self.type_ == 'v1_box' or self.type_ == 'v2_box':
            torch.save(self.vae_box.state_dict(), os.path.join(exp, outf, 'model_box_{}.pth'.format(epoch)))
        elif self.type_ == 'v1_full':
            torch.save(self.vae, os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
        elif self.type_ == 'v2_full':
            torch.save(self.vae_v2.state_dict(epoch, counter), os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
