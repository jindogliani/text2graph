import torch.nn as nn
import torch
import torch.nn.functional as F


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class ObjBoxDiscriminator(nn.Module):
    """
    Discriminator that considers a bounding box and an object class label and judges if its a
    plausible configuration
    """
    def __init__(self, box_dim, obj_dim):
        super(ObjBoxDiscriminator, self).__init__()

        self.obj_dim = obj_dim

        self.D = nn.Sequential(nn.Linear(box_dim+obj_dim, 512),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(),
                               nn.Linear(512, 512),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(),
                               nn.Linear(512, 1),
                               nn.Sigmoid())

        self.D.apply(_init_weights)

    def forward(self, objs, boxes, with_grad=False, is_real=False):

        objectCats = to_one_hot_vector(self.obj_dim, objs)

        x = torch.cat([objectCats, boxes], 1)
        reg = None
        if with_grad:
            x.requires_grad = True
            y = self.D(x)
            reg = discriminator_regularizer(y, x, is_real)
            x.requires_grad = False
        else:
            y = self.D(x)
        return y, reg


class ShapeAuxillary(nn.Module):
    """
    Auxiliary discriminator that receives a shape encoding and judges if it is plausible and
    simultaneously predicts a class label for the given shape encoding
    """
    # __init__ 함수에서 shape_dim과 num_classes를 받아서 내부 네트워크를 구성
    def __init__(self, shape_dim, num_classes): 
        super(ShapeAuxillary, self).__init__()

        # 딥러닝 모델의 layer 정의
        self.D = nn.Sequential(nn.Linear(shape_dim, 512), # Fully Connected(FC) Layer  # 입력: shape_dim 차원 → 출력: 512 차원
                               nn.BatchNorm1d(512), # Batch Normalization Layer # 입력 데이터를 평균 0, 분산 1로 정규화하여 학습을 더 빠르고 안정적으로 진행
                               nn.LeakyReLU(), # 비선형성을 추가하여 모델이 더 복잡한 패턴을 학습
                               nn.Linear(512, 512), # 입력: 512 차원 → 출력: 512 차원
                               nn.BatchNorm1d(512), # Batch Normalization Layer # 모델의 수렴 속도를 높이고 overfitting을 방지
                               nn.LeakyReLU()
                               )
        self.classifier = nn.Linear(512, num_classes) # 512차원 feature를 입력받아 객체의 클래스(class) 예측
        self.discriminator = nn.Linear(512, 1) # 512차원 feature를 입력받아 객체의 shape이 실제(real)인지 가짜(fake)인지 판별

        self.D.apply(_init_weights)
        self.classifier.apply(_init_weights)
        self.discriminator.apply(_init_weights)

    def forward(self, shapes):

        backbone = self.D(shapes) # shape_dim 차원의 객체 shape encoding이 들어옴. 현재는 256차원 feature. # backbone이라는 512차원 feature가 생성
        # logits은 객체가 어떤 카테고리에 속할 가능성이 있는지 나타내는 값 (이 값에 softmax()를 적용하면 확률이 됨.)
        logits = self.classifier(backbone) # self.classifier 레이어를 통과 class를 예측
        realfake = torch.sigmoid(self.discriminator(backbone)) # self.discriminator 레이어를 통과 Real(1)/Fake(0) 판별

        return logits, realfake


class BoxDiscriminator(nn.Module):
    """
    Relationship discriminator based on bounding boxes. For a given object pair, it takes their
    semantic labels, the relationship label and the two bounding boxes of the pair and judges
    whether this is a plausible occurence.
    """
    # __init__ 함수에서 box_dim(6), rel_dim(num_relationships), obj_dim(num_classes)을 받아서 내부 네트워크를 구성
    def __init__(self, box_dim, rel_dim, obj_dim, with_obj_labels=True):
        super(BoxDiscriminator, self).__init__()

        self.rel_dim = rel_dim
        self.obj_dim = obj_dim
        self.with_obj_labels = with_obj_labels
        if self.with_obj_labels:
          in_size = box_dim*2+rel_dim+obj_dim*2
        else:
          in_size = box_dim*2+rel_dim

        self.D = nn.Sequential(nn.Linear(in_size, 512),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(),
                               nn.Linear(512, 512),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(),
                               nn.Linear(512, 1),
                               nn.Sigmoid())

        self.D.apply(_init_weights)

    def forward(self, objs, triples, boxes, keeps=None, with_grad=False, is_real=False):

        s_idx, predicates, o_idx = triples.chunk(3, dim=1)
        predicates = predicates.squeeze(1)
        s_idx = s_idx.squeeze(1)
        o_idx = o_idx.squeeze(1)
        subjectBox = boxes[s_idx]
        objectBox = boxes[o_idx]

        if keeps is not None: # keeps는 "학습에서 제외해야 할 관계들을 걸러내는 필터 역할"을 한다!
            subjKeeps = keeps[s_idx]
            objKeeps = keeps[o_idx]
            keep_t = ((1 - subjKeeps) + (1 - objKeeps)) > 0

        predicates = to_one_hot_vector(self.rel_dim, predicates) # 숫자 라벨(3)을 One-hot벡터([0, 0, 1, 0, 0])로 변환

        if self.with_obj_labels:
            subjectCat = to_one_hot_vector(self.obj_dim, objs[s_idx]) #One-hot Encoding을 사용해서 Discriminator가 "객체의 종류"를 직접 이해
            objectCat = to_one_hot_vector(self.obj_dim, objs[o_idx])

            x = torch.cat([subjectCat, objectCat, predicates, subjectBox, objectBox], 1) # 여러개의 텐서를 하나로 합침

        else:
            x = torch.cat([predicates, subjectBox, objectBox], 1)

        reg = None
        # Regularization을 추가하는 이유
        # GAN에서 Discriminator가 너무 강해지는 것을 방지하고, Generator가 안정적으로 학습할 수 있도록 하기 위해 사용됨.
        # Gradient Penalty를 사용하면 Discriminator가 더 부드러운 결정을 내릴 수 있음.
        if with_grad: # Discriminator의 출력(y)에 대한 Gradient Penalty 적용 # Gradient Penalty를 추가하면 학습이 더 안정적으로 진행
            x.requires_grad = True
            y = self.D(x)
            reg = discriminator_regularizer(y, x, is_real) # Discriminator의 Regularization 값 (일반적으로 Gradient Penalty를 의미)
            x.requires_grad = False
        else:
            y = self.D(x)
        if keeps is not None and reg is not None:
            return y[keep_t], reg[keep_t]
        elif keeps is not None and reg is None:
            return y[keep_t], reg
        else:
            return y, reg


def discriminator_regularizer(logits, arg, is_real):

    logits.backward(torch.ones_like(logits), retain_graph=True)
    grad_logits = arg.grad
    grad_logits_norm = torch.norm(grad_logits, dim=1).unsqueeze(1)

    assert grad_logits_norm.shape == logits.shape

    # tf.multiply -> element-wise mul
    if is_real:
        reg = (1.0 - logits)**2 * (grad_logits_norm)**2
    else:
        reg = (logits)**2 * (grad_logits_norm)**2

    return reg


def to_one_hot_vector(num_class, label):
    """ Converts a label to a one hot vector

    :param num_class: number of object classes
    :param label: integer label values
    :return: a vector of the length num_class containing a 1 at position label, and 0 otherwise
    """
    return torch.nn.functional.one_hot(label, num_class).float()
