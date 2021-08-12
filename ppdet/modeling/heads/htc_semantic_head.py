import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform, KaimingUniform
from paddle.regularizer import L2Decay

from paddle import ParamAttr

from ppdet.core.workspace import register, create
from ppdet.modeling import ops

from .bbox_head import BBoxHead, TwoFCHead, XConvNormHead
from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta, delta2bbox, clip_bbox, nonempty_bbox

__all__ = ['FusedSemanticHead']

@register
class FusedSemanticHead(nn.Layer):
    def __init__(self,
                 semantic_num_class=183):
        super(FusedSemanticHead, self).__init__()

        self.semantic_num_class = semantic_num_class

        self.lateral_convs = []
        self.convs = []

        for i in range(5):
            lateral_name = 'lateral_convs.{}.conv'.format(i)
            lateral = self.add_sublayer(
                lateral_name,
                nn.Conv2D(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    weight_attr=ParamAttr(
                        initializer=KaimingUniform())))
            self.lateral_convs.append(lateral)

        for i in range(4):
            fpn_name = 'convs.{}.conv'.format(i)
            fpn_conv = self.add_sublayer(
                fpn_name,
                nn.Conv2D(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=KaimingUniform())))
            self.convs.append(fpn_conv)

        self.conv_embedding = self.add_sublayer(
            'conv_embedding',
            nn.Conv2D(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                weight_attr=ParamAttr(
                    initializer=KaimingUniform())))

        self.conv_logits = self.add_sublayer(
            'conv_logits',
            nn.Conv2D(
                in_channels=256,
                out_channels=self.semantic_num_class,
                kernel_size=1,
                weight_attr=ParamAttr(
                    initializer=KaimingUniform())))

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
    # @classmethod
    # def from_config(cls, cfg, input_shape):
    #     s = input_shape
    #     s = s[0] if isinstance(s, (list, tuple)) else s
    #     return {'in_channel': s.channels}
    #
    # @property
    # def out_shape(self):
    #     return [ShapeSpec(channels=self.out_channel, )]

    def forward(self, body_feats):
        x = F.relu(self.lateral_convs[1](body_feats[1]))
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(body_feats):
            if i != 1:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += F.relu(self.lateral_convs[i](feat))

        for i in range(4):
            x = F.relu(self.convs[i](x))

        mask_pred = self.conv_logits(x)
        x = F.relu(self.conv_embedding(x))
        return mask_pred, x

    def loss(self, mask_pred, labels):
        # labels = labels.squeeze(1)
        labels = paddle.transpose(labels, perm=[0, 2, 3, 1]).astype('int64')
        mask_pred = paddle.transpose(mask_pred, perm=[0, 2, 3, 1])
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= 0.2  # self.loss_weight
        return loss_semantic_seg