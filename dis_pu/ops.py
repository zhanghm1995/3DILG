from PUCRN.PUCRN import Conv1d as conv1d
from PUCRN.PUCRN import Conv2d as conv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k + 1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = torch.unsqueeze(point_cloud, dim=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = torch.cat(
        (point_cloud_central, point_cloud_neighbors - point_cloud_central), dim=-1)
    return edge_feature, idx


def dense_conv(feature, n=3, growth_rate=64, k=16, scope='dense_conv', **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = torch.cat((
                    conv2d(y, growth_rate, scope='l%d' % i, **kwargs),
                    tf.tile(torch.unsqueeze(feature, dim=2), [1, 1, k, 1])), dim=-1)
            elif i == n - 1:
                y = torch.cat((
                    conv2d(y, growth_rate, scope='l%d' % i, activation_fn=None, **kwargs),
                    y), dim=-1)
            else:

                y = torch.cat((
                    conv2d(y, growth_rate, scope='l%d' % i, **kwargs),
                    y), dim=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx

def feature_extraction_GCN(inputs, growth_rate=24, is_training=True, bn_decay=None,
                           use_bn=False, dense_block=2):
    # use_bn = False
    # use_ibn = False
    # growth_rate = 24

    dense_n = 3
    knn = 16
    comp = growth_rate * 2
    l0_features = torch.unsqueeze(inputs, dim=2)
    l0_features = conv2d(l0_features, 24, [1, 1],
                         padding='VALID', scope='layer0', is_training=is_training, bn=use_bn,
                         bn_decay=bn_decay, activation_fn=None)
    l0_features = torch.squeeze(l0_features, dim=2)  # 24

    # encoding layer
    l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                     scope="layer1", is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)
    out_feat = torch.cat((l1_features, l0_features), dim=-1)  # In+ (comp + 3*F) =  24 + (24 + 24*3) = 120

    if dense_block > 1:
        l2_features = conv1d(out_feat, comp, 1,  # 48
                             padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        out_feat = torch.cat((l2_features, out_feat), dim=-1)  # In+ (2F + 3*F) =  120 + (48 + 24*3) = 240

    if dense_block > 2:
        l3_features = conv1d(out_feat, comp, 1,  # 48
                             padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        out_feat = torch.cat((l3_features, out_feat), dim=-1)  # In+ (2F + 3*F) =  240 + (48 + 24*3) = 360

    if dense_block > 3:
        l4_features = conv1d(out_feat, comp, 1,  # 48
                             padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                         scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        out_feat = torch.cat((l4_features, out_feat), dim=-1)  # In+ (2F + 3*F) =  240 + (48 + 24*3) = 480

    # l4_features = torch.unsqueeze(l4_features, dim=2)

    return out_feat