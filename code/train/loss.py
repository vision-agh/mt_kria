# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PART OF THIS FILE AT ALL TIMES.

# This file has been modified by Maciej Baczmański for usage in https://github.com/maciekbaczmanski/mt_kria repository, under Apache 2.0 license.
# All files were modified to update the repository to work with newest versions of libraries, and to train and evaluate our own MultiTask V3 model.

from typing import Optional

import torch
import torch.nn.functional as F

from lovasz_losses import lovasz_softmax


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    # loss = F.nll_loss(log_p, target, ignore_index=250,
    loss = F.nll_loss(log_p, target, ignore_index=255,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, y_preds, y_true):
        '''
        :param y_preds: (N, C, H, W), Variable of FloatTensor
        :param y_true:  (N, H, W), Variable of LongTensor
        # :param weights: sample weights, (N, H, W), Variable of FloatTensor
        :return:
        '''
        logp = F.log_softmax(y_preds, dim=1)  # (N, C, H, W)
        ylogp = torch.gather(logp, 1, y_true.view(y_true.size(0), 1, y_true.size(1), y_true.size(2)))  # (N, 1, H, W)
        return -(ylogp.squeeze(1)).mean()


def softmax_focal_loss2d(input, target, gamma=2, size_average=True, use_focal_loss=True, use_weight=True):
    n, c, h, w = input.size()
    if use_focal_loss:
        scores = F.softmax(input, dim=1)
        factor = torch.pow(1. - scores, gamma)
        log_score = F.log_softmax(input, dim=1)
        log_p = factor * log_score
    else:
        log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    if use_weight:
        histc = torch.histc(target, bins=c, min=0, max=c - 1).float()
        weight = 1. - histc / target.size(0)
        divd = (histc * weight).sum()
    else:
        divd = mask.data.sum().float()
        weight = None
    # loss = F.nll_loss(log_p, target, ignore_index=250,
    loss = F.nll_loss(log_p, target, ignore_index=255,
                      weight=weight, size_average=False)
    if size_average:
        # loss /= mask.data.sum().float()
        loss /= divd
    return loss


# Depth

class SilogLoss(torch.nn.Module):
    def __init__(self, variance_focus=0.85, eps=1e-12):
        super(SilogLoss, self).__init__()
        self.eps = eps
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask] + self.eps) - torch.log(depth_gt[mask] + self.eps)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum, dtype=torch.float32)
    s = torch.ones_like(ssum, dtype=torch.float32)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum().long(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


def trimmed_mae_loss(prediction, target, mask, trim=0.2, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
                 : int(len(res) * (1.0 - trim))
                 ]

    image_loss = trimmed.sum() / 2

    return reduction(image_loss, M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class GradientLoss(torch.nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class TrimmedProcrustesLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)

        total = trimmed_mae_loss(self.__prediction_ssi, target_, mask, trim=0.2, reduction=self.__reduction)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.
    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(-dice_score + 1.)


class DiceLoss(torch.nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return dice_loss(input, target, self.eps)


class Criterion(torch.nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, weight=None, use_weight=True, reduce=True):
        super(Criterion, self).__init__()
        class_wts = torch.ones(len(weight))
        for i in range(0, len(weight)):
            class_wts[i] = weight[i]
        # class_wts[0] = weight[0]
        # class_wts[1] = weight[1]
        # class_wts[2] = weight[2]
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_wts, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class LovaszSoftmaxLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=255, reduce=None, reduction="mean",
                 classes="present", per_image=False):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.per_image = per_image
        self.weight = weight
        self.ce_loss = Criterion(ignore_index=ignore_index, weight=weight, use_weight=True, reduce=True)

    def forward(self, pred, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=True)

        iou_loss = lovasz_softmax(F.softmax(scale_pred, dim=1), target, ignore=self.ignore_index, classes=self.classes,
                                  per_image=self.per_image)
        ce_loss = self.ce_loss(pred, target)
        return iou_loss + ce_loss
