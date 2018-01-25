import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def eval_net(net, dataset):
    total = 0
    for i, image in enumerate(dataset):
        X = Variable(image['image'])
        y = Variable(image['combined_mask'])

        y_pred = net(X)

        y_pred = (F.sigmoid(y_pred) > 0.5).float()

        score = get_iou(y_pred, y.float())
        total += score
    return total / len(dataset)


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print(f'Prediction shape: {pred.shape}, Ground Truth Shape{gt.shape}')
    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou
