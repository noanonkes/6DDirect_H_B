# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from pytorch3d.transforms import rotation_6d_to_matrix


class GeodesicLoss(nn.Module):
    """ 
    Input: Rotations matrices of shape batch x 3 x 3
           Both matrices are orthogonal rotation matrices

    Output: Average theta between 0 to pi radians
    """
    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, m1, m2):
        if len(m1.shape) < 3:
            m1 = m1[None]
        if len(m2.shape) < 3:
            m2 = m2[None]

        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
        
        if self.reduction == "mean":
            return torch.mean(theta)
        elif self.reduction == "sum":
            return torch.sum(theta)

        return theta
    

class RotationLoss(nn.Module):
    """
    A custom loss function to measure the error between predicted and target rotations using 
    both L2 norm and geodesic distance. Designed for head and body pose estimation tasks.
    
    Attributes:
    -----------
    nc : int
        Number of classes (default is 2: head and body).
    conf_thres : float
        General confidence threshold for filtering predictions.
    conf_thres_head : float
        Confidence threshold specifically for head predictions.
    conf_thres_body : float
        Confidence threshold specifically for body predictions.
    GEOobj : GeodesicLoss
        Instance of GeodesicLoss to measure geodesic distance between rotation matrices.
    head_idx : int
        Index of the head class predictions.
    body_idx : int
        Index of the body class predictions.
    """

    def __init__(self, nc=2, conf_thres=0.4, conf_thres_head=0.6, conf_thres_body=0.4):
        super().__init__()
        if nc != 1 and nc != 2:
            raise ValueError("This class only works with 1 or 2 classes. Amount given:", nc)
        self.nc = nc
        self.GEOobj = GeodesicLoss()  # Measures the geodesic distance between predicted and ground truth rotation matrices
        self.head_idx = 0  # Index of head class for selecting valid predictions
        self.body_idx = 1  # Index of body class for selecting valid predictions

        # Confidence thresholds for filtering predictions
        self.conf_thres = conf_thres
        self.conf_thres_head = conf_thres_head
        self.conf_thres_body = conf_thres_body
    
    def forward(self, prot, trot, pobj, pcls):
        """
        Forward pass to compute the rotation loss.
        
        Parameters:
        ----------
        prot : torch.Tensor
            Predicted rotations in 6D representation.
        trot : torch.Tensor
            Target rotations in 6D representation.
        pobj : torch.Tensor
            Predicted occupancy scores.
        pcls : torch.Tensor
            Predicted class scores.
        
        Returns:
        -------
        tuple of (torch.Tensor, torch.Tensor, int, int) or bool
            L2 loss, geodesic loss, count of valid head predictions, count of valid body predictions.
            Returns False if no valid predictions are found.
        """
        # Get indices of all valid predictions
        left_index, heads, bodies = self.valid_predictions(pcls, pobj)

        # Check if there are any valid predictions
        if torch.sum(left_index) > 0:
            
            # Filter valid 6D rotations predictions and targets, and scale them to [-1, 1]
            prot = prot[left_index].sigmoid()
            trot = trot[left_index]

            t_6d = (trot - 0.5) * 2
            p_6d = (prot - 0.5) * 2

            # Convert 6D representations to rotation matrices
            p_rot = rotation_6d_to_matrix(p_6d)
            t_rot = rotation_6d_to_matrix(t_6d)
            
            # Calculate L2 and geodesic losses
            ll2 = torch.mean(torch.linalg.norm(p_rot - t_rot, dim=-1))
            lgeo = self.GEOobj(p_rot, t_rot)

            return ll2, lgeo, heads, bodies
        
        # No valid predictions
        return False

    def valid_predictions(self, pcls, pobj):
        """
        Determine valid predictions based on class probabilities and occupancy scores.

        Applies sigmoid activation to predicted class scores (`pcls`) and occupancy scores (`pobj`), 
        and compares them against confidence thresholds for body and head.

        Parameters:
        ----------
        pcls : torch.Tensor
            Predicted class scores of shape `(N, C)`.
        pobj : torch.Tensor
            Predicted objectness scores of shape `(N, 1)`.

        Returns:
        -------
        tuple of (torch.Tensor, int, int)
            A boolean tensor of shape `(N, 1)` indicating valid predictions, 
            and the counts of valid body and head predictions.
        """
        if self.nc == 1:
            # Apply sigmoid to class scores and objectness scores
            pcls_head = pcls.sigmoid()[:, self.head_idx].unsqueeze(dim=-1)
            pcls_body = pcls.sigmoid()[:, self.body_idx].unsqueeze(dim=-1)

            # Determine valid head and body predictions based on confidence thresholds
            conf_head = (pobj.sigmoid() > self.conf_thres) & (pcls_head > self.conf_thres_head)
            conf_body = (pobj.sigmoid() > self.conf_thres) & (pcls_body > self.conf_thres_body)

            # Count valid head and body predictions
            n_heads = conf_head.sum()
            n_bodies = conf_body.sum()

            # Combine valid head and body predictions
            left_index = conf_head | conf_body
        else:
            # Only one class (head), apply threshold
            left_index = pobj.sigmoid() > self.conf_thres
            n_heads = left_index.sum()
            n_bodies = 0

        return left_index.squeeze(), n_heads, n_bodies
    

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, num_rot=6, conf_thres=0., conf_thres_head=0., conf_thres_body=0.):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        # Define criteria
        # Measures the error for the classification task
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device)) # 1.0 in our case
        # Calculates the error in detecting whether an object is present in a particular grid cell or not
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device)) # 1.0 in our case
        # Measures L2 and Geodesic distance; rotation loss
        ROTobj = RotationLoss(nc=det.nc, conf_thres=conf_thres, conf_thres_head=conf_thres_head, conf_thres_body=conf_thres_body)
        
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # label smoothing is 0. in our case
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma, 0. in our case
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.ROTobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, ROTobj, 1.0, h, autobalance

        if self.autobalance:
            self.loss_coeffs = model.module.loss_coeffs if is_parallel(model) else model.loss_coeffs[-1]
       
        self.num_rot = num_rot
        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors

    def __call__(self, p, targets):  # predictions, targets, model
        """
        P3-P5 output strides of 4, 8, 16, 32 to get grid sizes of 
        160x160, 80x80, 40x40 and 20x20
        
        len p always 4; Detect(P3, P4, P5, P6)
        batch size x number of anchors x grid size x grid size x (obj, cls, x, y, w, h, a1-a6)
        """
        # x, y, w, h, obj, cls, a1-a6 = p[0], p[1], p[2], p[3], p[4], p[5], p[6:]

        device = targets.device
        lcls, lbox, lobj, lgeo, ll2, n_heads, n_bodies = self.initialize_losses(device)
        tcls, tbox, trot, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # which image of batch, which anchor, which grid cell

            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # predictions (p) for bounding box (x, y, w, h), objectsness (obj), classification (cls)
                # and rotation 6D (rot)
                pxy, pwh, pobj, pcls, prot = pi[b, a, gj, gi].split((2, 2, 1, self.nc, self.num_rot), 1)  # prediction subset corresponding to targets
                # Regression
                pxy = pxy[:, :2].sigmoid() * 2. - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]  # range [0, 4] * anchor
                pbox = torch.cat((pxy, pwh), 1)  # predicted box

                # Measures the error in localizing the object within the grid cell
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Measures the rotation loss
                if self.num_rot:
                    out = self.ROTobj(prot, trot[i], pobj, pcls)
                    # If out = False -> no prediction was certain enough
                    if out is not False:
                        l2_loss, geo_loss, heads, bodies = out

                        ll2 += l2_loss
                        lgeo += geo_loss

                        n_heads += heads
                        n_bodies += bodies          

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        if self.hyp["geo"] != 0.:
            lgeo *= self.hyp["geo"]
        if self.hyp["l2"] != 0.:
            ll2 *= self.hyp["l2"]

        if self.autobalance:
            loss = (lbox + lobj + lcls) / (torch.exp(2 * self.loss_coeffs[0])) + self.loss_coeffs[0]
            loss += lgeo / (torch.exp(2 * self.loss_coeffs[1])) + self.loss_coeffs[1]
        elif self.hyp["geo"] != 0. and self.hyp["l2"] != 0.:
            loss = lbox + lobj + lcls + lgeo + ll2
        elif self.hyp["geo"] != 0.:
            loss = lbox + lobj + lcls + lgeo
        else:
            loss = lbox + lobj + lcls + ll2

        bs = tobj.shape[0]  # batch size

        return loss * bs, torch.cat((lbox, lobj, lcls, lgeo, ll2, n_bodies, n_heads)).detach()

    def initialize_losses(self, device):
        """Initialize the loss components and counters."""
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lgeo = torch.zeros(1, device=device)
        ll2 = torch.zeros(1, device=device)
        n_heads = torch.zeros(1, device=device)
        n_bodies = torch.zeros(1, device=device)
        return lcls, lbox, lobj, lgeo, ll2, n_heads, n_bodies

    def build_targets(self, p, targets):
        """ Understand anchors 
        https://blog.csdn.net/flyfish1986/article/details/117594265 
        https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html
        """
        # Build targets for compute_loss(), input targets(image, class, x, y, w, h, a1, a2, a3, a4, a5, a6)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, trot, indices, anch = [], [], [], [], []
        gain = torch.ones(7 + self.num_rot, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            xy_gain = torch.tensor(p[i].shape)[[3, 2]]
            gain[2:4] = xy_gain
            gain[4:6] = xy_gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp["iou_t"]  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b = t[:, 0].long()  # image
            c = t[:, 1].long()  # class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            if self.num_rot:
                trot.append(t[:, 5+1 : 5+1+self.num_rot])  # 6 angles
            
            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, trot, indices, anch
