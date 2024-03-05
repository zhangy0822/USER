import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


class UTO(nn.Module):
    def __init__(self, opt):
        super(UTO, self).__init__()
        self.opt = opt
        self.l_alpha = opt.mu
        self.l_ep = opt.gama

    def forward(self, im, s ):

        bsize = im.size()[0]
        scores = get_sim(im, s)
    
        tmp  = torch.eye(bsize).cuda()   
        s_diag = tmp * scores        
        scores_ = scores - s_diag      
        S_ = torch.exp(self.l_alpha * (scores_ - self.l_ep))
    
        loss_diag_1 = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum( torch.log(1 + S_.sum(0)) / self.l_alpha + torch.log(1 + S_.sum(1)) / self.l_alpha + loss_diag_1) / bsize

        return loss
    def moco_forward(self, v_q, t_k, t_q, v_k, v_queue, t_queue):
        # v positive logits: Nx1
        v_pos = torch.einsum("nc,nc->n", [v_q, t_k]).unsqueeze(-1)
        # v negative logits: NxK
        t_queue = t_queue.clone().detach()    
        v_neg = torch.einsum("nc,ck->nk", [v_q, t_queue])

        # # t positive logits: Nx1
        t_pos = torch.einsum("nc,nc->n", [t_q, v_k]).unsqueeze(-1)
        # t negative logits: NxK
        v_queue = v_queue.clone().detach()
        t_neg = torch.einsum("nc,ck->nk", [t_q, v_queue])

        v_pos_diag = torch.diag_embed(v_pos.squeeze(-1))
        v_bsize = v_pos_diag.size()[0]
        v_loss_diag = - torch.log(1 + F.relu(v_pos_diag.sum(0)))
        v_S_ = torch.exp((v_neg - self.l_ep) * self.l_alpha)
        v_S_T = v_S_.T
        v_loss = torch.sum(torch.log(1 + v_S_T.sum(0)) / self.l_alpha + v_loss_diag) / v_bsize

        t_pos_diag = torch.diag_embed(t_pos.squeeze(-1))
        t_bsize = t_pos_diag.size()[0]
        t_loss_diag = - torch.log(1 + F.relu(t_pos_diag.sum(0)))
        t_S_ = torch.exp((t_neg - self.l_ep) * self.l_alpha)
        t_S_T = t_S_.T
        t_loss = torch.sum(torch.log(1 + t_S_T.sum(0)) / self.l_alpha + t_loss_diag) / t_bsize

        return v_loss + t_loss