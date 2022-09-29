import torch
from torch import nn
from torch.nn import functional as F

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class ConstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        self.mask_dtype = torch.bool

    def forward(self, feat_q,feat_k):
        feat_q = Normalize()(feat_q)
        feat_k = Normalize()(feat_k)
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = 1

        loss_co2 = 0
        qsim = []
        ksim = []
        for i in range(feat_q.size(0)):
            q_temp = F.cosine_similarity(feat_q[i].unsqueeze(0),feat_q)
            k_temp = F.cosine_similarity(feat_k[i].unsqueeze(0),feat_k)
            for j in range(feat_q.size(0)):
                if i!=j:
                    qsim.append(q_temp[j].unsqueeze(0))
                    ksim.append(k_temp[j].unsqueeze(0))

        qsim = torch.cat(qsim,dim=0)
        ksim = torch.cat(ksim,dim=0)

        loss_co2 = torch.mean((qsim-ksim)**2)

        return loss_co2

    
class PatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        feat_q_norm = feat_q.clone().norm(p=2,dim=-1,keepdim=True)
        feat_k_norm = feat_k.clone().norm(p=2,dim=-1,keepdim=True)
        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos_norm = feat_k_norm*feat_q_norm
        l_pos = l_pos.view(batchSize, 1) / l_pos_norm
        batch_dim_for_bmm = 1
        # if self.opt.nce_allbatch:
        #     batch_dim_for_bmm = 1
        # else:
        #     batch_dim_for_bmm = self.opt.batch

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        
        feat_q_norm = feat_q_norm.view(batch_dim_for_bmm, -1, 1)
        feat_k_norm = feat_k_norm.view(batch_dim_for_bmm, -1, 1)
        
        npatches = feat_q.size(1)
        l_neg_norm = torch.bmm(feat_q_norm, feat_k_norm.transpose(2, 1))
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) / l_neg_norm
        

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss