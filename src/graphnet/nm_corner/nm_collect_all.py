from flash_attn.modules.mha import MHA

from graphnet.models import Model

from graphnet.models.easy_model import EasySyntax
from graphnet.models import Model
from graphnet.models import StandardModel

from graphnet.training.labels import Direction

from torch_geometric.data import Data

from graphnet.data.dataset import SQLiteDataset
from graphnet.data import GraphNeTDataModule

from graphnet.data.utilities.sqlite_utilities import query_database

from einops import rearrange, repeat

from graphnet.models.data_representation.graphs import KNNGraph
#from graphnet.models.graphs import KNNGraph
from graphnet.models.detector import IceCube86

from typing import Set, Union, List, Type, Optional, Dict, Any
from graphnet.models.gnn.gnn import GNN

from graphnet.training.loss_functions import MSELoss

from graphnet.models.task import IdentityTask
import torch
import numpy as np
from torch.optim.adam import Adam
import os


from torch.functional import Tensor
from torch_geometric.nn.pool import (
    knn_graph,
)

from torch_geometric.utils import dropout_node
from torch_geometric.utils import mask_select

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


from pytorch_lightning import LightningModule

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.select.topk import topk

import torch_geometric.transforms as T


from graphnet.models.components.layers import DropPath
from graphnet.models.data_representation.graphs.edges import KNNEdges


from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like

from torch.optim import RAdam


from graphnet.training.loss_functions import LossFunction

class custom_EnergyReconstruction(StandardLearnedTask):
    """Reconstructs energy using stable method."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        return x + eps_like(x)

class DirectionRecoNM(StandardLearnedTask):
    """Reconstructs direction."""

    # Requires three features: untransformed points in (x,y,z)-space.
    default_target_labels = ["direction"]  # contains dir_x, dir_y, dir_z
    # see Direction label in /src/graphnet/training/labels.py
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        vec_x = x[:, 0]
        vec_y = x[:, 1]
        vec_z = x[:, 2]
        return torch.stack((vec_x, vec_y, vec_z), dim=1)

class OpeningAngleLoss(LossFunction):

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """

        Args:
            prediction: Output of the model. Must have shape [N, 3] where
                columns 0, 1, 2 are predictions of `direction`
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von opening angle loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 3
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        target_norm = torch.nn.functional.normalize(target, dim=1, eps=1e-5)
        prediction_norm = torch.nn.functional.normalize(prediction, dim=1, eps=1e-5)

        elements = torch.acos(torch.clamp((prediction_norm*target_norm).sum(axis=1),-1,1))

        return elements

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = torch.nn.Parameter(torch.zeros(dim))
        torch.nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return torch.nn.functional.normalize(x, dim = -1) * self.scale * gamma


#region ### Block for sub_sample functions ###
def hlc_sub_sample(data, max_length, columns = [0, 1, 2], nb_nearest=8, hlc_pos=6):
    x = data.x
    btch = data.batch
    x = x.view(-1, 1) if x.dim() == 1 else x
    score = x[:,hlc_pos-1]

    node_index = topk(score, max_length, btch)
    edge_ind = knn_graph(x=x[:, columns][node_index], k=nb_nearest, batch=btch[node_index])
    return node_index , edge_ind

def custom_sub_sample(data, max_length, score, pos, nb_nearest=8):
    #x = data.x
    btch = data.batch
    #x = x.view(-1, 1) if x.dim() == 1 else x

    node_index = topk(score, max_length, btch)
    edge_ind = knn_graph(x=pos[node_index], k=nb_nearest, batch=btch[node_index])
    return node_index , edge_ind

def simple_sub_sample(batchv, max_length, score, pos, nb_nearest=8):
    node_index = topk(score, max_length, batchv)
    edge_ind = knn_graph(x=pos[node_index], k=nb_nearest, batch=batchv[node_index])
    return node_index , edge_ind

#endregion

#region ### Block for contrastive pretraining
def generate_representation_simple(x: torch.Tensor,
                                   bv: torch.Tensor):
    
    maximize = scatter(src=x, index=bv, dim = 0, reduce='max')
    minimize = scatter(src=x, index=bv, dim = 0, reduce='min')
    summation = scatter(src=x, index=bv, dim = 0, reduce='sum')
    averaging = scatter(src=x, index=bv, dim = 0, reduce='mean')

    #rep = torch.cat((maximize,minimize,summation,averaging), dim=1)
    rep = maximize + minimize + summation + averaging

    return rep

def batched_mse_loss(reco, orig, bv):
    reco = to_dense_batch(reco, bv)[0]
    orig = to_dense_batch(orig, bv)[0]

    loss = torch.mean((reco - orig) ** 2, dim=[1,2]).view(-1,1)
    return loss

class Projector(Model):
    """ Projection Head for SimSiam """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        act = torch.nn.SELU(inplace=True)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            act
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            act
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
class Predictor(Model):
    """ Predictor for SimSiam """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        act = torch.nn.SELU(inplace=True)
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            act
        )
        self.layer2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def negative_cosine_similarity(p, z):
    """ Negative Cosine Similarity """
    z = z.detach()
    p = torch.nn.functional.normalize(p, dim=1)
    z = torch.nn.functional.normalize(z, dim=1)

    return -(p*z).sum(dim=1).view(-1, 1)#.mean() #mean is put into the shared_step


class phys_augment2(Model):
    def __init__(self, mask_prob = 0.2, dropout_chance = 0.3):
        super().__init__()
        self.p = dropout_chance
        self.m = mask_prob

        self.transform = T.remove_isolated_nodes.RemoveIsolatedNodes()
        self.reconnect = KNNEdges(8) #take care that the n of nearest neighbours is the same as in the original dataloading
        #note to expruns: {exprun1: only dropping, exprun2: only masking, exprun3: only wiggle} also dropping is with fixed nn=8
    def forward(
        self,
        data: Data
    ) -> Data:
        auged = data.clone()

        #next idea for more randomness: take the stds over a random amount of nodes or leave out wiggling entirely?
        rand_start = np.random.randint(low=0, high=auged.x.shape[0]/2)
        rand_end = np.random.randint(low=auged.x.shape[0]/2 + 1, high=auged.x.shape[0]+1)
        stds = torch.std(auged.x[rand_start:rand_end], dim=0)

        # stds = torch.std(auged.x, dim=0)
        #print(stds)

        ###dropping nodes and randomized connectivity###
        auged.edge_index, _, _ = dropout_node(edge_index=data.edge_index, p=self.p)
        auged = self.transform(auged)
        #version with fixed nn
        #auged = self.reconnect(auged)
        #version with randomized nn
        nn = np.random.randint(low=4, high=12)
        reconnect = KNNEdges(nn).to(device=self.device)
        auged = reconnect(auged)

        ###wiggle time and charge (or other features)### first try adding value around std across events
        #create mask for selection of features: x,y,z,t,q,hlc
        wiggle_mask = torch.tensor([0,0,0,1,1,0], device=self.device)
        #create mask of random_values, shape like input data, previous prefactor:1
        rand_mask = 2*torch.randn_like(auged.x)
        #print((rand_mask*wiggle_mask)[0:5])
        #add stds multiplied with random values to feature data
        auged.x = auged.x + (stds*wiggle_mask*rand_mask)


        ###masking nodes###
        # f_mask = (torch.rand((auged.x.shape[0], 1)) > self.m).to(device=self.device).float()
        # for i in range(1, auged.x.shape[1]):
        #     f_mask = torch.cat((f_mask, (torch.rand((auged.x.shape[0], 1)) > self.m).to(device=self.device).float()), dim=1)
        # auged.x = auged.x*f_mask


        
        return auged
    

class cont_frame(EasySyntax):
    def __init__(self,
                 enc_net,
                 lat_feat,
                 projection_dim=1000, 
                 hidden_dim_proj=2000, 
                 hidden_dim_pred=2000,
                 optimizer_class: Type[torch.optim.Optimizer] = Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[type] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 scheduler_config: Optional[Dict] = None,) -> None:
        
        task = IdentityTask(nb_outputs = 1, target_labels=['energy'], hidden_size=1, loss_function=MSELoss())
        super().__init__(
            tasks=task,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )

        self.backbone = enc_net

        self.projector = Projector(lat_feat, hidden_dim=hidden_dim_proj, out_dim=projection_dim)
        self.predictor = Predictor(in_dim=projection_dim, hidden_dim=hidden_dim_pred, out_dim=projection_dim)

        self.augment = phys_augment2()
        

    def forward(self, data: Union[Data, List[Data]]):
        if not isinstance(data, Data):
            data = data[0]
        #augment twice
        aug1 = self.augment(data)
        aug2 = self.augment(data)

        if torch.any(aug1.x.isnan()) or torch.any(aug2.x.isnan()):
            print('nan in augments')

        #generate two different latents
        rep1 = self.backbone(aug1)
        rep2 = self.backbone(aug2)

        #calculate contrastive loss between two different latents
        z1 = self.projector(rep1)
        z2 = self.projector(rep2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2

        #loss = self.id_task(loss)

        if torch.any(p1.isnan()) or torch.any(p2.isnan()):
            print('nan in predictor/projector')

        if torch.any(loss.isnan()):
            print('nan in loss')

        #loss is returned as list to comply with the graphnet predict functionality
        return [loss]


    def validate_tasks(self) -> None:
        accepted_tasks = IdentityTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def shared_step(self, batch: List[Data], batch_idx: int):
        loss = self(batch)
        if isinstance(loss, list):
            assert len(loss) == 1
            loss = loss[0]
        return torch.mean(loss, dim=0)
    
    def give_encoder_model(self):
        #function to return the encoder model
        #as a way to transport the pretrained encoder
        #into another learning context or saving the parameters manually
        return self.encoder
    
    def save_pretrained_model(self, save_path):
        model = self.backbone

        run_name = 'pretrained_model'

        save_path = os.path.join(save_path, run_name)
        print('saving to', save_path)
        os.makedirs(save_path, exist_ok=True)

        model.save_state_dict(f"{save_path}/state_dict.pth")
        model.save_config(f"{save_path}/model_config.yml")

#endregion


#region ### Block for mask_pred_pretraining (maybe later) ###

def dense_mse_loss(reco, orig, bv):
    squared_errs = (reco - orig)**2
    losses = torch.mean(scatter(src=squared_errs, index=bv, reduce='mean', dim=0), dim=1)

    return losses.view(-1,1)

def neg_cosine_loss(reco, orig, bv):
    reco_norm = torch.nn.functional.normalize(reco, dim=1)
    orig_norm = torch.nn.functional.normalize(orig, dim=1)
    cos = -(reco_norm*orig_norm).sum(dim=1)
    losses = scatter(src=cos, index=bv, reduce='mean', dim=0)

    return losses.view(-1,1)

class standard_maskpred_net(Model):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 1000,
                 out_dim: int = 5,
                 nb_linear: int = 5,
                 ):
        super().__init__()

        self.activation = torch.nn.SELU()
        
        self.lin_net = torch.nn.ModuleList()
        for i in range(nb_linear):
            if i == 0:
                self.lin_net.append(torch.nn.Linear(in_dim,hidden_dim))
            else:
                self.lin_net.append(torch.nn.Linear(hidden_dim,hidden_dim))

        self.final_proj = torch.nn.Linear(hidden_dim, out_dim)
        

    def forward(self, data:Union[Data, Tensor]):
        if isinstance(data, Data):
            x_hat = data.x
        else:
            x_hat = data
        x_hat = self.lin_net[0](x_hat)
        x_hat = self.activation(x_hat)
        for i in range(1,len(self.lin_net)):
            x_hat = x_hat + self.lin_net[i](x_hat)
            x_hat = self.activation(x_hat)

        x_hat = self.final_proj(x_hat)
        
        return x_hat

class mask_pred_augment(Model):
    def __init__(self, 
                 masked_ratio: float = 0.25,
                 masked_feat: List[int] = [0,1,2,3,4],
                 learned_masking_value: bool = True,
                 hlc_pos: int = None,
                 ):
        super().__init__()
        self.ratio = masked_ratio
        self.hlc_pos = hlc_pos
        self.masked_feat = masked_feat
        self.learned_value = learned_masking_value

        if self.learned_value:
            print('warning: can currently only mask adjacent features, e.g. only (x,y,z) or only (t,q) but not e.g. (x,t,q)')
            self.values = torch.nn.Parameter(torch.randn(1,len(self.masked_feat)))

    def forward(self, data: Data):
        auged = data.clone()

        rand_score = torch.rand_like(data.batch.to(dtype=torch.bfloat16))
        if self.hlc_pos is not None:
            rand_score = rand_score + auged.x[:,self.hlc_pos].view(1,-1)
            rand_score = rand_score.view(-1)

        ind = topk(x=rand_score, ratio=self.ratio, batch=data.batch)

        mask = torch.ones_like(data.batch.to(dtype=torch.bfloat16))
        mask[ind] = 0

        target = mask_select(src=auged.x, dim=0, mask=~mask.bool())[:,self.masked_feat]
        if not self.learned_value:
            auged.x[:,self.masked_feat] = auged.x[:,self.masked_feat]*mask.view(-1,1)
        else:
            auged.x[ind,self.masked_feat[0]:self.masked_feat[-1]+1] = self.values
        # print('orig', data.x[0:5])
        # print('auged', auged.x[0:5])
        # print('target', target[0:5])

        #returned mask is zero at the target position and 1 else
        return auged, target, mask

class mask_pred_frame(EasySyntax):
    def __init__(self,
                 encoder: Model,
                 encoder_out_dim: int = None,
                 masked_ratio: float = 0.25,
                 masked_feat: List[int] = [0,1,2,3,4],
                 learned_masking_value: bool = True,
                 hlc_pos: int = None,
                 mask_pred_net: Model = None,
                 default_hidden_dim: int = 1000, 
                 default_nb_linear: int = 5,
                 final_loss: str = 'mse',
                 add_charge_pred: bool = False,
                 optimizer_class: Type[torch.optim.Optimizer] = Adam,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_class: Optional[type] = None,
                 scheduler_kwargs: Optional[Dict] = None,
                 scheduler_config: Optional[Dict] = None,) -> None:
        
        #just because I need to specify a task
        task = IdentityTask(nb_outputs=1, target_labels=['skip'], hidden_size=2, loss_function=MSELoss())

        super().__init__(
            tasks=task,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )
        self.backbone = encoder

        self.ratio = masked_ratio

        self.augment = mask_pred_augment(masked_ratio=masked_ratio,
                                         masked_feat=masked_feat,
                                         learned_masking_value=learned_masking_value,
                                         hlc_pos=hlc_pos
                                         )

        if encoder_out_dim is None:
            assert encoder.nb_outputs > 0, 'make sure to either specify \"encoder_out_dim\" or have a \".nb_outputs\" in your encoder'
            lat_dim = encoder.nb_outputs
        else:
            lat_dim = encoder_out_dim

        if mask_pred_net is None:
            print('no custom net for mask prediction specified; using a standard net')
            self.rep = standard_maskpred_net(in_dim=lat_dim,
                                             hidden_dim=default_hidden_dim,
                                             out_dim=len(masked_feat),
                                             nb_linear=default_nb_linear)
        else:
            assert mask_pred_net.nb_outputs == len(masked_feat), f'make sure that your \"mask_pred_net\" has number of output feats equal to nb of masked feats ({len(masked_feat)})'
            self.rep = mask_pred_net

        # self.scorer = standard_maskpred_net(in_dim=len(masked_feat),
        #                                     hidden_dim=default_hidden_dim,
        #                                     out_dim=1,
        #                                     nb_linear=default_nb_linear)
        
        self.custom_loss = True
        assert final_loss in ['cosine', 'mse'], f'can only choose from {['cosine', 'mse']} for loss function'
        if final_loss == 'cosine':
            self.loss_func = neg_cosine_loss
        elif final_loss == 'mse':
            self.loss_func = dense_mse_loss

        if add_charge_pred:
            self.add_charge_pred = True
            self.charge_net = torch.nn.Linear(lat_dim, 1)
        else:
            self.add_charge_pred = False

            
    def forward(self, data: Union[Data, List[Data]]):
        if not isinstance(data, Data):
            data = data[0]

        aug, target, mask = self.augment(data)

        data_hat, cls_tensor = self.backbone(aug)

        rep = self.rep(data_hat)

        nodes = rep[~mask.bool()]
        btch = data.batch[~mask.bool()]

        loss = self.loss_func(reco=nodes, orig=target, bv=btch) #data.batch[ind]

        if self.add_charge_pred:
            #print('doing charge prediction')
            charge_tensor = torch.pow(10, data.x[:,4]).view(-1,1)
            charge_sums = torch.log10(scatter(src=charge_tensor, index=data.batch, dim = 0, reduce='sum'))
            pred_charge = self.charge_net(cls_tensor)
            loss = loss + (charge_sums - pred_charge)**2

        #loss is returned as a list to comply with the graphnet predict functionality
        return [loss]

    def validate_tasks(self) -> None:
        accepted_tasks = IdentityTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def shared_step(self, batch: List[Data], batch_idx: int):
        loss = self(batch)
        if isinstance(loss, list):
            assert len(loss) == 1
            loss = loss[0]
        return torch.mean(loss, dim=0)
    
    def give_encoder_model(self):
        #function to return the encoder model
        #as a way to transport the pretrained encoder
        #into another learning context or saving the parameters manually
        return self.encoder
    
    def save_pretrained_model(self, save_path):
        model = self.backbone

        run_name = 'pretrained_model'

        save_path = os.path.join(save_path, run_name)
        print('saving to', save_path)
        os.makedirs(save_path, exist_ok=True)

        model.save_state_dict(f"{save_path}/state_dict.pth")
        model.save_config(f"{save_path}/model_config.yml")

class simple_model(Model):
    def __init__(self,
                 ):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4,10),
                                       torch.nn.SELU(),
                                       torch.nn.Linear(10,6))
        self.nb_outputs=6

    def forward(self, data:Data):
        data.x = self.net(data.x)
        return data

#endregion
    
#region ### Block for Compression Modules ###

class random_scorer(Model):
    def __init__(self,
                 hlc_pos=None
                 ):
        self.hlc_pos = hlc_pos
        super().__init__()

    def forward(self, data:Data):
        if self.hlc_pos is None:
            x = torch.rand_like(data.x[:,0])
            self.hlc_pos = 0
        else:
            x = data.x[:,self.hlc_pos]
        
        return x

class minimum_linear_embedding(Model):
    def __init__(self,
                 in_dim,
                 lat_dim,
                 ):
        super().__init__()
        self.in_dim = in_dim

        self.lin = torch.nn.Linear(in_dim, lat_dim)

    def forward(self, data:Data):
        x_hat = self.lin(data.x[:,0:self.in_dim])
        
        return x_hat

class simple_linear_encoder(Model):
    def __init__(self,
                 in_dim,
                 lat_dim,
                 ratio=1.0,
                 lin_net_length=5,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.nb_outputs = lat_dim

        hidden_dim = int(lat_dim*ratio)

        if ratio < 1:
            self.need_proj = True
            self.final_proj = torch.nn.Linear(hidden_dim, lat_dim)
        else:
            self.need_proj = True
        
        self.lin_net = torch.nn.ModuleList()
        for i in range(lin_net_length):
            if i == 0:
                self.lin_net.append(torch.nn.Linear(in_dim,hidden_dim))
            else:
                self.lin_net.append(torch.nn.Linear(hidden_dim,hidden_dim))

        self.activation = torch.nn.SELU()

    def forward(self, data:Data):
        x_hat = self.lin_net[0](data.x[:,0:self.in_dim])
        x_hat = self.activation(x_hat)
        for i in range(1,len(self.lin_net)):
            x_hat = x_hat + self.lin_net[i](x_hat)
            x_hat = self.activation(x_hat)
        if self.need_proj:
            x_hat = self.final_proj(x_hat)
        
        return x_hat

class simple_gnn_encoder(Model):
    def __init__(self,
                 in_dim,
                 lat_dim,
                 ratio=1.0,
                 nb_messages=1,
                 lin_net_length=5,
                 nb_nearest=8,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.nb_outputs = lat_dim
        self.nb_nearest = nb_nearest

        hidden_dim = int(lat_dim*ratio)

        if ratio < 1:
            self.need_proj = True
            self.final_proj = torch.nn.Linear(hidden_dim, lat_dim)
        else:
            self.need_proj = False
        
        self.lin_net = torch.nn.ModuleList()
        for i in range(lin_net_length):
            if i == 0:
                self.lin_net.append(torch.nn.Linear(in_dim,hidden_dim))
            else:
                self.lin_net.append(torch.nn.Linear(hidden_dim,hidden_dim))

        self.conv_net = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(nb_messages):
            self.conv_net.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, add_self_loops=False))
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        self.has_custom_message = False
        self.activation = torch.nn.SELU()

    def forward(self, data:Data):
        if self.nb_nearest != 8:
            data.edge_index = knn_graph(x=data.x[:,0:3], k=self.nb_nearest, batch=data.batch)
        x_hat = self.lin_net[0](data.x[:,0:self.in_dim])
        x_hat = self.activation(x_hat)
        for i in range(1,len(self.lin_net)):
            x_hat = x_hat + self.lin_net[i](x_hat)
            x_hat = self.activation(x_hat)
        for i in range(len(self.conv_net)):
            x_hat = x_hat + self.conv_net[i](x_hat, data.edge_index)
            x_hat = self.activation(x_hat)
            x_hat = self.norms[i](x_hat)
            data.edge_index = knn_graph(x=x_hat[:,0:3], k=self.nb_nearest, batch=data.batch)

        if self.need_proj:
            x_hat = self.final_proj(x_hat)
        
        return x_hat

class chain_gnn_encoders(Model):
    def __init__(self,
                 in_dim,
                 lat_dim,
                 nb_encoders,
                 nb_nearest=8,
                 nb_messages=1,
                 lin_net_length=5,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.nb_near = nb_nearest
        self.nb_outputs = lat_dim
        
        self.encoders = torch.nn.ModuleList()
        for i in range(nb_encoders):
            self.encoders.append(simple_gnn_encoder(in_dim,
                                                    lat_dim,
                                                    nb_messages,
                                                    lin_net_length))

    def forward(self, data:Data):
        x_hat = self.encoders[0](data)
        ei = knn_graph(x=x_hat[:,0:3], k=self.nb_near, batch=data.batch)
        data_hat = Data(x=x_hat, edge_index=ei, batch=data.batch)
        for i in range(1,len(self.encoders)-1):
            x_hat = self.encoders[i](data_hat)
            ei = knn_graph(x=x_hat[:,0:3], k=self.nb_near, batch=data.batch)
            data_hat = Data(x=x_hat, edge_index=ei, batch=data.batch)
        x_hat = self.encoders[-1](data_hat)
        
        return x_hat

class general_compression_module(Model):
    def __init__(self,
                 max_length=3840,
                 lat_dim=192,
                 nb_nearest=8,
                 net_list=[],
                 scorer_length=5,
                 scorer_width=100,
                 hlc_pos=None,
                 random_subs_embedding=None
                 ):
        super().__init__()

        activation = torch.nn.SELU()

        self.max_l = max_length
        self.nb_near = nb_nearest
        self.nb_outputs = lat_dim

        if not net_list:
            #print('doing random')
            self.net_list = random_scorer(hlc_pos=hlc_pos)
            self.do_random = True
            if random_subs_embedding is None:
                print('using a minimal linear layer for embedding with random subsampling; data is assumed to have at least 5 feature dimensions')
                self.embedding = minimum_linear_embedding(in_dim=5,
                                                          lat_dim=lat_dim)
            else:
                self.embedding = random_subs_embedding
        else:
            #print('doing learned')
            for module in net_list:
                assert module.nb_outputs == lat_dim, f'all compression nets must have the same lat_dim {lat_dim}, {module} has output of {module.nb_outputs}'
            self.net_list = torch.nn.ModuleList(net_list)
            self.do_random = False
            self.scorers = torch.nn.ModuleList()
            
            for i in range(len(net_list)):
                #print('adding a scorer')
                submod = []
                submod.append(torch.nn.Sequential(torch.nn.Linear(net_list[i].nb_outputs, scorer_width),
                                                    activation))
                for j in range(1,scorer_length-1):
                    submod.append(torch.nn.Sequential(torch.nn.Linear(scorer_width, scorer_width),
                                                        activation))
                submod.append(torch.nn.Sequential(torch.nn.Linear(scorer_width, 1),
                                                    activation))
                self.scorers.append(torch.nn.Sequential(*submod))

            if len(net_list) > 1:
                self.concat_scores = True

                self.combine_scores = torch.nn.Sequential(torch.nn.Linear(len(net_list),5*len(net_list)),
                                                          activation,
                                                          torch.nn.Linear(5*len(net_list),1),
                                                          activation,)
            else:
                self.concat_scores = False

    def forward(self, data:Data):
        if self.do_random:
            score = self.net_list(data)
            x_hat = self.embedding(data)
        else:
            if self.concat_scores:
                score = []
                for i in range(len(self.net_list)):
                    feat = self.net_list[i](data)
                    score.append(self.scorers[i](feat))
                    if i == 0:
                        x_hat = feat
                    else:
                        x_hat = x_hat + feat #was += feat
                score = torch.cat(score, dim=1)
                score = self.combine_scores(score)
                
            else:
                x_hat = self.net_list[0](data)
                score = self.scorers[0](x_hat)

        node_index = topk(score, self.max_l, data.batch)
        new_b = data.batch[node_index]
        if self.do_random:
            x_hat = x_hat[node_index]
        else:
            x_hat = x_hat[node_index]*torch.sigmoid(score[node_index])
        new_ei = knn_graph(x=x_hat[:,0:3], k=self.nb_near, batch=new_b)

        new_data = Data(x=x_hat, edge_index=new_ei, batch=new_b)

        return new_data
    
class just_embedding_module(Model):
    def __init__(self,
                 encoder,
                 nb_nearest=8,
                 ):
        super().__init__()
        self.embedding = encoder
        self.nb_near = nb_nearest
        

    def forward(self, data:Data):
        x_hat = self.embedding(data)
        new_ei = knn_graph(x=x_hat[:,0:3], k=self.nb_near, batch=data.batch)
        new_data = Data(x=x_hat, edge_index=new_ei, batch=data.batch)
        return new_data

#endregion

#region ### Block for (modified) DeepIce auxiliaries ###

class flash_Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        This is mostly analogous to the Mlp for DeepIce other than the dtypes being chnaged to torch.float16

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features "
                f"{in_features} instead"
            )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = torch.nn.Linear(in_features, hidden_features, dtype=torch.bfloat16)
        self.activation = activation()
        self.output_projection = torch.nn.Linear(hidden_features, out_features, dtype=torch.bfloat16)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x
    
class flashMHA_block(LightningModule):
    """Implementation of BEiTv2 Block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        softm_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
    ):
        """Construct 'Block_rel'.

        Implements flash_attn's MHA module. Most of the arguments pertain to everything but theís module and as of now
        tweaking the actual attention mechanism can be done by adjusting the parameters passed to the MHA in the class initialization
        by hand.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key,
                and value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key
                matrices in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the
                `Attention_rel` layer.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=True)
        self.attn = MHA(embed_dim=input_dim,
                        num_heads=num_heads,
                        use_flash_attn=True,
                        softmax_scale=softm_scale,
                        dropout=attn_drop,
                        dtype=torch.bfloat16)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )
        self.norm2 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=True)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = flash_Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
            self.gamma_2 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
        max_seq: int,


    ) -> Tensor:
        """Forward pass."""
        #print('using MHA block')
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    cu_seqlens=cu_seqs,
                    max_seqlen=max_seq,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1.to(device=self.device)
                * self.drop_path(
                    self.attn(
                        xn,
                        cu_seqlens=cu_seqs,
                        max_seqlen=max_seq,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2.to(device=self.device) * self.mlp(self.norm2(x)))
        return x

class rope_embedder(Model):
    def __init__(self,
                 token_dim=384,
                 max_seqlen=2560,
                 ):
        super().__init__()
        if token_dim % 2 != 0:
            new_token_dim = token_dim + 1
        else:
            new_token_dim = token_dim

        self.lin = torch.nn.Sequential(torch.nn.Linear(token_dim, token_dim),
                                       torch.nn.SELU(),
                                       torch.nn.BatchNorm1d(token_dim),
                                       torch.nn.Linear(token_dim, new_token_dim),
                                       torch.nn.SELU())
        
        exp = -2*(torch.arange(1, new_token_dim/2 + 1).repeat_interleave(2) -1)/new_token_dim
        base = 10000
        #theta_vec = torch.pow(base, exp)
        theta_mat = torch.arange(1,max_seqlen+1).view(-1,1).expand(-1,new_token_dim)*torch.pow(base, exp)
        self.cos_theta_mat = torch.cos(theta_mat)
        self.sin_theta_mat = torch.sin(theta_mat)


    def forward(self, data:Data):
        x_hat = self.lin(data.x)
        batched_x, mask = to_dense_batch(x=x_hat, batch=data.batch)
        emb = batched_x*self.cos_theta_mat[0:batched_x.shape[1]].to(device=self.device)
        indeces_even = torch.arange(0,batched_x.shape[2],2).to(device=self.device)
        indeces_odd = torch.arange(1,batched_x.shape[2],2).to(device=self.device)
        batched_x[:,:,indeces_odd], batched_x[:,:,indeces_even] = -batched_x[:,:,indeces_even], batched_x[:,:,indeces_odd]
        emb += batched_x*self.sin_theta_mat[0:batched_x.shape[1]].to(device=self.device)
        return emb[mask]

#endregion

#region ### Block with DeepIce Modifications ###

class Theseus_DeepIce(GNN):
    """DeepIce model."""

    def __init__(
        self,
        compression_model,   #here the variables to the compression are given
        fix_compression: bool = False,
        max_length: int = 2560,
        hidden_dim: int = 384,   #here the original TheseusDeepIce variables are given (possibly with some changes)
        mlp_ratio: int = 4,
        depth_two: int = 12,
        head_size: int = 32,
        depth_one: int = 4,
        exit_early: bool = False,
    ):
        """Construct `DeepIce`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with the original Kaggle
                Competition settings. If `include_dynedge` is False, this
                argument have no impact.
            n_features: The number of features in the input data.
            have_rel_bias: choose whether to use rel_pos_bias or not.
                False is recommended for training with a compression method
        """
        super().__init__(max_length, hidden_dim)

        self.compression = compression_model
        
        self.fix = fix_compression

        self.embedding = rope_embedder(token_dim=hidden_dim, max_seqlen=max_length)
        
        #first attention block, first layer has seperately roped q and k if rope_qk_sep==True
        self.sandwich = torch.nn.ModuleList(
            [flashMHA_block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / (depth_one - 1)),
                    init_values=1,
                )
                for i in range(depth_one)
            ]
        )

        self.cls_token = torch.nn.Linear(hidden_dim, 1, bias=False, dtype=torch.bfloat16)
        #second attention block
        self.blocks = torch.nn.ModuleList(
            [
                flashMHA_block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / (depth_two - 1)),
                    init_values=1,
                )
                for i in range(depth_two)
            ]
        )

        self.exit_early = exit_early


    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        #run through the compression model and embedding
        if self.fix:
            with torch.no_grad():
                compressed_data = self.compression(data)
        else:
            compressed_data = self.compression(data)

        x = self.embedding(compressed_data).to(dtype=torch.bfloat16)
        compressed_batch = compressed_data.batch

        #auxiliary variables for the Transformer architectures
        _,seq_lengths = torch.unique_consecutive(compressed_batch, return_counts=True)
        batch_size = seq_lengths.shape[0]
        cu_seqs = torch.nn.functional.pad(seq_lengths.cumsum(0), pad=(1,0), value=0).to(torch.int32)
        max_seq = seq_lengths.max().to(device=self.device, dtype=torch.int32).item()
 
        #actual Transformer procedure
        for blk in self.sandwich:
            x = blk(x=x,
                    cu_seqs=cu_seqs,
                    max_seq=max_seq,)

        cls_token = self.cls_token.weight.expand(
        batch_size, -1
        )
        #create empty tensor and append cls_tokens at the beginning of the individual sequences; fill the rest with original x
        emp = torch.empty((x.shape[0]+batch_size, x.shape[1])).to(device=self.device, dtype=torch.bfloat16)
        cls_ind = cu_seqs + torch.arange(batch_size+1).to(device=self.device, dtype=torch.int32)
        ran = torch.arange(x.shape[0]+batch_size).to(device=self.device)
        emp[cls_ind[:-1],:] = cls_token
        emp[ran[~torch.isin(ran,cls_ind[:-1]).to(device=self.device)],:] = x
        x=emp

        for blk in self.blocks:
            x = blk(x=x,
                    cu_seqs=cls_ind,
                    max_seq=max_seq+1,)
            
        if self.exit_early:
            keep_index = torch.ones(x.shape[0], dtype=bool)
            keep_index[cls_ind[:-1]] = False
            #cls_collection = x[cls_ind[:-1], :]
            #cls_collection = cls_collection.repeat_interleave(seq_lengths, dim=0)
            return x[keep_index].to(dtype=torch.float32) , x[cls_ind[:-1], :].to(dtype=torch.float32)# + cls_collection
        else:
            return x[cls_ind[:-1], :].to(dtype=torch.float32) #cu_seqs-1

#endregion


def main(param_path, save_path, gpus=None
) -> None:
    
    #this is an example use that makes predictions with a model given via param_path


    features = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'hlc']
    truths = ['energy', 'azimuth', 'zenith', 'position_x', 'position_y', 'position_z', 'event_no']

    config: Dict[str, Any] = {
        "num_workers": 30,
        "early_stopping_patience": 5,
        "batch_size": 200,
        "dataset_reference": SQLiteDataset,}
    

    graph_definition = KNNGraph(
    detector=IceCube86(),
    nb_nearest_neighbours=8,
    input_feature_names = features,
    )
    
    nt_path = "/scratch/users/nikme/northern_tracks_data/dev_northern_tracks_muon_labels_v3/"

    num_sets = 6
    db_paths = [nt_path+f'dev_northern_tracks_muon_labels_v3_part_{i}.db'for i in range(1,num_sets+1)]
    sels = []
    for i in range(len(db_paths)):
        query = "select event_no from truth limit 1000000"
        out = query_database(database=db_paths[i], query=query)
        sels.append(out['event_no'].tolist())
    
    # target_nb_events = 1000000 #meaning 10 sets
    # starting_set = [22010, 6]
    # db_paths = []
    # sels = []
    # count = 0

    # len_directory = '/ptmp/mpp/nikme/python_files/snowstorm_seqlen_parquets'
    # l = sorted(os.listdir(len_directory), key=lambda x:(int(re.search(r'runid\d+',x).group().replace('runid', '')),int(re.search(r'part_\d+',x).group().replace('part_', ''))))
    # #print(len(l))
    # for i in range(len(l)):
    #     current_set = [int(re.search(r'runid\d+',l[i]).group().replace('runid', '')),int(re.search(r'part_\d+',l[i]).group().replace('part_', ''))]
    #     print(current_set)
    #     if (current_set[0]<starting_set[0] or current_set[1]<starting_set[1]):
    #         print('skipped')
    #         pass
    #     else:
    #         if current_set[0]<22042:
    #             correct_dir = '/scratch/users/smagel/data/SnowStormDataset/sqlite/'
    #         else:
    #             print('failure imminent')
    #             correct_dir = '/ptmp/nikme/data_files/SnowStormData/sqlite/'
    #         db_paths.append(correct_dir+f'{current_set[0]}'+f'/merged_part_{current_set[1]}.db')
    #         #print(db_paths)
    #         valid_nbs = pd.read_parquet(path=len_directory+'/'+l[i], columns=['event_no']).head(1000000)
    #         sels.append(valid_nbs['event_no'].tolist())
    #         current_nb = min(int(re.search(r'nbevents\d+',l[i]).group().replace('nbevents', '')), len(sels[-1]))
    #         target_nb_events = target_nb_events - min(2000000, current_nb)
    #         count += min(2000000, current_nb)
    #         print(current_nb, target_nb_events)
    #         if target_nb_events <= 0:
    #             break

    # print('nb of selected events', count)
    
    #defining the data
    dm = GraphNeTDataModule(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "path": db_paths[-1], #1
            "pulsemaps" : "InIceDSTPulses",#InIceDSTPulses
            "truth_table" : 'truth',
            "features": features,
            "truth":truths,
            "data_representation" : graph_definition,
            "labels": {'direction': Direction()},  
        },
        selection=sels[-1],
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": True,
        },
        validation_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": False,
        },
        test_selection=sels[-1],
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": False,
        },
        train_val_split= [0.9, 0.1]
    )
    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader
    test_dataloader = dm.test_dataloader

    #define general variables
    max_length=3840
    lat_dim=768 #384 for v1 768 for v3

    #define encoding nets
    in_dim=5
    nb_messages=1
    lin_net_length=5
    ratio=0.5 # 1 for v1 0.5 for v3
  
    enc1 = simple_gnn_encoder(in_dim=in_dim,
                              lat_dim=lat_dim,
                              ratio=ratio,
                              nb_messages=nb_messages,
                              lin_net_length=lin_net_length)
    
    
    nb_nearest_enc = 12
    enc2 = simple_gnn_encoder(in_dim=in_dim,
                              lat_dim=lat_dim,
                              ratio=ratio,
                              nb_messages=nb_messages,
                              lin_net_length=lin_net_length,
                              nb_nearest=nb_nearest_enc)
    nb_nearest_enc = 20
    enc3 = simple_gnn_encoder(in_dim=in_dim,
                              lat_dim=lat_dim,
                              ratio=ratio,
                              nb_messages=nb_messages,
                              lin_net_length=lin_net_length,
                              nb_nearest=nb_nearest_enc)


    #actually subsampling - net_list is a list for subsampled encodings; empty means random subsample or based on hlc
    net_list=[enc1,enc2,enc3] # for v3 [enc1,enc2,enc3]
    nb_nearest=8
    scorer_length=5
    scorer_width=100
    hlc_pos=None
    random_subs_embedding=None
    
    
    
    compression = general_compression_module(max_length=max_length,
                                             lat_dim=lat_dim,
                                             nb_nearest=nb_nearest,
                                             net_list=net_list,
                                             scorer_length=scorer_length,
                                             scorer_width=scorer_width,
                                             hlc_pos=hlc_pos,
                                             random_subs_embedding=random_subs_embedding,)

    
    
    
    
    # train Theseus_Deepice
    backbone = Theseus_DeepIce(
        compression_model=compression,
        max_length=max_length,
        hidden_dim=lat_dim,
        exit_early=False) 
    latent_dim = backbone.nb_outputs
    #backbone.load_state_dict("/ptmp/nikme/training_files/models/modelv1_retry_cont_pretraining/modelv1_just_wiggle/pretrained_model/state_dict.pth")

    # model = cont_frame(
    #     enc_net=backbone,
    #     lat_feat=lat_dim,
    #     optimizer_class = RAdam,
    #     optimizer_kwargs = {'eps': 1e-05, 'lr': 2e-04},)
    
    
    # #define task; choose appropriate
    # task = EnergyReconstruction(
    #     target_labels=['energy'],
    #     hidden_size=latent_dim,
    #     transform_prediction_and_target = lambda x: torch.log10(x),
    #     loss_function=LogCoshLoss()
    # )

    # task = custom_EnergyReconstruction(
    #     target_labels=['energy'],
    #     hidden_size=latent_dim,
    #     transform_target = lambda x: torch.log10(x),
    #     transform_inference = lambda x: torch.pow(10, x),
    #     loss_function=LogCoshLoss()
    # )

    # task = DirectionReconstructionWithKappa(
    #     hidden_size=latent_dim,
    #     target_labels=['direction'],
    #     loss_function=VonMisesFisher3DLoss(),
    # )

    task = DirectionRecoNM(
        hidden_size=latent_dim,
        target_labels=['direction'],
        loss_function=OpeningAngleLoss(),
    )


    model = StandardModel(data_representation = graph_definition,
                          backbone = backbone,
                          tasks = task,
                          optimizer_class = RAdam,
                          optimizer_kwargs = {'eps': 1e-05, 'lr': 2e-04},
                          )
    

    model.load_state_dict(param_path)

    os.makedirs(save_path, exist_ok=True)

    # #switch to appropriate name and cols
    # name = 'energy_pred'
    # cols = ['energy_pred']
    name = 'direction_prediction'
    #xyz = ['x','y','z','kappa']
    xyz = ['x','y','z']
    cols = [f'direction_pred_{xyz[i]}' for i in range(len(xyz))]

    df = model.predict_as_dataframe(dataloader = test_dataloader,
                                    additional_attributes = truths,
                                    prediction_columns = cols,
                                    gpus = gpus)
    
    df.to_parquet(f'{os.path.join(save_path, name)}.parquet')
    
    #print(df.head())

if __name__ == "__main__":
    #some_list = [200]
    #some_list = [i for i in range(47,51)]
    #some_list = ['pre', 'scratch']
    some_list = ['combo6040'] #1mil=1mil, 1milmore=2mil, 1milmoreagain=3mil, 1milmoryetagain=4mil, 1milmorelast=5mil

    #nohup python /ptmp/mpp/nikme/python_files/from_raven/py_scripts/raven_Theseus_pretrained_modelv3_script.py > /ptmp/mpp/nikme/my_logs/pred_log1.out & 
    for i in range(len(some_list)):
        print('predicting', some_list[i])
        s_path = f'/ptmp/mpp/nikme/predictions_parquet/modelv3_vMF_plus_opening_angle/opening_angle_{some_list[i]}'
        p_path = f'/ptmp/mpp/nikme/python_files/from_raven/model_pths/modelv3_vMF_openingangle/opening_angle_10mil_snows_pretrained_{some_list[i]}/state_dict.pth'
        main(param_path=p_path, save_path=s_path, gpus=[1])