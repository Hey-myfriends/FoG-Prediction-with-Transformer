from multiprocessing import reduction
import pdb, sys, os
print("Current path is ", os.getcwd())
# if os.getcwd() not in sys.path:
    # sys.path.append(os.getcwd())
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .backbone import build_backbone
from .encoder import build_encoder_FoG
from .positional_encoding import PositionalEncoding
from typing import Optional
import sklearn.metrics as metrics

class FoG_Net(nn.Module):
    def __init__(self, backbone, encoder, num_classes, cls_type="cls_token", aux_loss=False) -> None:
        super().__init__()

        assert cls_type == "cls_token" or cls_type == "global_ave", "cls_type error: must be cls_token or global_ave."

        self.backbone = backbone
        self.encoder = encoder
        hidden_dim = self.encoder.d_model
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, 1)
        self.num_classes = num_classes
        self.cls_type = cls_type
        self.aux_loss = aux_loss
        self.cls_embed = nn.Linear(hidden_dim, num_classes)
        self.cls_token = None
        self.pos_embed = PositionalEncoding(hidden_dim)
        if cls_type == "cls_token":
            # self.pos_embed = nn.Embedding(hidden_dim, 24+1).weight
            self.cls_token = torch.randn(hidden_dim, 1)

    def forward(self, x):
        # pdb.set_trace()
        features = self.backbone(x)
        features = self.input_proj(features)

        # mask = torch.full().to(features.device)
        bs, c, L = features.shape
        if self.cls_token is not None:
            cls_token = self.cls_token.repeat(bs, 1, 1).to(features.device) # (bs, L, d_model)
            features = torch.cat((cls_token, features), dim=2) # TODO
        # pos = self.pos_embed.unsqueeze(0).repeat(bs, 1, 1) # not properly
        pos = self.pos_embed(L+1 if self.cls_type == "cls_token" else L).unsqueeze(0).repeat(bs, 1, 1)
        hs = self.encoder(features, src_key_padding_mask=None, pos=pos)
        if self.cls_type == "cls_token":
            output_class = self.cls_embed(hs[:, :, 0, :])
        else:
            output_class = self.cls_embed(hs.mean(dim=2))
        out = {"pred_logits": output_class[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_outputs(output_class)
        return out

    @torch.jit.unused
    def _set_aux_outputs(self, out_class):
        return [{"pred_logits": a} for a in out_class[:-1]]


class SetCriterion(nn.Module):
    def __init__(self, num_class: int, loss: list, weight_dict: dict, cls_weight: Optional[Tensor], 
                focal_scaler: Optional[int] = None) -> None:
        super().__init__()
        assert num_class == 2 or num_class == 3
        self.num_classes = num_class
        self.loss = loss
        self.cls_weights = cls_weight
        self.focal_scaler = focal_scaler
        self.weight_dict = weight_dict

    def loss_labels(self, outputs: dict, targets: Tensor, log=True):
        # pdb.set_trace()
        assert "pred_logits" in outputs.keys()
        src_logits = outputs['pred_logits']
        self.cls_weights = self.cls_weights.to(src_logits.device)
        if self.num_classes == 3:
            loss_ce = F.cross_entropy(src_logits, targets, reduction="none")
            if log:
                loss_details = {
                    "loss_NW": loss_ce[targets == 0].mean(),
                    "loss_preFoG": loss_ce[targets == 1].mean(),
                    "loss_FoG": loss_ce[targets == 2].mean()
                }
            at = self.cls_weights.gather(dim=0, index=targets)
            if self.focal_scaler is not None:
                pt = torch.exp(-loss_ce)
                loss_ce = at * (1 - pt) ** self.focal_scaler * loss_ce
            else:
                loss_ce = at * loss_ce
            loss_ce = loss_ce.mean()
        elif self.num_classes == 2:
            targets_clone = targets.clone()
            targets_clone[targets_clone == 2] = 1
            loss_ce = F.cross_entropy(src_logits, targets_clone, reduction="none")
            if log:
                loss_details = {
                    "loss_NW": loss_ce[targets == 0].mean(),
                    "loss_preFoG": loss_ce[targets == 1].mean(),
                    "loss_FoG": loss_ce[targets == 2].mean()
                }
            at = self.cls_weights.gather(dim=0, index=targets)
            if self.focal_scaler is not None:
                pt = torch.exp(-loss_ce)
                loss_ce = at * (1 - pt) ** self.focal_scaler * loss_ce
            else:
                loss_ce = at * loss_ce
            loss_ce = loss_ce.mean()
        
        losses = {"loss_ce": loss_ce}

        if log:
            losses.update(loss_details)
            losses["class_error"] = 100 - accuracy(src_logits, targets) * 100 # TODO: log
        return losses

    def get_loss(self, loss: str, outputs: dict, targets: Tensor, **kwargs):
        loss_map = {
            "labels": self.loss_labels
        }
        assert loss in loss_map, "loss type error, please check it."
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs: dict, targets: Tensor):
        losses = {}
        for loss in self.loss:
            losses.update(self.get_loss(loss, outputs, targets))

        if "aux_outputs" in outputs.keys():
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.loss:
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, **kwargs)
                    l_dict = {k+"_{}".format(i): v for k,v in l_dict.items()}
                    losses.update(l_dict)
        return losses

@torch.no_grad()
def accuracy(outputs: Tensor, targets: Tensor):
    _, pred = outputs.max(dim=1)
    accu = metrics.accuracy_score(targets.cpu().numpy(), pred.cpu().numpy())
    return accu

def build(in_chan, d_model, num_class, cls_type="cls_token", cls_weight=torch.ones(3), aux_loss=True, focal_scaler=2):
    
    backbone = build_backbone(in_chan, d_model)
    encoder = build_encoder_FoG(d_model=d_model, nhead=8, num_encoder_layers=6, dim_feedforward=4*d_model, 
                                normalize_before=True, return_intermediate=True if aux_loss else False)
    model = FoG_Net(backbone, encoder, num_class, cls_type=cls_type, aux_loss=aux_loss)
    
    weight_dict = {"loss_ce": 1}
    if aux_loss:
        aux_weight_dict = {}
        for i in range(encoder.num_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels"]
    criterion = SetCriterion(num_class, losses, weight_dict, cls_weight, focal_scaler=focal_scaler)
    # criterion.to(device)
    # postprocessor = PostProcess()
    return model, criterion #, postprocessor

if __name__ == "__main__":
    x = torch.rand(4, 6, 1500)
    pdb.set_trace()
    in_chan, d_model, num_class = x.shape[1], 128, 3
    model, _ = build(in_chan, d_model, num_class, cls_type="global_ave")
    y = model(x)
    print(x.shape, y["pred_logits"].shape)