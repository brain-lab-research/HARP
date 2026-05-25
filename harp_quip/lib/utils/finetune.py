import torch
from torch import nn


def extract_susv_params(module):
    susv_params = []
    harp_params = []
    other_params = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        parts = name.split(".")

        if ("SU" in name) or ("SV" in name) or (parts[-1] in ("SU", "SV")):
            susv_params.append(param)

        elif "harp" in parts:
            harp_params.append(param)

        else:
            other_params.append(param)

    seen = set()
    for group in (susv_params, harp_params, other_params):
        for p in group:
            pid = id(p)
            if pid in seen:
                raise RuntimeError("Parameter appears in multiple optimizer groups.")
            seen.add(pid)

    return susv_params, harp_params, other_params


def get_susv_adam(susv_params, harp_params, other_params, args):
    param_groups = [
        {"params": susv_params, "lr": args.ft_susv_lr},
    ]
    if len(harp_params) > 0:
        param_groups.append({"params": harp_params, "lr": args.ft_harp_lr})
    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": args.ft_lr})
    
    return torch.optim.Adam(param_groups)


def save_susv(module, path):
    saved_layer = torch.load(path, map_location=torch.device('cpu'))
    saved_layer['SU'] = module.SU.data.to(torch.float16)
    saved_layer['SV'] = module.SV.data.to(torch.float16)

    incoh_mode = getattr(module, "incoh_mode", saved_layer.get("incoh_mode", "had"))
    if incoh_mode == "harp":
        saved_layer['incoh_mode'] = "harp"
        if hasattr(module, "harp_cfg") and module.harp_cfg is not None:
            saved_layer['harp_cfg'] = module.harp_cfg
        if hasattr(module, "harp") and module.harp is not None:
            saved_layer['harp_state'] = {k: v.detach().cpu() for k, v in module.harp.state_dict().items()}

    torch.save(saved_layer, path)


def calculate_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            total_loss += nn.MSELoss()(layer(source.to(device), position_ids=position_ids)[0],
                                       target.to(device))
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()


def calculate_ce_loss(layer, position_ids, attention_mask, dataloader):
    layer.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            output = layer(
                source,
                position_ids=position_ids,
                attention_mask=attention_mask.float())[:, :-1].contiguous()
            total_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                target.to(0).view(-1, target.shape[-1]),
            )
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()
