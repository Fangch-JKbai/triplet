# =============================================
# File: src/models/mtl_model.py
# Purpose: A complete, end-to-end multi-task learning model with an ESM-2 backbone.
# =============================================
from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from esm import pretrained


class MTLModel(nn.Module):
    def __init__(self,
                 esm_name: str,
                 cpt: bool = False,
                 cpt_checkpoint_path: str = None,
                 freeze_encoder: bool = True, # 控制是否冻结骨干网络
                 head_hidden: Optional[int] = None,
                 dropout: float = 0.1):

        super().__init__()
        # 将冻结选项传递给 Encoder
        self.encoder = ESM2Encoder(esm_name = esm_name,
                                   cpt = cpt,
                                   cpt_checkpoint_path = cpt_checkpoint_path, 
                                   freeze=freeze_encoder)
        
        embedding_dim = self.encoder.hidden
        self.ec_head = ECHead(input_dim=embedding_dim, d1=512, d2=256, d_z=128, dropout_rate=0.2)
        self.kcat_head = KCatHead(embedding_dim, dropout=0.25, hidden=512)
        self.ph_head = PHHead(embedding_dim, dropout=dropout, hidden=head_hidden)
        self.temp_head = TempHead(embedding_dim, dropout=dropout, hidden=head_hidden)

    @property
    def alphabet(self):
        return self.encoder.alphabet

    @property
    def batch_converter(self):
        return self.encoder.batch_converter
        
    # --- NEW METHOD ---
    # This is the only change you need to make in this file.
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)

    def unfreeze_last_k(self, k: int):
        self.encoder.unfreeze_last_k(k)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The 'encode' method is for embedding generation.
        # The 'forward' method is for end-to-end training.
        h = self.encode(tokens)

        logits = self.ec_head(h)
        pred_kcat = self.kcat_head(h)
        pred_ph = self.ph_head(h)
        pred_temp = self.temp_head(h)

        return {
            'logits': logits,
            'kcat': pred_kcat,
            'ph_opt': pred_ph,
            'temp_opt': pred_temp,
        }



class ESM2Encoder(nn.Module):
    def __init__(self,
                 esm_name: str = "esm2_t33_650M_UR50D",
                 cpt: bool = True,
                 cpt_checkpoint_path: str = None,
                 freeze: bool = True):
        super().__init__()

        model, alphabet = pretrained.load_model_and_alphabet(esm_name)
        self.esm = model
        self.alphabet = alphabet
        self.checkpoint_path = cpt_checkpoint_path
        if cpt and cpt_checkpoint_path and os.path.isfile(cpt_checkpoint_path):
            print(f"   - [ESM2Encoder] Loading CPT weights from: {cpt_checkpoint_path}")
            checkpoint = torch.load(cpt_checkpoint_path, map_location="cpu")
            
            # 你的 .bin 文件本身就是 state_dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # 直接加载到 self.esm 子模块
            missing, unexpected = self.esm.load_state_dict(state_dict, strict=False)
            
            # 添加检查，确保加载成功
            if not unexpected and not missing:
                print("   - [ESM2Encoder] CPT weights loaded with a perfect match.")
            elif not unexpected and len(missing) < 10: # 有时会少一些lm_head的键，是正常的
                print("   - [ESM2Encoder] CPT weights loaded successfully.")
            else:
                print("   - [ESM2Encoder] ⚠️ WARNING: CPT weights might have failed to load correctly.")
                print(f"     - Unexpected keys: {unexpected[:5]}")
                print(f"     - Missing keys: {missing[:5]}")
        else:
             print(f"   - [ESM2Encoder] Using official pre-trained weights for ESM2.")

        self.batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
        
        self.padding_idx = alphabet.padding_idx
        self.hidden = getattr(model, 'embed_dim', 0)
        self.num_layers = getattr(model, 'num_layers', 0)
        self.repr_layer = self.num_layers

        # 根据 freeze 参数决定是否冻结所有参数
        if freeze:
            print("ESM2Encoder is frozen.")
            for param in self.esm.parameters():
                param.requires_grad = False
        else:
            print("ESM2Encoder is trainable.")

    def unfreeze_last_k(self, k: int = 12):
        # 首先确保所有参数都被冻结
        for p in self.esm.parameters():
            p.requires_grad = False

        # 然后只解冻最后 k 层
        total_layers = len(self.esm.layers)
        if k > total_layers:
            k = total_layers
        
        for layer in self.esm.layers[total_layers - k:]:
            for p in layer.parameters():
                p.requires_grad = True
        
        print(f"Unfroze the last {k} layers of the ESM2 encoder.")
    
    def _pool(self, reps: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        is_amino_acid = (input_ids != self.alphabet.cls_idx) & \
                        (input_ids != self.alphabet.eos_idx) & \
                        (input_ids != self.padding_idx)
        mask = is_amino_acid.unsqueeze(-1).type_as(reps)
        summed = (reps * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.esm(input_ids, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]
        h = self._pool(reps, input_ids)
        return h

# --- ECHead ---
class ECHead(nn.Module):
    def __init__(self, input_dim, d1=512, d2=256, d_z=128, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d1)
        self.ln1 = nn.LayerNorm(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.ln2 = nn.LayerNorm(d2)
        self.fc3 = nn.Linear(d2, d_z)
        self.dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = self.dropout(h)
        z = self.fc3(h)
        return F.normalize(z, p=2, dim=-1)


# --- KineticsHead ---
class KCatHead(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.1, hidden: Optional[int] = None):
        super().__init__()
        out_dim = 1
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
    


# --- PHHead，专门负责 pH 预测并添加范围约束 ---
class PHHead(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.1, hidden: Optional[int] = None):
        super().__init__()
        out_dim = 1
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# --- 新建 TempHead，专门负责温度预测 ---
class TempHead(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.1, hidden: Optional[int] = None):
        super().__init__()
        out_dim = 1
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), 
                nn.Linear(d_in, hidden), 
                nn.GELU(), 
                nn.LayerNorm(hidden), 
                nn.Dropout(dropout), 
                nn.Linear(hidden, out_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout), 
                nn.Linear(d_in, out_dim)
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)