from abc import ABC, abstractmethod
from typing import Literal
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BaseSVDDFM(ABC):
    def __init__(self, device: torch.device, dfm: bool=False, dfm_num_layers: int=3,
                 dfm_aggr: Literal["SUM", "CONCAT"] ="SUM",
                 out_feat_type: Literal["HIDDEN", "CLS"] ="HIDDEN"):
        """
           params:
               device: torch.device, the device to which the model will be moved
               dfm: bool, whether to add DFM (Disentangled Fake Manifolds) layers
               dfm_num_layers: int, number of DFM layers
               dfm_aggr: Literal["SUM", "CONCAT"], aggregation method for DFM ("SUM" or "CONCAT")
               out_feat_type: Literal["HIDDEN", "CLS"], type of clip output features ("HIDDEN" or "CLS")
        """
        super(BaseSVDDFM, self).__init__()

        assert isinstance(device, torch.device), "device should be a torch.device"
        assert isinstance(dfm, bool), "dfm should be a boolean"
        assert isinstance(dfm_num_layers, int) and dfm_num_layers in [2, 3], "dfm_num_layers should be 2 or 3"
        # assert isinstance(dfm_num_mani, int) and 0 < dfm_num_mani <= 8, "dfm_num_mani should be between 1 and 8"
        assert dfm_aggr in ["SUM", "CONCAT"], f"dfm_aggr should be either 'SUM' or 'CONCAT' but got {dfm_aggr}"
        assert out_feat_type in ["HIDDEN", "CLS"], "out_feat_type should be either 'HIDDEN' or 'CLS'"

        self.device = device
        self.dfm = dfm
        dfm_num_mani = 2
        self.orthogonal_manifolds = nn.ModuleList([SingleManifoldLayer(dfm_num_layers) 
                                                   for _ in range(dfm_num_mani)]) if dfm else None

        if dfm:
            for manifold in self.orthogonal_manifolds:
                manifold.requires_grad_(True)
            self.dfm_aggr = dfm_aggr
            encoder_decoder_num_layers = 2
            if dfm_aggr == "SUM":
                hidden_dim = 1024
            else: # dfm_aggr == "CONCAT":
                hidden_dim = 1024 * dfm_num_mani
            self.dfm_encoder = SingleManifoldLayer(encoder_decoder_num_layers)
            self.dfm_encoder.requires_grad_(True)
            self.dfm_decoder = SingleManifoldLayer(encoder_decoder_num_layers, hidden_dim)
            self.dfm_decoder.requires_grad_(True)
        else:
            self.dfm_encoder = None
            self.dfm_decoder = None
        
        self.out_feat_type = out_feat_type

        self.feat_model = None  # To be defined in the subclass
        
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )
        self.head.requires_grad_(True)

    def forward_common(self, x: torch.Tensor):
        features = self.feat_model(x)

        if self.out_feat_type == "HIDDEN":
            features = features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
        else:  # self.out_feat_type == "CLS"
            features = features.pooler_output  # Use CLS token representation

        logger.debug(f"features shape after feat_model: {features.size()}")

        if self.dfm:
            results_manifolds_features = None
            features = self.dfm_encoder(features)
            encoder_features = features.view(features.size())

            if self.dfm_aggr == "SUM":
                manifolds_features = torch.tensor([]).to(features)
                for manifold in self.orthogonal_manifolds:
                    manifold_features = manifold(features)
                    manifolds_features = torch.cat((manifolds_features, manifold_features.unsqueeze(0)))  # Concatenate along dimension of manifolds

                logger.debug(f"manifolds_features shape after orthogonal manifolds: {manifolds_features.size()}")

                features = manifolds_features.sum(dim=0)  # Summation pooling across manifold dimension
                results_manifolds_features = manifolds_features.view(manifolds_features.size())

                logger.debug(f"features shape after summation pooling: {features.size()}")
                logger.debug(f"results_manifolds_features shape: {results_manifolds_features.size()}")
            else:  # self.dfm_aggr == "CONCAT":
                manifolds_features = []
                for manifold in self.orthogonal_manifolds:
                    manifold_features = manifold(features)
                    manifolds_features.append(manifold_features)

                logger.debug(f"manifolds_features list length after orthogonal manifolds: {len(manifolds_features)}")

                features = torch.hstack(manifolds_features) # Concatenate all the manifolds features
                results_manifolds_features = features.reshape(len(manifolds_features),
                                                              manifolds_features[0].size(0),
                                                              manifolds_features[0].size(1))

                logger.debug(f"features shape after concatenation pooling: {features.size()}")
                logger.debug(f"results_manifolds_features shape: {results_manifolds_features.size()}")

            features = self.dfm_decoder(features)

            logger.debug(f"features shape after decoder: {features.size()}")
            logger.debug(f"results_manifolds_features shape: {results_manifolds_features.size()}")

            return self.head(features), encoder_features, features, results_manifolds_features
        else:
            return self.head(features), features, None, None

    def to(self):
        self.head.to(self.device)
        if self.dfm:
            self.dfm_encoder.to(self.device)
            self.dfm_decoder.to(self.device)
            for manifold in self.orthogonal_manifolds:
                manifold.to(self.device)

    # Method to replace nn.Linear modules within attention modules with SVDResidualLinear
    @abstractmethod
    def replace_svd_residual_to_attn_linear(self, model, r):
        pass


class SingleManifoldLayer(nn.Module):
    def __init__(self, dfm_num_layers: int=2, hidden_dim: int=1024):
        """
            params:
                dfm_num_layers: int, number of DFM layers
                hidden_dim: int, hidden dimension size
        """
        super(SingleManifoldLayer, self).__init__()
        if dfm_num_layers == 2:
            self.manifold = nn.Sequential(
                nn.Linear(hidden_dim, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 1024, bias=True)
            )
        elif dfm_num_layers == 3:
            self.manifold = nn.Sequential(
                nn.Linear(hidden_dim, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, 1024, bias=True)
            )
        else:
            raise ValueError("Invalid number of DFM layers")

    def forward(self, x):
        x = self.manifold(x)

        return x


