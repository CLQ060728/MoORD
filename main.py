import json
from typing import Literal, Dict, Any
import torch
import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as LTN
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score
from torchmetrics.functional.classification import binary_precision, binary_recall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchmetrics.classification import BinaryPrecisionRecallCurve
from lightning.pytorch.cli import LightningCLI
from Model.DFDDFM import ClipSVDDFM, Dinov2SVDDFM, Dinov3SVDDFM
from Model.FeatureExtractors import ClipFeatureExtractor, Dinov2FeatureExtractor, Dinov3FeatureExtractor
from Loss.DFDDFMLosses import DFDLoss, ReconstructionLoss, SVDLoss, ConsistencyLoss, DistanceLoss
from Loss.DFDDFMLosses import SparsityLoss, ReconRegLoss
from Dataset.DatasetLoader import DFDDFMTrainDataModule
from Utils.UtilFunctions import ConfigDict
import logging, os

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')


class DFDDFMTrainer(LTN.LightningModule):
    def __init__(self,
                 model_mode: Literal["SVDDFM", "SVD_DFM", "SVD", "FEAT", "FEAT_LINEAR"] = "SVDDFM",
                 svd_dfm: Dict[str, Any] = {},
                 model_type: Literal["CLIP", "DINO_V2", "DINO_V3"] | None = "DINO_V3",
                 feat_extractor_type: Literal["CLIP", "DINO_V2", "DINO_V3"] | None = None,
                 model_configs: Dict[str, Any] = {},
                 loss_configs: Dict[str, Any] = {},
                 optim_configs: Dict[str, Any] = {},
                 inference_configs: Dict[str, Any] = {}):
        """
            Initialize the DFDDFMTrainer with the given configurations.
            Params:
                model_mode: The mode of the model (SVDDFM, SVD_DFM, SVD, FEAT, FEAT_LINEAR).
                svd_dfm: Configuration dictionary for the SVD_DFM model.
                model_type: The type of the DFM model (CLIP, DINO_V2, DINO_V3).
                feat_extractor_type: The type of the feature extractor (CLIP, DINO_V2, DINO_V3).
                model_configs: Configuration dictionary for the model.
                loss_configs: Configuration dictionary for the loss functions.
                optim_configs: Configuration dictionary for the optimizer.
                inference_configs: Configuration dictionary for inference/prediction settings.
        """
        super(DFDDFMTrainer, self).__init__()
        self.save_hyperparameters()
        if model_mode == "SVD_DFM":
            assert svd_dfm is not None and len(svd_dfm) > 0, "svd_dfm config must be provided for SVD_DFM model_mode"
            assert svd_dfm.get("svd_chkpt_path", None) is not None and svd_dfm.get("svd_chkpt_path", None) != "", "svd_chkpt_path must be provided for SVD_DFM model_mode"
        assert model_mode in ["SVDDFM", "SVD_DFM", "SVD", "FEAT", "FEAT_LINEAR"], f"Invalid model_mode: {model_mode}"
        assert model_type in ["CLIP", "DINO_V2", "DINO_V3"], f"Invalid model_type: {model_type}"

        self.model_mode = model_mode
        self.svd_dfm = ConfigDict(svd_dfm)
        self.svd_dfm_with_dfd = self.svd_dfm.svd_dfm_with_dfd
        self.learning_rate = optim_configs.get("learning_rate", 2e-4)
        self.do_reconstruction = optim_configs.get("do_reconstruction", False)
        self.use_recon_reg_loss = optim_configs.get("use_recon_reg_loss", False)
        self.use_recon_reg_loss = True if self.do_reconstruction else self.use_recon_reg_loss
        self.model_type = model_type
        self.feat_extractor_type = feat_extractor_type
        self.model_configs = ConfigDict(model_configs)
        self.loss_configs = ConfigDict(loss_configs)
        self.optim_configs = ConfigDict(optim_configs)
        self.inference_configs = ConfigDict(inference_configs)
        self.beta_1 = self.loss_configs.DistanceSparsity.beta_1_start
        self.__get_all_training_objects__()  # Initialize model and losses
    
    def __check_model_grad__(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logger.info(f"Total params: {total_params}")
        logger.info(f"Total trainable params: {trainable_params}")
        logger.info(f"Total frozen params: {frozen_params}")

    def __get_all_training_objects__(self):
        if self.model_mode == "SVD":
            self.model_configs.ClipSVDDFM.dfm = False
            self.model_configs.Dinov2SVDDFM.dfm = False
            self.model_configs.Dinov3SVDDFM.dfm = False
        elif self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            self.model_configs.ClipSVDDFM.dfm = True
            self.model_configs.Dinov2SVDDFM.dfm = True
            self.model_configs.Dinov3SVDDFM.dfm = True

        if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
            if self.model_type == "CLIP":
                self.model = ClipSVDDFM(self.device, self.model_configs.ClipSVDDFM.dfm,
                                        self.model_configs.ClipSVDDFM.dfm_num_layers,
                                        self.model_configs.ClipSVDDFM.dfm_aggr,
                                        self.model_configs.ClipSVDDFM.out_feat_type,
                                        self.model_configs.ClipSVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V2":
                self.model = Dinov2SVDDFM(self.device, self.model_configs.Dinov2SVDDFM.dfm,
                                        self.model_configs.Dinov2SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov2SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov2SVDDFM.out_feat_type,
                                        self.model_configs.Dinov2SVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V3":
                self.model = Dinov3SVDDFM(self.device, self.model_configs.Dinov3SVDDFM.dfm,
                                        self.model_configs.Dinov3SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov3SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov3SVDDFM.out_feat_type,
                                        self.model_configs.Dinov3SVDDFM.model_type,
                                        self.model_configs.Dinov3SVDDFM.chkpt_dir)
        
        if self.model_mode == "SVD_DFM":
            assert os.path.exists(self.svd_dfm.svd_chkpt_path), f"SVD checkpoint path does not exist: {self.svd_dfm.svd_chkpt_path}"
            logger.debug(f"Loading pretrained SVD model from {self.svd_dfm.svd_chkpt_path} for SVD_DFM training...")

            pre_trained_svd_feat_model = self.load_from_checkpoint(self.svd_dfm.svd_chkpt_path).model.feat_model
            pre_trained_svd_feat_model.requires_grad_(False)  # Freeze the feature extractor
            self.__check_model_grad__(pre_trained_svd_feat_model)

            if self.model_type == "CLIP":
                self.model = ClipSVDDFM(self.device, self.model_configs.ClipSVDDFM.dfm,
                                        self.model_configs.ClipSVDDFM.dfm_num_layers,
                                        self.model_configs.ClipSVDDFM.dfm_aggr,
                                        self.model_configs.ClipSVDDFM.out_feat_type,
                                        self.model_configs.ClipSVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V2":
                self.model = Dinov2SVDDFM(self.device, self.model_configs.Dinov2SVDDFM.dfm,
                                        self.model_configs.Dinov2SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov2SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov2SVDDFM.out_feat_type,
                                        self.model_configs.Dinov2SVDDFM.chkpt_dir)
            elif self.model_type == "DINO_V3":
                self.model = Dinov3SVDDFM(self.device, self.model_configs.Dinov3SVDDFM.dfm,
                                        self.model_configs.Dinov3SVDDFM.dfm_num_layers,
                                        self.model_configs.Dinov3SVDDFM.dfm_aggr,
                                        self.model_configs.Dinov3SVDDFM.out_feat_type,
                                        self.model_configs.Dinov3SVDDFM.model_type,
                                        self.model_configs.Dinov3SVDDFM.chkpt_dir)
            self.__check_model_grad__(self.model.feat_model)

            self.model.feat_model = pre_trained_svd_feat_model.to(self.model.device)

            self.__check_model_grad__(self.model.feat_model)
            self.__check_model_grad__(self.model)

        if self.model_mode == "FEAT" or self.model_mode == "FEAT_LINEAR":
            self.model_configs.ClipFeatureExtractor.as_linear_classifier =\
                True if self.model_mode == "FEAT_LINEAR" else False
            self.model_configs.Dinov2FeatureExtractor.as_linear_classifier =\
                True if self.model_mode == "FEAT_LINEAR" else False
            self.model_configs.Dinov3FeatureExtractor.as_linear_classifier =\
                True if self.model_mode == "FEAT_LINEAR" else False
            
            if self.feat_extractor_type == "CLIP":
                self.feat_extractor = ClipFeatureExtractor(self.device,
                                                self.model_configs.ClipFeatureExtractor.as_linear_classifier,
                                                self.model_configs.ClipFeatureExtractor.chkpt_dir)
            elif self.feat_extractor_type == "DINO_V2":
                self.feat_extractor = Dinov2FeatureExtractor(self.device,
                                                self.model_configs.Dinov2FeatureExtractor.as_linear_classifier,
                                                self.model_configs.Dinov2FeatureExtractor.chkpt_dir)
            elif self.feat_extractor_type == "DINO_V3":
                self.feat_extractor = Dinov3FeatureExtractor(self.device,
                                                self.model_configs.Dinov3FeatureExtractor.pre_ds,
                                                self.model_configs.Dinov3FeatureExtractor.as_linear_classifier,
                                                self.model_configs.Dinov3FeatureExtractor.chkpt_dir)
        
        if self.model_mode != "FEAT":
            self.dfd_loss = DFDLoss(coef=self.loss_configs.DFDLoss.coef)
            if not self.svd_dfm_with_dfd and (self.model_mode == "SVD_DFM"):
                self.dfd_loss = None
                del self.dfd_loss
            if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
                self.svd_loss = SVDLoss(coef=self.loss_configs.SVDLoss.coef)
            if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
                self.consistency_loss = ConsistencyLoss()
                self.distance_loss = DistanceLoss()
                self.sparsity_loss = SparsityLoss()
                if self.do_reconstruction:
                    self.recon_loss = ReconstructionLoss(coef=self.loss_configs.ReconstructionLoss.coef)
                    self.recon_reg_loss = ReconRegLoss(coef=self.loss_configs.ReconRegLoss.beta_3_max)
                else:
                    if self.use_recon_reg_loss:
                        self.recon_reg_loss = ReconRegLoss(coef=self.loss_configs.ReconRegLoss.beta_3_max)

    def forward(self, batch):
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            x_pair, x2_manifold_indices, y_pair = batch
            x_1, x_2 = x_pair
            y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1 = self.model(x_1)
            y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2 = self.model(x_2)
            return y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
                   y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
                   x2_manifold_indices, y_pair
        elif self.model_mode == "SVD":
            x, y = batch
            y_hat, _, _, _ = self.model(x)
            return y_hat, y
        elif self.model_mode == "FEAT_LINEAR":
            x, y = batch
            y_hat = self.feat_extractor(x)
            return y_hat, y
        else:
            return self.feat_extractor(batch)

    def __svd_linear_training_step__(self, batch, total_loss):
        x_pair, _, y_pair = batch  # x2_manifold_indices
        x_1, x_2 = x_pair
        y_1, y_2 = y_pair
        batch_1 = (x_1, y_1)
        batch_2 = (x_2, y_2)
        y_hat_1, y_1 = self(batch_1)
        y_hat_2, y_2 = self(batch_2)
        dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
        dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)

        logger.debug(f"dfd_loss_dict_1: {dfd_loss_dict_1}")
        logger.debug(f"dfd_loss_dict_2: {dfd_loss_dict_2}")

        dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
        dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
        total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
        if self.model_mode != "FEAT_LINEAR" and self.model_mode != "FEAT":
            svd_losses_dict = self.svd_loss(self.model)
            svd_losses_value = svd_losses_dict["svd_losses_orth_keepsv"]
            total_loss.update({"svd_losses": svd_losses_value})

    def __check_network_grad__(self):
        total_frozen_params = 0
        total_trainable_params = 0
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
            total_params = sum(p.numel() for p in self.model.parameters())
            total_trainable_params += sum(p.numel() for p in self.model.parameters()
                                          if p.requires_grad)
            total_frozen_params += total_params - total_trainable_params
        elif self.model_mode == "FEAT" or self.model_mode == "FEAT_LINEAR":
            total_params = sum(p.numel() for p in self.feat_extractor.parameters())
            total_trainable_params += sum(p.numel() for p in self.feat_extractor.parameters()
                                          if p.requires_grad)
            total_frozen_params += total_params - total_trainable_params
        
        logger.debug(f"{self.model_mode}, Total trainable params: {total_trainable_params}")
        logger.debug(f"{self.model_mode}, Total frozen params: {total_frozen_params}")

        # if self.model_mode == "SVDDFM" or self.model_mode == "SVD":
        #     logger.debug(f"Checking feat extractor parameters...")
        #     for param in network.feat_model.parameters():
        #         if param.requires_grad:
        #             total_trainable_params += 1
        #         else:
        #             total_frozen_params += 1
                
        # logger.debug(f"Total trainable params with feat extractor: {total_trainable_params}")
        # logger.debug(f"Total frozen params with feat extractor: {total_frozen_params}")

    def training_step(self, batch, batch_idx):
        total_loss = {}
        if self.model_mode == "SVDDFM":

            # self.__check_network_grad__()

            if self.current_epoch < self.optim_configs.dfm_start_epoch:
                logger.debug(f"Switching model mode from SVDDFM to SVD")

                self.model_mode = "SVD"
                self.model.dfm = False

                logger.debug(f"Model type set to: {self.model_type}; Model DFM set to: {self.model.dfm}")
                logger.debug(f"total_loss before __svd_linear_training_step__: {total_loss}")

                self.__svd_linear_training_step__(batch, total_loss)

                logger.debug(f"total_loss after __svd_linear_training_step__: {total_loss}")

                self.model_mode = "SVDDFM"
                self.model.dfm = True

                self.log_dict(total_loss, prog_bar=True, sync_dist=True)
                logger.debug(f"Model type set to: {self.model_type}; Model DFM set to: {self.model.dfm}")
            else:
                logger.debug(f"Epoch {self.current_epoch}: Training with full SVDDFM model")

                y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
                y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
                x2_manifold_indices, y_pair = self(batch)
                y_1, y_2 = y_pair
                
                # compute dfd and svd losses
                dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
                dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)
                dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
                dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
                total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
                svd_losses_dict = self.svd_loss(self.model)
                svd_losses_value = svd_losses_dict["svd_losses_orth_keepsv"]
                total_loss.update({"svd_losses": svd_losses_value})
                
                # compute dfm reconstruction and regularization losses
                if self.do_reconstruction:
                    recon_loss_dict_1 = self.recon_loss(decoder_features_1, encoder_features_1)
                    recon_loss_dict_2 = self.recon_loss(decoder_features_2, encoder_features_2)
                    recon_loss_value_1 = recon_loss_dict_1[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                    recon_loss_value_2 = recon_loss_dict_2[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                    total_loss.update({"recon_loss": (recon_loss_value_1 + recon_loss_value_2) / 2})
                    
                if self.current_epoch < self.optim_configs.dissparcons_start_epoch and self.use_recon_reg_loss:
                    recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                    recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                    total_loss.update({"recon_reg_loss": recon_reg_loss_value})
                
                # compute other dfm losses after 'dissparcons_start_epoch's, i.e., distance, sparsity, consistency
                if self.current_epoch >= self.optim_configs.dissparcons_start_epoch:
                    self.beta_1 = self.loss_configs.DistanceSparsity.beta_1_start * (1 + (1 - 1 / (batch_idx + 1)))\
                                if self.beta_1 < self.loss_configs.DistanceSparsity.beta_1_end\
                                else self.loss_configs.DistanceSparsity.beta_1_end
                    beta_2 = (self.beta_1 - 1 / (self.current_epoch + 1))\
                                if (self.beta_1 - 1 / (self.current_epoch + 1)) > 0 else 0
                    beta_2 = self.loss_configs.ConsistencyLoss.beta_2_max\
                                if beta_2 > self.loss_configs.ConsistencyLoss.beta_2_max\
                                else beta_2
                    beta_3 = self.loss_configs.ReconRegLoss.beta_3_max - beta_2
                    self.distance_loss.set_coef(self.beta_1)
                    self.sparsity_loss.set_coef(self.beta_1)
                    self.consistency_loss.set_coef(beta_2)
                    self.recon_reg_loss.set_coef(beta_3)

                    logger.debug(f"Distance loss coefficient set to: {self.distance_loss.coef}")
                    logger.debug(f"Sparsity loss coefficient set to: {self.sparsity_loss.coef}")
                    logger.debug(f"Consistency loss coefficient set to: {self.consistency_loss.coef}")
                    logger.debug(f"Reconstruction regularization loss coefficient set to: {self.recon_reg_loss.coef}")

                    distance_loss_dict = self.distance_loss(manifolds_features_1, manifolds_features_2,
                                                            x2_manifold_indices)
                    distance_loss_value = distance_loss_dict["distance_loss"]
                    total_loss.update({"distance_loss": distance_loss_value})

                    sparsity_loss_dict = self.sparsity_loss(manifolds_features_1, manifolds_features_2)
                    sparsity_loss_value = sparsity_loss_dict["sparsity_loss"]
                    total_loss.update({"sparsity_loss": sparsity_loss_value})

                    # prepare S_hat for all the manifolds, consistency loss
                    S_hat = torch.tensor([]).to(manifolds_features_1)
                    for manifold_idx in range(manifolds_features_1.size(0)):
                        remaining_indices = torch.tensor([idx for idx in range(manifolds_features_1.size(0))\
                                                          if idx != manifold_idx]).to(manifolds_features_1.device)

                        logger.debug(f"manifold_idx: {manifold_idx}; remaining_indices: {remaining_indices}")

                        if self.model.dfm_aggr == "SUM":
                            aggr_12 = torch.cat((manifolds_features_1[[manifold_idx], :, :],
                                                 manifolds_features_2[remaining_indices, :, :]),
                                                 dim=0).sum(dim=0)
                            
                            logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                        else: # self.model.dfm_aggr == "CONCAT"
                            aggr_12 = []
                            aggr_12.append(manifolds_features_1[manifold_idx, :, :])
                            for remaining_idx in remaining_indices:
                                aggr_12.append(manifolds_features_2[remaining_idx, :, :])
                            aggr_12 = torch.hstack(aggr_12)
                            
                            logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                        
                        X_s_hat = self.model.dfm_decoder(aggr_12)
                        f_12 = self.model.dfm_encoder(X_s_hat)
                        S_manifold_idx = self.model.orthogonal_manifolds[manifold_idx](f_12)
                        S_hat = torch.cat((S_hat, S_manifold_idx.unsqueeze(0)), dim=0)

                    logger.debug(f"S_hat shape: {S_hat.size()}")

                    consistency_loss_dict = self.consistency_loss(S_hat, manifolds_features_1)
                    consistency_loss_value = consistency_loss_dict["consistency_loss"]
                    total_loss.update({"consistency_loss": consistency_loss_value})

                    if self.use_recon_reg_loss:
                        recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                        recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                        total_loss.update({"recon_reg_loss": recon_reg_loss_value})
        elif self.model_mode == "SVD_DFM":
            logger.debug(f"Epoch {self.current_epoch}: Training with SVD_DFM model")

            y_hat_1, encoder_features_1, decoder_features_1, manifolds_features_1,\
            y_hat_2, encoder_features_2, decoder_features_2, manifolds_features_2,\
            x2_manifold_indices, y_pair = self(batch)
            y_1, y_2 = y_pair
            
            # compute dfd losses if applicable
            if self.svd_dfm_with_dfd:
                dfd_loss_dict_1 = self.dfd_loss(y_hat_1, y_1)
                dfd_loss_dict_2 = self.dfd_loss(y_hat_2, y_2)
                dfd_loss_value_1 = dfd_loss_dict_1["dfd_loss"]
                dfd_loss_value_2 = dfd_loss_dict_2["dfd_loss"]
                total_loss.update({"dfd_loss": (dfd_loss_value_1 + dfd_loss_value_2) / 2})
            
            # compute dfm reconstruction and regularization losses
            if self.do_reconstruction:
                recon_loss_dict_1 = self.recon_loss(decoder_features_1, encoder_features_1)
                recon_loss_dict_2 = self.recon_loss(decoder_features_2, encoder_features_2)
                recon_loss_value_1 = recon_loss_dict_1[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                recon_loss_value_2 = recon_loss_dict_2[f"reconstruction_loss_{self.recon_loss.loss_type}"]
                total_loss.update({"recon_loss": (recon_loss_value_1 + recon_loss_value_2) / 2})
                
            if self.current_epoch < self.optim_configs.dissparcons_start_epoch and self.use_recon_reg_loss:
                recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                total_loss.update({"recon_reg_loss": recon_reg_loss_value})
            
            # compute other dfm losses after 'dissparcons_start_epoch's, i.e., distance, sparsity, consistency
            if self.current_epoch >= self.optim_configs.dissparcons_start_epoch:
                self.beta_1 = self.loss_configs.DistanceSparsity.beta_1_start * (1 + (1 - 1 / (batch_idx + 1)))\
                            if self.beta_1 < self.loss_configs.DistanceSparsity.beta_1_end\
                            else self.loss_configs.DistanceSparsity.beta_1_end
                beta_2 = (self.beta_1 - 1 / (self.current_epoch + 1))\
                            if (self.beta_1 - 1 / (self.current_epoch + 1)) > 0 else 0
                beta_2 = self.loss_configs.ConsistencyLoss.beta_2_max\
                            if beta_2 > self.loss_configs.ConsistencyLoss.beta_2_max\
                            else beta_2
                beta_3 = self.loss_configs.ReconRegLoss.beta_3_max - beta_2
                self.distance_loss.set_coef(self.beta_1)
                self.sparsity_loss.set_coef(self.beta_1)
                self.consistency_loss.set_coef(beta_2)
                self.recon_reg_loss.set_coef(beta_3)

                logger.debug(f"Distance loss coefficient set to: {self.distance_loss.coef}")
                logger.debug(f"Sparsity loss coefficient set to: {self.sparsity_loss.coef}")
                logger.debug(f"Consistency loss coefficient set to: {self.consistency_loss.coef}")
                logger.debug(f"Reconstruction regularization loss coefficient set to: {self.recon_reg_loss.coef}")

                distance_loss_dict = self.distance_loss(manifolds_features_1, manifolds_features_2,
                                                        x2_manifold_indices)
                distance_loss_value = distance_loss_dict["distance_loss"]
                total_loss.update({"distance_loss": distance_loss_value})

                sparsity_loss_dict = self.sparsity_loss(manifolds_features_1, manifolds_features_2)
                sparsity_loss_value = sparsity_loss_dict["sparsity_loss"]
                total_loss.update({"sparsity_loss": sparsity_loss_value})

                # prepare S_hat for all the manifolds, consistency loss
                S_hat = torch.tensor([]).to(manifolds_features_1)
                for manifold_idx in range(manifolds_features_1.size(0)):
                    remaining_indices = torch.tensor([idx for idx in range(manifolds_features_1.size(0))\
                                                        if idx != manifold_idx]).to(manifolds_features_1.device)

                    logger.debug(f"manifold_idx: {manifold_idx}; remaining_indices: {remaining_indices}")

                    if self.model.dfm_aggr == "SUM":
                        aggr_12 = torch.cat((manifolds_features_1[[manifold_idx], :, :],
                                                manifolds_features_2[remaining_indices, :, :]),
                                                dim=0).sum(dim=0)
                        
                        logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                    else: # self.model.dfm_aggr == "CONCAT"
                        aggr_12 = []
                        aggr_12.append(manifolds_features_1[manifold_idx, :, :])
                        for remaining_idx in remaining_indices:
                            aggr_12.append(manifolds_features_2[remaining_idx, :, :])
                        aggr_12 = torch.hstack(aggr_12)
                        
                        logger.debug(f"aggr_12 shape: {aggr_12.size()}")
                    
                    X_s_hat = self.model.dfm_decoder(aggr_12)
                    f_12 = self.model.dfm_encoder(X_s_hat)
                    S_manifold_idx = self.model.orthogonal_manifolds[manifold_idx](f_12)
                    S_hat = torch.cat((S_hat, S_manifold_idx.unsqueeze(0)), dim=0)

                logger.debug(f"S_hat shape: {S_hat.size()}")

                consistency_loss_dict = self.consistency_loss(S_hat, manifolds_features_1)
                consistency_loss_value = consistency_loss_dict["consistency_loss"]
                total_loss.update({"consistency_loss": consistency_loss_value})

                if self.use_recon_reg_loss:
                    recon_reg_loss_dict = self.recon_reg_loss(manifolds_features_1, manifolds_features_2)
                    recon_reg_loss_value = recon_reg_loss_dict["recon_reg_loss"]
                    total_loss.update({"recon_reg_loss": recon_reg_loss_value})
        elif self.model_mode == "SVD":
            # self.__check_network_grad__()

            logger.debug(f"total_loss before __svd_linear_training_step__: {total_loss}")

            self.__svd_linear_training_step__(batch, total_loss)

            logger.debug(f"total_loss after __svd_linear_training_step__: {total_loss}")
        elif self.model_mode == "FEAT_LINEAR":
            # self.__check_network_grad__()

            logger.debug(f"{self.model_mode}, total_loss before __svd_linear_training_step__: {total_loss}")

            self.__svd_linear_training_step__(batch, total_loss)

            logger.debug(f"total_loss after __svd_linear_training_step__: {total_loss}")
            
        self.log_dict(total_loss, prog_bar=True, sync_dist=True)
        total_loss_value = sum(total_loss.values()) if len(total_loss) > 0 else None

        logger.debug(f"total_loss_value: {total_loss_value}")

        return total_loss_value

    # def __val_test_common_step__(self, batch: torch.Tensor,
    #                              step_mode: Literal["val", "test"] = "val"):

    def __compute_val_metrics_common__(self, y_hat_1, y_1, y_hat_2, y_2, total_performance):
        # compute accuracy f1, precision, recall
        accuracy_1 = binary_accuracy(F.sigmoid(y_hat_1), y_1.long())
        accuracy_2 = binary_accuracy(F.sigmoid(y_hat_2), y_2.long())
        f1_1 = binary_f1_score(F.sigmoid(y_hat_1), y_1.long())
        f1_2 = binary_f1_score(F.sigmoid(y_hat_2), y_2.long())
        precision_1 = binary_precision(F.sigmoid(y_hat_1), y_1.long())
        precision_2 = binary_precision(F.sigmoid(y_hat_2), y_2.long())
        recall_1 = binary_recall(F.sigmoid(y_hat_1), y_1.long())
        recall_2 = binary_recall(F.sigmoid(y_hat_2), y_2.long())
        accuracy = (accuracy_1 + accuracy_2) / 2
        f1 = (f1_1 + f1_2) / 2
        precision = (precision_1 + precision_2) / 2
        recall = (recall_1 + recall_2) / 2
        total_performance.update({"accuracy": accuracy, "f1": f1,
                                  "precision": precision, "recall": recall})

    def __svd_linear_validation_step__(self, batch, total_performance):
        x_pair, _, y_pair = batch  # x2_manifold_indices
        x_1, x_2 = x_pair
        y_1, y_2 = y_pair
        batch_1 = (x_1, y_1)
        batch_2 = (x_2, y_2)
        y_hat_1, y_1 = self(batch_1)
        y_hat_2, y_2 = self(batch_2)

        self.__compute_val_metrics_common__(y_hat_1, y_1, y_hat_2, y_2, total_performance)

    def validation_step(self, batch: torch.Tensor):
        total_performance = {}
        if self.model_mode == "SVDDFM":
            # self.__check_network_grad__()
            if self.current_epoch < self.optim_configs.dfm_start_epoch:
                logger.debug(f"Switching model mode from SVDDFM to SVD")

                self.model_mode = "SVD"
                self.model.dfm = False
                self.__svd_linear_validation_step__(batch, total_performance)
                self.model_mode = "SVDDFM"
                self.model.dfm = True
            else:
                y_hat_1, _, _, _, y_hat_2, _, _, _, _, y_pair = self(batch)
                y_1, y_2 = y_pair

                self.__compute_val_metrics_common__(y_hat_1, y_1, y_hat_2, y_2, total_performance)
        elif self.model_mode == "SVD_DFM":
            y_hat_1, _, _, _, y_hat_2, _, _, _, _, y_pair = self(batch)
            y_1, y_2 = y_pair

            self.__compute_val_metrics_common__(y_hat_1, y_1, y_hat_2, y_2, total_performance)
        elif self.model_mode == "SVD":
            # self.__check_network_grad__()

            self.__svd_linear_validation_step__(batch, total_performance)
        elif self.model_mode == "FEAT_LINEAR":
            # self.__check_network_grad__()

            self.__svd_linear_validation_step__(batch, total_performance)

        self.log_dict(total_performance, prog_bar=True, sync_dist=True)
    
    def on_test_epoch_start(self):
        self.test_total_accuracy = BinaryAccuracy().to(self.device)
        self.test_total_f1 = BinaryF1Score().to(self.device)
        self.test_total_precision = BinaryPrecision().to(self.device)
        self.test_total_recall = BinaryRecall().to(self.device)
        self.test_total_pr_curve = BinaryPrecisionRecallCurve().to(self.device)
        self.test_per_semantic_accuracy = {}
        self.test_per_semantic_f1 = {}
        self.test_per_semantic_precision = {}
        self.test_per_semantic_recall = {}
        self.test_per_semantic_pr_curve = {}

    def __compute_test_metrics_common__(self, y_hat_1, y_1, y_hat_2, y_2, semantic_indices,
                                        total_performance):
        semantic_indices_1, semantic_indices_2 = semantic_indices
        semidx_y_hat_dict_1 = {}
        semidx_y_dict_1 = {}
        semidx_y_hat_dict_2 = {}
        semidx_y_dict_2 = {}
        for idx, semidx in enumerate(semantic_indices_1):
            if semidx.item() not in semidx_y_hat_dict_1:
                semidx_y_hat_dict_1[semidx.item()] = y_hat_1[[idx], :].detach().clone().to(y_hat_1)
                semidx_y_dict_1[semidx.item()] = y_1[[idx], :].detach().clone().to(y_1)
            else:
                semidx_y_hat_dict_1[semidx.item()] = torch.cat([semidx_y_hat_dict_1[semidx.item()], 
                                                            y_hat_1[[idx], :].detach().clone().to(y_hat_1)],
                                                               dim=0)
                semidx_y_dict_1[semidx.item()] = torch.cat([semidx_y_dict_1[semidx.item()], 
                                                            y_1[[idx], :].detach().clone().to(y_1)],
                                                            dim=0)
            
            logger.debug(f"semidx_y_hat_dict_1[{semidx.item()}] size: {semidx_y_hat_dict_1[semidx.item()].size()}")
            logger.debug(f"semidx_y_dict_1[{semidx.item()}] size: {semidx_y_dict_1[semidx.item()].size()}")

        for idx, semidx in enumerate(semantic_indices_2):
            if semidx.item() not in semidx_y_hat_dict_2:
                semidx_y_hat_dict_2[semidx.item()] = y_hat_2[[idx], :].detach().clone().to(y_hat_2)
                semidx_y_dict_2[semidx.item()] = y_2[[idx], :].detach().clone().to(y_2)
            else:
                semidx_y_hat_dict_2[semidx.item()] = torch.cat([semidx_y_hat_dict_2[semidx.item()], 
                                                           y_hat_2[[idx], :].detach().clone().to(y_hat_2)],
                                                               dim=0)
                semidx_y_dict_2[semidx.item()] = torch.cat([semidx_y_dict_2[semidx.item()], 
                                                            y_2[[idx], :].detach().clone().to(y_2)],
                                                           dim=0)

            logger.debug(f"semidx_y_hat_dict_2[{semidx.item()}] size: {semidx_y_hat_dict_2[semidx.item()].size()}")
            logger.debug(f"semidx_y_dict_2[{semidx.item()}] size: {semidx_y_dict_2[semidx.item()].size()}")

        # compute accuracy f1, precision, recall
        accuracy_1 = binary_accuracy(F.sigmoid(y_hat_1), y_1.long())
        accuracy_2 = binary_accuracy(F.sigmoid(y_hat_2), y_2.long())
        f1_1 = binary_f1_score(F.sigmoid(y_hat_1), y_1.long())
        f1_2 = binary_f1_score(F.sigmoid(y_hat_2), y_2.long())
        precision_1 = binary_precision(F.sigmoid(y_hat_1), y_1.long())
        precision_2 = binary_precision(F.sigmoid(y_hat_2), y_2.long())
        recall_1 = binary_recall(F.sigmoid(y_hat_1), y_1.long())
        recall_2 = binary_recall(F.sigmoid(y_hat_2), y_2.long())
        accuracy = (accuracy_1 + accuracy_2) / 2
        f1 = (f1_1 + f1_2) / 2
        precision = (precision_1 + precision_2) / 2
        recall = (recall_1 + recall_2) / 2
        total_performance.update({"accuracy": accuracy, "f1": f1,
                                  "precision": precision, "recall": recall})
        self.test_total_accuracy.update(F.sigmoid(y_hat_1), y_1.long())
        self.test_total_accuracy.update(F.sigmoid(y_hat_2), y_2.long())
        self.test_total_f1.update(F.sigmoid(y_hat_1), y_1.long())
        self.test_total_f1.update(F.sigmoid(y_hat_2), y_2.long())
        self.test_total_precision.update(F.sigmoid(y_hat_1), y_1.long())
        self.test_total_precision.update(F.sigmoid(y_hat_2), y_2.long())
        self.test_total_recall.update(F.sigmoid(y_hat_1), y_1.long())
        self.test_total_recall.update(F.sigmoid(y_hat_2), y_2.long())
        self.test_total_pr_curve.update(F.sigmoid(y_hat_1), y_1.long())
        self.test_total_pr_curve.update(F.sigmoid(y_hat_2), y_2.long())

        for semidx, y_hat in semidx_y_hat_dict_1.items():
            y = semidx_y_dict_1[semidx]

            logger.debug(f"Semantic class {semidx} - y_hat_1 size: {y_hat.size()}; y_1 size: {y.size()}")

            if semidx not in self.test_per_semantic_accuracy:
                self.test_per_semantic_accuracy[semidx] = BinaryAccuracy().to(self.device)
                self.test_per_semantic_f1[semidx] = BinaryF1Score().to(self.device)
                self.test_per_semantic_precision[semidx] = BinaryPrecision().to(self.device)
                self.test_per_semantic_recall[semidx] = BinaryRecall().to(self.device)
                self.test_per_semantic_pr_curve[semidx] = BinaryPrecisionRecallCurve().to(self.device)
            self.test_per_semantic_accuracy[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_f1[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_precision[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_recall[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_pr_curve[semidx].update(F.sigmoid(y_hat), y.long())
        
        for semidx, y_hat in semidx_y_hat_dict_2.items():
            y = semidx_y_dict_2[semidx]

            logger.debug(f"Semantic class {semidx} - y_hat_2 size: {y_hat.size()}; y_2 size: {y.size()}")

            if semidx not in self.test_per_semantic_accuracy:
                self.test_per_semantic_accuracy[semidx] = BinaryAccuracy().to(self.device)
                self.test_per_semantic_f1[semidx] = BinaryF1Score().to(self.device)
                self.test_per_semantic_precision[semidx] = BinaryPrecision().to(self.device)
                self.test_per_semantic_recall[semidx] = BinaryRecall().to(self.device)
                self.test_per_semantic_pr_curve[semidx] = BinaryPrecisionRecallCurve().to(self.device)
            self.test_per_semantic_accuracy[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_f1[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_precision[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_recall[semidx].update(F.sigmoid(y_hat), y.long())
            self.test_per_semantic_pr_curve[semidx].update(F.sigmoid(y_hat), y.long())

    def __svd_linear_test_step__(self, batch, total_performance):
        x_pair, _, y_pair, semantic_indices = batch  # x2_manifold_indices
        x_1, x_2 = x_pair
        y_1, y_2 = y_pair
        batch_1 = (x_1, y_1)
        batch_2 = (x_2, y_2)
        y_hat_1, y_1 = self(batch_1)
        y_hat_2, y_2 = self(batch_2)

        self.__compute_test_metrics_common__(y_hat_1, y_1, y_hat_2, y_2, semantic_indices,
                                             total_performance)
        
    def test_step(self, batch: torch.Tensor):
        total_performance = {}
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            x_pair, x2_manifold_indices, y_pair, semantic_indices = batch
            forward_batch = (x_pair, x2_manifold_indices, y_pair)
            y_hat_1, _, _, _, y_hat_2, _, _, _, _, y_pair = self(forward_batch)
            y_1, y_2 = y_pair

            self.__compute_test_metrics_common__(y_hat_1, y_1, y_hat_2, y_2, semantic_indices,
                                                 total_performance)
        elif self.model_mode == "SVD":
            self.__svd_linear_test_step__(batch, total_performance)
        elif self.model_mode == "FEAT_LINEAR":
            self.__svd_linear_test_step__(batch, total_performance)

        self.log_dict(total_performance, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        self.log("test_total_Accuracy_epoch", self.test_total_accuracy.compute(), prog_bar=True, sync_dist=True)
        self.log("test_total_F1_epoch", self.test_total_f1.compute(), prog_bar=True, sync_dist=True)
        self.log("test_total_Precision_epoch", self.test_total_precision.compute(), prog_bar=True, sync_dist=True)
        self.log("test_total_Recall_epoch", self.test_total_recall.compute(), prog_bar=True, sync_dist=True)
        precision, recall, _ = self.test_total_pr_curve.compute()
        sorted_indices = torch.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]
        ap = torch.trapz(precision_sorted, recall_sorted)
        self.log("test_total_AP_epoch", ap, prog_bar=True, sync_dist=True)
        
        aps = []
        for semidx in self.test_per_semantic_accuracy.keys():
            self.log(f"test_semidx_{semidx}_Accuracy_epoch",
                     self.test_per_semantic_accuracy[semidx].compute(),
                     prog_bar=False, sync_dist=True)
            self.log(f"test_semidx_{semidx}_F1_epoch",
                     self.test_per_semantic_f1[semidx].compute(),
                     prog_bar=False, sync_dist=True)
            self.log(f"test_semidx_{semidx}_Precision_epoch",
                     self.test_per_semantic_precision[semidx].compute(),
                     prog_bar=False, sync_dist=True)
            self.log(f"test_semidx_{semidx}_Recall_epoch",
                     self.test_per_semantic_recall[semidx].compute(),
                     prog_bar=False, sync_dist=True)
            precision, recall, _ = self.test_per_semantic_pr_curve[semidx].compute()
            sorted_indices = torch.argsort(recall)
            recall_sorted = recall[sorted_indices]
            precision_sorted = precision[sorted_indices]
            ap = torch.trapz(precision_sorted, recall_sorted)
            aps.append(ap)
        mAP = torch.stack(aps).mean()
        self.log("test_total_mAP_epoch", mAP, prog_bar=True, sync_dist=True)

    def on_predict_epoch_start(self):
        assert self.inference_configs is not None, "inference_configs must be provided for prediction."
        assert self.inference_configs.metrics_eval is not None and \
               self.inference_configs.save_features is not None, "Both metrics_eval and save_features must be provided."
        assert isinstance(self.inference_configs.metrics_eval, bool) and \
               isinstance(self.inference_configs.save_features, bool), \
                "Both metrics_eval and save_features must be boolean."
        assert self.inference_configs.metrics_eval or self.inference_configs.save_features, \
               "At least one of metrics_eval or save_features must be set to True."
        if self.inference_configs.metrics_eval:
            self.predict_total_accuracy = BinaryAccuracy().to(self.device)
            self.predict_total_f1 = BinaryF1Score().to(self.device)
            self.predict_total_precision = BinaryPrecision().to(self.device)
            self.predict_total_recall = BinaryRecall().to(self.device)
            self.predict_total_pr_curve = BinaryPrecisionRecallCurve().to(self.device)
            self.predict_per_semantic_accuracy = {}
            self.predict_per_semantic_f1 = {}
            self.predict_per_semantic_precision = {}
            self.predict_per_semantic_recall = {}
            self.predict_per_semantic_pr_curve = {}
            self.predict_semantic_labels = []
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            if self.inference_configs.save_features:
                self.predict_feat_model_outputs = []
                self.predict_decoder_features = []
                self.predict_manifolds_features_0 = []
                self.predict_manifolds_features_1 = []
                self.predict_manifolds_decoder_features_0 = []
                self.predict_manifolds_decoder_features_1 = []
        else:
            if self.inference_configs.save_features:
                self.predict_feat_model_outputs = []

    def __compute_predict_metrics_common__(self, y_hat, y, semantic_indices):
        semidx_y_hat_dict = {}
        semidx_y_dict = {}
        for idx, semidx in enumerate(semantic_indices):
            if semidx.item() not in semidx_y_hat_dict:
                semidx_y_hat_dict[semidx.item()] = y_hat[[idx], :].detach().clone().to(y_hat)
                semidx_y_dict[semidx.item()] = y[[idx], :].detach().clone().to(y)
            else:
                semidx_y_hat_dict[semidx.item()] = torch.cat([semidx_y_hat_dict[semidx.item()], 
                                                            y_hat[[idx], :].detach().clone().to(y_hat)],
                                                               dim=0)
                semidx_y_dict[semidx.item()] = torch.cat([semidx_y_dict[semidx.item()], 
                                                            y[[idx], :].detach().clone().to(y)],
                                                            dim=0)

            logger.debug(f"semidx_y_hat_dict[{semidx.item()}] size: {semidx_y_hat_dict[semidx.item()].size()}")
            logger.debug(f"semidx_y_dict[{semidx.item()}] size: {semidx_y_dict[semidx.item()].size()}")

        # compute accuracy f1, precision, recall
        self.predict_total_accuracy.update(F.sigmoid(y_hat), y.long())
        self.predict_total_f1.update(F.sigmoid(y_hat), y.long())
        self.predict_total_precision.update(F.sigmoid(y_hat), y.long())
        self.predict_total_recall.update(F.sigmoid(y_hat), y.long())
        self.predict_total_pr_curve.update(F.sigmoid(y_hat), y.long())

        for semidx, y_hat in semidx_y_hat_dict.items():
            y = semidx_y_dict[semidx]

            logger.debug(f"Semantic class {semidx} - y_hat size: {y_hat.size()}; y size: {y.size()}")

            if semidx not in self.predict_per_semantic_accuracy:
                self.predict_per_semantic_accuracy[semidx] = BinaryAccuracy().to(self.device)
                self.predict_per_semantic_f1[semidx] = BinaryF1Score().to(self.device)
                self.predict_per_semantic_precision[semidx] = BinaryPrecision().to(self.device)
                self.predict_per_semantic_recall[semidx] = BinaryRecall().to(self.device)
                self.predict_per_semantic_pr_curve[semidx] = BinaryPrecisionRecallCurve().to(self.device)
            self.predict_per_semantic_accuracy[semidx].update(F.sigmoid(y_hat), y.long())
            self.predict_per_semantic_f1[semidx].update(F.sigmoid(y_hat), y.long())
            self.predict_per_semantic_precision[semidx].update(F.sigmoid(y_hat), y.long())
            self.predict_per_semantic_recall[semidx].update(F.sigmoid(y_hat), y.long())
            self.predict_per_semantic_pr_curve[semidx].update(F.sigmoid(y_hat), y.long())

    def predict_step(self, batch: torch.Tensor):
        if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
            if self.inference_configs.metrics_eval:
                    x, semantic_label, y = batch
                    self.predict_semantic_labels.append(semantic_label.detach().clone().to(x))
                    y_hat, _, decoder_features, manifolds_features = self.model(x)
                    self.__compute_predict_metrics_common__(y_hat, y, semantic_label)
            else:
                x = batch
                _, _, decoder_features, manifolds_features = self.model(x)
            if self.inference_configs.save_features:
                self.predict_manifolds_features_0.append(manifolds_features[0, :, :].detach().clone().to(x))
                manifold_decoder_features_0 = self.model.dfm_decoder(manifolds_features[0, :, :])
                self.predict_manifolds_decoder_features_0.append(manifold_decoder_features_0.detach().clone().to(x))
                self.predict_manifolds_features_1.append(manifolds_features[1, :, :].detach().clone().to(x))
                manifold_decoder_features_1 = self.model.dfm_decoder(manifolds_features[1, :, :])
                self.predict_manifolds_decoder_features_1.append(manifold_decoder_features_1.detach().clone().to(x))
                feat_model_outputs = self.model.feat_model(x)
                if self.model.out_feat_type == "HIDDEN":
                    features = feat_model_outputs.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
                else:  # self.model.out_feat_type == "CLS"
                    features = feat_model_outputs.pooler_output  # Use CLS token representation
                self.predict_feat_model_outputs.append(features.detach().clone().to(x))
                self.predict_decoder_features.append(decoder_features.detach().clone().to(x))
        elif self.model_mode == "SVD":
            if self.inference_configs.metrics_eval:
                x, semantic_label, y = batch
                self.predict_semantic_labels.append(semantic_label.detach().clone().to(x))
                y_hat, features, _, _ = self.model(x)
                self.__compute_predict_metrics_common__(y_hat, y, semantic_label)
            else:
                x = batch
                _, features, _, _ = self.model(x)
            if self.inference_configs.save_features:
                self.predict_feat_model_outputs.append(features.detach().clone().to(x))
        elif self.model_mode == "FEAT_LINEAR":
            if self.inference_configs.metrics_eval:
                x, semantic_label, y = batch
                self.predict_semantic_labels.append(semantic_label.detach().clone().to(x))
                y_hat = self.feat_extractor(x)
                self.__compute_predict_metrics_common__(y_hat, y, semantic_label)
            else:
                x = batch
            if self.inference_configs.save_features:
                dino_features = self.feat_extractor.dino_model(x)
                features = dino_features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
                self.predict_feat_model_outputs.append(features.detach().clone().to(x))
        elif self.model_mode == "FEAT":
            if self.inference_configs.metrics_eval:
                x, semantic_label, y = batch
            else:
                x = batch
            dino_features = self.feat_extractor(x)
            features = dino_features.last_hidden_state[:, 1:, :].mean(dim=1) # Average pooling excluding CLS token
            self.predict_feat_model_outputs.append(features.detach().clone().to(x))

    def on_predict_epoch_end(self):
        assert self.inference_configs.save_dir is not None and self.inference_configs.save_dir != "", \
            "save_dir must be provided for saving inference results."
        os.makedirs(self.inference_configs.save_dir, exist_ok=True)

        if self.inference_configs.metrics_eval:
            prediction_results = {}
            prediction_results["total_accuracy"] = self.predict_total_accuracy.compute().cpu().item()
            prediction_results["total_f1"] = self.predict_total_f1.compute().cpu().item()
            prediction_results["total_precision"] = self.predict_total_precision.compute().cpu().item()
            prediction_results["total_recall"] = self.predict_total_recall.compute().cpu().item()
            precision, recall, _ = self.predict_total_pr_curve.compute()
            sorted_indices = torch.argsort(recall)
            recall_sorted = recall[sorted_indices]
            precision_sorted = precision[sorted_indices]
            ap = torch.trapz(precision_sorted, recall_sorted)
            prediction_results["total_ap"] = ap.cpu().item()
            
            aps = []
            for semidx in self.predict_per_semantic_accuracy.keys():
                prediction_results[f"{semidx}_Accuracy"] = \
                    self.predict_per_semantic_accuracy[semidx].compute().cpu().item()
                prediction_results[f"{semidx}_F1"] = \
                    self.predict_per_semantic_f1[semidx].compute().cpu().item()
                prediction_results[f"{semidx}_Precision"] = \
                    self.predict_per_semantic_precision[semidx].compute().cpu().item()
                prediction_results[f"{semidx}_Recall"] = \
                    self.predict_per_semantic_recall[semidx].compute().cpu().item()
                precision, recall, _ = self.predict_per_semantic_pr_curve[semidx].compute()
                sorted_indices = torch.argsort(recall)
                recall_sorted = recall[sorted_indices]
                precision_sorted = precision[sorted_indices]
                ap = torch.trapz(precision_sorted, recall_sorted)
                prediction_results[f"{semidx}_AP"] = ap.cpu().item()
                aps.append(ap)
            mAP = torch.stack(aps).mean()
            prediction_results["mAP"] = mAP.cpu().item()
            
            with open(os.path.join(self.inference_configs.save_dir, "prediction_results.json"), "w") as predict_file:
                json.dump(prediction_results, predict_file, indent=4, ensure_ascii=False)

        # concatenate all the collected tensors and save to disk
        if self.inference_configs.save_features:
            if self.inference_configs.metrics_eval:
                torch.save(torch.cat(self.predict_semantic_labels, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_semantic_labels.pt"))
            if self.model_mode == "SVDDFM" or self.model_mode == "SVD_DFM":
                torch.save(torch.cat(self.predict_feat_model_outputs, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_feat_model_outputs.pt"))
                torch.save(torch.cat(self.predict_decoder_features, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_decoder_features.pt"))
                torch.save(torch.cat(self.predict_manifolds_features_0, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_manifolds_features_0.pt"))
                torch.save(torch.cat(self.predict_manifolds_decoder_features_0, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_manifolds_decoder_features_0.pt"))
                torch.save(torch.cat(self.predict_manifolds_features_1, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_manifolds_features_1.pt"))
                torch.save(torch.cat(self.predict_manifolds_decoder_features_1, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_manifolds_decoder_features_1.pt"))
            else:
                torch.save(torch.cat(self.predict_feat_model_outputs, dim=0).cpu(),
                        os.path.join(self.inference_configs.save_dir, "predict_feat_model_outputs.pt"))

    def configure_optimizers(self):
        if self.model_mode == "FEAT_LINEAR":
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                                self.feat_extractor.parameters()),
                                         lr=self.learning_rate)
        elif self.model_mode == "SVD" or self.model_mode == "SVDDFM":
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                                self.model.parameters()),
                                         lr=self.learning_rate)

        return optimizer
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.optim_configs.scheduler_T_max)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch"  # or "step"
        #     }
        # }


def cli_main():
    cli = LightningCLI(DFDDFMTrainer, DFDDFMTrainDataModule, seed_everything_default=88)


if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    # logging.basicConfig(filename='./logs/main.log', level=logging.INFO)
    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # configure logging on module level, redirect to file
    # logger = logging.getLogger("pytorch_lightning.core")
    # logger.addHandler(logging.FileHandler("./logs/ltn_core.log"))

    cli_main()