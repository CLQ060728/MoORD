import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Literal
from Model.DFDDFM import SVDResidualLinear, ClipSVDDFM, Dinov2SVDDFM, Dinov3SVDDFM
from Loss.BaseLoss import BaseLoss
import logging, json

logger = logging.getLogger(__name__)


class DFDLoss(BaseLoss):
    """
    A class representing a Deep Fake Detection (DFD) classification loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="dfd",
                 reduction="mean",
                 check_nan=False
    ):
        super(DFDLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
        self.loss_function = F.binary_cross_entropy

    def __call__(self,
                 logits: Tensor,
                 targets: Tensor
    ):
        """
        Compute the DFD loss for a given DFDDFM model.

        Args:
            logits (Tensor): The predicted logits from the model.
            targets (Tensor): The ground truth labels.

        Returns:
            dict: Dictionary containing the DFD loss value.
        """
        loss = self.loss_function(F.sigmoid(logits), targets, reduction='none')

        return self.return_loss(self.name, loss)


class SVDLoss(BaseLoss):
    """
    A class representing singular value decomposition (SVD)-related losses function.

    Args:
        coef (float): The coefficient to multiply the loss by.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="svd",
                 reduction="none",
                 check_nan=False
    ):
        super(SVDLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_losses_orth_keepsv"
    
    def __call__(self,
                 dfddfm_model
    ):
        """
        Compute the orthogonal and keepsv losses for a given DFDDFM model.

        Args:
            dfddfm_model: The DFDDFM model to compute the losses for.
        Returns:
            dict: Dictionary containing the orthogonal and keepsv losses values.
        """
        reg_term = 0.0
        num_reg = 0

        if isinstance(dfddfm_model, ClipSVDDFM):
            logger.debug("Computing SVD losses for ClipSVDDFM model...")

            for name, module in dfddfm_model.feat_model.named_children():
                if name == "encoder":
                    for _, module1 in module.named_children():
                        for _, clip_encoder_layer in enumerate(module1):
                            for name2, module2 in clip_encoder_layer.named_children():
                                if 'self_attn' == name2:
                                    # compute orthogonal and keepsv losses
                                    for _, sub_module in module2.named_children():
                                        if isinstance(sub_module, SVDResidualLinear):
                                            logger.debug(f"Found SVDResidualLinear layer: {sub_module}")

                                            reg_term += self.__compute_orthogonal_loss__(sub_module)

                                            logger.debug(f"orthogonal reg_term: {reg_term}")

                                            reg_term += self.__compute_keepsv_loss__(sub_module)
                                            
                                            logger.debug(f"keepsv reg_term: {reg_term}")

                                            num_reg += 1
        elif isinstance(dfddfm_model, Dinov2SVDDFM):
            logger.debug("Computing SVD losses for Dinov2SVDDFM model...")

            for name, module in dfddfm_model.feat_model.named_children():
                if name == "encoder":
                    for _, module1 in module.named_children():
                        for _, Dinov2Layer in enumerate(module1):
                            for name2, module2 in Dinov2Layer.named_children():
                                if 'attention' == name2:
                                    for name3, module3 in module2.named_children():
                                        if 'attention' == name3 or 'output' == name3:
                                            # compute orthogonal and keepsv losses
                                            for _, sub_module in module3.named_children():
                                                if isinstance(sub_module, SVDResidualLinear):
                                                    logger.debug(f"Found SVDResidualLinear layer: {sub_module}")

                                                    reg_term += self.__compute_orthogonal_loss__(sub_module)

                                                    logger.debug(f"orthogonal reg_term: {reg_term}")

                                                    reg_term += self.__compute_keepsv_loss__(sub_module)

                                                    logger.debug(f"keepsv reg_term: {reg_term}")

                                                    num_reg += 1
        elif isinstance(dfddfm_model, Dinov3SVDDFM):
            logger.debug("Computing SVD losses for Dinov3SVDDFM model...")

            for name, module in dfddfm_model.feat_model.named_children():
                if name == "layer":
                    for _, DINOv3ViTLayer in enumerate(module):
                        for name1, module1 in DINOv3ViTLayer.named_children():
                            if 'attention' == name1:
                                # compute orthogonal and keepsv losses
                                for _, sub_module in module1.named_children():
                                    if isinstance(sub_module, SVDResidualLinear):
                                        logger.debug(f"Found SVDResidualLinear layer: {sub_module}")

                                        reg_term += self.__compute_orthogonal_loss__(sub_module)

                                        logger.debug(f"orthogonal reg_term: {reg_term}")

                                        reg_term += self.__compute_keepsv_loss__(sub_module)

                                        logger.debug(f"keepsv reg_term: {reg_term}")

                                        num_reg += 1
        
        loss = reg_term / num_reg
        # loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        logger.debug(f"number of svd loss layers: {num_reg}; final loss: {loss}")

        return self.return_loss(self.name, loss)

    def __compute_orthogonal_loss__(self, svd_residual_layer: SVDResidualLinear):
        logger.debug(f"inside __compute_orthogonal_loss__")

        if svd_residual_layer.S_residual is not None:
            logger.debug(f"S_residual is not None, computing orthogonal loss...")
            # According to the properties of orthogonal matrices: A^TA = I
            UUT = torch.cat((svd_residual_layer.U_r, svd_residual_layer.U_residual), dim=1) @ torch.cat((svd_residual_layer.U_r, svd_residual_layer.U_residual), dim=1).t()
            VVT = torch.cat((svd_residual_layer.V_r, svd_residual_layer.V_residual), dim=0) @ torch.cat((svd_residual_layer.V_r, svd_residual_layer.V_residual), dim=0).t()
            # print(self.U_r.size(), self.U_residual.size())  # torch.Size([1024, 1023]) torch.Size([1024, 1])
            # print(self.V_r.size(), self.V_residual.size())  # torch.Size([1023, 1024]) torch.Size([1, 1024])
            # UUT = self.U_residual @ self.U_residual.t()
            # VVT = self.V_residual @ self.V_residual.t()
            
            # Construct an identity matrix
            UUT_identity = torch.eye(UUT.size(0)).to(UUT)
            VVT_identity = torch.eye(VVT.size(0)).to(VVT)

            # Using frobenius norm to compute loss
            loss = 0.5 * torch.norm(UUT - UUT_identity, p='fro') + 0.5 * torch.norm(VVT - VVT_identity, p='fro')
        else:
            loss = torch.tensor(0.0)

        return loss

    def __compute_keepsv_loss__(self, svd_residual_layer: SVDResidualLinear):
        logger.debug(f"inside __compute_keepsv_loss__")

        if (svd_residual_layer.S_residual is not None) and (svd_residual_layer.weight_original_fnorm is not None):
            logger.debug(f"S_residual and weight_original_fnorm are not None, computing keepsv loss...")

            # Total current weight is the fixed main weight plus the residual
            weight_current = svd_residual_layer.weight_r + svd_residual_layer.U_residual @ torch.diag(svd_residual_layer.S_residual) @ svd_residual_layer.V_residual
            # Frobenius norm of current weight
            weight_current_fnorm = torch.norm(weight_current, p='fro')

            loss = torch.abs(weight_current_fnorm ** 2 - svd_residual_layer.weight_original_fnorm ** 2)
        else:
            loss = torch.tensor(0.0)
        
        return loss


class ReconstructionLoss(BaseLoss):
    """
    A class representing a reconstruction loss function.

    Args:
        loss_type (Literal["l1", "l2", "smooth_l1"]): The type of loss function to use.
        coef (float): The coefficient to multiply the loss by.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["l1", "l2", "smooth_l1"]): The type of loss function being used.
        loss_fn (function): The loss function being used.
        name (str): The name of the loss function.
    """

    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        coef: float = 1.0,
        name="reconstruction",
        reduction="mean",
        check_nan=False
    ):
        super(ReconstructionLoss, self).__init__(coef, check_nan, reduction)
        self.loss_type = loss_type
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Unknown loss type: {loss_type}")
        self.name = f"{name}_loss_{self.loss_type}"

    def __call__(
        self,
        decoder_features: Tensor,
        encoder_features: Tensor
    ):
        """
        Compute the loss between the predicted and ground truth values.

        Args:
            decoder_features (Tensor): The predicted values.
            encoder_features (Tensor): The ground truth values.
        Returns:
            dict: Dictionary containing the reconstruction loss value.
        """
        encoder_features, decoder_features = encoder_features.squeeze(), decoder_features.squeeze()
        loss = self.loss_fn(decoder_features, encoder_features, reduction="none")
        
        return self.return_loss(self.name, loss)


class DistanceLoss(BaseLoss):
    """
    A class representing a distance loss function for the disentangled manifolds.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="distance",
                 reduction="mean",
                 check_nan=False):
        super(DistanceLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(self,
                 manifolds_features1: Tensor,
                 manifolds_features2: Tensor,
                 manifolds_indices2: Tensor
    ):
        """
        Compute the distance loss for disentangled manifolds.

        Args:
            manifolds_features1 (Tensor): The predicted single disentangled manifold features.
            manifolds_features2 (Tensor): The original single disentangled manifold features.
            manifolds_indices2 (Tensor): The indices of the disentangled manifolds to disentangle.
        Returns:
            dict: Dictionary containing the distance loss value.
        """
        dis_manifolds_features1 = F.normalize(manifolds_features1, p=2, dim=-1)
        dis_manifolds_features2 = F.normalize(manifolds_features2, p=2, dim=-1)
        sim_matrix = F.cosine_similarity(dis_manifolds_features1, dis_manifolds_features2, dim=-1).T
        sim_matrix = F.softmax(sim_matrix, dim=-1)

        logger.debug(f"sim_matrix shape: {sim_matrix.size()}; min value: {sim_matrix.min()}; max value: {sim_matrix.max()};")

        sim_mask = torch.zeros_like(sim_matrix)
        for manifolds_idx in torch.arange(manifolds_indices2.size(0)):
            sim_mask[manifolds_idx, manifolds_indices2[manifolds_idx]] = 1.0
        
        loss = sim_matrix * sim_mask

        logger.debug(f"sim_mask shape: {sim_mask.size()};")
        logger.debug(f"loss shape: {loss.size()}; min value: {loss.min()}; max value: {loss.max()};")

        return self.return_loss(self.name, loss)


class SparsityLoss(BaseLoss):
    """
    A class representing a sparsity loss function for the disentangled manifolds.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="sparsity",
                 reduction="mean",
                 check_nan=False):
        super(SparsityLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(self,
                 manifolds_features1: Tensor,
                 manifolds_features2: Tensor
    ):
        """
        Compute the sparsity loss for disentangled manifolds.

        Args:
            manifolds_features1 (Tensor): The predicted single disentangled manifold features.
            manifolds_features2 (Tensor): The original single disentangled manifold features.
        Returns:
            dict: Dictionary containing the sparsity loss value.
        """
        spar_manifolds_features1 = torch.zeros(1, manifolds_features1.size(1),
                                               manifolds_features1.size(2)).to(manifolds_features1.device)
        spar_manifolds_features2 = torch.zeros(1, manifolds_features2.size(1),
                                               manifolds_features2.size(2)).to(manifolds_features2.device)

        logger.debug(f"spar_manifolds_features1 shape: {spar_manifolds_features1.size()}")
        logger.debug(f"spar_manifolds_features2 shape: {spar_manifolds_features2.size()}")

        for manifold_idx in range(manifolds_features1.size(0)):
            remaining_indices = torch.tensor([idx for idx in range(manifolds_features1.size(0)) if idx != manifold_idx]).to(manifolds_features1.device)

            logger.debug(f"manifolds_features1 shape: {manifolds_features1.size()}; manifold_idx: {manifold_idx}; remaining_indices: {remaining_indices}")

            sum_manifolds_features1 = manifolds_features1[remaining_indices, :, :].sum(dim=0, keepdim=True)
            sum_manifolds_features2 = manifolds_features2[remaining_indices, :, :].sum(dim=0, keepdim=True)

            logger.debug(f"sum_manifolds_features1 shape: {sum_manifolds_features1.size()}; sum_manifolds_features2 shape: {sum_manifolds_features2.size()}")

            abs_manifolds_features1 = torch.abs(manifolds_features1[[manifold_idx], :, :] * sum_manifolds_features1)
            abs_manifolds_features2 = torch.abs(manifolds_features2[[manifold_idx], :, :] * sum_manifolds_features2)

            logger.debug(f"abs_manifolds_features1 shape: {abs_manifolds_features1.size()}; abs_manifolds_features2 shape: {abs_manifolds_features2.size()}")

            spar_manifolds_features1 += abs_manifolds_features1
            spar_manifolds_features2 += abs_manifolds_features2

        loss = spar_manifolds_features1 + spar_manifolds_features2

        logger.debug(f"loss shape: {loss.size()}; min value: {loss.min()}; max value: {loss.max()};")

        return self.return_loss(self.name, loss)


class ConsistencyLoss(BaseLoss):
    """
    A class representing a consistency loss function.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="consistency",
                 reduction="mean",
                 check_nan=False):
        super(ConsistencyLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"
        self.loss_fn = F.mse_loss

    def __call__(self,
                 S_hat: Tensor,
                 S_original: Tensor
    ):
        """
        Compute the consistency loss for a given SVDResidualLinear layer.

        Args:
            S_hat (Tensor): The predicted disentangled manifolds features.
            S_original (Tensor): The original disentangled manifolds features.
        Returns:
            dict: Dictionary containing the consistency loss value.
        """
        loss = self.loss_fn(S_hat, S_original, reduction='none')

        return self.return_loss(self.name, loss)


class ReconRegLoss(BaseLoss):
    """
    A class representing a reconstruction regularization loss function for the disentangled manifolds.

    Args:
        coef (float): The coefficient for the loss.
        name (str): The name of the loss function.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        name (str): The name of the loss function.
    """

    def __init__(self,
                 coef: float = 1.0,
                 name="recon_reg",
                 reduction="mean",
                 check_nan=False):
        super(ReconRegLoss, self).__init__(coef, check_nan, reduction)
        self.name = f"{name}_loss"

    def __call__(self,
                 manifolds_features1: Tensor,
                 manifolds_features2: Tensor
    ):
        """
        Compute the reconstruction regularization loss for disentangled manifolds.

        Args:
            manifolds_features1 (Tensor): The disentangled manifolds features for the first input.
            manifolds_features2 (Tensor): The disentangled manifolds features for the second input.
        Returns:
            dict: Dictionary containing the reconstruction regularization loss value.
        """
        dis_manifolds_features1 = F.normalize(manifolds_features1, p=2, dim=-1)
        dis_manifolds_features2 = F.normalize(manifolds_features2, p=2, dim=-1)
        dis_matrix = 1 - F.cosine_similarity(dis_manifolds_features1, dis_manifolds_features2, dim=-1).T
        dis_matrix = F.softmax(dis_matrix, dim=-1)

        logger.debug(f"Inside {self.name} function.")
        logger.debug(f"dis_matrix shape: {dis_matrix.size()}; min value: {dis_matrix.min()}; max value: {dis_matrix.max()};")

        loss = (dis_matrix.mean(dim=0, keepdim=True) - 1 / dis_matrix.size(1))

        logger.debug(f"loss shape: {loss.size()}; loss: {loss}")

        loss = torch.pow(loss, 2).sum(dim=1)

        logger.debug(f"loss shape: {loss.size()}; loss: {loss}")

        return self.return_loss(self.name, loss)
