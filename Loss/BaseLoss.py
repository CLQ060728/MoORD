import torch
from torch import nn, Tensor
import logging

logger = logging.getLogger(__name__)


class BaseLoss(nn.Module):
    """
    Base class for defining custom loss functions.

    Args:
        coef (float): Coefficient to scale the loss by.
        check_nan (bool): Whether to check if the loss is NaN.
        reduction (str): Type of reduction to apply to the loss. Can be "mean" or "none".

    Methods:
        __call__(self, *args, name: str, **kwargs): Computes the loss.
        set_coef(self, coef: float): Sets the coefficient to scale the loss by.
        return_loss(self, name: str, loss: Tensor): Returns the loss scaled by the coefficient.
    """

    def __init__(self,
                 coef: float = 1.0,
                 check_nan: bool = False,
                 reduction="mean"):
        super(BaseLoss, self).__init__()
        self.coef = coef
        self.check_nan = check_nan
        assert reduction in ["mean", "none"]
        self.reduction = reduction

    def __call__(self, *args, name: str, **kwargs):
        """
        Computes the loss.

        Args:
            *args: Variable length argument list.
            name (str): Name of the loss.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def set_coef(self, coef: float):
        """
        Sets the coefficient to scale the loss by.

        Args:
            coef (float): Coefficient to scale the loss by.
        """
        logger.debug(f"loss coefficient before setting: {self.coef}")
        self.coef = coef
        logger.debug(f"Set loss coefficient to {self.coef}")

    def return_loss(self, name: str, loss: Tensor):
        """
        Returns the loss scaled by the coefficient.

        Args:
            name (str): Name of the loss.
            loss (Tensor): Loss tensor.

        Returns:
            dict: Dictionary containing the scaled loss.
        """
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            loss = loss
        else:
            raise NotImplementedError()
        if self.check_nan:
            if torch.isnan(loss):
                raise ValueError(f"Loss {name} is NaN.")
            
        return {name: loss * self.coef}

