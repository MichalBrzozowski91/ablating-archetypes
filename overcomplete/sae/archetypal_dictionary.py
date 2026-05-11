"""
Module dedicated to everything around the Dictionary Layer of SAE.
"""

import torch
import torch.nn.functional as F
from torch import nn


class RelaxedArchetypalDictionary(nn.Module):
    """
    Dictionary used for Relaxed Archetypal SAE (RA-SAE).

    Constructs a dictionary where each atom is a convex combination of data
    points from C, with a small relaxation term constrained by delta.

    For more details, see the paper:
    "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction
    in Large Vision Models" by Fel et al. (2025), https://arxiv.org/abs/2502.12892

    Parameters
    ----------
    in_dimensions : int
        Dimensionality of the input data (e.g number of channels).
    nb_concepts : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > in_dimensions.
    points : tensors
        Real data points (or point in the convex hull) used to find the candidates archetypes.
    delta : float, optional
        Constraint on the relaxation term, by default 1.0.
    use_multiplier : bool, optional
        Whether to train a positive scalar to multiply the dictionary after convex combination,
        making the dictionary in the conical hull (and not convex hull) of the data points,
        by default True.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
    """

    def __init__(
        self, in_dimensions, nb_concepts,
        points, delta=1.0, use_multiplier=True,
        device="cpu",
        projection: bool = True,
    ):
        super().__init__()
        self.in_dimensions = in_dimensions
        self.nb_concepts = nb_concepts
        self.device = device
        self.delta = delta

        # follow the eq.4 of the paper
        self.register_buffer("C", points)
        self.nb_candidates = self.C.shape[0]
        if projection:
            self.W = nn.Parameter(torch.eye(nb_concepts, self.nb_candidates, device=device))
        else:
            self.W = torch.eye(nb_concepts, self.nb_candidates, device=device)
        if projection:
            self.Relax = nn.Parameter(torch.zeros(nb_concepts, self.in_dimensions, device=device))
        else:
            self.Relax = nn.Parameter(self.W @ self.C)

        # init multiplier
        if use_multiplier:
            self.multiplier = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        else:
            # by default, the multiplier will be exp(0) = 1 and not trainable
            self.register_buffer("multiplier", torch.tensor(0.0, device=device))

        self._fused_dictionary = None
        self.projection = projection

    def forward(self, z):
        """
        Reconstruct input data from latent representation.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, dimensions).
        """
        dictionary = self.get_dictionary()
        x_hat = torch.matmul(z, dictionary)
        return x_hat

    def get_dictionary(self):
        """
        Get the dictionary.

        Returns
        -------
        torch.Tensor
            The dictionary tensor of shape (nb_components, dimensions).
        """
        if not self.projection:
            return self.Relax
        if self.training:
            # we are in .train() mode, compute the dictionary on the fly
            if self.projection:
                with torch.no_grad():
                    # ensure W remains row-stochastic (positive and row sum to one)
                    W = torch.relu(self.W)
                    W /= (W.sum(dim=-1, keepdim=True) + 1e-8)
                    self.W.data = W

                    # enforce the norm constraint on Lambda to limit deviation from conv(C)
                    norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)  # norm per row
                    scaling_factor = torch.clamp(self.delta / norm_Lambda, max=1)  # safe scaling factor
                    self.Relax.data *= scaling_factor  # scale Lambda to satisfy ||Lambda|| < delta

                # compute the dictionary as a convex combination plus relaxation
                D = self.W @ self.C + self.Relax
                return D * torch.exp(self.multiplier)
        else:
            # we are in .eval() mode, return the fused dictionary
            assert self._fused_dictionary is not None, "Dictionary is not initialized."
            return self._fused_dictionary

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to fuse W, C, Relax and multiplier into a single dictionary.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if not mode:
            # we are in .eval() mode, fuse the dictionary
            self._fused_dictionary = self.get_dictionary()
        super().train(mode)


class SoftmaxArchetypalDictionary(nn.Module):
    """
    Fully differentiable archetypal dictionary using softmax weights.

    Dictionary atoms are convex combinations of data points C, with weights
    parameterized via softmax — no projected gradient, no in-place .data
    mutation, no no_grad blocks. A learnable multiplier allows atoms to scale
    beyond the convex hull (conical hull), which is important for reconstruction.

    Parameters
    ----------
    in_dimensions : int
        Dimensionality of the input data.
    nb_concepts : int
        Number of dictionary atoms.
    points : torch.Tensor
        Data points used to define archetypes, shape (nb_candidates, in_dimensions).
    use_multiplier : bool, optional
        Whether to train a positive scalar multiplier, by default True.
    device : str, optional
        Device to run the model on, by default 'cpu'.
    """

    def __init__(self, in_dimensions, nb_concepts, points,
                 use_multiplier=True, device="cpu"):
        super().__init__()
        self.in_dimensions = in_dimensions
        self.nb_concepts = nb_concepts
        self.device = device

        self.register_buffer("C", points)
        # W is unconstrained; softmax is applied in get_dictionary()
        self.W = nn.Parameter(
            torch.eye(nb_concepts, points.shape[0], device=device)
        )

        if use_multiplier:
            self.multiplier = nn.Parameter(
                torch.tensor(0.0, device=device), requires_grad=True
            )
        else:
            self.register_buffer("multiplier", torch.tensor(0.0, device=device))

        self._fused_dictionary = None

    def get_dictionary(self):
        if self.training:
            W_soft = F.softmax(self.W, dim=-1)          # rows sum to 1, all positive
            return (W_soft @ self.C) * torch.exp(self.multiplier)
        else:
            assert self._fused_dictionary is not None, "Dictionary is not initialized."
            return self._fused_dictionary

    def forward(self, z):
        return torch.matmul(z, self.get_dictionary())

    def train(self, mode=True):
        if not mode:
            self._fused_dictionary = self.get_dictionary()
        super().train(mode)
