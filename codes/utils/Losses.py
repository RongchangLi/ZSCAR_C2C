import torch
import numpy as np
# loss_hsic_f += self.hsic_factor * self.hsic_loss(feat_unbias, feat_bias1.detach(), unbiased=True)
def hsic_loss(self, input1, input2, unbiased=False):
    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = self._kernel(input1, sigma_x)
    kernel_YY = self._kernel(input2, sigma_y)

    if unbiased:
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
        tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic if self.alternative else hsic / (N * (N - 3))
    else:
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss
