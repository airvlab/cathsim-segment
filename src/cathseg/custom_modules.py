import torch
import torch.nn as nn


class BSpline(nn.Module):
    def __init__(self, degree=3):
        super(BSpline, self).__init__()
        self.degree = degree

    def bspline_basis(self, t, knots, degree):
        n_knots = knots.shape[0] - degree - 1
        B = torch.zeros((t.shape[0], n_knots), device=t.device)
        for i in range(n_knots):
            B[:, i] = ((knots[i] <= t) & (t < knots[i + 1])).float()
        for d in range(1, degree + 1):
            for i in range(n_knots - d):
                denom1 = knots[i + d] - knots[i]
                denom2 = knots[i + d + 1] - knots[i + 1]
                term1 = ((t - knots[i]) / denom1) * B[:, i] if denom1 != 0 else 0
                term2 = ((knots[i + d + 1] - t) / denom2) * B[:, i + 1] if denom2 != 0 else 0
                B[:, i] = term1 + term2
        return B

    def forward(self, t, c, delta):
        t_min, t_max = 0, t[-1].item()
        padded_t = torch.cat([torch.zeros(self.degree + 1, device=t.device), t])
        sample_points = torch.arange(t_min, t_max, delta, device=t.device)
        B = self.bspline_basis(sample_points, padded_t, self.degree)
        spline_values = torch.matmul(B, c.T)
        return sample_points, spline_values
