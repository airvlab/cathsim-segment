import torch
import torch.nn as nn


class BSplineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.degree = 3
        self.bspline = BSpline(self.degree)
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, pred_seq, true_seq, true_masks):
        pred_knots = pred_seq[:, :, 0]
        pred_coeffs = pred_seq[:, :, 1:3]
        true_knots = true_seq[:, :, 0]
        true_coeffs = true_seq[:, :, 1:3]

        sampled_true, sampled_mask = self.bspline(
            true_coeffs, true_knots, true_masks, num_samples=20, batched=True, max_len=20
        )
        sampled_pred, sampled_mask = self.bspline(
            pred_coeffs, pred_knots, true_masks, num_samples=20, batched=True, max_len=20
        )
        # print(sampled_true.shape)
        # sampled_true = sampled_true[0][sampled_mask[0] == 1].cpu().numpy() * 1024
        # print(sampled_true.shape)
        # plt.imshow(np.ones((1024, 1024)), cmap="gray")
        # print(sampled_true)
        # plt.plot(sampled_true[:, 0], sampled_true[:, 1], "ro")
        # plt.show()
        # exit()

        loss = (self.loss(sampled_pred, sampled_true) * sampled_mask.unsqueeze(-1)).sum() / sampled_mask.sum()
        return loss


class BSpline(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(
        self, coefficients, knots, masks=None, num_samples=None, delta=None, weights=None, batched=False, max_len=100
    ):
        if not batched:
            return self._interpolate(coefficients, knots, num_samples=num_samples, delta=delta), None
        return self._interpolate_batch(
            coefficients, knots, masks, num_samples=num_samples, delta=delta, max_output_len=max_len
        )

    def _interpolate(self, coefficients, knots, num_samples=None, delta=None):
        knots = torch.cat([torch.zeros(self.degree + 1, device=knots.device), knots])
        t_max = knots[-1]
        num_samples = int(t_max / delta) + 1 if delta else num_samples
        t = torch.linspace(0, 1, num_samples, device=knots.device)

        n = coefficients.shape[0]  # Number of control points
        d = coefficients.shape[1]  # Dimensionality of control points

        if self.degree < 1:
            raise ValueError("Degree must be at least 1 (linear).")
        if self.degree > (n - 1):
            raise ValueError("Degree must be less than or equal to point count - 1.")

        weights = torch.ones(n, device=coefficients.device)

        if knots.shape[0] != n + self.degree + 1:
            raise ValueError("Bad knot vector length.")

        # Remap t to the domain of the spline
        domain = [self.degree, knots.shape[0] - 1 - self.degree]
        low, high = knots[domain[0]], knots[domain[1]]
        t = t * (high - low) + low

        if (t < low).any() or (t > high).any():
            raise ValueError("Out of bounds")

        # Find the correct spline segment s
        s = torch.searchsorted(knots[domain[0] : domain[1]], t, right=True) + domain[0] - 1

        # Initialize the result tensor
        result = torch.zeros((t.shape[0], d), device=t.device)

        # Convert control points to homogeneous coordinates
        v = torch.cat((coefficients * weights.unsqueeze(-1), weights.unsqueeze(-1)), dim=-1)

        # Perform de Boor recursion to evaluate the spline
        for idx in range(t.shape[0]):
            t_val = t[idx]
            s_val = int(s[idx])

            # Make a copy of v for this specific t value
            v_temp = v.clone()

            for l in range(1, self.degree + 2):
                for i in range(s_val, s_val - self.degree - 1 + l, -1):
                    denom = knots[i + self.degree + 1 - l] - knots[i]
                    if denom != 0:
                        alpha = (t_val - knots[i]) / denom
                    else:
                        alpha = 0

                    # Avoid in-place operation by creating new tensors instead of assigning in-place
                    v_temp[i] = (1 - alpha) * v_temp[i - 1].clone() + alpha * v_temp[i].clone()

            # Convert back to Cartesian coordinates for the result
            result[idx] = v_temp[s_val, :-1] / v_temp[s_val, -1]

        return result

    def _interpolate_batch(self, coefficients, knots, masks, num_samples=None, delta=None, max_output_len=100):
        batch_size = coefficients.shape[0]
        result = torch.zeros((batch_size, max_output_len, coefficients.shape[2]), device=coefficients.device)
        result_masks = torch.zeros((batch_size, max_output_len), device=coefficients.device)

        for batch_idx in range(batch_size):
            try:
                idx_result = self._interpolate(
                    coefficients=coefficients[batch_idx][masks[batch_idx] == 1],
                    knots=knots[batch_idx][masks[batch_idx] == 1],
                    num_samples=num_samples,
                    delta=delta,
                )
            except ValueError:
                idx_result = torch.ones((max_output_len, coefficients.shape[2]), device=coefficients.device) * 100

            result[batch_idx][: idx_result.shape[0]] = idx_result
            result_masks[batch_idx][: idx_result.shape[0]] = 1

        return result, result_masks
