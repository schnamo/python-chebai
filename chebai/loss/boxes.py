import torch

class BoxLoss(torch.nn.Module):
    def __init__(
        self, base_loss: torch.nn.Module = None
    ):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, input, target, **kwargs):
        b = input["boxes"]
        points = input["embedded_points"]
        target = target.float().unsqueeze(-1)
        left_borders, lind = torch.min(b, dim=-1)
        right_borders, rind = torch.max(b, dim=-1)
        width = (right_borders - left_borders) / 2

        # We want some safety margins around boxes. (False) positives should be drawn
        # further into the box, whilst (false) negatives should be pushed further outside.
        # Therefore, we use different borders for (false) positives and negatives.
        r_fp = right_borders + 0.1 * width
        r_fn = right_borders - 0.1 * width
        l_fp = left_borders - 0.1 * width
        l_fn = left_borders + 0.1 * width
        inside_fp = (l_fp < points) * (points < r_fp)
        inside_fn = (l_fn < points) * (points < r_fn)

        # False positive and negatives, w.r.t. the adapted box borders
        fn_per_dim = ~inside_fn * target
        fp_per_dim = inside_fp * (1 - target)

        # We also want to penalise wrong memberships in multiple dimensions. This
        # is important, because a false positive in a single dimension is not wrong,
        # if at least one dimension is true negative.
        false_per_dim = fn_per_dim + fp_per_dim
        number_of_false_dims = torch.sum(false_per_dim, dim=-1, keepdim=True)


        # We calculate the gradient for left and right border simultaneously, but we only need the one
        # closest to the point. Therefore, we create a filter for that.
        dl = torch.abs(left_borders - points)
        dr = torch.abs(right_borders - points)
        closer_to_l_than_r = dl < dr

        # The scaling factor encodes the conjunction of whether the respective dimension is false and whether the respective
        # border is the closest to the point.
        r_scale_fp = number_of_false_dims * torch.rand_like(fp_per_dim) * (fp_per_dim * ~closer_to_l_than_r)
        l_scale_fp = number_of_false_dims * torch.rand_like(fp_per_dim) * (fp_per_dim * closer_to_l_than_r)
        r_scale_fn = number_of_false_dims * torch.rand_like(fn_per_dim) * (fn_per_dim * ~closer_to_l_than_r)
        l_scale_fn = number_of_false_dims * torch.rand_like(fn_per_dim) * (fn_per_dim * closer_to_l_than_r)

        # The loss for a border is then the mean of the scaled vector between the points for which the model would
        # produce a wrong prediction and the closest border of the box
        r_loss = torch.mean(torch.sum(torch.abs(r_scale_fp * (r_fp - points)), dim=-1) + torch.sum(
            torch.abs(r_scale_fn * (r_fn - points)), dim=-1))
        l_loss = torch.mean(torch.sum(torch.abs(l_scale_fp * (l_fp - points)), dim=-1) + torch.sum(
            torch.abs(l_scale_fn * (l_fn - points)), dim=-1))
        return l_loss + r_loss

    def _calculate_implication_loss(self, l, r):
        capped_difference = torch.relu(l - r)
        return torch.mean(
            torch.sum(
                (torch.softmax(capped_difference, dim=-1) * capped_difference), dim=-1
            ),
            dim=0,
        )
