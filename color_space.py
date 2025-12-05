import torch.nn.functional as F
import torch
import torch.nn as nn


class Mapping(nn.Module):
    def __init__(self, M=16, a_min=0.5, a_max=5.0):
        super(Mapping, self).__init__()
        self.M = M
        self.a_min = a_min
        self.a_max = a_max
        self.u_i = nn.Parameter(torch.zeros(M))
        # self.u_i = nn.Parameter(torch.rand(M) * 20 - 10)
        self.this_u_i = 0

        # Pre-calculate segment boundaries
        self.register_buffer('bounds', torch.linspace(0, 1, M + 1))

    def _get_slopes(self, ui):
        """Get slopes for each segment."""
        return self.a_min + (self.a_max - self.a_min) * torch.sigmoid(ui)

    def forward(self, S):
        """
        S: Input values (e.g., saturation) [B, H, W]
        Returns: Mapped values S' [B, H, W]
        """
        u_i = self.u_i
        this_u_i = u_i.tolist()  # Result: [0.1, 0.2, 0.3]

        # Use torch.tensor() directly when converting back to tensor
        self.this_u_i = torch.tensor(this_u_i, device=S.device)  # Result: tensor([0.1000, 0.2000, 0.3000])
        a_i = self._get_slopes(u_i)  # [M]
        delta = 1.0 / self.M
        Y = torch.sum(a_i * delta)  # Normalization factor

        # Find the segment index for each element in S
        j = torch.bucketize(S, self.bounds, right=True) - 1
        j = torch.clamp(j, 0, self.M - 1)

        # Initialize cum_before
        cum_before = torch.zeros_like(S)

        # Calculate cum_before element-wise
        for i in range(self.M):
            cum_before += (j > i).float() * a_i[i] * delta

        # Calculate contribution of the current segment
        segment_contrib = a_i[j] * (S - self.bounds[j])

        # Total mapping
        S_total = cum_before + segment_contrib
        S_prime = S_total / Y

        return S_prime

    def inverse(self, S_prime):
        """
        S_prime: Mapped values [B, H, W]
        Returns: Original values S [B, H, W]
        """
        u_i = self.this_u_i
        a_i = self._get_slopes(u_i)
        delta = 1.0 / self.M
        Y = torch.sum(a_i * delta)

        S_total = S_prime * Y

        # Calculate cumulative sum
        cum_sum = torch.cumsum(a_i * delta, dim=0)
        new_delta = a_i * delta
        
        # Find the segment index for S_total
        j = torch.bucketize(S_total, cum_sum, right=False)
        j = torch.clamp(j, 0, self.M - 1)

        # Calculate cum_before element-wise
        cum_before = torch.zeros_like(S_total)
        for i in range(self.M):
            cum_before += (j > i).float() * new_delta[i]

        # Calculate original S within the current segment
        S = self.bounds[j] + (S_total - cum_before) / a_i[j]

        return S

    def get_mapping_info(self):
        """Get mapping information."""
        a_i = self._get_slopes(self.u_i) # Fixed: passed self.u_i to _get_slopes
        delta = 1.0 / self.M
        Y = torch.sum(a_i * delta)

        return {
            'slopes': a_i.detach().cpu().numpy(),
            'total_length': Y.item(),
            'segment_lengths': (a_i * delta).detach().cpu().numpy()
        }


class ColorSpace(nn.Module):
    def __init__(self):
        super(ColorSpace, self).__init__()
        self.normal_vector_bias = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)
        # self.normal_vector_bias = nn.Parameter(torch.tensor([0.9,0.5,-0.8]), requires_grad=True)
        self.initial_normal_vector = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                                                  requires_grad=False)
        self.this_normal_vector_bias = 0
        self.hue_trans = Mapping(M=20)
        self.sat_trans = Mapping(M=20)
        self.lum_trans = Mapping(M=20)

    def value_encode(self, value, max_val, min_val):
        # Avoid division by zero risk
        return (value - min_val) / (max_val - min_val + 1e-6)

    def value_decode(self, value, max_val, min_val):
        # Avoid numerical instability
        return value * (max_val - min_val) + min_val

    def calculate_orthogonal_vectors(self, normal1):
        """
        Calculate two vectors normal2 and normal3 that are orthogonal to normal1.
        """
        # Ensure normal1 is normalized to avoid numerical instability
        normal1 = normal1 / ((normal1[0] ** 2 + normal1[1] ** 2 + normal1[2] ** 2 + 1e-8) ** (1 / 2) + 1e-8)
        d = -0.5 * (normal1[0] + normal1[1] + normal1[2])

        # Calculate normal2
        point = torch.tensor([1, 0, 0], dtype=torch.float32, device=normal1.device)
        t = -(point[0] * normal1[0] + point[1] * normal1[1] + point[2] * normal1[2] + d)
        normal2 = normal1 * t + point - 0.5
        normal2 = normal2 / ((normal2[0] ** 2 + normal2[1] ** 2 + normal2[2] ** 2 + 1e-8) ** (1 / 2) + 1e-8)

        # Ensure consistent direction for normal2
        if torch.dot(normal2, point) < 0:
            normal2 = -normal2

        # Calculate normal3 via cross product to get orthogonal vector
        normal3 = torch.cross(normal1, normal2)
        normal3 = normal3 / ((normal3[0] ** 2 + normal3[1] ** 2 + normal3[2] ** 2 + 1e-8) ** (1 / 2) + 1e-8)

        return [normal1, normal2, normal3]

    def gen_range(self, normal):
        """
        Generate range and angle information (maintain original logic, eliminate inplace operations).
        """
        points = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
                              dtype=torch.float32, device=normal.device)

        d = -0.5 * (normal[0] + normal[1] + normal[2])  # Maintain original d calculation logic

        n_points = points.shape[0]

        # Use lists to collect results (replacing inplace assignment) ensuring numerical logic remains unchanged
        point_flat_list = []
        ranges_list = []
        angles_list = []

        # Calculate projection and distance for each point (maintain original formulas)
        for i in range(n_points):
            # Original t calculation: dot product of normal and points[i] + d, then negated
            t = -(normal[0] * points[i][0] + normal[1] * points[i][1] + normal[2] * points[i][2] + d)
            # Original point_flat[i] calculation: normal*t + points[i] - 0.5
            point_flat_i = normal * t + points[i] - 0.5
            point_flat_list.append(point_flat_i)

            # Original ranges[i] calculation: L2 norm of point_flat_i
            range_i = (point_flat_i[0] ** 2 + point_flat_i[1] ** 2 + point_flat_i[2] ** 2 + 1e-8) ** (1 / 2)
            ranges_list.append(range_i)

        # Convert lists to tensors (shapes consistent with original point_flat and ranges)
        point_flat = torch.stack(point_flat_list)  # Shape: (n_points, 3)
        ranges = torch.stack(ranges_list)  # Shape: (n_points,)

        # Calculate angles between adjacent points (maintain original formulas)
        for i in range(n_points - 1):
            v1 = point_flat[i]
            v2 = point_flat[i + 1]
            # Original d_v1, d_v2 calculation: norms of v1, v2
            d_v1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2 + 1e-8) ** (1 / 2)
            d_v2 = (v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2 + 1e-8) ** (1 / 2)
            # Original cos_angle calculation: dot product of v1 and v2 / (d_v1*d_v2)
            cos_angle = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (d_v1 * d_v2 + 1e-8)
            cos_angle = cos_angle.clamp(-1, 1)
            # Original angles[i] calculation: arccos
            angle_i = torch.acos(cos_angle)
            angles_list.append(angle_i)

        # Convert list to tensor (shape consistent with original angles)
        angles = torch.stack(angles_list)  # Shape: (n_points-1,)

        # Accumulate angles (maintain original logic, use cumsum instead of loop inplace accumulation)
        # Original logic: cumulative_sums[0] = 0, cumulative_sums[i] = sum(angles[0..i-1])
        cumulative_sums = torch.zeros(n_points, dtype=torch.float32, device=normal.device)
        if n_points > 1:
            cumulative_sums[1:] = torch.cumsum(angles, dim=0)  # cumsum matches loop accumulation exactly
        cumulative_sums[-1] = torch.pi * 2

        # Maintain original t_min, t_max calculation logic
        t_min = (normal[0] + normal[1] + normal[2] + d)
        t_max = -(normal[0] + normal[1] + normal[2] + d)

        return t_min, t_max, ranges, angles, cumulative_sums

    def project_to_h_l_s(self, img, normals, t_min, t_max, ranges, angles, cumulative_sums):
        normal1, normal2, normal3 = normals
        # print(normals)
        img = img.permute(0, 2, 3, 1)  # (B,H,W,C)
        R, G, B = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
        d = -0.5 * (normal1[0] + normal1[1] + normal1[2])

        t = -(normal1[0] * R + normal1[1] * G + normal1[2] * B + d)

        R_new = normal1[0] * t + R
        G_new = normal1[1] * t + G
        B_new = normal1[2] * t + B
        vector = torch.stack([R_new - 0.5, G_new - 0.5, B_new - 0.5], dim=-1)
        distance = (vector[:, :, :, 0] ** 2 + vector[:, :, :, 1] ** 2 + vector[:, :, :, 2] ** 2 + 1e-8) ** (1 / 2)

        cos = (vector[:, :, :, 0] * normal2[0] + vector[:, :, :, 1] * normal2[1] + vector[:, :, :, 2] * normal2[2]) / (
                distance + 1e-8)
        cos = cos.clamp(-1 + 1e-5, 1 - 1e-5)
        sin = (vector[:, :, :, 0] * normal3[0] + vector[:, :, :, 1] * normal3[1] + vector[:, :, :, 2] * normal3[2]) / (
                distance + 1e-8)
        sin = sin.clamp(-1 + 1e-5, 1 - 1e-5)

        phase = torch.acos(cos)
        phase = torch.where(sin < 0, 2 * torch.pi - phase, phase)
        phase = phase % (cumulative_sums[-1])
        phase = phase / (torch.pi * 2)
        phase = self.hue_trans.forward(phase)
        phase = phase * 2 * torch.pi
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        rate = torch.max(ranges)

        # Normalization
        distance = distance / (rate + 1e-8)
        distance[rate == 0] = 0
        distance = self.sat_trans.forward(distance)
        hs1 = distance * cos
        hs2 = distance * sin
        t = self.value_encode(t, t_max, t_min)
        t = self.lum_trans.forward(t)
        hs1 = hs1.clamp(-1, 1)
        hs2 = hs2.clamp(-1, 1)

        return t, hs1, hs2

    def rgb2hsv(self, img):
        normal_bias = self.normal_vector_bias.clamp(-0.9, 0.9)
        this_normal_bias = normal_bias.tolist()  # Result: [0.1, 0.2, 0.3]

        # Use torch.tensor() directly when converting back to tensor
        self.this_normal_vector_bias = torch.tensor(this_normal_bias,
                                                    device=img.device)  # Result: tensor([0.1000, 0.2000, 0.3000])
        normal = self.initial_normal_vector + normal_bias
        normal = normal / ((normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2 + 1e-8) ** (1 / 2) + 1e-8)
        t_min, t_max, ranges, angles, cumulative_sums = self.gen_range(normal)
        normals = self.calculate_orthogonal_vectors(normal)
        t, hs1, hs2 = self.project_to_h_l_s(img, normals, t_min, t_max, ranges, angles, cumulative_sums)

        t = t.unsqueeze(1)  # (B, 1, H, W)
        hs1 = hs1.unsqueeze(1)  # (B, 1, H, W)
        hs2 = hs2.unsqueeze(1)  # (B, 1, H, W)

        output = torch.cat([t, hs1, hs2], dim=1)  # (B, 3, H, W)
        return output

    def hsv2rgb(self, hsv):
        hsv = hsv.permute(0, 2, 3, 1)
        t, hs1, hs2 = hsv[:, :, :, 0], hsv[:, :, :, 1], hsv[:, :, :, 2]
        t = t.clamp(0, 1)
        t = self.lum_trans.inverse(t)

        hs1 = hs1.clamp(-1, 1)
        hs2 = hs2.clamp(-1, 1)

        normal_bias = self.this_normal_vector_bias.clamp(-0.9, 0.9)
        normal = self.initial_normal_vector + normal_bias
        normal = normal / ((normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2 + 1e-8) ** (1 / 2) + 1e-8)

        t_min, t_max, ranges, angles, cumulative_sums = self.gen_range(normal)
        normals = self.calculate_orthogonal_vectors(normal)
        normal1, normal2, normal3 = normals
        B, H, W = t.shape
        # rate = torch.zeros(B, H, W, device=t.device)
        rate = torch.max(ranges)
        t = self.value_decode(t, t_max, t_min)

        distance = (hs1 ** 2 + hs2 ** 2 + 1e-8) ** 0.5
        cos = hs1 / (distance + 1e-8)
        sin = hs2 / (distance + 1e-8)
        cos = cos.clamp(-1 + 1e-5, 1 - 1e-5)
        sin = sin.clamp(-1 + 1e-5, 1 - 1e-5)

        phase = torch.acos(cos)
        phase = torch.where(sin < 0, 2 * torch.pi - phase, phase)
        phase = phase % (cumulative_sums[-1])
        phase = phase / (2 * torch.pi)
        phase = self.hue_trans.inverse(phase)
        phase = phase * 2 * torch.pi
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        # Normalization
        distance = self.sat_trans.inverse(distance)
        distance = distance * rate

        # 2D Plane Reconstruction
        img_2D = 0.5 + distance.unsqueeze(-1) * cos.unsqueeze(-1) * normal2 + \
                 distance.unsqueeze(-1) * sin.unsqueeze(-1) * normal3

        # 3D Coordinate Restoration
        R_new, G_new, B_new = img_2D[:, :, :, 0], img_2D[:, :, :, 1], img_2D[:, :, :, 2]
        R_re = R_new - normal1[0] * t
        G_re = G_new - normal1[1] * t
        B_re = B_new - normal1[2] * t

        # Range constraints and dimension adjustment
        img_out = torch.stack([R_re, G_re, B_re], dim=-1)
        img_out = img_out.clamp(0, 1)
        img_out = img_out.permute(0, 3, 1, 2)
        return img_out, normal