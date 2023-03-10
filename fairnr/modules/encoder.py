
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
from numba import jit
import math
import sys
import os
import math
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from plyfile import PlyData, PlyElement

from fairnr.data.data_utils import load_matrix
from fairnr.data.geometry import (
    trilinear_interp, splitting_points, offset_points,
    get_edge, build_easy_octree, discretize_points
)
from fairnr.clib import (
    aabb_ray_intersect, triangle_ray_intersect,
    uniform_ray_sampling, svo_ray_intersect
)
from fairnr.modules.linear import FCBlock, Linear, Embedding

MAX_DEPTH = 10000.0
ENCODER_REGISTRY = {}

def register_encoder(name):
    def register_encoder_cls(cls):
        if name in ENCODER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        ENCODER_REGISTRY[name] = cls
        return cls
    return register_encoder_cls


def get_encoder(name):
    if name not in ENCODER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return ENCODER_REGISTRY[name]


@register_encoder('abstract_encoder')
class Encoder(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass

@jit(nopython=True)
def get_voxel_color(np_fine_points, pc_points, color_feat, voxel_size, min_color=-1):
    def point_in_voxel(point, voxel_center, voxel_size):
        half_voxel = voxel_size / 2.0
        if point[0] < voxel_center[0] - half_voxel or point[0] > voxel_center[0] + half_voxel:
            return False
        if point[1] < voxel_center[1] - half_voxel or point[1] > voxel_center[1] + half_voxel:
            return False
        if point[2] < voxel_center[2] - half_voxel or point[2] > voxel_center[2] + half_voxel:
            return False

        return True

    voxel_color = np.zeros((np_fine_points.shape[0],3), dtype=np.float32)
    for j in range(np_fine_points.shape[0]):
        count = 0
        for i in range(pc_points.shape[0]):
            if point_in_voxel(pc_points[i], np_fine_points[j,:], voxel_size):
                voxel_color[j,:] += color_feat[i,:] / 255.0
                # voxel_include_point[j, count] = i
                count += 1
        voxel_color[j,:] /= (count + 0.00001)
    if min_color == -1:
        voxel_color -= 0.5
        voxel_color *= 2.
    return voxel_color

@register_encoder('sparsevoxel_encoder')
class SparseVoxelEncoder(Encoder):

    def __init__(self, args, voxel_path=None, pc_path=None, voxel_color_path=None, bbox_path=None, shared_values=None):
        super().__init__(args)
        # read initial voxels or learned sparse voxels
        self.voxel_path = voxel_path if voxel_path is not None else args.voxel_path
        self.pc_path = pc_path if pc_path is not None else args.pc_path
        self.voxel_color_path = voxel_color_path if voxel_color_path is not None else args.voxel_color_path
        self.bbox_path = bbox_path if bbox_path is not None else getattr(args, "initial_boundingbox", None)
        assert (self.bbox_path is not None) or (self.voxel_path is not None), \
            "at least initial bounding box or pretrained voxel files are required."
        self.voxel_index = None
        if self.voxel_path is not None:
            assert os.path.exists(self.voxel_path), "voxel file must exist"
            assert getattr(args, "voxel_size", None) is not None, "final voxel size is essential."
            
            voxel_size = args.voxel_size

            if Path(self.voxel_path).suffix == '.ply':
                from plyfile import PlyData, PlyElement
                plydata = PlyData.read(self.voxel_path)['vertex']
                np_fine_points = np.stack([plydata['x'], plydata['y'], plydata['z']]).astype('float32').T
                fine_points = torch.from_numpy(np_fine_points)
                try:
                    self.voxel_index = torch.from_numpy(plydata['quality']).long()
                except ValueError:
                    pass
            else:
                # supporting the old version voxel points
                np_fine_points = np.loadtxt(self.voxel_path)[:, 3:].astype('float32')
                fine_points = torch.from_numpy(np_fine_points)
        else:
            bbox = np.loadtxt(self.bbox_path)
            voxel_size = bbox[-1]
            np_fine_points = bbox2voxels(bbox[:6], voxel_size)
            fine_points = torch.from_numpy(np_fine_points)
        half_voxel = voxel_size * .5

        self.load_pc = getattr(args, "load_pc", False)
        if self.load_pc and self.voxel_color_path is not None:
            if os.path.exists(self.voxel_color_path):
                voxel_color = np.loadtxt(self.voxel_color_path)
            else:
                assert self.pc_path is not None, "point cloud file must be given"
                assert os.path.exists(self.pc_path), "point cloud file must exist"
                print("we load point cloud!!!")
                if Path(self.pc_path).suffix == '.ply':
                    from plyfile import PlyData, PlyElement
                    plydata = PlyData.read(self.pc_path)['vertex']
                    pc_points = np.stack([plydata['x'], plydata['y'], plydata['z']]).astype('float32').T
                    try:
                        color_feat = np.stack([plydata['red'], plydata['green'], plydata['blue']]).astype('int32').T
                    except Exception as e:
                        print('color feature obtained error!')
                        color_feat = np.zeros(pc_points.shape, dtype=np.int32)
                else:
                    # supporting the old version voxel points
                    color_feat = np.loadtxt(self.pc_path)[:, 3:].astype('int32')

                # pre-compute each voxel include which point cloud
                print("gathering voxel color ...")
                voxel_color = get_voxel_color(np_fine_points, pc_points, color_feat, voxel_size, min_color=self.args.min_color)
                np.savetxt(self.voxel_color_path, voxel_color)
            voxel_color = torch.from_numpy(voxel_color)

        # transform from voxel centers to voxel corners (key/values)
        fine_coords, _ = discretize_points(fine_points, half_voxel)
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)
        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()

        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * voxel_size
        else:
            step_size = args.raymarching_stepsize

        # register parameters (will be saved to checkpoints)
        self.register_buffer("points", fine_points)          # voxel centers
        self.register_buffer("keys", fine_keys.long())       # id used to find voxel corners/embeddings
        self.register_buffer("feats", fine_feats.long())     # for each voxel, 8 voxel corner ids
        self.register_buffer("num_keys", num_keys)
        self.register_buffer("keep", fine_feats.new_ones(fine_feats.size(0)).long())  # whether the voxel will be pruned
        if self.load_pc:
            self.register_buffer("voxel_color", voxel_color)

        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        # set-up other hyperparameters and initialize running time caches
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.use_octree = getattr(args, "use_octree", False)
        self.track_max_probs = getattr(args, "track_max_probs", False)    
        self._runtime_caches = {
            "flatten_centers": None,
            "flatten_children": None,
            "max_voxel_probs": None
        }

        # sparse voxel embeddings     
        if shared_values is None and self.embed_dim > 0:
            self.values = Embedding(num_keys, self.embed_dim, None)
        else:
            self.values = shared_values

    def upgrade_state_dict_named(self, state_dict, name):
        # update the voxel embedding shapes
        if self.values is not None:
            loaded_values = state_dict[name + '.values.weight']
            self.values.weight = nn.Parameter(self.values.weight.new_zeros(*loaded_values.size()))
            self.values.num_embeddings = self.values.weight.size(0)
            self.total_size = self.values.weight.size(0)
            self.num_keys = self.num_keys * 0 + self.total_size
        
        if self.voxel_index is not None:
            state_dict[name + '.points'] = state_dict[name + '.points'][self.voxel_index]
            state_dict[name + '.feats'] = state_dict[name + '.feats'][self.voxel_index]
            state_dict[name + '.keep'] = state_dict[name + '.keep'][self.voxel_index]
        
        # update the buffers shapes
        self.points = self.points.new_zeros(*state_dict[name + '.points'].size())
        self.feats  = self.feats.new_zeros(*state_dict[name + '.feats'].size())
        self.keys   = self.keys.new_zeros(*state_dict[name + '.keys'].size())
        self.keep   = self.keep.new_zeros(*state_dict[name + '.keep'].size())
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--initial-boundingbox', type=str, help='the initial bounding box to initialize the model')
        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial')
        parser.add_argument('--voxel-path', type=str, help='path for pretrained voxel file. if provided no update')
        parser.add_argument('--pc-path', type=str, 
                            help='path for point cloud file. if provided compute voxel color from beginning', default=None)
        parser.add_argument('--voxel-color-path', type=str, 
                            help='path for voxle color file. if provided load voxel color', default=None)
        parser.add_argument('--load-pc', action="store_true", help='load point cloud or not', default=False)
        parser.add_argument('--pc-pose-dim', type=int, metavar='N', help="pc pose dim", default=2)
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-stepsize-ratio', type=float, metavar='D',
                            help='if the concrete step size is not given (=0), we use the ratio to the voxel size as step size.')
        parser.add_argument('--use-octree', action='store_true', help='if set, instead of looping over the voxels, we build an octree.')
        parser.add_argument('--track-max-probs', action='store_true', help='if set, tracking the maximum probability in ray-marching.')
        parser.add_argument('--octree-path', type=str, help='path for octree file. if provided skip time-consuming build process', default=None)

    def reset_runtime_caches(self):
        if self.use_octree:
            octree_path = self.args.octree_path
            if octree_path is not None and os.path.exists(octree_path):
                octree = np.load(octree_path)
                np_centers, np_children = octree['centers'], octree['children']
                centers, children = torch.from_numpy(np_centers).to(self.points.device), torch.from_numpy(np_children).to(self.points.device).int()
                print('load octree success!!!')
            else:
                centers, children = build_easy_octree(self.points[self.keep.bool()], self.voxel_size / 2.0)
                if octree_path is not None:
                    np_centers = centers.clone().detach().cpu().numpy()
                    np_children = children.clone().detach().cpu().numpy()
                    np.savez(octree_path, centers=np_centers, children=np_children)
            # centers, children = build_easy_octree(self.points[self.keep.bool()], self.voxel_size / 2.0)
            self._runtime_caches['flatten_centers'] = centers
            self._runtime_caches['flatten_children'] = children
        self._runtime_caches['max_voxel_probs'] = self.points.new_zeros(self.points.size(0))

    def clean_runtime_caches(self):
        for name in self._runtime_caches:
            self._runtime_caches[name] = None

    def precompute(self, id=None, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        values = self.values.weight[: self.num_keys] if self.values is not None else None
        
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if self.load_pc:
            encoder_states['voxel_color'] = self.voxel_color

        if self.use_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            if id is not None:
                flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
                flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children
        return encoder_states

    @torch.no_grad()
    def export_voxels(self, return_mesh=False):
        logger.info("exporting learned sparse voxels...")
        voxel_idx = torch.arange(self.keep.size(0), device=self.keep.device)
        voxel_idx = voxel_idx[self.keep.bool()]
        voxel_pts = self.points[self.keep.bool()]
        if not return_mesh:
            # HACK: we export the original voxel indices as "quality" in case for editing
            points = [
                (voxel_pts[k, 0], voxel_pts[k, 1], voxel_pts[k, 2], voxel_idx[k])
                for k in range(voxel_idx.size(0))
            ]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])
            return PlyData([PlyElement.describe(vertex, 'vertex')])
        
        else:
            # generate polygon for voxels
            center_coords, residual = discretize_points(voxel_pts, self.voxel_size / 2)
            offsets = torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1]], device=center_coords.device)
            vertex_coords = center_coords[:, None, :] + offsets[None, :, :]
            vertex_points = vertex_coords.type_as(residual) * self.voxel_size / 2 + residual
            
            faceidxs = [[1,6,7,5],[7,6,2,4],[5,7,4,3],[1,0,2,6],[1,5,3,0],[0,3,4,2]]
            all_vertex_keys, all_vertex_idxs  = {}, []
            for i in range(vertex_coords.shape[0]):
                for j in range(8):
                    key = " ".join(["{}".format(int(p)) for p in vertex_coords[i,j]])
                    if key not in all_vertex_keys:
                        all_vertex_keys[key] = vertex_points[i,j]
                        all_vertex_idxs += [key]
            all_vertex_dicts = {key: u for u, key in enumerate(all_vertex_idxs)}
            all_faces = torch.stack([torch.stack([vertex_coords[:, k] for k in f]) for f in faceidxs]).permute(2,0,1,3).reshape(-1,4,3)
    
            all_faces_keys = {}
            for l in range(all_faces.size(0)):
                key = " ".join(["{}".format(int(p)) for p in all_faces[l].sum(0) // 4])
                if key not in all_faces_keys:
                    all_faces_keys[key] = all_faces[l]

            vertex = np.array([tuple(all_vertex_keys[key].cpu().tolist()) for key in all_vertex_idxs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            face = np.array([([all_vertex_dicts["{} {} {}".format(*b)] for b in a.cpu().tolist()],) for a in all_faces_keys.values()],
                dtype=[('vertex_indices', 'i4', (4,))])
            return PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')])

    @torch.no_grad()
    def export_surfaces(self, field_fn, th, bits):
        """
        extract triangle-meshes from the implicit field using marching cube algorithm
            Lewiner, Thomas, et al. "Efficient implementation of marching cubes' cases with topological guarantees." 
            Journal of graphics tools 8.2 (2003): 1-15.
        """
        logger.info("marching cube...")
        encoder_states = self.precompute(id=None)
        points = encoder_states['voxel_center_xyz']

        scores = self.get_scores(field_fn, th=th, bits=bits, encoder_states=encoder_states)
        coords, residual = discretize_points(points, self.voxel_size)
        A, B, C = [s + 1 for s in coords.max(0).values.cpu().tolist()]
    
        # prepare grids
        full_grids = points.new_ones(A * B * C, bits ** 3)
        full_grids[coords[:, 0] * B * C + coords[:, 1] * C + coords[:, 2]] = scores
        full_grids = full_grids.reshape(A, B, C, bits, bits, bits)
        full_grids = full_grids.permute(0, 3, 1, 4, 2, 5).reshape(A * bits, B * bits, C * bits)
        full_grids = 1 - full_grids

        # marching cube
        from skimage import measure
        space_step = self.voxel_size.item() / bits
        verts, faces, normals, _ = measure.marching_cubes_lewiner(
            volume=full_grids.cpu().numpy(), level=0.5,
            spacing=(space_step, space_step, space_step)
        )
        verts += (residual - (self.voxel_size / 2)).cpu().numpy()
        verts = np.array([tuple(a) for a in verts.tolist()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        faces = np.array([(a, ) for a in faces.tolist()], dtype=[('vertex_indices', 'i4', (3,))])
        return PlyData([PlyElement.describe(verts, 'vertex'), PlyElement.describe(faces, 'face')])

    def get_edge(self, ray_start, ray_dir, samples, encoder_states):
        outs = get_edge(
            ray_start + ray_dir * samples['sampled_point_depth'][:, :1], 
            encoder_states['voxel_center_xyz'].reshape(-1, 3)[samples['sampled_point_voxel_idx'][:, 0].long()], 
            self.voxel_size).type_as(ray_dir)   # get voxel edges/depth (for visualization)
        outs = (1 - outs[:, None].expand(outs.size(0), 3)) * 0.7
        return outs

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()

        if self.use_octree:  # ray-voxel intersection with SVO
            flatten_centers = encoder_states['voxel_octree_center_xyz']
            flatten_children = encoder_states['voxel_octree_children_idx']
            pts_idx, min_depth, max_depth = svo_ray_intersect(
                self.voxel_size, self.max_hits, flatten_centers, flatten_children,
                ray_start, ray_dir)
        else:   # ray-voxel intersection with all voxels
            pts_idx, min_depth, max_depth = aabb_ray_intersect(
                self.voxel_size, self.max_hits, point_xyz, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        
        if S > 1:  # extend the point-index to multiple shapes (just in case)
            pts_idx = (pts_idx + H * torch.arange(S, 
                device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                ).masked_fill_(pts_idx.eq(-1), -1)

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits

    def ray_sample(self, intersection_outputs):
        min_depth = intersection_outputs['min_depth']
        max_depth = intersection_outputs['max_depth']
        pts_idx = intersection_outputs['intersected_voxel_idx']

        max_ray_length = (max_depth.masked_fill(max_depth.eq(MAX_DEPTH), 0).max(-1)[0] - min_depth.min(-1)[0]).max()
        sampled_idx, sampled_depth, sampled_dists = uniform_ray_sampling(
            pts_idx, min_depth, max_depth, self.step_size, max_ray_length, 
            self.deterministic_step or (not self.training))
        sampled_dists = sampled_dists.clamp(min=0.0)
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)

        samples = {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,
        }
        return samples

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        # encoder states
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']

        # ray point samples
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        sampled_xyz = samples['sampled_point_xyz'].requires_grad_(True)
        sampled_dir = samples['sampled_point_ray_direction']

        # prepare inputs for implicit field
        inputs = {'pos': sampled_xyz, 'ray': sampled_dir}
        if values is not None:
            # resample point features
            point_xyz = F.embedding(sampled_idx, point_xyz)
            point_feats = F.embedding(F.embedding(sampled_idx, point_feats), values).view(point_xyz.size(0), -1)

            # tri-linear interpolation
            p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
            q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5   # BUG (FIX)
            inputs.update({'emb': trilinear_interp(p, q, point_feats)})

        return inputs

    def track_voxel_probs(self, voxel_idxs, voxel_probs):
        voxel_idxs = voxel_idxs.masked_fill(voxel_idxs.eq(-1), self.max_voxel_probs.size(0))
        max_voxel_probs = self.max_voxel_probs.new_zeros(voxel_idxs.size(0), self.max_voxel_probs.size(0) + 1).scatter_add_(
            dim=-1, index=voxel_idxs, src=voxel_probs).max(0)[0][:-1].data
        self.max_voxel_probs = torch.max(self.max_voxel_probs, max_voxel_probs)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, encoder_states=None, train_stats=False):
        if not train_stats:
            logger.info("pruning...")
            scores = self.get_scores(field_fn, th=th, bits=16, encoder_states=encoder_states)
            keep = (1 - scores.min(-1)[0]) > th
        else:
            logger.info("pruning based on training set statics (e.g. probs)...")
            if dist.get_world_size() > 1:  # sync on multi-gpus
                dist.all_reduce(self.max_voxel_probs, op=dist.ReduceOp.MAX)
            keep = self.max_voxel_probs > th

        self.keep.masked_scatter_(self.keep.bool(), keep.long())
        logger.info("pruning done. # of voxels before: {}, after: {} voxels".format(keep.size(0), keep.sum()))
    
    def get_scores(self, field_fn, th=0.5, bits=16, encoder_states=None):
        if encoder_states is None:
            encoder_states = self.precompute(id=None)
        
        feats = encoder_states['voxel_vertex_idx'] 
        points = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']
        chunk_size = 64

        def get_scores_once(feats, points, values):
            # sample points inside voxels
            sampled_xyz = offset_points(points, self.voxel_size / 2.0, bits=bits)
            sampled_idx = torch.arange(points.size(0), device=points.device)[:, None].expand(*sampled_xyz.size()[:2])
            sampled_xyz, sampled_idx = sampled_xyz.reshape(-1, 3), sampled_idx.reshape(-1)
            
            field_inputs = self.forward(
                {'sampled_point_xyz': sampled_xyz, 
                 'sampled_point_voxel_idx': sampled_idx,
                 'sampled_point_ray_direction': None}, 
                {'voxel_vertex_idx': feats,
                 'voxel_center_xyz': points,
                 'voxel_vertex_emb': values})  # get field inputs
     
            # evaluation with density
            field_outputs = field_fn(field_inputs, outputs=['sigma'])
            free_energy = -torch.relu(field_outputs['sigma']).reshape(-1, bits ** 3)
            
            # return scores
            return torch.exp(free_energy)

        return torch.cat([get_scores_once(feats[i: i + chunk_size], points[i: i + chunk_size], values) 
            for i in range(0, points.size(0), chunk_size)], 0)

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        encoder_states = self.precompute(id=None)
        feats, points, values = encoder_states['voxel_vertex_idx'], encoder_states['voxel_center_xyz'], encoder_states['voxel_vertex_emb']
        new_points, new_feats, new_values, new_keys = splitting_points(points, feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)
        
        # set new voxel embeddings
        if new_values is not None:
            self.values.weight = nn.Parameter(new_values)
            self.values.num_embeddings = self.values.weight.size(0)
        
        self.total_size = new_num_keys
        self.num_keys = self.num_keys * 0 + self.total_size

        self.points = new_points
        self.feats = new_feats
        self.keep = self.keep.new_ones(new_point_length)
        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(points.size(0), self.keep.sum()))
        
    @property
    def flatten_centers(self):
        if self._runtime_caches['flatten_centers'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_centers']
    
    @property
    def flatten_children(self):
        if self._runtime_caches['flatten_children'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['flatten_children']

    @property
    def max_voxel_probs(self):
        if self._runtime_caches['max_voxel_probs'] is None:
            self.reset_runtime_caches()
        return self._runtime_caches['max_voxel_probs']

    @max_voxel_probs.setter
    def max_voxel_probs(self, x):
        self._runtime_caches['max_voxel_probs'] = x

    @property
    def feature_dim(self):
        return self.embed_dim

    @property
    def dummy_loss(self):
        if self.values is not None:
            return self.values.weight[0,0] * 0.0
        return 0.0
    
    @property
    def num_voxels(self):
        return self.keep.long().sum()


def bbox2voxels(bbox, voxel_size):
    vox_min, vox_max = bbox[:3], bbox[3:]
    steps = ((vox_max - vox_min) / voxel_size).round().astype('int64') + 1
    x, y, z = [c.reshape(-1).astype('float32') for c in np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]))]
    x, y, z = x * voxel_size + vox_min[0], y * voxel_size + vox_min[1], z * voxel_size + vox_min[2]
    return np.stack([x, y, z]).T.astype('float32')


