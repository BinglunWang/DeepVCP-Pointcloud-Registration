import os
from utils import *
from torch.utils.data import Dataset, DataLoader
import trimesh

'''
Downsample point cloud to N points 
@params  src: original source point cloud
           N: number of points desired
@returns src: sampled point cloud
'''


def downsample(src, N):
    num_src = src.shape[0]
    src_downsample_indices = np.arange(num_src)
    if num_src > N:
        src_downsample_indices = np.random.choice(num_src, N, replace=False)
    return src[src_downsample_indices, :]


class CustomDataset(Dataset):
    def __init__(self, root, augment=True, rotate=True, split="train",
                 N=10000):
        self.root = root
        self.split = split
        self.augment = augment
        self.N = N
        self.files = []
        self.points = []
        # path to pointclouds + poses
        path = f"{self.root}meshes/"
        for file in os.listdir(path):
            if not file.endswith('.obj'):
                continue
            print(f"Processing {file}")
            src_pts = trimesh.load(os.path.join(path, file))
            self.files.append(file)
            self.points.append(src_pts)

        print('# Total clouds', len(self.points))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        # source pointcloud
        src_points = self.points[index].sample(self.N).T
        # 3 x N
        # print("Loading file: ", self.files[index])

        # data augmentation
        # generate random angles for rotation matrices
        theta_x = np.random.uniform(0, np.pi * 2)
        theta_y = np.random.uniform(0, np.pi * 2)
        theta_z = np.random.uniform(0, np.pi * 2)

        # generate random translation
        translation_max = 1.0
        translation_min = -1.0
        t = np.random.uniform(translation_min, translation_max, (3, 1))

        # Generate target point cloud by doing a series of random
        # rotations on source point cloud
        Rx = RotX(theta_x)
        Ry = RotY(theta_y)
        Rz = RotZ(theta_z)
        R = Rx @ Ry @ Rz

        # rotate source point cloud
        target_points = R @ src_points + t

        src_points = torch.from_numpy(src_points)
        target_points = torch.from_numpy(target_points)
        R = torch.from_numpy(R)

        # return source point cloud and transformed (target) point cloud 
        # src, target: B x 3 x N, reflectance : B x 1 x N 

        # return (src_points, target_points, R, t, src_reflectance)
        #
        return (src_points, target_points, R, t)


if __name__ == "__main__":
    data = CustomDataset(
        root='./', N=10000, augment=True, split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    for src, target, R, t in DataLoader:
        print('Source:', src.shape)  # B x 3 x N
        print('Target:', target.shape)  # B x 3 x N
        print('R', R.shape)
