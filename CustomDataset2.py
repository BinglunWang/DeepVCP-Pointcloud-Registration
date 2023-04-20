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


class CustomDataset2(Dataset):
    def __init__(self, root, augment=True, rotate=True, split="train",
                 N=10000):
        self.root = root
        self.split = split
        self.augment = augment
        self.N = N
        self.files = []
        self.points = []
        self.augnum = 5
        self.diag_len_bounding_boxes = []
        # path to pointclouds + poses
        path = f"{self.root}"
        for file in os.listdir(path):
            if not file.endswith('.ply'):
                continue
            print(f"Processing {file}")
            src_pts = trimesh.load(os.path.join(path, file))
            self.files.append(file)
            self.points.append(src_pts)
            print(src_pts.vertices.shape)
            # Calculate diagonal length of bounding box
            x_len = np.amax(src_pts.vertices[:,0]) - np.amin(src_pts.vertices[:,0])
            y_len = np.amax(src_pts.vertices[:,1]) - np.amin(src_pts.vertices[:,1])
            z_len = np.amax(src_pts.vertices[:,2]) - np.amin(src_pts.vertices[:,2])
            diag_len_bounding_box = np.sqrt(x_len * x_len + y_len * y_len + z_len * z_len)
            self.diag_len_bounding_boxes.append(diag_len_bounding_box)

        print('# Total clouds', len(self.points))

    def __len__(self):
        # Each object need to learn different rotation. For each dimension we aims to learn 15 degree.
        return len(self.points) * self.augnum

    def __getitem__(self, index):
        # source pointcloud
        
        points_ind = index // self.augnum
        src_points = self.points[points_ind].sample(self.N).T
        # 3 x N
        # print("Loading file: ", self.files[index])

        # data augmentation
        # generate random angles for rotation matrices
        theta_x = np.random.uniform(0, np.pi)
        theta_y = np.random.uniform(0, np.pi)
        theta_z = np.random.uniform(0, np.pi)
        
        # generate random translation
        translation_max = self.diag_len_bounding_boxes[points_ind] * 0.3
        translation_min = self.diag_len_bounding_boxes[points_ind]  * 0.3
        t = np.random.uniform(translation_min, translation_max, (3, 1))

        # Generate target point cloud by doing a series of random
        # rotations on source point cloud
        Rx = RotX(theta_x)
        Ry = RotY(theta_y)
        Rz = RotZ(theta_z)
        R = Rx @ Ry @ Rz

        prior_x = theta_x + np.random.uniform(-np.pi/4, np.pi/4)
        prior_y = theta_y + np.random.uniform(-np.pi/4, np.pi/4)
        prior_z = theta_z + np.random.uniform(-np.pi/4, np.pi/4)
        R_prior_x = RotX(prior_x)
        R_prior_y = RotY(prior_y)
        R_prior_z = RotZ(prior_z)
        R_prior = R_prior_x @ R_prior_y @ R_prior_z

        # noise_theta_x = theta_x + np.random.uniform(-np.pi / 4, np.pi / 4)
        # noise_theta_y = theta_y + np.random.uniform(-np.pi / 4, np.pi / 4)
        # noise_theta_z = theta_x + np.random.uniform(-np.pi / 4, np.pi / 4)
        # Rx = RotX(noise_theta_x)
        # Ry = RotY(noise_theta_y)
        # Rz = RotZ(noise_theta_z)
        # R_noise = Rx @ Ry @ Rz


        # rotate source point cloud
        target_points = R @ src_points + t

        src_points = torch.from_numpy(src_points)
        target_points = torch.from_numpy(target_points)
        R = torch.from_numpy(R)
        R_prior = torch.from_numpy(R_prior)

        # return source point cloud and transformed (target) point cloud 
        # src, target: B x 3 x N, reflectance : B x 1 x N 

        # return (src_points, target_points, R, t, src_reflectance)
        #
        return (src_points, target_points, R, t, R_prior)


if __name__ == "__main__":
    data = CustomDataset(
        root='./', N=10000, augment=True, split="train")
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    for src, target, R, t in DataLoader:
        print('Source:', src.shape)  # B x 3 x N
        print('Target:', target.shape)  # B x 3 x N
        print('R', R.shape)
