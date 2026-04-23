import csv
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision.models.inception import inception_v3
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from tqdm import tqdm
from PIL import Image
import argparse
import os

def get_frechet_inception_distance(dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception_model = inception_v3(pretrained=True, transform_input=True).to(device)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    
    def get_inception_features(image_batch):
        inception_output = inception_model(image_batch)
        return inception_output.data.cpu().numpy()
    
    inception_feature_batches_fake = []
    for _, inpainted_image in tqdm(dataloader, desc=f'FID - Fake Data Feature Extraction', total=len(dataloader)):
        image_batch = torch.Tensor(inpainted_image).to(device)
        inception_feature_batch = get_inception_features(image_batch)
        inception_feature_batches_fake.append(inception_feature_batch)
    inception_features_fake = np.concatenate(inception_feature_batches_fake)

    inception_feature_batches_real = []
    for target_image, _ in tqdm(dataloader, desc=f'FID - Real Data Feature Extraction', total=len(dataloader)):
        image_batch = torch.Tensor(target_image).to(device)
        inception_feature_batch = get_inception_features(image_batch)
        inception_feature_batches_real.append(inception_feature_batch)
    inception_features_real= np.concatenate(inception_feature_batches_real)
    
    mu_fake, sigma_fake = inception_features_fake.mean(axis=0), cov(inception_features_fake, rowvar=False)
    mu_real, sigma_real = inception_features_real.mean(axis=0), cov(inception_features_real, rowvar=False)
    ssdiff = np.sum((mu_fake - mu_real)**2.0)
    cov_mean = sqrtm(sigma_fake.dot(sigma_real))
    if iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    frechet_inception_distance = ssdiff + trace(sigma_fake + sigma_real - 2.0 * cov_mean)
    return frechet_inception_distance


class InferenceDataset(Dataset):
    def __init__(self, datadir, inference_dir, eval_resolution=512, img_suffix='.jpg', inpainted_suffix='_removed.png'):
        self.inference_dir = inference_dir
        self.datadir = datadir
        if not datadir.endswith('/'):
            datadir += '/'
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.file_names = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.img_filenames]
        self.eval_resolution = eval_resolution
        self.ids = [file_name.rsplit('/', 1)[1].rsplit('_mask.png', 1)[0] for file_name in self.mask_filenames]

    def __len__(self):
        return len(self.ids)
    
    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.eval_resolution,self.eval_resolution), Image.Resampling.BICUBIC)
        img = np.array(img, dtype=float) / 255
        img = np.moveaxis(img, [0,1,2], [1,2,0])
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, idx):
        #scene_id = self.ids[idx]
        
        target_image = self.read_image(self.img_filenames[idx])
        #target_image = self.read_image(self.test_filenames[idx])
        inpainted_image = self.read_image(self.file_names[idx])
        return target_image, inpainted_image
    
class Inferencedataset_local(InferenceDataset):
    def __init__(self, datadir, inference_dir, test_scene, eval_resolution=512, img_suffix='.jpg', inpainted_suffix='_removed.png'):
        super().__init__(datadir, inference_dir, eval_resolution, img_suffix, inpainted_suffix)
        self.test_scene = self.read_csv_to_dict(test_scene)

    def read_csv_to_dict(self,file_path):
        data_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            header = next(reader)  # Skip header if there is one
            for row in reader:
                id = row[0].rsplit('.', 1)[0]
                LabelName = row[1]
                BoxXMin = float(row[2])
                BoxXMax = float(row[3])
                BoxYMin = float(row[4])
                BoxYMax = float(row[5])
                
                data_dict[id] = {
                    'LabelName': LabelName,
                    'BoxXMin': BoxXMin,
                    'BoxXMax': BoxXMax,
                    'BoxYMin': BoxYMin,
                    'BoxYMax': BoxYMax
                }
        return data_dict
    
    def read_image(self, path, object_bbox):
        img = Image.open(path).crop(object_bbox)
        img = img.convert('RGB')
        img = img.resize((self.eval_resolution,self.eval_resolution), Image.Resampling.BICUBIC)
        img = np.array(img, dtype=float) / 255
        img = np.moveaxis(img, [0,1,2], [1,2,0])
        img = torch.from_numpy(img).float()
        return img
    
    def __getitem__(self, idx):
        scene_id = self.ids[idx]
        object_bbox = (int(self.test_scene[scene_id]["BoxXMin"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxYMin"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxXMax"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxYMax"]*self.eval_resolution))
        
        target_image = self.read_image(self.img_filenames[idx], object_bbox)
        #target_image = self.read_image(self.test_filenames[idx], object_bbox)
        inpainted_image = self.read_image(self.file_names[idx], object_bbox)
        return target_image, inpainted_image
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        default=".DATA/original/",
        help="Directory of the original images and masks",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="outputs/inference/",
        help="Directory of the inference results",
    )
    parser.add_argument(
        "--test_scene",
        type=str,
        default="./DATA/fetch_output.csv",
        help="path of the test scene",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/fid",
        help="Directory of evaluation outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Bath size of Inception v3 forward pass",
    )
    parser.add_argument(
        "--inpainted_suffix",
        type=str,
        default='_removed.png',
        help="inference_dir's suffix",
    )
    args = parser.parse_args()

    dataset = InferenceDataset(args.datadir, args.inference_dir, eval_resolution=512, img_suffix='.jpg', inpainted_suffix=args.inpainted_suffix)
    dataset_local = Inferencedataset_local(args.datadir, args.inference_dir, args.test_scene, eval_resolution=512, img_suffix='.jpg', inpainted_suffix=args.inpainted_suffix)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    dataloader_local = torch.utils.data.DataLoader(dataset_local, batch_size=args.batch_size, shuffle=False)
    print('start to calculate FID_local')
    fid_local = get_frechet_inception_distance(dataloader_local)
    print(f"FID_local: {fid_local}")
    
    print('start to calculate FID')
    fid = get_frechet_inception_distance(dataloader)
    print(f"FID: {fid}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    fid_str = f"FID: {fid}"
    fid_local_str = f"FID_local: {fid_local}"
    output_path = os.path.join(output_dir, f"fid_{dataset.eval_resolution}.txt")
    f = open(output_path, "w")
    f.write(fid_str + '\n')
    f.write(fid_local_str)
    f.close()

    print(fid_str)
    print(fid_local_str)





