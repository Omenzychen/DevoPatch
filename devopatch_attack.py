import cv2 
import time
from numpy.core.records import record
import timm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import argparse
import torchvision.transforms as transforms
import random
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
import sys
import time


parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack!')
parser.add_argument('--model', type=str, default='resnet50', help='cnn')
parser.add_argument('--pop_size', default=10, type=int, help='random seed')
parser.add_argument('--steps', default=10000, type=int, help='random seed')
parser.add_argument('--init_rate', default=0.1, type=float, help='random seed')
parser.add_argument('--mutation_rate', default=1, type=int, help='random seed')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--norm', default=0, type=int, help='random seed')
parser.add_argument('--bs', default=16, type=int, help='random seed')
parser.add_argument('--targeted', default=False, action='store_true')



args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
CUDA_VISIBLE_DEVICES=0 python3 devopatch_attack.py --model resnet50 --bs 1000 --pop_size 10 --steps 10000 --init_rate 0.35 --mutation_rate 1

'''

class DevoPatch():
    def __init__(self, model, model_name, pop_size=10, init_rate=0.1, mutation_rate=1, steps=10000, targeted=False, device='cuda', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], norm=0, seed=0):
        self.model = model
        self.model_name = model_name
        self.pop_size = pop_size
        self.init_rate = init_rate
        self.mutation_rate = mutation_rate
        self.steps = steps
        self.now_queries = 0
        self.targeted = targeted
        self.device = device
        self.mean = mean
        self.std = std
        self.save_dir = 'attack_results'
        self.seed = seed
        self.norm = norm
        save_path = '{}-{}-{}-{}-{}-{}-{}-'.format(model_name, str(pop_size), str(init_rate), str(mutation_rate), str(steps), str(self.norm), str(seed))
        if targeted:
            save_path += 'targeted'
        else:
            save_path += 'untargeted'
        self.save_path = os.path.join(self.save_dir, save_path)
        print(self.save_path)

    def initialise_population(self, inputs, source_labels, starting_imgs, target_labels):
        print('Initialize Population...')
        bs, c, h, w = inputs.shape
        self.now_queries = 0
        V = torch.zeros(bs, self.pop_size, 4).long().to(self.device)
        G = torch.ones(bs, self.pop_size).to(self.device) * torch.Tensor([float('Inf')]).to(self.device)
        C = torch.ones(bs, self.pop_size).to(self.device)

        for i in range(self.pop_size):
            h_margin = int(h * self.init_rate
            w_margin = int(w * self.init_rate)
            cnt = 0
            while True:
                v_tmp = torch.zeros(bs, 4).long().to(self.device)
                v_tmp[:, 0] = torch.randint(0, h_margin, size=(bs,)).to(self.device) # x1
                v_tmp[:, 1] = torch.randint(0, w_margin, size=(bs,)).to(self.device) # y1
                v_tmp[:, 2] = torch.randint(h-h_margin+1, h+1, size=(bs,)).to(self.device) # x2
                v_tmp[:, 3] = torch.randint(w-w_margin+1, w+1, size=(bs,)).to(self.device) # y2
                v_mask = self.coord2mask(v_tmp)
                x_tmp = v_mask * starting_imgs + (~v_mask) * inputs
                flag = self.predict(x_tmp, source_labels, target_labels)
                fitness = self.get_fitness(x_tmp, inputs)
                idx = ((G[:, i] > fitness) & flag & torch.isinf(G[:, i]))
                print(flag.sum(), torch.isinf(G[:, i]).sum(), idx.sum())
                G[:, i][idx] = fitness[idx]
                V[:, i][idx] = v_tmp[idx].clone()
                if torch.isinf(G[:, i]).sum() == 0:
                    break
                cnt += 1
                if cnt > 10:
                    h_margin = w_margin = 1

        C = C * self.now_queries
        return V, G, C

    def coord2mask(self, v):
        mask = torch.zeros(v.shape[0], 1, self.h, self.w).bool().to(self.device)
        for i in range(v.shape[0]):
            mask[i, 0, v[i][0]:v[i][2], v[i][1]:v[i][3]] = True
        return mask
    
    def bound_handle(self, v_tmp):
        idx = (v_tmp[:, 0] > v_tmp[:, 2])
        v_tmp[:, 0][idx], v_tmp[:, 2][idx] = v_tmp[:, 2][idx], v_tmp[:, 0][idx]
        idx = (v_tmp[:, 1] > v_tmp[:, 3])
        v_tmp[:, 1][idx], v_tmp[:, 3][idx] = v_tmp[:, 3][idx], v_tmp[:, 1][idx]
        idx = (v_tmp[:, 3] >= self.w)
        v_tmp[:, 3][idx] = self.w - 1
        idx = (v_tmp[:, 2] >= self.h)
        v_tmp[:, 2][idx] = self.h - 1
        idx = (v_tmp[:, 1] < 0)
        v_tmp[:, 1][idx] = 0
        idx = (v_tmp[:, 0] < 0)
        v_tmp[:, 0][idx] = 0
        return v_tmp

    def crossover(self, v_best, v_j, v_q):
        sub = (v_j - v_q) * self.mutation_rate
        sub = sub.long()
        v_tmp = v_best.clone()
        v_tmp = v_tmp + sub
        v_tmp = self.bound_handle(v_tmp)
        return v_tmp.long()
    
    def mutation(self, v):
        noise = torch.randint(-self.mutation_rate, self.mutation_rate+1, (self.bs, 4)).to(self.device)
        v_tmp = v.clone() + noise
        v_tmp = self.bound_handle(v_tmp)
        return v_tmp.long()

    def get_fitness(self, inputs, x_adv):
        return torch.norm(inputs - x_adv, self.norm, dim=(1, 2, 3))

    def predict(self, inputs, source_labels, target_labels):
        self.now_queries += 1
        outputs = self.model(inputs)
        _, predicted = outputs.max(1)
        if self.targeted:
            flag = predicted.eq(target_labels)
        else:
            flag = (~predicted.eq(source_labels))
        return flag

                
    def perturb(self, inputs, source_labels, starting_imgs, target_labels):
        start = time.time()
        bs, c, h, w = inputs.shape
        self.h = h
        self.w = w
        self.bs = bs
        V, G, C = self.initialise_population(inputs, source_labels, starting_imgs, target_labels)
        k_worst = G.argmax(dim=1)
        k_best = G.argmin(dim=1)
        record_area = torch.zeros(self.steps).to(self.device)
        record_query = torch.zeros(self.steps).to(self.device)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        # self.record_imgs(0, inputs, torch.Tensor([100]), source_labels, target_labels, torch.Tensor([12344]))
        # self.record_imgs(1, starting_imgs, torch.Tensor([99]), source_labels, target_labels, torch.Tensor([13000]))


        bs_list = torch.arange(self.bs).long().to(self.device)
        with tqdm(total=self.steps, unit='iter') as pbar:
            for i in range(self.steps-self.pop_size):
                k = torch.randint(0, self.pop_size, size=(self.bs, 2)).to(self.device)
                cnt = random.randint(1, self.pop_size-1)
                idx = (k[:, 0] == k[:, 1])
                k[:, 1][idx] = (k[:, 1][idx] + 1) % self.pop_size
                idx = (k[:, 0] == k_best)
                if k[:, 0][idx].shape[0] != 0:
                    k[:, 0][idx] = (k[:, 0][idx] + cnt) % self.pop_size
                assert (k[:, 0] == k_best).sum() == 0, 'j is not best'
                idx = (k[:, 1] == k_best)
                if k[:, 1][idx].shape[0] != 0:
                    k[:, 1][idx] = (k[:, 1][idx] + cnt) % self.pop_size
                assert (k[:, 1] == k_best).sum() == 0, 'q is not best'

                j, q = k[:, 0], k[:, 1]
                v_r = self.crossover(V[range(self.bs), k_best], V[range(self.bs), j], V[range(self.bs), q])
                v_m = self.mutation(v_r)
                v_m_mask = self.coord2mask(v_m)
                x_adv = (~v_m_mask) * inputs + v_m_mask * starting_imgs
                flag = self.predict(x_adv, source_labels, target_labels)
                fitness = self.get_fitness(x_adv, inputs)
                
                update_idx = (G[range(self.bs), k_worst] * 1000 > fitness * 1000) & flag
                 
                if update_idx.sum() != 0:
                    G[bs_list[update_idx], k_worst[update_idx]] = fitness[update_idx]
                    V[bs_list[update_idx], k_worst[update_idx]] = v_m.clone()[update_idx]
                    C[bs_list[update_idx], k_worst[update_idx]] = self.now_queries

                k_worst = G.argmax(dim=1)
                k_best = G.argmin(dim=1)
        
                v_area = V[range(bs), k_best]
                ttt_area = ((v_area[:, 2] - v_area[:, 0]) * (v_area[:, 3] - v_area[:, 1]))/ self.w / self.h 
                ttt_queries = C[range(self.bs), k_best].mean()
                pbar.set_postfix(**{'Avg. Area': ttt_area.mean().item(), 'Avg. Queries':ttt_queries.item(), 'Max Area':ttt_area.max().item()})
                pbar.update(1)

                # Record
                record_query[i] = ttt_queries
                record_area[i] = ttt_area.mean()
                if (i) % 10000 ==0:
                    x_adv = (~self.coord2mask(V[range(bs), k_best])) * inputs + self.coord2mask(V[range(bs), k_best]) * starting_imgs
                    self.record_imgs(i, x_adv, ttt_area, source_labels, target_labels, C[range(self.bs), k_best])
        
        v_area = V[range(bs), k_best]
        ttt_area = ((v_area[:, 2] - v_area[:, 0]) * (v_area[:, 3] - v_area[:, 1]))/ self.w / self.h

        print('Total queries = ', self.now_queries)

        # Total Record
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        x_adv = (~self.coord2mask(V[range(bs), k_best])) * inputs + self.coord2mask(V[range(bs), k_best]) * starting_imgs
        self.record_imgs(i, x_adv, ttt_area, source_labels, target_labels, C[range(self.bs), k_best])

        torch.save(record_area, os.path.join(self.save_path, 'record_area.pt'))
        torch.save(record_query, os.path.join(self.save_path, 'record_query.pt'))
        torch.save(x_adv, os.path.join(self.save_path, 'adv.pt'))
        torch.save(self.coord2mask(V[range(bs), k_best]), os.path.join(self.save_path, 'mask.pt'))

        with open(os.path.join(self.save_path, 'log.txt'), 'w') as f:
            f.write('Average Area = ' + str(ttt_area.mean().item()) + '\n')
            f.write('Average Queries = ' + str(C[range(self.bs), k_best].mean().item()) +'\n')
            f.write('Total Time = ' + str(time.time() - start) +'\n')

        print('Finish Attack!')
        return x_adv
    
    def record_imgs(self, i, x_adv, area, source_labels, target_labels, best_cnt):
        print('Begin save imgs!')
        mean = self.mean
        std = self.std
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        area = area.cpu().numpy()
        source_labels = source_labels.cpu().numpy()
        target_labels = target_labels.cpu().numpy()
        best_cnt = best_cnt.cpu().numpy()

        x_adv = x_adv.cpu()
        x_adv = x_adv * std.view(-1, 3, 1, 1) + mean.view(-1, 3, 1, 1)
        x_adv = (255 * x_adv).numpy().astype('uint8').transpose((0, 2, 3, 1))
        x_adv = x_adv[:, :, :, ::-1]

        save_path = os.path.join(self.save_path, 'imgs')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_img_path =  '%04d-%d-%d-%.1f-%d.png' % (i, source_labels, target_labels, area*100, best_cnt)
        print(save_img_path)
        cv2.imwrite(os.path.join(save_path, save_img_path), x_adv[0])


if __name__ == '__main__':
    torch.set_printoptions(precision=6)

    # Model
    print('==> Preparing Model..', args.model)
    print('==> Building model...')
    print('==> Using ' + args.model)
    if args.model == 'resnet50':
        # https://download.pytorch.org/models/resnet50-0676ba61.pth
        net = torchvision.models.resnet50(pretrained=True)


    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        net.to(device)
        cudnn.benchmark = True

    net.eval()
    print(net.__class__.__name__)

    # Data
    print('==> Preparing data..')
    if args.model == 'vit' or args.model == 'mixer_mlp':
        print('Using 0.5 Nor...')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Construct a dataset, here we take one single image as an example
    img_path = "images/937.png"
    img = transform_test(Image.open(img_path)).unsqueeze(0).to(device)
    target_path = 'images/931.png'
    _, label = net(img).max(1)
    target_image = transform_test(Image.open(target_path)).unsqueeze(0).to(device)
    _, target_label = net(target_image).max(1)
    print(img.shape, img.max(), target_image.max(), label, target_label)

    adversary = DevoPatch(model=net, model_name=args.model, pop_size=args.pop_size, init_rate=args.init_rate, mutation_rate=args.mutation_rate, steps=args.steps, targeted=args.targeted, device='cuda', mean=mean, std=std, norm=args.norm, seed=args.seed)
 

    correct = 0
    with torch.no_grad():
        x_adv = adversary.perturb(img, label, target_image, target_label)
        outputs = net(x_adv)
        _, predicted = outputs.max(1)
        if args.targeted:
            correct += predicted.eq(target_label).sum().item()
        else:
            correct += (predicted!=label).sum().item()
    
    if correct:
        print("Attack Successfully!")
    else:
        print("Failure!")




