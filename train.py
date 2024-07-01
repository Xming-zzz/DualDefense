import math
import torch
import os
import random
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2
from DiffJPEG import DiffJPEG
from dataset160 import CustomDataset, split_dataset
from DualDefense_gan_fs160 import DualDefense
from facenet_pytorch import MTCNN, InceptionResnetV1

resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet.classify = True

# jpeg = DiffJPEG(height=64, width=64, differentiable=True, quality=50)
mode = 'train'
date = '.'
f_print_path = "./save/loss/" + date + "_print.txt"
save_result = ['encode', 'encode_fake', 'fake', 'real', 'val', 'tmp']
if not os.path.exists('./save/img/' + date):
    os.mkdir('./save/img/' + date)
for path in save_result:
    if not os.path.exists('./save/img/' + date + '/' + path):
        os.mkdir('./save/img/' + date + '/' + path)
ori = 0.9
adv = 0.1


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input ids must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# separate the ycbcr images for the loss calculation
def ycbcr_images(cover_ybr,stego_ybr):
    dwtimage = DWTForward_Init(J=1, mode='zero', wave='haar').to(device)
    idwtimage = DWTInverse_Init(mode='zero', wave='haar').to(device)
    stego_Yl, stego_Yh = dwtimage(stego_ybr[:, 0, :, :].unsqueeze(1).to(device))
    cover_Yl, cover_Yh = dwtimage(cover_ybr[:, 0, :, :].unsqueeze(1).to(device))
    stego_color = torch.cat((stego_ybr[:, 1, :, :], stego_ybr[:, 2, :, :]), 1).unsqueeze(1)
    cover_color = torch.cat((cover_ybr[:, 1, :, :], cover_ybr[:, 2, :, :]), 1).unsqueeze(1)
    stego_YH = torch.tensor([item.cpu().detach().numpy() for item in stego_Yh]).to(device).squeeze()
    cover_YH = torch.tensor([item.cpu().detach().numpy() for item in cover_Yh]).to(device).squeeze()

    return stego_Yl, stego_Yh, cover_Yl, cover_Yh, stego_color, cover_color, stego_YH, cover_YH

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = c.w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)
set_seed(0)

parser = argparse.ArgumentParser(description='TAW')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--epoch', default=2000, type=int, help='epochs')
parser.add_argument('--start_decode', default=30, type=int, help='epoch to start training decoder')
parser.add_argument('--clip', default=15, type=int, help='clip')
parser.add_argument('--message_size', default=15, type=int, help='msg size')
parser.add_argument('--lambda_val', default=1, type=float, help='weight of msg loss')
parser.add_argument('--alpha_val', default=1, type=float, help='weight of image loss')
parser.add_argument('--T_max', default=50, type=int, help='cosine annealing LR scheduler t_max')
parser.add_argument('--name', default='ckpt-new', type=str, help='name to save')
parser.add_argument('--gpus', default='1', type=str, help='id of gpus to use')
# parser.add_argument('--num_gpus', default=2, type=int, help='numbers of gpus to use')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
weight = '_' + str(args.lambda_val) + '_' + str(args.alpha_val) + '_'
ckpt_best = weight + 'best.pt'
ckpt_img_best = weight + 'best_img.pt'
ckpt_final = weight + 'final.pt'


A_PATH = './'
CAGE_PATH = './'


SAVE_PATH = './save/'
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation((-10, 10)),
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1))
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trump_train_dataset, trump_val_dataset, trump_test_dataset = split_dataset(TRUMP_PATH, train_transform=transform_train,
                                                                           test_transform=transform_test)
trump_train_loader = DataLoader(trump_train_dataset, batch_size=args.batch_size, shuffle=True)
trump_val_loader = DataLoader(trump_val_dataset, batch_size=args.batch_size, shuffle=False)
trump_test_loader = DataLoader(trump_test_dataset, batch_size=args.batch_size, shuffle=False)

cage_train_dataset, cage_val_dataset, cage_test_dataset = split_dataset(CAGE_PATH, train_transform=transform_train,
                                                                        test_transform=transform_test)
cage_train_loader = DataLoader(cage_train_dataset, batch_size=args.batch_size, shuffle=True)
cage_val_loader = DataLoader(cage_val_dataset, batch_size=args.batch_size, shuffle=False)
cage_test_loader = DataLoader(cage_test_dataset, batch_size=args.batch_size, shuffle=False)

save_epoch = [0, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
              1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 1999]
if mode == 'train':
    model = DualDefense(args.message_size, in_channels=3, device=device)
    opt = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.lr, weight_decay=1e-5)
    opt_discriminator = optim.Adam(model.adv_model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.T_max, eta_min=1e-8)
    criterion = torch.nn.MSELoss(reduction='sum')
    message_criterion = torch.nn.BCELoss(reduction='sum')
    adversarial_loss = torch.nn.BCELoss(reduction='sum')

    min_val_loss = float('inf')
    min_val_image_loss = float('inf')
    train_loss_plot = []
    val_loss_plot = []
    val_acc_plot = []
    val_df_acc_plot = []
    train_ed_loss = []
    train_adv_loss = []
    val_ed_loss = []
    val_adv_loss = []
    train_gan_loss_plot = []
    val_gan_loss = []
    for i in tqdm(range(args.epoch)):
        lr = None
        for param_group in opt.param_groups:
            lr = param_group["lr"]

        train_image_loss, train_image_df_loss, train_gan_loss, val_image_loss, val_image_df_loss = 0, 0, 0, 0, 0
        train_message_loss, train_message_correct, train_df_message_correct = 0, 0, 0
        val_message_loss, val_message_correct, val_df_message_correct = 0, 0, 0
        train_size, val_size = 0, 0

        model.encoder.train()
        model.decoder.train()
        model.adv_model.train()

        for trump_train_x, cage_train_x, in zip(trump_train_loader, cage_train_loader):
            trump_train_x = trump_train_x.to(device)
            cage_train_x = cage_train_x.to(device)
            trump_valid = Variable(torch.Tensor(len(trump_train_x), 1).to(device).fill_(1.0),
                                   requires_grad=False)
            trump_fake = Variable(torch.Tensor(len(trump_train_x), 1).to(device).fill_(0.0), requires_grad=False)
            cage_valid = Variable(torch.Tensor(len(cage_train_x), 1).to(device).fill_(1.0),
                                  requires_grad=False)
            cage_fake = Variable(torch.Tensor(len(cage_train_x), 1).to(device).fill_(0.0),
                                 requires_grad=False)
            trump_message = torch.randint(0, 2, (trump_train_x.shape[0], args.message_size), dtype=torch.float).to(
                device).detach()
            cage_message = torch.randint(0, 2, (cage_train_x.shape[0], args.message_size), dtype=torch.float).to(
                device).detach()

            # 判别器loss---------------------------------------------------------------
            encoded_trump = model.encode(trump_train_x, trump_message)
            encoded_cage = model.encode(cage_train_x, cage_message)
            opt_discriminator.zero_grad()
            trump_discriminator_loss = adversarial_loss(model.adv(trump_train_x.float().to(device)), trump_valid).to(
                device) + adversarial_loss(model.adv(encoded_trump.float().to(device)), trump_fake.to(device))  # 原始图像
            cage_discriminator_loss = adversarial_loss(model.adv(cage_train_x.float().to(device)),
                                                       cage_valid.to(device)) + adversarial_loss(
                model.adv(encoded_cage.float().to(device)), cage_fake.to(device))  # 原始图像
            discriminator_loss = (trump_discriminator_loss + cage_discriminator_loss) / 2
            train_gan_loss += discriminator_loss.item()

            discriminator_loss.backward()
            opt_discriminator.step()
            # 判别器loss---------------------------------------------------------------

            opt.zero_grad()
            encoded_trump = model.encode(trump_train_x, trump_message)
            encoded_cage = model.encode(cage_train_x, cage_message)
            # original logits AB
            x1_trump, logits_ori_trump, encoded_trump_ori = model.deepfake1(encoded_trump, 'A')
            x2_cage, logits_ori_cage, encoded_cage_ori = model.deepfake1(encoded_cage, 'B')

            # df logits BA
            x1_trump_endf, logits_df_trump, encoded_trump_df = model.deepfake1(encoded_trump, 'B')
            x2_cage_endf, logits_df_cage, encoded_cage_df = model.deepfake1(encoded_cage, 'A')

            # 判别器new-----------------------------------------------------------
            encoded_adversarial_loss = adversarial_loss(model.adv(encoded_trump), trump_valid.to(device)) + \
                                       adversarial_loss(model.adv(encoded_cage), cage_valid.to(device))
            # 判别器new-----------------------------------------------------------
            encoded_loss = (criterion(encoded_trump, trump_train_x) + criterion(encoded_cage, cage_train_x))
            image_adv_logits_loss = (
                    criterion(logits_ori_trump, logits_df_trump) + criterion(logits_ori_cage, logits_df_cage))
            image_loss = 0.15 * encoded_adversarial_loss + 0.8 * encoded_loss + 0.05 * image_adv_logits_loss
            # image_loss = ori * enc_loss + adv * adv_loss
            image_loss *= args.alpha_val
            loss = image_loss

            train_image_loss += encoded_loss.item()
            train_image_df_loss += image_adv_logits_loss.item()

            if i >= args.start_decode:
                # encoded_msg
                encoded_trump_message = model.decode(encoded_trump)
                encoded_cage_message = model.decode(encoded_cage)

                # df_message
                encoded_trump_df_message = model.decode(encoded_trump_df)
                encoded_cage_df_message = model.decode(encoded_cage_df)

                message_loss = message_criterion(encoded_trump_df_message, trump_message) + message_criterion(
                    encoded_cage_df_message, cage_message) + \
                               message_criterion(encoded_trump_message, trump_message) + message_criterion(
                    encoded_cage_message, cage_message)

                train_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + (
                        (encoded_cage_df_message > 0.5) == cage_message).sum().item()
                train_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + (
                        (encoded_cage_message > 0.5) == cage_message).sum().item()

                message_loss *= args.lambda_val
                loss += message_loss
                train_message_loss += message_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            loss.backward()
            opt.step()
            train_size += trump_train_x.shape[0] + cage_train_x.shape[0]

        train_image_loss /= train_size
        train_image_df_loss /= train_size
        train_gan_loss /= train_size

        train_message_loss /= train_size

        train_loss_plot.append(train_message_loss)

        train_ed_loss.append(train_image_loss)
        train_adv_loss.append(train_image_df_loss)
        train_gan_loss_plot.append(train_gan_loss)

        train_df_message_acc = train_df_message_correct / (train_size * args.message_size)
        train_message_acc = train_message_correct / (train_size * args.message_size)

        lr_scheduler.step()

        model.encoder.eval()
        model.decoder.eval()

        with torch.no_grad():
            for (trump_val_x, cage_val_x) in zip(trump_val_loader, cage_val_loader):

                trump_val_x = trump_val_x.to(device)
                cage_val_x = cage_val_x.to(device)

                trump_message = torch.randint(0, 2, (trump_val_x.shape[0], args.message_size), dtype=torch.float).to(
                    device).detach()
                cage_message = torch.randint(0, 2, (cage_val_x.shape[0], args.message_size), dtype=torch.float).to(
                    device).detach()

                # encode msg
                encoded_trump = model.encode(trump_val_x, trump_message)
                encoded_cage = model.encode(cage_val_x, cage_message)


                # ori logits
                x1_ori_trump_val, logits_ori_trump_val, encoded_trump_ori = model.deepfake1(encoded_trump, 'A')
                x2_ori_cage_val, logits2_ori_cage_val, encoded_cage_ori = model.deepfake1(encoded_cage, 'B')
                # df logits
                x1_val_endf_trump, logits1_trump_endf_val, encoded_trump_df = model.deepfake1(encoded_trump, 'B')
                x2_val_endf_cage, logits2_cage_endf_val, encoded_cage_df = model.deepfake1(encoded_cage, 'A')

                encoded_loss = criterion(encoded_trump, trump_val_x) + criterion(encoded_cage, cage_val_x)
                image_adv_logits_loss = (
                        criterion(logits_ori_trump_val, logits1_trump_endf_val) + criterion(logits2_ori_cage_val,
                                                                                            logits2_cage_endf_val))


                image_loss = ori * encoded_loss + adv * image_adv_logits_loss
                image_loss *= args.alpha_val

                if i >= args.start_decode:
                    # decode df msg
                    encoded_trump_df_message = model.decode(encoded_trump_df)
                    encoded_cage_df_message = model.decode(encoded_cage_df)

                    # decode encoded msg
                    encoded_trump_message = model.decode(encoded_trump)
                    encoded_cage_message = model.decode(encoded_cage)

                    # compute message_loss
                    message_loss = message_criterion(encoded_trump_df_message, trump_message) + message_criterion(
                        encoded_cage_df_message, cage_message) + \
                                   message_criterion(encoded_trump_message, trump_message) + message_criterion(
                        encoded_cage_message, cage_message)
                    val_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + (
                            (encoded_cage_df_message > 0.5) == cage_message).sum().item()
                    val_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + (
                            (encoded_cage_message > 0.5) == cage_message).sum().item()
                    message_loss *= args.lambda_val
                    val_message_loss += message_loss.item()

                val_image_loss += encoded_loss.item()
                val_image_df_loss += image_adv_logits_loss.item()
                val_size += trump_val_x.shape[0] + cage_val_x.shape[0]

            val_image_loss /= val_size
            val_image_df_loss /= val_size

            val_message_loss /= val_size
            val_loss = val_image_loss + val_image_df_loss + val_message_loss

            val_message_acc = val_message_correct / (val_size * args.message_size)
            val_df_message_acc = val_df_message_correct / (val_size * args.message_size)

            val_ed_loss.append(val_image_loss)
            val_adv_loss.append(val_image_df_loss)

            val_loss_plot.append(val_message_loss)
            val_acc_plot.append(val_message_acc)
            val_df_acc_plot.append(val_df_message_acc)
            f_print = open(f_print_path, "a")
            print(f"Til: {train_image_loss}, Tdl: {train_image_df_loss}, Tga: {train_image_df_loss},",
                  f"Vil: {val_image_loss}, Vdl:{val_image_df_loss}",
                  f"Tml: {train_message_loss}, Vml: {val_message_loss}",
                  f"Ta: {train_message_acc}, TFa:{train_df_message_acc}",
                  f"Va: {val_message_acc}, VFa: {val_df_message_acc}", file=f_print)
            f_print.close()

            path = os.path.join(SAVE_PATH, args.name)
            if min_val_loss > val_loss and i > args.start_decode:
                print(f'(img+msg)model saved at epoch {i}')
                min_val_loss = val_loss
                if not os.path.isdir(path):
                    os.makedirs(path)
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": i
                }, os.path.join(path, date + ckpt_best))

            if min_val_image_loss > val_image_loss and i > args.start_decode:
                print(f'(img)model saved at epoch {i}')
                min_val_image_loss = val_image_loss
                if not os.path.isdir(path):
                    os.makedirs(path)
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": i
                }, os.path.join(path, date + ckpt_img_best))

            if i == args.epoch - 1:
                torch.save({
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": i
                }, os.path.join(path, date + ckpt_final))

        if i == args.epoch - 1:
            plt.plot(np.arange(len(train_loss_plot)), train_loss_plot, label="train loss")
            plt.plot(np.arange(len(val_loss_plot)), val_loss_plot, label="valid loss")
            plt.plot(np.arange(len(val_acc_plot)), val_acc_plot, label="valid acc")
            plt.plot(np.arange(len(val_df_acc_plot)), val_df_acc_plot, label="valid df acc")
            plt.legend()
            plt.savefig("./save/loss/loss_acc_" + date + ".png")
            plt.close()

            plt.plot(np.arange(len(train_ed_loss)), train_ed_loss, label="train_ed_loss")
            plt.plot(np.arange(len(train_adv_loss)), train_adv_loss, label="train_adv_loss")
            plt.plot(np.arange(len(val_ed_loss)), val_ed_loss, label="val_ed_loss")
            plt.plot(np.arange(len(val_adv_loss)), val_adv_loss, label="val_adv_loss")
            plt.savefig("./save/loss/loss_img_" + date + ".png")
            plt.legend()
            plt.close()

            plt.plot(np.arange(len(train_gan_loss_plot)), train_gan_loss_plot, label="train_gan_loss_plot")
            plt.savefig("./save/loss/loss_gan_" + date + ".png")
            plt.legend()
            plt.close()

            plt.plot(np.arange(len(train_ed_loss)), train_ed_loss, label="train_ed_loss")
            plt.plot(np.arange(len(train_adv_loss)), train_adv_loss, label="train_adv_loss")
            plt.plot(np.arange(len(train_message_loss)), train_message_loss, label="train_message_loss")
            plt.savefig("./save/loss/loss_enc_msg" + date + ".png")
            plt.legend()
            plt.close()

# ----------------test----------------
PATH_1 = os.path.join(SAVE_PATH, args.name)
LOAD_PATH = os.path.join(PATH_1, date + ckpt_final)
# LOAD_PATH = '/faketagger/FT-new/result/save/ckpt-new/0510_3_32_1_01_final.pt'  # gan-best
model = FaceTagger(args.message_size, in_channels=3, device=device)
model.encoder.load_state_dict(torch.load(LOAD_PATH)['encoder'])
model.decoder.load_state_dict(torch.load(LOAD_PATH)['decoder'])

model.encoder.eval()
model.decoder.eval()

test_message_correct = 0
test_df_message_correct = 0
test_size = 0

trump_psnr_sum = 0
cage_psnr_sum = 0
trump_ssim_sum = 0
cage_ssim_sum = 0
trump_df_psnr_sum = 0
cage_df_psnr_sum = 0
trump_df_ssim_sum = 0
cage_df_ssim_sum = 0
i = 0
trump_suc = 0
cage_suc = 0
trump_suc_ori = 0
cage_suc_ori = 0
trump_suc_real = 0
cage_suc_real = 0
trump_attack_success = 0
cage_attack_success = 0

with torch.no_grad():
    for (trump_test_x, cage_test_x) in zip(trump_test_loader, cage_test_loader):
        trump_test_x = trump_test_x.to(device)
        cage_test_x = cage_test_x.to(device)
        print(len(trump_test_x), len(cage_test_x))

        trump_message = torch.randint(0, 2, (trump_test_x.shape[0], args.message_size), dtype=torch.float).to(
            device).detach()
        cage_message = torch.randint(0, 2, (cage_test_x.shape[0], args.message_size), dtype=torch.float).to(
            device).detach()

        encoded_trump = model.encode(trump_test_x, trump_message)
        encoded_cage = model.encode(cage_test_x, cage_message)

        x1, _, encoded_trump_df = model.deepfake1(encoded_trump, 'B')
        x2, _, encoded_cage_df = model.deepfake1(encoded_cage, 'A')

        x3, _, trump_df = model.deepfake1(trump_test_x, 'B')
        x4, _, cage_df = model.deepfake1(cage_test_x, 'A')


        for k in range(len(trump_test_x)):
            trump_psnr_sum += calculate_psnr((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                             (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            trump_ssim_sum += calculate_ssim((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                             (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
        for k in range(len(cage_test_x)):
            cage_psnr_sum += calculate_psnr((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                            (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            cage_ssim_sum += calculate_ssim((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                            (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
        for k in range(len(trump_test_x)):
            trump_df_psnr_sum += calculate_psnr((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            trump_df_ssim_sum += calculate_ssim((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
        for k in range(len(cage_test_x)):
            cage_df_psnr_sum += calculate_psnr((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                               (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            cage_df_ssim_sum += calculate_ssim((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                               (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())

        for k in range(len(trump_df)):
            attack_l2 = torch.nn.functional.mse_loss(trump_df[k], encoded_trump_df[k], reduction='sum')
            trump_attack_success = trump_attack_success + 1 if attack_l2 >= 0.05 else trump_attack_success
        for k in range(len(cage_df)):
            attack_l2 = torch.nn.functional.mse_loss(cage_df[k], encoded_cage_df[k], reduction='sum')
            cage_attack_success = cage_attack_success + 1 if attack_l2 >= 0.05 else cage_attack_success

        i += 1
        encoded_trump_df_message = model.decode(encoded_trump_df)
        encoded_cage_df_message = model.decode(encoded_cage_df)

        encoded_trump_message = model.decode(encoded_trump)
        encoded_cage_message = model.decode(encoded_cage)

        test_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + (
                (encoded_cage_df_message > 0.5) == cage_message).sum().item()
        test_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + (
                (encoded_cage_message > 0.5) == cage_message).sum().item()

        trump_test_x_160 = trump_test_x.clone()
        img_probs = resnet(trump_test_x_160)
        img_probs_softmax = torch.softmax(img_probs, dim=1)
        label_tensor = torch.argmax(img_probs_softmax, dim=1)
        for i in range(len(label_tensor)):
            if label_tensor[i] == 38:
                trump_suc_real = trump_suc_real + 1
        cage_test_x_160 = cage_test_x.clone()
        img_probs = resnet(cage_test_x_160)
        img_probs_softmax = torch.softmax(img_probs, dim=1)
        label_tensor = torch.argmax(img_probs_softmax, dim=1)
        for i in range(len(label_tensor)):
            if label_tensor[i] == 397:
                cage_suc_real = cage_suc_real + 1

        encoded_trump_df_160 = encoded_trump_df.clone()
        img_probs = resnet(encoded_trump_df_160)  # n000004:2   换脸后为n000005:3 换脸成功=3
        img_probs_softmax = torch.softmax(img_probs, dim=1)
        label_tensor = torch.argmax(img_probs_softmax, dim=1)
        for i in range(len(label_tensor)):
            if label_tensor[i] != 397:
                trump_suc = trump_suc + 1
        encoded_cage_df_160 = encoded_cage_df.clone()
        img_probs = resnet(encoded_cage_df_160)  # n000005:3    换脸后为n000004:2 换脸成功=2
        img_probs_softmax = torch.softmax(img_probs, dim=1)
        label_tensor = torch.argmax(img_probs_softmax, dim=1)
        for i in range(len(label_tensor)):
            if label_tensor[i] != 38:
                cage_suc = cage_suc + 1

        trump_df_160 = trump_df.clone()
        img_probs = resnet(trump_df_160)  # n000004:2   换脸后为n000005:3 换脸成功=3
        img_probs_softmax = torch.softmax(img_probs, dim=1)
        label_tensor = torch.argmax(img_probs_softmax, dim=1)
        for i in range(len(label_tensor)):
            if label_tensor[i] == 397:
                trump_suc_ori = trump_suc_ori + 1
        cage_df_160 = cage_df.clone()
        img_probs = resnet(cage_df_160)  # n000005:3    换脸后为n000004:2 换脸成功=2
        img_probs_softmax = torch.softmax(img_probs, dim=1)
        label_tensor = torch.argmax(img_probs_softmax, dim=1)
        for i in range(len(label_tensor)):
            if label_tensor[i] == 38:
                cage_suc_ori = cage_suc_ori + 1

        test_size += trump_test_x.shape[0] + cage_test_x.shape[0]

    test_message_acc = test_message_correct / (test_size * args.message_size)
    test_df_message_acc = test_df_message_correct / (test_size * args.message_size)
    trump_psnr_avg = trump_psnr_sum / len(trump_test_loader.dataset)
    cage_psnr_avg = cage_psnr_sum / len(cage_test_loader.dataset)
    trump_ssim_avg = trump_ssim_sum / len(trump_test_loader.dataset)
    cage_ssim_avg = cage_ssim_sum / len(cage_test_loader.dataset)

    trump_df_psnr_avg = trump_df_psnr_sum / len(trump_test_loader.dataset)
    cage_df_psnr_avg = cage_df_psnr_sum / len(cage_test_loader.dataset)
    trump_df_ssim_avg = trump_df_ssim_sum / len(trump_test_loader.dataset)
    cage_df_ssim_avg = cage_df_ssim_sum / len(cage_test_loader.dataset)

    trump_suc_p = trump_suc / len(trump_test_loader.dataset)
    cage_suc_p = cage_suc / len(cage_test_loader.dataset)
    trump_suc_p_ori = trump_suc_ori / len(trump_test_loader.dataset)
    cage_suc_p_ori = cage_suc_ori / len(cage_test_loader.dataset)
    trump_suc_p_real = trump_suc_real / len(trump_test_loader.dataset)
    cage_suc_p_real = cage_suc_real / len(cage_test_loader.dataset)

    trump_attack_success_p = trump_attack_success / len(trump_test_loader.dataset)
    cage_attack_success_p = cage_attack_success / len(cage_test_loader.dataset)

    f_print = open(f_print_path, "a")
    print(f"Trump PSNR : {trump_psnr_avg}, Cage PSNR : {cage_psnr_avg}\n",
          f"Trump SSIM : {trump_ssim_avg}, Cage SSIM : {cage_ssim_avg}\n",
          f"Trump_df PSNR : {trump_df_psnr_avg}, Cage_df PSNR : {cage_df_psnr_avg}\n",
          f"Trump_df SSIM : {trump_df_ssim_avg}, Cage_df SSIM : {cage_df_ssim_avg}\n",
          f"Test encoded msg accuracy : {test_message_acc}, Test DF msg accuracy : {test_df_message_acc}\n",
          f"enADFtoX---trump_suc_p : {trump_suc_p}, cage_suc_p : {cage_suc_p}\n",
          f"ADFtoB---trump_suc_p_ori : {trump_suc_p_ori}, cage_suc_p_ori : {cage_suc_p_ori}\n",
          f"AtoA---trump_suc_p_real : {trump_suc_p_real}, cage_suc_p_real : {cage_suc_p_real}\n",
          f"l2---trump_suc_l2 : {trump_attack_success_p}, cage_suc_l2 : {cage_attack_success_p}\n",
          file=f_print)
    f_print.close()
