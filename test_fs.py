import math
import torch
import os
import random
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
from DiffJPEG import DiffJPEG
from dataset160 import CustomDataset, split_dataset
from DualDefense_gan_fs160 import DualDefense as DualDefense_160
# from DualDefense_gan_fs160_no_attention import DualDefense as DualDefense_160
# from DualDefense_gan_fs160_no_all import DualDefense as DualDefense_160
from facenet_pytorch import InceptionResnetV1
import lpips


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


def salt_noise(SNR, img_tensor):
    mask_noise = np.random.choice((0, 1, 2), size=(1, HW, HW), p=[SNR, (1 - SNR) / 2, (1 - SNR) / 2])
    mask_noise = np.repeat(mask_noise, 3, axis=0)
    # mask_noise = np.expand_dims(mask_noise, 0).repeat(len(img_tensor), axis=0)
    img_np = img_tensor.cpu().numpy()
    for img_np_i in img_np:
        img_np_i[mask_noise == 1] = 255
        img_np_i[mask_noise == 2] = 0
    return torch.from_numpy(img_np).to(device)


def sp_noise(img_tensor, prob):
    # 添加椒盐噪声
    # prob:噪声比例
    # output = np.zeros(image.shape, np.uint8)
    # prob_1 = 1 - prob
    p_1 = prob / 2
    p_2 = 1 - p_1
    l_l = int(img_tensor.shape[0])
    c = int(img_tensor.shape[1])
    h_l = int(img_tensor.shape[2])
    w_l = int(img_tensor.shape[3])
    for num in range(l_l):
        for h in range(h_l):
            for w in range(w_l):
                rdn = random.random()
                if rdn < p_1:
                    img_tensor[num, :, h, w] = 0
                elif rdn > p_2:
                    img_tensor[num, :, h, w] = 1  # 255
    return img_tensor


def gaussian_noise(image, mean=0, var=0.001):
    # 添加高斯噪声
    # mean : 均值
    # var : 方差
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def sharpen(image):
    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    out = cv2.filter2D(image, -1, kernel=kernel)
    return out


def MedianFilter(img_tensor, k_size):
    img_np = img_tensor.permute(0, 2, 3, 1).cpu().numpy()
    for idx in range(len(img_np)):
        img = cv2.medianBlur(img_np[idx], k_size)
        img = img.transpose(2, 0, 1)
        img_tensor[idx] = torch.from_numpy(img).data
    return img_tensor


set_seed(0)
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--message_size', default=15, type=int, help='msg size')
parser.add_argument('--gpus', default='0', type=str, help='id of gpus to use')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
loss_vgg = lpips.LPIPS(net='vgg').to(device)  # closer to "traditional" perceptual loss
SAVE_PATH = ''  
transform_test = transforms.Compose([transforms.ToTensor()])


def test_process_vgg():
    test_message_correct, test_df_message_correct, test_size = 0, 0, 0
    trump_psnr_sum, cage_psnr_sum, trump_ssim_sum, cage_ssim_sum = 0, 0, 0, 0
    trump_df_psnr_sum, cage_df_psnr_sum, trump_df_ssim_sum, cage_df_ssim_sum = 0, 0, 0, 0
    trump_suc, cage_suc, trump_suc_ori, cage_suc_ori = 0, 0, 0, 0
    trump_suc_real, cage_suc_real, trump_attack_success, cage_attack_success = 0, 0, 0, 0
    attack_l1_trump_sum, attack_l1_cage_sum, attack_l2_trump_sum, attack_l2_cage_sum = 0, 0, 0, 0
    trump_true_size, cage_true_size = 0, 0
    lpips_alex_trump, lpips_alex_cage, lpips_vgg_trump, lpips_vgg_cage = 0, 0, 0, 0
    cage_dist, trump_dist = 0, 0
    img_i = 0
    with torch.no_grad():
        for (trump_test_x, cage_test_x) in zip(trump_test_loader, cage_test_loader):
            trump_test_x = trump_test_x.to(device)
            cage_test_x = cage_test_x.to(device)
            print(len(trump_test_x), len(cage_test_x))
            trump_true_size = trump_true_size + len(trump_test_x)
            cage_true_size = cage_true_size + len(cage_test_x)
            trump_message = torch.randint(0, 2, (trump_test_x.shape[0], args.message_size), dtype=torch.float).to(
                device).detach()
            cage_message = torch.randint(0, 2, (cage_test_x.shape[0], args.message_size), dtype=torch.float).to(
                device).detach()
            encoded_trump = model.encode(trump_test_x, trump_message)
            encoded_cage = model.encode(cage_test_x, cage_message)

            if attack_way == 0:
                encoded_trump = jpeg(encoded_trump).to(device)
                encoded_cage = jpeg(encoded_cage).to(device)
            elif attack_way == 1:
                transform_resize = transforms.Compose([
                    transforms.Resize((new_hw, new_hw)),
                    transforms.Resize((HW, HW)),
                ])
                for idx in range(len(encoded_trump)):
                    encoded_trump[idx] = transform_resize(encoded_trump[idx])
                for idx in range(len(encoded_cage)):
                    encoded_cage[idx] = transform_resize(encoded_cage[idx])
            elif attack_way == 2:
                print(ker, cir_len)
                transform_blur = transforms.Compose([
                    transforms.GaussianBlur(ker, cir_len)
                ])
                for idx in range(len(encoded_trump)):
                    encoded_trump[idx] = transform_blur(encoded_trump[idx])
                for idx in range(len(encoded_cage)):
                    encoded_cage[idx] = transform_blur(encoded_cage[idx])
            elif attack_way == 3:
                print(noise_v)
                for idx in range(len(encoded_trump)):
                    mask = np.random.normal(0, noise_v, (3, HW, HW))
                    mask = torch.from_numpy(mask).to(device)
                    encoded_trump[idx] = encoded_trump[idx] + mask
                for idx in range(len(encoded_cage)):
                    mask = np.random.normal(0, noise_v, (3, HW, HW))
                    mask = torch.from_numpy(mask).to(device)
                    encoded_cage[idx] = encoded_cage[idx] + mask
            elif attack_way == 4:
                print(salt_SNR)
                encoded_trump = sp_noise(encoded_trump, salt_SNR)
                encoded_cage = sp_noise(encoded_cage, salt_SNR)
            _, _, encoded_trump_df = model.deepfake1(encoded_trump, 'B')
            _, _, encoded_cage_df = model.deepfake1(encoded_cage, 'A')
            _, _, trump_df = model.deepfake1(trump_test_x, 'B')
            _, _, cage_df = model.deepfake1(cage_test_x, 'A')

            for k in range(len(trump_test_x)):
                cv2.imwrite(os.path.join(
                    '' + date + '/real/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode_fake/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/fake/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_psnr_sum += calculate_psnr((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_ssim_sum += calculate_ssim((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_psnr_sum += calculate_psnr((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_ssim_sum += calculate_ssim((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            for k in range(len(cage_test_x)):
                cv2.imwrite(os.path.join(
                    '' + date + '/real/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode_fake/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/fake/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_psnr_sum += calculate_psnr((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_ssim_sum += calculate_ssim((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_psnr_sum += calculate_psnr((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                   (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_ssim_sum += calculate_ssim((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                   (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            for k in range(len(trump_df)):
                mask = abs(trump_df[k] - encoded_trump_df[k])
                mask = mask[0, :, :] + mask[1, :, :] + mask[2, :, :]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if (((trump_df[k] * mask - encoded_trump_df[k] * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                    trump_dist += 1

                attack_l1 = torch.nn.functional.l1_loss(trump_df[k], encoded_trump_df[k])
                attack_l2 = torch.nn.functional.mse_loss(trump_df[k], encoded_trump_df[k])
                attack_l1_trump_sum = attack_l1_trump_sum + attack_l1
                attack_l2_trump_sum = attack_l2_trump_sum + attack_l2
                lpips_alex_trump = lpips_alex_trump + loss_alex(trump_df[k].unsqueeze(0),
                                                                encoded_trump_df[k].unsqueeze(0))
                lpips_vgg_trump = lpips_vgg_trump + loss_vgg(trump_df[k].unsqueeze(0), encoded_trump_df[k].unsqueeze(0))
                trump_attack_success = trump_attack_success + 1 if attack_l2 >= 0.05 else trump_attack_success
            for k in range(len(cage_df)):
                mask = abs(cage_df[k] - encoded_cage_df[k])
                mask = mask[0, :, :] + mask[1, :, :] + mask[2, :, :]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if (((cage_df[k] * mask - encoded_cage_df[k] * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                    cage_dist += 1

                attack_l1 = torch.nn.functional.l1_loss(cage_df[k], encoded_cage_df[k])
                attack_l2 = torch.nn.functional.mse_loss(cage_df[k], encoded_cage_df[k])
                attack_l1_cage_sum = attack_l1_cage_sum + attack_l1
                attack_l2_cage_sum = attack_l2_cage_sum + attack_l2
                lpips_alex_cage = lpips_alex_cage + loss_alex(cage_df[k].unsqueeze(0), encoded_cage_df[k].unsqueeze(0))
                lpips_vgg_cage = lpips_vgg_cage + loss_vgg(cage_df[k].unsqueeze(0), encoded_cage_df[k].unsqueeze(0))
                cage_attack_success = cage_attack_success + 1 if attack_l2 >= 0.05 else cage_attack_success

            img_i += 1
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
                if label_tensor[i] == min_label:
                    trump_suc_real = trump_suc_real + 1
            cage_test_x_160 = cage_test_x.clone()
            img_probs = resnet(cage_test_x_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == max_label:
                    cage_suc_real = cage_suc_real + 1
            encoded_trump_df_160 = encoded_trump_df.clone()
            img_probs = resnet(encoded_trump_df_160)  # n000004:2   换脸后为n000005:3 换脸成功=3
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] != max_label:
                    trump_suc = trump_suc + 1
            encoded_cage_df_160 = encoded_cage_df.clone()
            img_probs = resnet(encoded_cage_df_160)  # n000005:3    换脸后为n000004:2 换脸成功=2
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] != min_label:
                    cage_suc = cage_suc + 1
            trump_df_160 = trump_df.clone()
            img_probs = resnet(trump_df_160)  # n000004:2   换脸后为n000005:3 换脸成功=3
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == max_label:
                    trump_suc_ori = trump_suc_ori + 1
            cage_df_160 = cage_df.clone()
            img_probs = resnet(cage_df_160)  # n000005:3    换脸后为n000004:2 换脸成功=2
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == min_label:
                    cage_suc_ori = cage_suc_ori + 1

            test_size += trump_test_x.shape[0] + cage_test_x.shape[0]

        test_message_acc = test_message_correct / (test_size * args.message_size)
        test_df_message_acc = test_df_message_correct / (test_size * args.message_size)
        trump_psnr_avg = trump_psnr_sum / trump_true_size
        cage_psnr_avg = cage_psnr_sum / cage_true_size
        trump_ssim_avg = trump_ssim_sum / trump_true_size
        cage_ssim_avg = cage_ssim_sum / cage_true_size
        trump_df_psnr_avg = trump_df_psnr_sum / trump_true_size
        cage_df_psnr_avg = cage_df_psnr_sum / cage_true_size
        trump_df_ssim_avg = trump_df_ssim_sum / trump_true_size
        cage_df_ssim_avg = cage_df_ssim_sum / cage_true_size
        trump_suc_p = trump_suc / trump_true_size
        cage_suc_p = cage_suc / cage_true_size
        trump_suc_p_ori = trump_suc_ori / trump_true_size
        cage_suc_p_ori = cage_suc_ori / cage_true_size
        trump_suc_p_real = trump_suc_real / trump_true_size
        cage_suc_p_real = cage_suc_real / cage_true_size
        trump_attack_success_p = trump_attack_success / trump_true_size
        cage_attack_success_p = cage_attack_success / cage_true_size
        attack_l1_trump_res = attack_l1_trump_sum / trump_true_size
        attack_l1_cage_res = attack_l1_cage_sum / cage_true_size
        attack_l2_trump_res = attack_l2_trump_sum / trump_true_size
        attack_l2_cage_res = attack_l2_cage_sum / cage_true_size
        lpips_alex_trump_res = lpips_alex_trump / trump_true_size
        lpips_alex_cage_res = lpips_alex_cage / cage_true_size
        lpips_vgg_trump_res = lpips_vgg_trump / trump_true_size
        lpips_vgg_cage_res = lpips_vgg_cage / cage_true_size
        trump_dist_mask = trump_dist / trump_true_size
        cage_dist_mask = cage_dist / cage_true_size
        print(trump_true_size, cage_true_size)
        f_print_vgg = open(f_print_path, "a")
        print(f"Trump PSNR : {trump_psnr_avg}, Cage PSNR : {cage_psnr_avg}\n",
              f"Trump SSIM : {trump_ssim_avg}, Cage SSIM : {cage_ssim_avg}\n",
              f"Trump_df PSNR : {trump_df_psnr_avg}, Cage_df PSNR : {cage_df_psnr_avg}\n",
              f"Trump_df SSIM : {trump_df_ssim_avg}, Cage_df SSIM : {cage_df_ssim_avg}\n",
              f"Test encoded msg accuracy : {test_message_acc}, Test DF msg accuracy : {test_df_message_acc}\n",
              f"enADFtoX---trump : {trump_suc_p}, cage : {cage_suc_p}\n",
              f"ADFtoB---trump : {trump_suc_p_ori}, cage : {cage_suc_p_ori}\n",
              f"AtoA---trump : {trump_suc_p_real}, cage : {cage_suc_p_real}\n",
              f"l2 suc---trump : {trump_attack_success_p}, cage : {cage_attack_success_p}\n",
              f"l1---trump : {attack_l1_trump_res}, cage : {attack_l1_cage_res}\n",
              f"l2---trump : {attack_l2_trump_res}, cage : {attack_l2_cage_res}\n",
              f"lpips_alex---trump : {lpips_alex_trump_res.item()}, cage : {lpips_alex_cage_res.item()}\n",
              f"lpips_vgg---trump : {lpips_vgg_trump_res.item()}, cage : {lpips_vgg_cage_res.item()}\n",
              f"mask---trump : {trump_dist_mask}, cage : {cage_dist_mask}\n",
              file=f_print_vgg)
        f_print_vgg.close()


def test_process_casia():
    test_message_correct, test_df_message_correct, test_size = 0, 0, 0
    trump_psnr_sum, cage_psnr_sum, trump_ssim_sum, cage_ssim_sum = 0, 0, 0, 0
    trump_df_psnr_sum, cage_df_psnr_sum, trump_df_ssim_sum, cage_df_ssim_sum = 0, 0, 0, 0
    trump_suc, cage_suc, trump_suc_ori, cage_suc_ori = 0, 0, 0, 0
    trump_suc_real, cage_suc_real, trump_attack_success, cage_attack_success = 0, 0, 0, 0
    attack_l1_trump_sum, attack_l1_cage_sum, attack_l2_trump_sum, attack_l2_cage_sum = 0, 0, 0, 0
    trump_true_size, cage_true_size = 0, 0
    lpips_alex_trump, lpips_alex_cage, lpips_vgg_trump, lpips_vgg_cage = 0, 0, 0, 0
    cage_dist, trump_dist = 0, 0
    img_i = 0
    with torch.no_grad():
        for (trump_test_x, cage_test_x) in zip(trump_test_loader, cage_test_loader):
            trump_test_x = trump_test_x.to(device)
            cage_test_x = cage_test_x.to(device)
            print(len(trump_test_x), len(cage_test_x))
            trump_true_size = trump_true_size + len(trump_test_x)
            cage_true_size = cage_true_size + len(cage_test_x)
            trump_message = torch.randint(0, 2, (trump_test_x.shape[0], args.message_size), dtype=torch.float).to(
                device).detach()
            cage_message = torch.randint(0, 2, (cage_test_x.shape[0], args.message_size), dtype=torch.float).to(
                device).detach()
            encoded_trump = model.encode(trump_test_x, trump_message)
            encoded_cage = model.encode(cage_test_x, cage_message)

            if attack_way == 0:
                encoded_trump = jpeg(encoded_trump).to(device)
                encoded_cage = jpeg(encoded_cage).to(device)
            elif attack_way == 1:
                transform_resize = transforms.Compose([
                    transforms.Resize((new_hw, new_hw)),
                    transforms.Resize((HW, HW)),
                ])
                for idx in range(len(encoded_trump)):
                    encoded_trump[idx] = transform_resize(encoded_trump[idx])
                for idx in range(len(encoded_cage)):
                    encoded_cage[idx] = transform_resize(encoded_cage[idx])
            elif attack_way == 2:
                print(ker, cir_len)
                transform_blur = transforms.Compose([
                    transforms.GaussianBlur(ker, cir_len)
                ])
                for idx in range(len(encoded_trump)):
                    encoded_trump[idx] = transform_blur(encoded_trump[idx])
                for idx in range(len(encoded_cage)):
                    encoded_cage[idx] = transform_blur(encoded_cage[idx])
            elif attack_way == 3:
                print(noise_v)
                for idx in range(len(encoded_trump)):
                    mask = np.random.normal(0, noise_v, (3, HW, HW))
                    mask = torch.from_numpy(mask).to(device)
                    encoded_trump[idx] = encoded_trump[idx] + mask
                for idx in range(len(encoded_cage)):
                    mask = np.random.normal(0, noise_v, (3, HW, HW))
                    mask = torch.from_numpy(mask).to(device)
                    encoded_cage[idx] = encoded_cage[idx] + mask
            elif attack_way == 4:
                print(salt_SNR)
                encoded_trump = sp_noise(encoded_trump, salt_SNR)
                encoded_cage = sp_noise(encoded_cage, salt_SNR)

            '''
            resize_160_64 = transforms.Compose([transforms.Resize((64, 64))])
            _, encoded_trump_df = model.deepfake1(resize_160_64(encoded_trump), 'B')
            _, encoded_cage_df = model.deepfake1(resize_160_64(encoded_cage), 'A')
            _, trump_df = model.deepfake1(resize_160_64(trump_test_x), 'B')
            _, cage_df = model.deepfake1(resize_160_64(cage_test_x), 'A')
            resize_64_160 = transforms.Compose([transforms.Resize((160, 160))])
            encoded_trump_df = resize_64_160(encoded_trump_df)
            encoded_cage_df = resize_64_160(encoded_cage_df)
            trump_df = resize_64_160(trump_df)
            cage_df = resize_64_160(cage_df)
            '''
            _, _, encoded_trump_df = model.deepfake1(encoded_trump, 'B')
            _, _, encoded_cage_df = model.deepfake1(encoded_cage, 'A')
            _, _, trump_df = model.deepfake1(trump_test_x, 'B')
            _, _, cage_df = model.deepfake1(cage_test_x, 'A')

            for k in range(len(trump_test_x)):
                cv2.imwrite(os.path.join(
                    '' + date + '/real/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode_fake/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/fake/' + str(img_i) + '_' + str(k) + '_A.png'),
                    (trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_psnr_sum += calculate_psnr((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_ssim_sum += calculate_ssim((trump_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                 (encoded_trump[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_psnr_sum += calculate_psnr((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                trump_df_ssim_sum += calculate_ssim((trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                    (encoded_trump_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            for k in range(len(cage_test_x)):
                cv2.imwrite(os.path.join(
                    '' + date + '/real/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/encode_fake/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cv2.imwrite(os.path.join(
                    '' + date + '/fake/' + str(img_i) + '_' + str(k) + '_B.png'),
                    (cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_psnr_sum += calculate_psnr((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_ssim_sum += calculate_ssim((cage_test_x[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                (encoded_cage[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_psnr_sum += calculate_psnr((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                   (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
                cage_df_ssim_sum += calculate_ssim((cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy(),
                                                   (encoded_cage_df[k] * 255).permute(1, 2, 0).detach().cpu().numpy())
            for k in range(len(trump_df)):
                mask = abs(trump_df[k] - encoded_trump_df[k])
                mask = mask[0, :, :] + mask[1, :, :] + mask[2, :, :]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if (((trump_df[k] * mask - encoded_trump_df[k] * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                    trump_dist += 1

                attack_l1 = torch.nn.functional.l1_loss(trump_df[k], encoded_trump_df[k])
                attack_l2 = torch.nn.functional.mse_loss(trump_df[k], encoded_trump_df[k])
                attack_l1_trump_sum = attack_l1_trump_sum + attack_l1
                attack_l2_trump_sum = attack_l2_trump_sum + attack_l2
                lpips_alex_trump = lpips_alex_trump + loss_alex(trump_df[k].unsqueeze(0),
                                                                encoded_trump_df[k].unsqueeze(0))
                lpips_vgg_trump = lpips_vgg_trump + loss_vgg(trump_df[k].unsqueeze(0), encoded_trump_df[k].unsqueeze(0))
                trump_attack_success = trump_attack_success + 1 if attack_l2 >= 0.05 else trump_attack_success
            for k in range(len(cage_df)):
                mask = abs(cage_df[k] - encoded_cage_df[k])
                mask = mask[0, :, :] + mask[1, :, :] + mask[2, :, :]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if (((cage_df[k] * mask - encoded_cage_df[k] * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                    cage_dist += 1

                attack_l1 = torch.nn.functional.l1_loss(cage_df[k], encoded_cage_df[k])
                attack_l2 = torch.nn.functional.mse_loss(cage_df[k], encoded_cage_df[k])
                attack_l1_cage_sum = attack_l1_cage_sum + attack_l1
                attack_l2_cage_sum = attack_l2_cage_sum + attack_l2
                lpips_alex_cage = lpips_alex_cage + loss_alex(cage_df[k].unsqueeze(0), encoded_cage_df[k].unsqueeze(0))
                lpips_vgg_cage = lpips_vgg_cage + loss_vgg(cage_df[k].unsqueeze(0), encoded_cage_df[k].unsqueeze(0))
                cage_attack_success = cage_attack_success + 1 if attack_l2 >= 0.05 else cage_attack_success

            encoded_trump_df_message = model.decode(encoded_trump_df)
            encoded_cage_df_message = model.decode(encoded_cage_df)

            encoded_trump_message = model.decode(encoded_trump)
            encoded_cage_message = model.decode(encoded_cage)

            test_df_message_correct += ((encoded_trump_df_message > 0.5) == trump_message).sum().item() + (
                    (encoded_cage_df_message > 0.5) == cage_message).sum().item()
            test_message_correct += ((encoded_trump_message > 0.5) == trump_message).sum().item() + (
                    (encoded_cage_message > 0.5) == cage_message).sum().item()

            img_i += 1

            trump_test_x_160 = trump_test_x.clone()
            img_probs = resnet(trump_test_x_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == min_label:
                    trump_suc_real = trump_suc_real + 1
            cage_test_x_160 = cage_test_x.clone()
            img_probs = resnet(cage_test_x_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == max_label:
                    cage_suc_real = cage_suc_real + 1
            encoded_trump_df_160 = encoded_trump_df.clone()
            img_probs = resnet(encoded_trump_df_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] != max_label:
                    trump_suc = trump_suc + 1
            encoded_cage_df_160 = encoded_cage_df.clone()
            img_probs = resnet(encoded_cage_df_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] != min_label:
                    cage_suc = cage_suc + 1
            trump_df_160 = trump_df.clone()
            img_probs = resnet(trump_df_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == max_label:
                    trump_suc_ori = trump_suc_ori + 1
            cage_df_160 = cage_df.clone()
            img_probs = resnet(cage_df_160)
            img_probs_softmax = torch.softmax(img_probs, dim=1)
            label_tensor = torch.argmax(img_probs_softmax, dim=1)
            for i in range(len(label_tensor)):
                if label_tensor[i] == min_label:
                    cage_suc_ori = cage_suc_ori + 1

            test_size += trump_test_x.shape[0] + cage_test_x.shape[0]

        test_message_acc = test_message_correct / (test_size * args.message_size)
        test_df_message_acc = test_df_message_correct / (test_size * args.message_size)
        trump_psnr_avg = trump_psnr_sum / trump_true_size
        cage_psnr_avg = cage_psnr_sum / cage_true_size
        trump_ssim_avg = trump_ssim_sum / trump_true_size
        cage_ssim_avg = cage_ssim_sum / cage_true_size

        trump_df_psnr_avg = trump_df_psnr_sum / trump_true_size
        cage_df_psnr_avg = cage_df_psnr_sum / cage_true_size
        trump_df_ssim_avg = trump_df_ssim_sum / trump_true_size
        cage_df_ssim_avg = cage_df_ssim_sum / cage_true_size

        trump_suc_p = trump_suc / trump_true_size
        cage_suc_p = cage_suc / cage_true_size
        trump_suc_p_ori = trump_suc_ori / trump_true_size
        cage_suc_p_ori = cage_suc_ori / cage_true_size
        trump_suc_p_real = trump_suc_real / trump_true_size
        cage_suc_p_real = cage_suc_real / cage_true_size

        trump_attack_success_p = trump_attack_success / trump_true_size
        cage_attack_success_p = cage_attack_success / cage_true_size

        attack_l1_trump_res = attack_l1_trump_sum / trump_true_size
        attack_l1_cage_res = attack_l1_cage_sum / cage_true_size
        attack_l2_trump_res = attack_l2_trump_sum / trump_true_size
        attack_l2_cage_res = attack_l2_cage_sum / cage_true_size

        lpips_alex_trump_res = lpips_alex_trump / trump_true_size
        lpips_alex_cage_res = lpips_alex_cage / cage_true_size
        lpips_vgg_trump_res = lpips_vgg_trump / trump_true_size
        lpips_vgg_cage_res = lpips_vgg_cage / cage_true_size

        trump_dist_mask = trump_dist / trump_true_size
        cage_dist_mask = cage_dist / cage_true_size

        print(trump_true_size, cage_true_size)
        f_print_casia = open(f_print_path, "a")
        print(f"Trump PSNR : {trump_psnr_avg}, Cage PSNR : {cage_psnr_avg}\n",
              f"Trump SSIM : {trump_ssim_avg}, Cage SSIM : {cage_ssim_avg}\n",
              f"Trump_df PSNR : {trump_df_psnr_avg}, Cage_df PSNR : {cage_df_psnr_avg}\n",
              f"Trump_df SSIM : {trump_df_ssim_avg}, Cage_df SSIM : {cage_df_ssim_avg}\n",
              f"Test encoded msg accuracy : {test_message_acc}, Test DF msg accuracy : {test_df_message_acc}\n",
              f"enADFtoX---trump : {trump_suc_p}, cage : {cage_suc_p}\n",
              f"ADFtoB---trump : {trump_suc_p_ori}, cage : {cage_suc_p_ori}\n",
              f"AtoA---trump : {trump_suc_p_real}, cage : {cage_suc_p_real}\n",
              f"l2 suc---trump : {trump_attack_success_p}, cage : {cage_attack_success_p}\n",
              f"l1---trump : {attack_l1_trump_res}, cage : {attack_l1_cage_res}\n",
              f"l2---trump : {attack_l2_trump_res}, cage : {attack_l2_cage_res}\n",
              f"lpips_alex---trump : {lpips_alex_trump_res.item()}, cage : {lpips_alex_cage_res.item()}\n",
              f"lpips_vgg---trump : {lpips_vgg_trump_res.item()}, cage : {lpips_vgg_cage_res.item()}\n",
              f"mask---trump : {trump_dist_mask}, cage : {cage_dist_mask}\n",
              file=f_print_casia)
        f_print_casia.close()


if __name__ == '__main__':
    date = '0202-lfw-test'  # 'Ours-NOISE'
    test_str = 'casia'  # vgg casia
    f_print_path = "" + date + "_print.txt"
    save_result = ['encode', 'encode_fake', 'fake', 'real']
    if not os.path.exists('/f/0626/result/save/img/' + date):
        os.mkdir('' + date)
    for path in save_result:
        if not os.path.exists('' + date + '/' + path):
            os.mkdir('' + date + '/' + path)

    # LOAD_PATH = '/f/0626/result/save/ckpt-160/0523-160_1_1_final.pt'  # vgg
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0613-160-no-attention_1_1_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0613-160-no-gan_1_1_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0628-160-casia_1_1_final.pt'  # casia
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0702-160-casia-new-train_1_0.5_best.pt'  # casia-new
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0706-casia-0004266-0010736_2_0.5_best.pt'  # casia-new---11
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0720-casia-0004266-0010736-no-adv_2_0.5_final.pt'
    LOAD_PATH = '/f/0626/result/save/ckpt-new/0714-casia-0004266-0010736_1_0.5_final.pt'  # img+msg
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/071414-casia-0004266-0010736_2_1_final.pt'  # img_msg
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/071430-casia-0004266-0010736_2_0.5_final.pt'  # 30
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0715-casia-0004266-0010736_3_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0715-no_attention-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0715-no-gan-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/071611-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0716-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/07161111-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0716-no-all-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0717-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0717-160-vgg-38-397_1_1_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0719-no-all_1_1_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/071911-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/07191111-casia-0004266-0010736_2_0.5_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/0719-160-vgg-38-397_1_1_final.pt'
    # LOAD_PATH = '/f/0626/result/save/ckpt-new/072245-casia-0004266-0010736_2_0.5_final.pt'
    # lfw -------------
    # LOAD_PATH = '/f/0626/result/save/ckpt-lfw/lfw-George_W_Bush-Colin_Powell-0127-160_1_1_final.pt'
    # lfw -------------
    model = DualDefense_160(args.message_size, in_channels=3, device=device)
    model.encoder.load_state_dict(torch.load(LOAD_PATH)['encoder'], strict=False)
    model.decoder.load_state_dict(torch.load(LOAD_PATH)['decoder'], strict=False)
    model.encoder.eval()
    model.decoder.eval()

    HW = 160

    attack_num_list = [3]  # 0:JPEG 1:RESIZE 2:BLUR 3:NOISE
    attack_list = ['JPEG', 'RESIZE', 'BLUR', 'NOISE']
    quality_list = [50]
    new_hw_list = [80]
    ker_list = [5]
    cir_len_list = [50]
    noise_v_0_list = [0.005]

    # robust
    '''
    attack_num_list = [1]  # 0:JPEG 1:RESIZE 2:BLUR 3:NOISE 0, 1, 2, 3,
    attack_list = ['JPEG', 'RESIZE', 'BLUR', 'NOISE', 'SALT', 'MEDIAN']
    quality_list = [10, 30, 50, 70, 90]
    new_hw_list = [280]  # 40, 80, 120, 160, 200, 240,
    ker_list = [5]
    cir_len_list = [10, 30, 50, 70, 90]
    noise_v_0_list = [0.001, 0.002, 0.003, 0.004, 0.005]
    salt_ls = [0.005, 0.004, 0.003, 0.002, 0.001]
    # k_ls = [3, 5, 7, 9]
    '''

    if test_str == 'vgg':
        print('test_process_vgg')
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        resnet.classify = True
        resnet = resnet.to(device)

        TRUMP_PATH = '/f/0626/faceswap_data/n000043-clean/'  # 38
        CAGE_PATH = '/f/0626/faceswap_data/n000418-clean/'  # 397
        # A_PATH = '/f/0626/faceswap_data/n000270/'  # 254
        # B_PATH = '/f/0626/faceswap_data/n000395/'  # 376
        min_label = 38
        max_label = 397
        _, _, trump_test_dataset = split_dataset(TRUMP_PATH, test_transform=transform_test, val_ratio=0, test_ratio=1)
        trump_test_loader = DataLoader(trump_test_dataset, batch_size=args.batch_size, shuffle=False)
        _, _, cage_test_dataset = split_dataset(CAGE_PATH, test_transform=transform_test, val_ratio=0, test_ratio=1)
        cage_test_loader = DataLoader(cage_test_dataset, batch_size=args.batch_size, shuffle=False)

        # attack_way = 100
        # test_process_vgg()

        for i in attack_num_list:
            attack_way = i
            if attack_way == 0:
                for quality in quality_list:
                    quality = quality
                    jpeg = DiffJPEG(height=HW, width=HW, differentiable=True, quality=quality).to(device)
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], quality, file=f_print)
                    f_print.close()
                    test_process_vgg()
            elif attack_way == 1:
                for new_hw in new_hw_list:
                    # new_hw = new_hw
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], new_hw, file=f_print)
                    f_print.close()
                    test_process_vgg()
            elif attack_way == 2:
                for ker in ker_list:
                    # ker = ker
                    for cir_len in cir_len_list:
                        # cir_len = cir_len
                        f_print = open(f_print_path, "a")
                        print(attack_list[attack_way], ker, cir_len, file=f_print)
                        f_print.close()
                        test_process_vgg()
            elif attack_way == 3:
                for noise_v_0 in noise_v_0_list:
                    noise_v = noise_v_0 ** 0.5
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], noise_v_0, file=f_print)
                    f_print.close()
                    test_process_vgg()
            elif attack_way == 4:
                for salt_SNR in salt_ls:
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], salt_SNR, file=f_print)
                    f_print.close()
                    test_process_vgg()

    elif test_str == 'casia':
        resnet = InceptionResnetV1(pretrained='lfw').eval()
        resnet.classify = True
        resnet = resnet.to(device)
        # A_PATH = '/f/0626/casia-webface-data/0004266/'  # 819
        # B_PATH = '/f/0626/casia-webface-data/0519456/'  # 3411
        # A_PATH = '/f/0626/casia-webface-data/0000439/'  # 138
        # B_PATH = '/f/0626/casia-webface-data/0010736/'  # 1321
        # A_PATH = '/f/0626/casia-webface-data/0424060/'  # 3025
        # B_PATH = '/f/0626/casia-webface-data/0515116/'  # 3385
        # TRUMP_PATH = '/f/0626/casia-webface-data/0004266/'  # 819
        # CAGE_PATH = '/f/0626/casia-webface-data/0010736/'  # 1321
        # lfw ---------------
        TRUMP_PATH = '/f/0626/LFW_TOP_5/160_160/George_W_Bush/'
        CAGE_PATH = '/f/0626/LFW_TOP_5/160_160/Colin_Powell/'
        # lfw ---------------
        min_label = 3743
        max_label = 229
        _, _, trump_test_dataset = split_dataset(TRUMP_PATH, test_transform=transform_test, val_ratio=0, test_ratio=1)
        trump_test_loader = DataLoader(trump_test_dataset, batch_size=args.batch_size, shuffle=False)
        _, _, cage_test_dataset = split_dataset(CAGE_PATH, test_transform=transform_test, val_ratio=0, test_ratio=1)
        cage_test_loader = DataLoader(cage_test_dataset, batch_size=args.batch_size, shuffle=False)

        attack_way = 100
        test_process_casia()
        '''      
        for i in attack_num_list:
            attack_way = i
            if attack_way == 0:
                for quality in quality_list:
                    quality = quality
                    jpeg = DiffJPEG(height=HW, width=HW, differentiable=True, quality=quality).to(device)
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], quality, file=f_print)
                    f_print.close()
                    test_process_casia()
            elif attack_way == 1:
                for new_hw in new_hw_list:
                    # new_hw = new_hw
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], new_hw, file=f_print)
                    f_print.close()
                    test_process_casia()
            elif attack_way == 2:
                for ker in ker_list:
                    # ker = ker
                    for cir_len in cir_len_list:
                        # cir_len = cir_len
                        f_print = open(f_print_path, "a")
                        print(attack_list[attack_way], ker, cir_len, file=f_print)
                        f_print.close()
                        test_process_casia()
            elif attack_way == 3:
                for noise_v_0 in noise_v_0_list:
                    noise_v = noise_v_0 ** 0.5
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], noise_v_0, file=f_print)
                    f_print.close()
                    test_process_casia()
            elif attack_way == 4:
                for salt_SNR in salt_ls:
                    f_print = open(f_print_path, "a")
                    print(attack_list[attack_way], salt_SNR, file=f_print)
                    f_print.close()
                    test_process_casia()
        '''
