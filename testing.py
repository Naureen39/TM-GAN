#-------------------------------------------------------------------------------#
# File name         : Testing.py
# Purpose           : model (Tone mapping).

# Usage (command)   : !python testing.py --config config_supervised.yaml
                      # --model snapshot/Gan/model_epoch_num_1.pth --input_folder unpaired_HDR/
# Authors           : Naureen Mujtaba 
# Email             : 19060039@lums.edu.pk

# Last Date         : March 19, 2024
#------------------------------------------------------------------------------#




import os, sys, glob, argparse, logging, imageio, time, random, csv, yaml, torch,shutil, numpy as np

from PIL                                import Image
from TMQI                               import TMQI
from tqdm                               import tqdm
from gan.utils                          import initialize_weights, parse_yaml_config
from torchvision                        import transforms
from torch.utils.data                   import DataLoader
# from gan.gan_network                    import define_G
# from gan.unpaired_gen                   import create_G_net
# from gan.generator                     import UnetGenerator
from small_unet                         import UNet
from dataset.HDR_LDR_Loader             import HDR_LDR_Loader
from dataset.pre_process                import back_to_color2


device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")


class TestSession:
    """class for managing testing sessions"""
    def __init__(self, config_file, model_path):

        self.config = parse_yaml_config(config_file)
        self.model_path = model_path
        self.data_config = self.config['data']

        # self.netG = define_G(input_nc=1, output_nc=1, ngf=self.config['generator']['ngf'], netG=self.config['generator']['netG']).to(device)
        # self.netG = UnetGenerator().to(device)
        # self.netG = create_G_net("unet", device, 0, 1, 'sigmoid',32, "square_and_square_root_manual_d",
        #                   4, 0, 'none', "none", 'relu', 1, 1, 1, 0, "replicate", 2, 0).to(device)
        self.netG = UNet(in_chns=1, class_num=1, initialization=None).to(device)

    def preprocess_input(self, input_image):

        im      = imageio.imread(input_image[0], format="hdr").astype('float32')
        img = np.array(im)
        L = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
        L=L/np.max(L)
        transform = transforms.Compose([transforms.ToTensor()])
        hdr_image = transform(L)
        # input_image = torch.from_numpy(input_image).float().unsqueeze(0).unsqueeze(0)
        return hdr_image, im

    def predict(self, input_image):

        # self.netG.load_state_dict(torch.load(self.model_path))
        self.netG.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.netG.eval()

        input_image = glob.glob(input_image)
        input_image_tensor, hdr_im = self.preprocess_input(input_image)
        input_image_tensor = input_image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.netG(input_image_tensor)
            output = output.squeeze().cpu().detach().numpy() * 255.0
            output = back_to_color2(hdr_im, output)
            output = np.clip(output, 0, 255).astype(np.uint8)
            score = TMQI(hdr_im, output)
            print(score)

        return output, score


if __name__ == "__main__": # get the start time st = time.time()
    st      = time.time()
    parser = argparse.ArgumentParser(description='Adebra-Kadebra')
    parser.add_argument('--config', type=str, required=True, help='.yaml config file')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--input_folder', type=str, required=True, help='input image folder path')
    args = parser.parse_args()

    test_session = TestSession(args.config, args.model)

    # List all input images in the specified folder
    input_folder = args.input_folder
    input_images = os.listdir(input_folder)

    # Create an output folder if it doesn't exist
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)
    TMQI_scores=[]
    # Iterate through HDR images in the input folder
    for input_image_path in glob.glob(input_folder + "/*.hdr"):
        output_image_name = os.path.splitext(os.path.basename(input_image_path))[0] + "_output.png"
        output_image_path = os.path.join(output_folder, output_image_name)

        # Perform inference for the current input image
        output, score = test_session.predict(input_image_path)
        TMQI_scores.append(score)

        # Convert the output to a PIL Image
        # output = np.clip(output, 0, 255).astype(np.uint8)
        output = Image.fromarray(output)

        # Save the output image
        output.save(output_image_path)
        print(f"Output image saved: {output_image_path}")

    print(np.mean(TMQI_scores))
  # get the end time
    et = time.time()

    # get the execution time
    res = et - st
    final_res = res / 60

    print('Execution time:', final_res, 'minutes')