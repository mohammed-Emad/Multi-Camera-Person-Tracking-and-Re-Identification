# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from facenet_pytorch import InceptionResnetV1
from PIL import Image
import cv2, torch
from torchvision.transforms import functional as F
import numpy as np



# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def netproc(img,p=0):
    tsimg = F.to_tensor(np.float32(img))
    if p!=0:
       return tsimg
    return (tsimg - 127.5) / 128.0
    
def netEM(listor):
    _im = torch.stack(listor)
    return resnet(_im).detach().numpy()



def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

alpha = 1 
beta = 0.6 # transparency for the segmentation map
gamma = 0 # scalar added to each sum
#[topleft_x, topleft_y, w, h]
def crop_mask(imager, masks,boxes,labels, sizeim):
    phlist = []
    boxes2 = []
    for i in range(len(masks)):
        if labels[i]=='person':
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            try:
                # apply a randon color mask to each object
                red_map[masks[i] == 1] = 255
                
                res = cv2.bitwise_and(imager,imager, mask= red_map)
                x00 = (boxes[i][0][0], boxes[i][1][0])
                x11 = (boxes[i][0][1], boxes[i][1][1])
                x = min(x00)
                y = min(x11)
                width = max(x00)
                height = max(x11)
                crop_img = res[y:height, x:width]
                crop_img = netproc(cv2.resize(crop_img, sizeim), 0)
                phlist.append(crop_img)
                boxx = [x,y,int(width-x), int(height-y)]
                boxes2.append(boxx)
            except:
                print(masks[i].shape)
    return phlist ,np.array(boxes2)

def crop_mask_tf(imager, masks,boxes,labels, sizeim):
    phlist = []
    boxes2 = []
    for i in range(len(masks)):
        if labels[i]=='person':
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            try:
                # apply a randon color mask to each object
                red_map[masks[i] == 1] = 255
                
                res = cv2.bitwise_and(imager,imager, mask= red_map)
                x00 = (boxes[i][0][0], boxes[i][1][0])
                x11 = (boxes[i][0][1], boxes[i][1][1])
                x = min(x00)
                y = min(x11)
                width = max(x00)
                height = max(x11)
                crop_img = res[y:height, x:width]
                #crop_img = netproc(cv2.resize(crop_img, sizeim), 0)
                phlist.append(cv2.resize(crop_img, tuple(sizeim[::-1])))
                boxx = [x,y,int(width-x), int(height-y)]
                boxes2.append(boxx)
            except:
                print(masks[i].shape)
    return phlist ,np.array(boxes2)

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

# get everyone feature in a frame
# def encoder : return nx128 array
# n means people nums of this frame

import torch

from pytorch_sift import SIFTNet 

SIFT = SIFTNet(patch_size = 110)
SIFT.eval() 


def getpy(img):
    img = cv2.resize(img, (110,110))
    img =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    patches = np.ndarray((1, 1, 110, 110), dtype=np.float32)
    patches[0,0,:,:] = img
    with torch.no_grad():
        torch_patches = torch.from_numpy(patches)
        res = SIFT(torch_patches)
        sift = np.round(512. * res.data.cpu().numpy()).astype(np.int32)
    return sift


def crop_mask_torch(imager, masks,boxes,labels, sizeim):
    phlist = []
    boxes2 = []
    for i in range(len(masks)):
        if labels[i]=='person':
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            try:
                # apply a randon color mask to each object
                red_map[masks[i] == 1] = 255
                
                res = cv2.bitwise_and(imager,imager, mask= red_map)
                x00 = (boxes[i][0][0], boxes[i][1][0])
                x11 = (boxes[i][0][1], boxes[i][1][1])
                x = min(x00)
                y = min(x11)
                width = max(x00)
                height = max(x11)
                crop_img = res[y:height, x:width]
                patch = getpy(cv2.resize(crop_img, tuple(sizeim[::-1])))
                phlist.append(patch)

                boxx = [x,y,int(width-x), int(height-y)]
                boxes2.append(boxx)
            except:
                print(masks[i].shape)
    out = np.zeros((len(boxes), 128), np.float32)
    for ii in range(len(boxes2)):
        out[ii] = phlist[ii]
    return out ,np.array(boxes2)

sift_op = cv2.xfeatures2d.SIFT_create() # cv2.ORB_create()#

#import pysift


def exfea_sift(area):
    kp, des = sift_op.detectAndCompute(area,None)
    #kp, des = pysift.computeKeypointsAndDescriptors(area)
    return des

def crop_mask_siftorg(imager, masks,boxes,labels, sizeim):
    phlist = []
    boxes2 = []
    for i in range(len(masks)):
        if labels[i]=='person':
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            try:
                # apply a randon color mask to each object
                red_map[masks[i] == 1] = 255
                
                res = cv2.bitwise_and(imager,imager, mask= red_map)
                x00 = (boxes[i][0][0], boxes[i][1][0])
                x11 = (boxes[i][0][1], boxes[i][1][1])
                x = min(x00)
                y = min(x11)
                width = max(x00)
                height = max(x11)
                crop_img = res[y:height, x:width]
                #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                #patch = exfea_sift(crop_img)
                patch = exfea_sift(cv2.resize(crop_img, tuple(sizeim[::-1])))

                #print(patch.flatten().shape)
                phlist.append(patch)

                boxx = [x,y,int(width-x), int(height-y)]
                boxes2.append(boxx)
            except:
                print(masks[i].shape)
    '''
    out = np.zeros((len(boxes), 200), np.float32)
    for ii in range(len(boxes2)):
        out[ii] = phlist[ii]
        out[ii] = out[ii] / out.max()
    '''
    out=np.vstack(phlist)
    return out ,np.array(boxes2)


     
#Sift opencv
def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):

    def encoder(image, masks,boxes,labels):
        image_patches,boxes2 = crop_mask_siftorg(image, masks,boxes,labels, (160,160))
        return image_patches, boxes2

    return encoder


#sift torch
def create_box_encoder0torch(model_filename, input_name="images",
                       output_name="features", batch_size=32):

    def encoder(image, masks,boxes,labels):
        image_patches,boxes2 = crop_mask_torch(image, masks,boxes,labels, (110,110))
        return image_patches, boxes2

    return encoder


#no faceNet
def create_box_encode0r(model_filename, input_name="images",
                       output_name="features", batch_size=32):

    def encoder(image, masks,boxes,labels):
        image_patches,boxes2 = crop_mask(image, masks,boxes,labels, (160,160))
        if len(image_patches) > 0:
           sma = netEM(image_patches)
        else:
           sma = []
        return sma, boxes2
    return encoder

def create_box_encoder01(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, masks,boxes,labels):
        image_patches = []
        boxes2 = []
        for box in boxes:
            #patch = extract_image_patch(image, box, image_shape[:2])
            #image_patches,boxes2 = crop_mask(image, masks,boxes,labels)
            image_patches,boxes2 = crop_mask_tf(image, masks,boxes,labels, image_shape[:2])
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size), boxes2

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
