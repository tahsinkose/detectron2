import sys
import unittest
import torch
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.config import instantiate, LazyCall as L, get_cfg
import cv2
import numpy as np
from detectron2 import model_zoo

BASE_MODEL_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

def save_feature(output, key):
  feature = output[key][0][0].cpu().detach().numpy()
  feature = (255*(feature - np.min(feature))/np.ptp(feature)).astype(np.uint8)
  feature = cv2.resize(feature, (800, 800))
  cv2.imwrite(key+".png",feature)

class FPNTest(unittest.TestCase):
  def test_fpn(self):
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    )
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        BASE_MODEL_FILE))
    fpn = instantiate(backbone)
    inp_im = cv2.imread("test.png")
    inp_im = cv2.resize(inp_im, (1600, 1600))
    inp = np.zeros([1, 3, 1600, 1600], dtype=np.float32)
    inp[0] = np.reshape(inp_im, [3, 1600, 1600])
    inp = torch.from_numpy(inp)
    output = fpn(inp)
    save_feature(output, "p2")
    save_feature(output, "p3")
    save_feature(output, "p4")
    save_feature(output, "p5")
    save_feature(output, "p6")
if __name__ == "__main__":
    unittest.main()
