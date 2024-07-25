import os, os.path as osp
import argparse
import yaml
from tqdm import tqdm

from pycocotools.coco import COCO

def write_yolov5_labels(data):
    
    assert not osp.isdir(osp.join(data["path"], data["labels"])), \
        "Labels already generated. Remove or choose new name for labels."

    splits = [osp.splitext(osp.split(data[s])[-1])[0] for s in ["train", "val", "test"] if s in data]
    annotations = [osp.join(data["path"], data["{}_annotations".format(s)]) for s in ["train", "val", "test"] if s in data]
    test_split = [0 if s in ["train", "val"] else 1 for s in ["train", "val", "test"] if s in data]
    img_txt_dir = osp.join(data["path"], data["labels"], "img_txt")
    os.makedirs(img_txt_dir, exist_ok=True)

    nc = data["nc"]
    if nc > 1:
        print("Only writinig head labels.")
    else:
        print("Writing head and body labels.")

    for split, annot, is_test in zip(splits, annotations, test_split):
        head, body = 0, 0
        img_txt_path = osp.join(img_txt_dir, "{}.txt".format(split))
        labels_path = osp.join(data["path"], "{}/{}".format(data["labels"], split))
        if not is_test:
            os.makedirs(labels_path, exist_ok=True)
        coco = COCO(annot)
        if not is_test:
            pbar = tqdm(coco.anns.keys(), total=len(coco.anns.keys()))
            pbar.desc = "Writing {} labels to {}".format(split, labels_path)
            for id in pbar:
                a = coco.anns[id]

                if a["category_id"] == 1:
                    head += 1
                elif a["category_id"] == 2:
                    body += 1
                else:
                    continue

                # We only want head labels, so we skip this body
                if nc == 1 and a["category_id"] == 2:
                    continue

                if a["image_id"] not in coco.imgs:
                    continue

                if "train" in split and a["iscrowd"]:
                    continue

                img_info = coco.imgs[a["image_id"]]
                img_h, img_w = img_info["height"], img_info["width"]
                x, y, w, h = a["bbox"]
                xc, yc = x + w / 2, y + h / 2
                xc /= img_w
                yc /= img_h
                w /= img_w
                h /= img_h

                xc = max(min(1., xc), 0.)
                yc = max(min(1., yc), 0.)
                w = max(min(1., w), 0.)
                h = max(min(1., h), 0.)
                
                [a1, a2, a3, a4, a5, a6] = a["6D"]
                a1 = (a1 + 1) / 2 # [-1, 1] range to [0, 1] range
                a2 = (a2 + 1) / 2 # [-1, 1] range to [0, 1] range
                a3 = (a3 + 1) / 2 # [-1, 1] range to [0, 1] range
                a4 = (a4 + 1) / 2 # [-1, 1] range to [0, 1] range
                a5 = (a5 + 1) / 2 # [-1, 1] range to [0, 1] range
                a6 = (a6 + 1) / 2 # [-1, 1] range to [0, 1] range

                yolov5_label_txt = "{}.txt".format(str(img_info["id"]))
                with open(osp.join(labels_path, yolov5_label_txt), "a") as f:
                    f.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        a["category_id"]-1, xc, yc, w, h, a1, a2, a3, a4, a5, a6))
            pbar.close()

        with open(img_txt_path, "w") as f:
            for img_info in coco.imgs.values():
                f.write(osp.join(data["path"], "images", "{}".format(split), str(img_info["id"]) + ".jpg") + "\n")
        print(split, "-> HEAD:", head)
        print(split, "-> BODY:", body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/agora_coco.yaml")
    args = parser.parse_args()

    assert osp.isfile(args.data), "Data config file not found at {}".format(args.data)

    with open(args.data, "rb") as f:
        data = yaml.safe_load(f)
    write_yolov5_labels(data)