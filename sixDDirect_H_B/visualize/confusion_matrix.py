import json
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ANGLE = "yaw" # pitch, yaw or roll
ANGLE_IDX = 1 # pitch = 0, yaw = 1, roll = 2
TASK = "train"


with open(f"../AGORA/H_B/annotations/full_body_head_coco_style_{TASK}.json", "r") as f:
    data = json.load(f)

id2ann = defaultdict(dict)

for entry in data["annotations"]:
    id2ann[entry["id"]][entry["category_id"]] = entry


head_angles, body_angles = [], []
for id, value in id2ann.items():
    if len(value) == 2:
        if id2ann[id][2]["euler_angles"][ANGLE_IDX] > 180 or id2ann[id][2]["euler_angles"][ANGLE_IDX] < -180:
            continue
        body_angles.append(id2ann[id][2]["euler_angles"][ANGLE_IDX])
        head_angles.append(id2ann[id][1]["euler_angles"][ANGLE_IDX])


body_angles = [int(math.floor(angle / 10.0)) * 10 for angle in body_angles]
head_angles = [int(math.floor(angle / 10.0)) * 10 for angle in head_angles]

print("Min, max body:", min(body_angles), max(body_angles))
print("Min, max head:", min(head_angles), max(head_angles))

confmat = confusion_matrix(body_angles, head_angles)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat,
                              display_labels=list(range(-180, 180, 10)))

plt.figure(figsize=(30, 30))
disp.plot()
plt.xlabel(f"Body {ANGLE}")
plt.ylabel(f"Head {ANGLE}")
plt.xticks(rotation=90)
plt.savefig(f"visualize/conf_mat_{ANGLE}.png")