import cv2
import os

from tqdm import tqdm

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(file_dir, '../input/sharp')

if not os.path.exists(src_dir):
    print(f"Error: Source directory '{src_dir}' not found.")
    exit()

images = os.listdir(src_dir)
dst_dir = os.path.join(file_dir, '../input/gaussian_blurred')
os.makedirs(dst_dir, exist_ok=True)

for i in tqdm(range(len(images))):
    img = cv2.imread(os.path.join(src_dir, images[i]))
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (51, 51), 0)
    cv2.imwrite(os.path.join(dst_dir, images[i]), blur)

print('DONE')