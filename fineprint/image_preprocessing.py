import os
import cv2
from tqdm import tqdm
from typing import Tuple

def image_resizer(image_path: str, size: Tuple[int, int], output: str):
  img = cv2.imread(image_path)
  img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(output, img)

if __name__ == '__main__':
  dirs = [f"../output/{d}" for d in os.listdir("../output/")]
  OUTPUT = "../output_small"
  SIZE = (50, 50)
  os.makedirs(OUTPUT, exist_ok=True)
  
  for dir in dirs:
    old_dir_path = os.path.join("../output", os.path.basename(dir))
    new_dir_path = os.path.join(OUTPUT, os.path.basename(dir))
    os.makedirs(new_dir_path)
    print(f"Outputting to {new_dir_path}...")
    
    files = os.listdir(old_dir_path)
    for i in tqdm(range(len(files))):
      file = files[i]
      image_resizer(os.path.join(old_dir_path, file), SIZE, os.path.join(new_dir_path, file))
    print()
