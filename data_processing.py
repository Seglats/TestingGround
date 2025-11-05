from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
import logging

data_dir   = Path('coco_dataset')
output_dir = Path('images')
max_per_class = 30000        


logging.basicConfig(
    filename='convert.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


coco_train = COCO(str(data_dir / 'annotations' / 'instances_train2017.json'))
coco_val   = COCO(str(data_dir / 'annotations' / 'instances_val2017.json'))

train_person_ids   = coco_train.getImgIds(catIds=[1])[:max_per_class]
train_all_ids      = set(coco_train.getImgIds())
train_no_person_ids = list(train_all_ids - set(train_person_ids))[:max_per_class]

val_person_ids   = coco_val.getImgIds(catIds=[1])[:max_per_class]
val_all_ids      = set(coco_val.getImgIds())
val_no_person_ids = list(val_all_ids - set(val_person_ids))[:max_per_class]

(output_dir / 'with_people').mkdir(parents=True, exist_ok=True)
(output_dir / 'without_people').mkdir(parents=True, exist_ok=True)



def compress_to_gray_png(src_path: Path, dst_path: Path) -> None:
    try:
        with Image.open(src_path) as img:
            if img.mode != "L":
                img = img.convert("L")      # grayscale conversion

            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top  = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))

            img = img.resize((96, 96), Image.LANCZOS)
            img.info.pop('icc_profile', None) #drop icc
            img.save(dst_path, format='PNG')

        logging.info(f"{src_path.name} → {dst_path.name}")
    except Exception as exc:
        logging.error(f"Failed to process {src_path}: {exc}")
        raise
    


def process_images(img_ids, coco_obj, src_folder: Path, dst_folder: Path):
    for img_id in tqdm(img_ids,
                       desc=f'{dst_folder.name} ← {src_folder.name}',
                       unit='img'):
        try:
            img_info = coco_obj.loadImgs(img_id)[0]
        except Exception:
            tqdm.write(f"Image id {img_id} missing in COCO index – skipping.")
            continue

        src_path = src_folder / img_info["file_name"]
        if not src_path.is_file():
            tqdm.write(f"Source file not found: {src_path} – skipping.")
            continue

        dst_path = dst_folder / f'{Path(img_info["file_name"]).stem}.png'

        try:
            compress_to_gray_png(src_path, dst_path)   # <-- grayscale helper
        except Exception:
            continue
        

process_images(val_person_ids, coco_val,
               data_dir / 'val2017', output_dir / 'with_people')
process_images(val_no_person_ids, coco_val,
               data_dir / 'val2017', output_dir / 'without_people')
process_images(train_person_ids, coco_train,
               data_dir / 'train2017', output_dir / 'with_people')
process_images(train_no_person_ids, coco_train,
               data_dir / 'train2017', output_dir / 'without_people')

print("\n Done, storage loc = ", output_dir)