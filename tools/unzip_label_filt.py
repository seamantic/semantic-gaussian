import os
import zipfile
from tqdm import tqdm
import traceback

LABEL_ROOT = "/ocean/projects/cis250226p/shared/semantic-gaussian/dataset/data/scans/"
OUT_ROOT = "/ocean/projects/cis250226p/shared/semantic-gaussian/dataset/data/"


for split in [""]:
    ls = os.listdir(os.path.join(OUT_ROOT, split))
    ls.sort()
    for scene in tqdm(ls[21:]):
        print("SCENE: ", scene)
        if scene != "scans" and scene != "processed" and scene != "validation_scenes":
            img_path = os.path.join(OUT_ROOT, split, scene, "color")
            ext_imgs = os.listdir(img_path)
            ext_imgs.sort()

            out_path = os.path.join(OUT_ROOT, split, scene)
            os.makedirs(out_path, exist_ok=True)
            label_zip = os.path.join(LABEL_ROOT, split, scene, f"{scene}_2d-label-filt.zip")
            with zipfile.ZipFile(label_zip, "r") as zip_ref:
                for img in ext_imgs:
                    try:
                        zip_ref.extract(f"label-filt/{img}".replace("jpg", "png"), out_path)
                    except Exception:
                        print(traceback.format_exc())
                        print(scene)
