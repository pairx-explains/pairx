import torch
import os
from PIL import Image

from xai_dataset import XAIDataset, get_img_pair_from_paths
from example_loaders import toy_df, wildme_multispecies_miewid
from core import explain
import cv2
from datetime import datetime


torch.cuda.empty_cache()

def main():
    torch.cuda.reset_peak_memory_stats()
    start_time = datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, img_size, img_transforms = wildme_multispecies_miewid(device)

    df = toy_df()
    dataset = XAIDataset(df, img_size, img_transforms)

    imgs_0 = []
    imgs_1 = []
    imgs_np_0 = []
    imgs_np_1 = []

    data_paths = [("cow_0_0.jpg", "cow_0_1.jpg"), ("smores_shoes.jpg", "smores_sink.jpg"), ("bowie_airport.jpg", "bowie_blanket.jpg"), ("bowie_bow.jpg", "bowie_prairie_dog.jpg")]
    for data_0, data_1 in data_paths:
        img_0, img_1, img_np_0, img_np_1 = get_img_pair_from_paths(device, f"data/{data_0}", f"data/{data_1}", img_size, img_transforms)
        imgs_0.append(img_0)
        imgs_1.append(img_1)
        imgs_np_0.append(img_np_0)
        imgs_np_1.append(img_np_1)


    #img_0_0, img_1_0, img_np_0_0, img_np_1_0 = dataset.get_img_pair(device, "cow_0_0", "cow_0_1")
    #img_0_1, img_1_1, img_np_0_1, img_np_1_1 = get_img_pair_from_paths(device, "data/smores_radiator.jpg", "data/smores_sink.jpg", img_size, img_transforms)

    pairx_imgs = explain(torch.cat(imgs_0),                  # transformed image 0
                        torch.cat(imgs_1),                  # transformed image 1
                        imgs_np_0,               # untransformed image 0
                        imgs_np_1,               # untransformed image 1
                        model,                  # model
                        ["backbone.blocks.3"],  # intermediate layer to visualize
                        k_lines=20,             # number of matches to visualize as lines
                        k_colors=10,            # number of matches to visualize as colors
                        )
    
    #raw_img = cv2.hconcat((img_np_0, img_np_1))
    #pairx_img = cv2.vconcat((raw_img, pairx_img))

    for i, pairx_img in enumerate(pairx_imgs):
        pairx_img = Image.fromarray(pairx_img)
        pairx_img.save(f"examples/cow_pairx_example_{i}.png")

    peak_memory = torch.cuda.max_memory_allocated()  # in bytes
    print(f"Peak memory allocated: {peak_memory / (1024 ** 3):.2f} GiB")
    print(f"Time taken: {(datetime.now()-start_time).total_seconds():.2f} seconds")

if __name__ == "__main__":
    main()


