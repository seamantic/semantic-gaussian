import os
import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf

from model import GaussianModel, render_chn
from scene import Scene
from utils.system_utils import set_seed, searchForMaxIteration

# from model.mink_unet import mink_unet
from model.render_utils import get_mapped_label, get_text_features, render_palette
from dataset.fusion_utils import Voxelizer
from dataset.scannet.label_mapping import read_label_mapping
from utils import metric

# import MinkowskiEngine as ME
from model_ptv3 import PointTransformerV3
from distill import Model3DWithLinear


def init_dir(config):
    print("Distill folder: {}".format(config.distill.model_dir))
    weights_dir = os.path.join(config.distill.model_dir, "weights")
    if config.distill.iteration == -1:
        weights = os.listdir(weights_dir)
        weights.sort(key=lambda x: int(x))
        iteration = weights[-1]
    else:
        iteration = config.distill.iteration
    ckpt_path = os.path.join(weights_dir, str(iteration), "model.pth")
    return ckpt_path


def evaluate(config):
    if config.scene.dataset_name == "scannet20":
        config.scene.num_classes = 19
        label_mapping = read_label_mapping("./dataset/scannet/scannetv2-labels.modified.tsv", label_to="scannetid")
    elif config.scene.dataset_name == "cocomap":
        config.scene.num_classes = 20
        label_mapping = read_label_mapping("./dataset/scannet/scannetv2-labels.modified.tsv", label_to="cocomapid")

    if config.distill.feature_type == "all":
        # model_3d = mink_unet(in_channels=56, out_channels=768, D=3, arch=config.distill.model_3d).cuda()
        model_3dbase = PointTransformerV3(
            in_channels=56,
            enc_depths=(1,1,1,1,1),
            enc_channels=(32,64,96,128,160),
            enc_num_head=(2,4,6,8,10),
            enc_patch_size=(256,256,256,256,256),
            dec_depths=(1,1,1,1),
            dec_channels=(64,64,96,128),
            dec_num_head=(4,4,6,8),
            dec_patch_size=(256,256,256,256),
            drop_path=0.1,
            enable_flash=False
        ).cuda()
    elif config.distill.feature_type == "color":
        # model_3d = mink_unet(in_channels=48, out_channels=768, D=3, arch=config.distill.model_3d).cuda()
        model_3dbase = PointTransformerV3(
            in_channels=56,
            enc_depths=(1,1,1,1,1),
            enc_channels=(32,64,96,128,160),
            enc_num_head=(2,4,6,8,10),
            enc_patch_size=(256,256,256,256,256),
            dec_depths=(1,1,1,1),
            dec_channels=(64,64,96,128),
            dec_num_head=(4,4,6,8),
            dec_patch_size=(256,256,256,256),
            drop_path=0.1,
            enable_flash=False
        ).cuda()

    ckpt_path = init_dir(config)
    model_3d = Model3DWithLinear(model_3dbase)
    model_3d.load_state_dict(torch.load(ckpt_path))

    # if config.eval.eval_mode == "2d":
    #     eval_fusion(config, model_3d, label_mapping)
    if config.eval.eval_mode == "3d":
        eval_mink(config, model_3d, label_mapping)
    # elif config.eval.eval_mode == "2d_and_3d":
    #     eval_mink_and_fusion(config, model_3d, label_mapping)
    # elif config.eval.eval_mode == "pretrained":
    #     eval_seg_model(config, model_3d, label_mapping)
    # elif config.eval.eval_mode == "labelmap":
    #     eval_labelmap(config, model_3d, label_mapping)


def eval_mink(config, model_3d, label_mapping):
    print("HELLOBROTHER: ", config.model.model_dir)
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()
    print("HELLLLO: ", eval_scene)

    model_2d_name = config.distill.text_model.lower().replace("_", "")
    if model_2d_name == "lseg":
        from model.lseg_predictor import LSeg

        model_2d = LSeg(None)
    else:
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)
    print(eval_scene)
    for i, scene_name in enumerate(tqdm(eval_scene)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.abspath(os.path.join(eval_config.scene.scene_path, scene_name))
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            loaded_iter = config.model.load_iteration
            if loaded_iter == -1:
                loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    f"iteration_{loaded_iter}",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(model_2d.embedding_dim)

            # locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_type)
            # voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
            # locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
            # locs = torch.from_numpy(locs).int()
            # locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            # # features = torch.from_numpy(features).float()
            # # vox_ind = torch.from_numpy(vox_ind).cuda()

            # # sinput = ME.SparseTensor(features.cuda(), locs.cuda())
            # sinput = {
            #     "feat": features.cuda(),
            #     "grid_coord": grid_coord.cuda(),
            #     "coord": coord.cuda(),
            #     "offset": torch.tensor([features.shape[0]]).cuda()
            # }
            # output = model_3d(sinput).F[:, model_2d.embedding_dim * 0 : model_2d.embedding_dim * 1]

            # 1. Get locs, features from gaussians
            locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_type)

            voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)

            # 2. voxelizer returns numpy arrays
            locs, features, _, _, vox_ind = voxelizer.voxelize(
                locs, features, None, return_ind=True
            )

            # 3. Convert to torch
            grid_coord = torch.from_numpy(locs).int().cuda()     # (N, 3)
            coord = grid_coord.float().cuda()                    # (N, 3)
            features = torch.from_numpy(features).float().cuda() # (N, C)

            # 4. Build PTv3 input dict (same format as training)
            sinput = {
                "feat": features,
                "grid_coord": grid_coord,
                "coord": coord,
                "offset": torch.tensor([features.shape[0]], device=features.device)
            }

            # 5. For eval, mask=None
            output = model_3d(sinput, mask=None)

            # 6. Same slice as training
            output = output[:, model_2d.embedding_dim * 0 : model_2d.embedding_dim * 1]


            # print("output: ", output.shape)
            output /= torch.clamp(output.norm(dim=-1, keepdim=True), min=1e-8)
            
            # # Save output as image
            # output_np = output.cpu().numpy()
            # n_features, embedding_dim = output_np.shape
            
            # # Reduce to 3 channels for RGB visualization using PCA or first 3 channels
            # if embedding_dim >= 3:
            #     # Use first 3 channels
            #     output_rgb = output_np[:, :3]
            # else:
            #     # Pad with zeros if embedding_dim < 3
            #     output_rgb = np.zeros((n_features, 3))
            #     output_rgb[:, :embedding_dim] = output_np
            
            # # Normalize to [0, 1] range
            # output_rgb = (output_rgb - output_rgb.min()) / (output_rgb.max() - output_rgb.min() + 1e-8)
            
            # # Reshape to a square grid for visualization
            # grid_size = int(np.ceil(np.sqrt(n_features)))
            # padded_size = grid_size * grid_size
            # output_rgb_padded = np.zeros((padded_size, 3))
            # output_rgb_padded[:n_features] = output_rgb
            # output_image = output_rgb_padded.reshape(grid_size, grid_size, 3)
            
            # # Save as image
            # os.makedirs("path", exist_ok=True)
            # output_image_path = "path/output_features.png"
            # plt.imsave(output_image_path, output_image)
            # # print(f"Saved output image to: {output_image_path}")

            views = scene.getTrainCameras()
            # print("Views: ", views)
            out_path = os.path.join("eval_render_new_new", scene_name)
            os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "3d"), exist_ok=True)  
            gaussians._features_semantic[vox_ind] = output
            palette, text_features = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            # print("PEEPEEPOOPOO: ", len(views))
            for idx, view in enumerate((tqdm(views))):
                # if idx % 5 != 0:
                #     continue
                view.cuda()
                gt_path = str(view.image_path)
                # print("PATH: ", gt_path)
                # print("THING: ", views.camera_info[idx].image_name)
                # print("gt_path: ", gt_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    # print("hellooooooo")
                    continue
                # Ensure label_img is 2D (if it's RGB, convert to grayscale)
                if len(label_img.shape) == 3:
                    # If RGB, convert to grayscale by taking mean or first channel
                    # For label images, typically all channels are the same, so we can take any channel
                    label_img = label_img[:, :, 0] if label_img.shape[2] >= 1 else label_img.mean(axis=2)
                label_img = torch.from_numpy(label_img).int().cpu()

                

                if config.eval.pred_on_3d:
                    sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
                    label_soft = sim.softmax(dim=1)
                    label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=label_soft.shape[1],
                        override_color=label_soft,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    label = rendering[1:].argmax(dim=0).cpu()
                else:
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=model_2d.embedding_dim,
                        override_color=gaussians._features_semantic,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    rendering = rendering / torch.clamp(rendering.norm(dim=0, keepdim=True), min=1e-8)
                    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
                    # print(f"DEBUG: text_features shape: {text_features.shape}, rendering shape: {rendering.shape}, sim shape: {sim.shape}")
                    # print(f"DEBUG: sim[1:] shape: {sim[1:].shape}, sim min/max: {sim.min():.4f}/{sim.max():.4f}")
                    label = sim[1:].argmax(dim=0).cpu()
                

                # print("label: ", label.shape, label_img.shape)
                # print(f"DEBUG: label unique values: {torch.unique(label)}, label_img unique values: {torch.unique(label_img)}")
                # print(label)
                # print(label_img)

                # Visualize label and label_img side by side
                # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # # Convert to numpy for visualization
                # label_np = label.numpy()
                # label_img_np = label_img.numpy()
                
                # # Display label (2D)
                # axes[0].imshow(label_np, cmap='tab20')
                # axes[0].set_title(f'Label (shape: {label.shape})')
                # axes[0].axis('off')
                
                # # Display label_img (3D - if it's RGB, display directly; otherwise take first channel)
                # if len(label_img_np.shape) == 3:
                #     if label_img_np.shape[2] == 3:
                #         # RGB image - normalize to [0, 1] if needed
                #         img_display = label_img_np.astype(np.float32)
                #         if img_display.max() > 1.0:
                #             img_display = img_display / 255.0
                #         axes[1].imshow(np.clip(img_display, 0, 1))
                #     else:
                #         # Multi-channel, show first channel
                #         axes[1].imshow(label_img_np[:, :, 0], cmap='tab20')
                # else:
                #     axes[1].imshow(label_img_np, cmap='tab20')
                # axes[1].set_title(f'Label Img (shape: {label_img.shape})')
                # axes[1].axis('off')
                
                # plt.tight_layout()
                # # Save to the same output directory
                # vis_path = "path/thing.png"
                # try:
                #     plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
                #     plt.close()
                #     # if os.path.exists(vis_path):
                #     #     print(f"Successfully saved visualization to: {vis_path}")
                #     # else:
                #     #     print(f"ERROR: File was not created at {vis_path}")
                # except Exception as e:
                #     # print(f"ERROR saving visualization: {e}")
                #     plt.close()

                label += 1
                sem = render_palette(label, palette)
                sem_gt = render_palette(label_img, palette)

                # print("pepepepEPPEPEPEPEOEOEOEPOOOOOOPOOOOPEEEPEEEEPEEEPEPEPEPEPEPPEPEPEOAWIOFSIFAISADFPIHAFSDIHAFIDFADSHIFDSHIUFSDIUAFSHIU")
                # print(sem.shape, sem_gt.shape)

                torchvision.utils.save_image(sem, os.path.join(out_path, "3d", f"{views.camera_info[idx].image_name}.jpg"))
                torchvision.utils.save_image(sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


# def eval_fusion(config, model_3d, label_mapping):
#     eval_scene = os.listdir(config.model.model_dir)
#     eval_scene.sort()

#     model_2d_name = config.fusion.model_2d.lower().replace("_", "")
#     if model_2d_name == "lseg":
#         from model.lseg_predictor import LSeg

#         model_2d = LSeg(None)
#     else:
#         from model.openseg_predictor import OpenSeg

#         model_2d = OpenSeg(None, "ViT-L/14@336px")

#     bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#     confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

#     for i, scene_name in enumerate(tqdm(eval_scene)):
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             eval_config = deepcopy(config)
#             eval_config.scene.scene_path = os.path.abspath(os.path.join(eval_config.scene.scene_path, scene_name))
#             eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
#             scene = Scene(eval_config.scene)
#             gaussians = GaussianModel(eval_config.model.sh_degree)
#             loaded_iter = config.model.load_iteration
#             if loaded_iter == -1:
#                 loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
#             gaussians.load_ply(
#                 os.path.join(
#                     eval_config.model.model_dir,
#                     "point_cloud",
#                     f"iteration_{loaded_iter}",
#                     "point_cloud.ply",
#                 )
#             )
#             gaussians.create_semantic(model_2d.embedding_dim)

#             feature_path = os.path.join(eval_config.fusion.out_dir, scene_name, "0.pt")
#             gt = torch.load(feature_path)
#             feat, mask_full = gt["feat"], gt["mask_full"]

#             views = scene.getTrainCameras()
#             out_path = os.path.join("eval_render", scene_name)
#             os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
#             os.makedirs(os.path.join(out_path, "2d"), exist_ok=True)  
#             gaussians._features_semantic[mask_full] = feat.float().cuda()
#             palette, text_features = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
#             for idx, view in enumerate(views):
#                 # if idx % 5 != 0:
#                 #     continue
#                 view.cuda()
#                 gt_path = str(view.image_path)
#                 label_img = get_mapped_label(config, gt_path, label_mapping)
#                 if label_img is None:
#                     continue
#                 label_img = torch.from_numpy(label_img).int().cpu()

#                 if config.eval.pred_on_3d:
#                     sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
#                     label_soft = sim.softmax(dim=1)
#                     label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
#                     rendering = render_chn(
#                         view,
#                         gaussians,
#                         eval_config.pipeline,
#                         background,
#                         num_channels=label_soft.shape[1],
#                         override_color=label_soft,
#                         override_shape=[config.eval.width, config.eval.height],
#                     )["render"]
#                     label = rendering[1:].argmax(dim=0).cpu()
#                 else:
#                     rendering = render_chn(
#                         view,
#                         gaussians,
#                         eval_config.pipeline,
#                         background,
#                         num_channels=model_2d.embedding_dim,
#                         override_color=gaussians._features_semantic,
#                         override_shape=[config.eval.width, config.eval.height],
#                     )["render"]
#                     rendering = rendering / torch.clamp(rendering.norm(dim=0, keepdim=True), min=1e-8)
#                     sim = torch.einsum("cq,qhw->chw", text_features, rendering)
#                     label = sim[1:].argmax(dim=0).cpu()

#                 label += 1
#                 sem = render_palette(label, palette)
#                 sem_gt = render_palette(label_img, palette)
#                 torchvision.utils.save_image(sem, os.path.join(out_path, "2d", f"{views.camera_info[idx].image_name}.jpg"))
#                 torchvision.utils.save_image(sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg"))
#                 confusion += metric.confusion_matrix(
#                     label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
#                 )

#     metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


# def eval_mink_and_fusion(config, model_3d, label_mapping):
#     eval_scene = os.listdir(config.model.model_dir)
#     eval_scene.sort()

#     performance_dict = {}

#     model_2d_name = config.fusion.model_2d.lower().replace("_", "")
#     if model_2d_name == "lseg":
#         from model.lseg_predictor import LSeg

#         model_2d = LSeg(None)
#     else:
#         from model.openseg_predictor import OpenSeg

#         model_2d = OpenSeg(None, "ViT-L/14@336px")

#     text_model_name = config.distill.text_model.lower().replace("_", "")
#     if text_model_name == model_2d_name:
#         text_model = model_2d
#     elif text_model_name == "lseg":
#         from model.lseg_predictor import LSeg

#         text_model = LSeg(None)
#     else:
#         from model.openseg_predictor import OpenSeg

#         text_model = OpenSeg(None, "ViT-L/14@336px")

#     bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#     confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

#     for i, scene_name in enumerate(tqdm(eval_scene)):
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             eval_config = deepcopy(config)
#             eval_config.scene.scene_path = os.path.abspath(os.path.join(eval_config.scene.scene_path, scene_name))
#             eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
#             scene = Scene(eval_config.scene)
#             gaussians = GaussianModel(eval_config.model.sh_degree)
#             loaded_iter = config.model.load_iteration
#             if loaded_iter == -1:
#                 loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
#             gaussians.load_ply(
#                 os.path.join(
#                     eval_config.model.model_dir,
#                     "point_cloud",
#                     f"iteration_{loaded_iter}",
#                     "point_cloud.ply",
#                 )
#             )
#             gaussians.create_semantic(model_2d.embedding_dim + text_model.embedding_dim)

#             locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_type)
#             voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
#             locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
#             locs = torch.from_numpy(locs).int()
#             locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
#             features = torch.from_numpy(features).float()
#             vox_ind = torch.from_numpy(vox_ind).cuda()

#             sinput = ME.SparseTensor(features.cuda(), locs.cuda())
#             output = model_3d(sinput).F[:, text_model.embedding_dim * 0 : text_model.embedding_dim * 1]

#             feature_path = os.path.join(eval_config.fusion.out_dir, scene_name, "0.pt")
#             gt = torch.load(feature_path)
#             feat, mask_full = gt["feat"].float(), gt["mask_full"]

#             views = scene.getTrainCameras()
#             out_path = os.path.join("eval_render", scene_name)
#             os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
#             os.makedirs(os.path.join(out_path, "2d_and_3d"), exist_ok=True)
#             feat /= torch.clamp(feat.norm(dim=-1, keepdim=True), min=1e-8)
#             output /= torch.clamp(output.norm(dim=-1, keepdim=True), min=1e-8)
#             gaussians._features_semantic[mask_full, :model_2d.embedding_dim] = feat.float().cuda()
#             gaussians._features_semantic[vox_ind, model_2d.embedding_dim:] = output
#             palette, text_features_2d = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
#             palette, text_features_3d = get_text_features(text_model, dataset_name=config.scene.dataset_name)

#             for idx, view in enumerate(views):
#                 # if idx % 5 != 0:
#                 #     continue
#                 view.cuda()
#                 gt_path = str(view.image_path)
#                 label_img = get_mapped_label(config, gt_path, label_mapping)
#                 if label_img is None:
#                     continue
#                 label_img = torch.from_numpy(label_img).int().cpu()

#                 if config.eval.feature_fusion == "concat":
#                     cat_text_features = torch.cat([text_features_2d, text_features_3d], dim=1)
#                     if config.eval.pred_on_3d:
#                         sim = torch.einsum("cq,dq->dc", cat_text_features, gaussians._features_semantic)
#                         label_soft = sim.softmax(dim=1)
#                         rendering = render_chn(
#                             view,
#                             gaussians,
#                             eval_config.pipeline,
#                             background,
#                             num_channels=label_soft.shape[1],
#                             override_color=label_soft,
#                             override_shape=[config.eval.width, config.eval.height],
#                         )["render"]
#                         label = rendering[1:].argmax(dim=0).cpu()
#                     else:
#                         rendering1 = render_chn(
#                             view,
#                             gaussians,
#                             eval_config.pipeline,
#                             background,
#                             num_channels=model_2d.embedding_dim,
#                             override_color=gaussians._features_semantic[:, :model_2d.embedding_dim],
#                             override_shape=[config.eval.width, config.eval.height],
#                         )["render"]
#                         rendering2 = render_chn(
#                             view,
#                             gaussians,
#                             eval_config.pipeline,
#                             background,
#                             num_channels=text_model.embedding_dim,
#                             override_color=gaussians._features_semantic[:, model_2d.embedding_dim:],
#                             override_shape=[config.eval.width, config.eval.height],
#                         )["render"]
#                         rendering = torch.cat([rendering1, rendering2], dim=0)
#                         rendering = rendering / torch.clamp(rendering.norm(dim=0, keepdim=True), min=1e-8)
#                         sim = torch.einsum("cq,qhw->chw", cat_text_features, rendering)
#                         label = sim[1:].argmax(dim=0).cpu()
#                 elif config.eval.feature_fusion == "argmax":
#                     if config.eval.pred_on_3d:
#                         sim_2d = torch.einsum(
#                             "cq,dq->dc", text_features_2d, gaussians._features_semantic[:, :model_2d.embedding_dim]
#                         )
#                         sim_3d = torch.einsum(
#                             "cq,dq->dc", text_features_3d, gaussians._features_semantic[:, model_2d.embedding_dim:]
#                         )
#                         sim = torch.stack([sim_2d, sim_3d], dim=1)
#                         label_soft = sim.max(dim=1).values.softmax(dim=1)
#                         rendering = render_chn(
#                             view,
#                             gaussians,
#                             eval_config.pipeline,
#                             background,
#                             num_channels=label_soft.shape[1],
#                             override_color=label_soft,
#                             override_shape=[config.eval.width, config.eval.height],
#                         )["render"]
#                         label = rendering[1:].argmax(dim=0).cpu()
#                     else:
#                         rendering1 = render_chn(
#                             view,
#                             gaussians,
#                             eval_config.pipeline,
#                             background,
#                             num_channels=model_2d.embedding_dim,
#                             override_color=gaussians._features_semantic[:, :model_2d.embedding_dim],
#                             override_shape=[config.eval.width, config.eval.height],
#                         )["render"]
#                         rendering2 = render_chn(
#                             view,
#                             gaussians,
#                             eval_config.pipeline,
#                             background,
#                             num_channels=text_model.embedding_dim,
#                             override_color=gaussians._features_semantic[:, model_2d.embedding_dim:],
#                             override_shape=[config.eval.width, config.eval.height],
#                         )["render"]
#                         rendering1 = rendering1 / torch.clamp(rendering1.norm(dim=0, keepdim=True), min=1e-8)
#                         rendering2 = rendering2 / torch.clamp(rendering2.norm(dim=0, keepdim=True), min=1e-8)
#                         sim_2d = torch.einsum("cq,qhw->chw", text_features_2d, rendering1)
#                         sim_3d = torch.einsum("cq,qhw->chw", text_features_3d, rendering2)
#                         sim = torch.stack([sim_2d, sim_3d], dim=1)
#                         label = sim[1:].max(dim=1).values.argmax(dim=0).cpu()

#                 label += 1
#                 sem = render_palette(label, palette)
#                 sem_gt = render_palette(label_img, palette)
#                 torchvision.utils.save_image(
#                     sem, os.path.join(out_path, "2d_and_3d", f"{views.camera_info[idx].image_name}.jpg")
#                 )
#                 torchvision.utils.save_image(
#                     sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg")
#                 )
#                 confusion_img = metric.confusion_matrix(
#                     label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
#                 )
#                 confusion += confusion_img

#     metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


# def eval_seg_model(config, model_3d, label_mapping):
#     eval_scene = os.listdir(config.model.model_dir)
#     eval_scene.sort()

#     model_2d_name = config.fusion.model_2d.lower().replace("_", "")
#     if model_2d_name == "openseg":
#         from model.openseg_predictor import OpenSeg

#         model_2d = OpenSeg("./weights/openseg_exported_clip", "ViT-L/14@336px")
#     elif model_2d_name == "lseg":
#         from model.lseg_predictor import LSeg

#         model_2d = LSeg("./weights/lseg/demo_e200.ckpt")
#     elif model_2d_name == "samclip":
#         from model.samclip_predictor import SAMCLIP

#         model_2d = SAMCLIP("./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px")
#     elif model_2d_name == "vlpart":
#         from model.vlpart_predictor import VLPart

#         model_2d = VLPart(
#             "./weights/vlpart/swinbase_part_0a0000.pth",
#             "./weights/vlpart/sam_vit_h_4b8939.pth",
#             "ViT-L/14@336px",
#         )

#     confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

#     for i, scene_name in enumerate(tqdm(eval_scene)):
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             eval_config = deepcopy(config)
#             eval_config.scene.scene_path = os.path.abspath(os.path.join(eval_config.scene.scene_path, scene_name))
#             eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
#             scene = Scene(eval_config.scene)

#             views = scene.getTrainCameras()
#             out_path = os.path.join("eval_render", scene_name)
#             os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
#             os.makedirs(os.path.join(out_path, model_2d_name), exist_ok=True)
#             palette, text_features = get_text_features(model_2d, config.scene.dataset_name)
#             for idx, view in enumerate(tqdm(views)):
#                 # if idx % 5 != 0:
#                 #     continue
#                 view.cuda()
#                 gt_path = str(view.image_path)
#                 label_img = get_mapped_label(config, gt_path, label_mapping)
#                 if label_img is None:
#                     continue
#                 label_img = torch.from_numpy(label_img).int().cpu()

#                 features = model_2d.extract_image_feature(
#                     gt_path,
#                     [label_img.shape[0], label_img.shape[1]],
#                 ).float()
#                 sim = torch.einsum("cq,qhw->chw", text_features.cpu(), features.cpu())
#                 label = sim.argmax(dim=0)



#                 sem = render_palette(label, palette)
#                 sem_gt = render_palette(label_img, palette)
#                 torchvision.utils.save_image(
#                     sem, os.path.join(out_path, model_2d_name, f"{views.camera_info[idx].image_name}.jpg")
#                 )
#                 torchvision.utils.save_image(
#                     sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg")
#                 )
#                 confusion += metric.confusion_matrix(
#                     label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
#                 )
#                 print(confusion)

#     metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)

# def eval_labelmap(config, model_3d, label_mapping):
#     eval_scene = os.listdir(config.model.model_dir)
#     eval_scene.sort()

#     confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)
#     from model.openseg_predictor import OpenSeg

#     model_2d = OpenSeg(None, "ViT-L/14@336px")

#     for i, scene_name in enumerate(tqdm(eval_scene)):
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             eval_config = deepcopy(config)
#             eval_config.scene.scene_path = os.path.abspath(os.path.join(eval_config.scene.scene_path, scene_name))
#             eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name, "2")
#             scene = Scene(eval_config.scene)

#             views = scene.getTrainCameras()
#             out_path = os.path.join("eval_render_langsplat_2", scene_name)
#             os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
#             os.makedirs(os.path.join(out_path, "labelmap"), exist_ok=True)
#             palette, _ = get_text_features(model_2d, config.scene.dataset_name)

#             label_map_pts = os.listdir(eval_config.model.model_dir)
#             for idx, view in enumerate(tqdm(views)):
#                 name = view.image_name.split(".")[0]
#                 if f"{name}.pt" not in label_map_pts:
#                     continue

#                 gt_path = str(view.image_path)
#                 label_img = get_mapped_label(config, gt_path, label_mapping)
#                 if label_img is None:
#                     continue
#                 label_img = torch.from_numpy(label_img).int().cpu()

#                 label = torch.load(os.path.join(eval_config.model.model_dir, f"{name}.pt"))
#                 label += 1

#                 sem = render_palette(label, palette)
#                 sem_gt = render_palette(label_img, palette)
#                 torchvision.utils.save_image(
#                     sem, os.path.join(out_path, "labelmap", f"{views.camera_info[idx].image_name}.jpg")
#                 )
#                 torchvision.utils.save_image(
#                     sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg")
#                 )
#                 confusion += metric.confusion_matrix(
#                     label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
#                 )

#     metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)

if __name__ == "__main__":
    config = OmegaConf.load("./config/eval.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    evaluate(config)