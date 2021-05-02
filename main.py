# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch
import torch.nn as nn
import numpy as np

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False


def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    # result_folder = args.pop('result_folder', 'results')
    # result_folder = osp.join(output_folder, result_folder)
    # if not osp.exists(result_folder):
    #     os.makedirs(result_folder)
    #
    # mesh_folder = args.pop('mesh_folder', 'meshes')
    # mesh_folder = osp.join(output_folder, mesh_folder)
    # if not osp.exists(mesh_folder):
    #     os.makedirs(mesh_folder)
    #
    # out_img_folder = osp.join(output_folder, 'images')
    # if not osp.exists(out_img_folder):
    #     os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, **args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)


    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        # camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    img_list = list()
    keypoints_list = list()
    camera_list = list()
    view_num = len(dataset_obj)
    view_interval = int(round(view_num / 20.0))

    for idx, data in enumerate(dataset_obj):
        if idx % view_interval != 0:
            continue

        img = data['img']
        fn = data['fn']
        keypoints = data['keypoints'][[0]]
        # Create the camera object
        focal_length = args.get('focal_length')
        camera = create_camera(focal_length_x=focal_length,
                               focal_length_y=focal_length,
                               dtype=dtype,
                               **args)

        # sets up camera
        cam_R = data['cam_R']
        cam_t = data['cam_t']
        cam_fx = data['cam_fx']
        cam_fy = data['cam_fy']
        cam_cx = data['cam_cx']
        cam_cy = data['cam_cy']

        camera.focal_length_x = torch.full([1], cam_fx)
        camera.focal_length_y = torch.full([1], cam_fy)
        camera.center = torch.tensor([cam_cx, cam_cy], dtype=dtype).unsqueeze(0)
        camera.rotation.data = torch.from_numpy(cam_R).unsqueeze(0)
        camera.translation.data = torch.from_numpy(cam_t).unsqueeze(0)
        camera.rotation.requires_grad = False
        camera.translation.requires_grad = False

        if use_cuda:
            camera = camera.to(device)

        img_list.append(img)
        keypoints_list.append(keypoints)
        camera_list.append(camera)

        print('Processing: {}'.format(data['img_path']))

    curr_result_fn = osp.join(output_folder, 'smpl_param.pkl')
    curr_mesh_fn = osp.join(output_folder, 'smpl_mesh.obj')

    gender = input_gender
    if gender == 'neutral':
        body_model = neutral_model
    elif gender == 'female':
        body_model = female_model
    elif gender == 'male':
        body_model = male_model

    fit_single_frame(img_list, keypoints_list,
                     body_model=body_model,
                     camera_list=camera_list,
                     joint_weights=joint_weights,
                     dtype=dtype,
                     output_folder=output_folder,
                     result_fn=curr_result_fn,
                     mesh_fn=curr_mesh_fn,
                     shape_prior=shape_prior,
                     expr_prior=expr_prior,
                     body_pose_prior=body_pose_prior,
                     left_hand_prior=left_hand_prior,
                     right_hand_prior=right_hand_prior,
                     jaw_prior=jaw_prior,
                     angle_prior=angle_prior,
                     **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
