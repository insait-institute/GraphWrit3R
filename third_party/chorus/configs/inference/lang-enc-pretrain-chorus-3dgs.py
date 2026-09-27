# Standalone inference config for Chorus LangPretrainerMultiTeacher.

model = dict(
    type="LangPretrainerMultiTeacher",
    backbone=dict(
        type="PT-v3m2",
        in_channels=11,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
    projector_in_channels=1232,
    training_mode="joint",
    teachers=[
        dict(
            name="lang",
            target_key="lang_feat",
            mask_key="valid_feat_mask",
            segment_key="segment",
            projector=dict(
                out_channels=1152,
                depth=1,
                num_heads=32,
                patch_size=1024,
                drop_path_rate=0.1,
                block_type="mlp",
                input_norm=True,
                output_norm=False,
            ),
            criteria=[
                dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
                dict(type="SmoothL1Loss", reduction="mean", loss_weight=1.0),
                dict(
                    type="AggregatedContrastiveLoss",
                    temperature=0.2,
                    reduction="mean",
                    loss_weight=0.02,
                    schedule="last_75",
                ),
            ],
            teacher_norm=dict(enabled=False),
        ),
        dict(
            name="dino",
            target_key="dino_feat",
            mask_key=None,
            segment_key=None,
            projector=dict(
                out_channels=1024,
                depth=1,
                num_heads=16,
                patch_size=1024,
                drop_path_rate=0.1,
                block_type="mlp",
                input_norm=True,
                output_norm=False,
            ),
            criteria=[
                dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
                dict(type="SmoothL1Loss", reduction="mean", loss_weight=1.0),
            ],
            teacher_norm=dict(enabled=False),
        ),
    ],
)

feat_keys = ("color", "opacity", "quat", "scale")
grid_sample_keys = feat_keys + ("coord", "normal")
grid_sample_keys_test = feat_keys + ("coord", "normal")
collect_keys_test = (
    "coord",
    "grid_coord",
    "index",
)

inference = dict(
    checkpoint_hub=dict(
        repo_id="SceneSplatPro/chorus",
        revision="main",
    ),
    input_reader=dict(
        outlier_filter=dict(
            enabled=True, 
            method="mad",
            scale_mad_k=5.0,
            scale_threshold_factor=0.5,
            scale_max_real=10.0,
            scale_max_log=None,
            floater_candidate_quantile=0.995,
            floater_min_neighbors=3,
            floater_radius_mode="sqrt_scale",
            floater_radius_alpha=0.25,
            floater_radius_max=2.0,
            workers=1,
            verbose=True,
        ),
    ),
    transform=[
        dict(type="CenterShift", apply_z=True),
        dict(type="NormalizeColor"),
        dict(
            type="Copy",
            keys_dict=dict(
                coord="origin_coord",
            ),
        ),
        dict(
            type="GridSample",
            grid_size=0.025,
            hash_type="fnv",
            mode="train",
            keys=grid_sample_keys,
            apply_to_pc=False,
            return_inverse=True,
        ),
    ],
    test_cfg=dict(
        voxelize=dict(
            type="GridSample",
            grid_size=0.025,
            hash_type="fnv",
            mode="test",
            keys=grid_sample_keys_test,
            apply_to_pc=False,
            return_grid_coord=True,
        ),
        crop=None,
        post_transform=[
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=collect_keys_test,
                feat_keys=feat_keys,
            ),
        ],
        aug_transform=[
            [
                dict(
                    type="RandomRotateTargetAngle",
                    angle=[0],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ]
        ],
    ),
    chunk_size=600_000,
    save_features=dict(
        output_dir=None,
        backbone=dict(enabled=False),
        teachers=dict(
            lang=dict(enabled=True),
            dino=dict(enabled=False),
            pe_spatial=dict(enabled=False)
        ),
    ),
    default_scene_name="awesome_scene",
    return_numpy=True,
)
