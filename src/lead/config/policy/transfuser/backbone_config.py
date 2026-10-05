"""TransFuser backbone and fusion-transformer architecture."""

from lead.config.node import ConfigNode


class TransfuserBackboneConfig(ConfigNode):
    """Image/LiDAR encoders and the GPT fusion layers."""

    # Backbone class as "module:ClassName"; the throughput variants under
    # lead.policy.transfuser.encoder swap in here.
    backbone_target: str = (
        "lead.policy.transfuser.encoder.transfuser_backbone:TransfuserBackbone"
    )
    # If true freeze the backbone weights during training.
    freeze_backbone: bool = False
    # If true run all normalization layers in fp32 under autocast; if false they
    # follow the autocast dtype.
    norm_layers_in_fp32: bool = True
    # Architecture name for image encoder backbone.
    image_architecture: str = "resnet34"
    # Architecture name for LiDAR encoder backbone.
    lidar_architecture: str = "resnet34"
    # Latent TF
    LTF: bool = False

    # GPT Encoder
    # Block expansion factor for GPT layers.
    block_exp: int = 4
    # Number of transformer layers used in the vision backbone.
    n_layer: int = 2
    # Number of attention heads in transformer.
    n_head: int = 4
    # Embedding dropout probability.
    embd_pdrop: float = 0.1
    # Residual connection dropout probability.
    resid_pdrop: float = 0.1
    # Attention dropout probability.
    attn_pdrop: float = 0.1
    # Deformable fusion attention; read only by the backbone_deformable_fusion
    # variant, ignored by every other backbone_target.
    # Points each query samples per attention head, per modality. The attention
    # cost is linear in this, where the dense operator is quadratic in the token
    # count.
    deformable_num_points: int = 4
    # If true each query predicts a refinement of its reference points from its
    # own content, which is what lets it learn where on the other modality's
    # grid to read. If false the reference points stay at the query's own cell
    # and the other grid's centre.
    deformable_learn_cross_reference: bool = True
    # If true seed the cross-modal reference points from the rig's calibration:
    # each BEV cell anchors where it projects into the stitched image, and each
    # image token where its ray meets the ground plane. If false they start at
    # the other grid's centre, which is the ablation isolating what the
    # geometric prior contributes.
    deformable_calibrated_reference: bool = False
    # Height above the ego's ground plane at which the two grids are put into
    # correspondence, in meters. Roughly half a car, so a BEV cell anchors near
    # the middle of whatever occupies it rather than at its footprint.
    deformable_reference_height_meter: float = 0.8

    # If true the camera branch is trained to predict the BEV grid the LiDAR
    # branch produces, so a damaged LiDAR can be stood in for rather than only
    # weighted down. The gate moves attention between what the encoders made;
    # this rebuilds what one of them stopped making. Adds a head and one
    # auxiliary loss, and changes nothing at inference on its own.
    use_cross_modal_hallucination: bool = False
    # Weight of that auxiliary loss relative to the other tasks.
    hallucination_loss_weight: float = 1.0

    # Mean of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_mean: float = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_std: float = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight: float = 1.0
