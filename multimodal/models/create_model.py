from models.transformer_AS import EmbCLSFusionTransformerASMasked, EmbCLSFusionTransformerASMasked_cross_attention, EmbCLSFusionTransformerASMasked_gate
from models.transformer_VS import EmbCLSFusionTransformerVSMasked, EmbCLSFusionTransformerVSMasked_cross_attention, EmbCLSFusionTransformerVSMasked_gate
from models.transformer_AV import EmbCLSFusionTransformerAV, EmbCLSFusionTransformerAV_cross_attention, EmbCLSFusionTransformerAV_gate
from models.transformer_AVS import EmbCLSFusionTransformerAVSMasked, EmbCLSFusionTransformerAVSMasked_cross_attention, EmbCLSFusionTransformerAVSMasked_gate
from models.gated import GMU_Transformer

def create_model(dim_a, dim_v, dim_s, num_labels, config):
    if config['model_name'] == "seq_transformer_avs_masked":
        if config['transformer'] == "simple":
           return EmbCLSFusionTransformerAVSMasked(dim_a, dim_v, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "gated":
           return EmbCLSFusionTransformerAVSMasked_gate(dim_a, dim_v, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "cross_attention":
           return EmbCLSFusionTransformerAVSMasked_cross_attention(dim_a, dim_v, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'])

    if config['model_name'] == "seq_transformer_as_masked":
        if config['transformer'] == "simple":
           return EmbCLSFusionTransformerASMasked(dim_a, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "gated":
           return EmbCLSFusionTransformerASMasked_gate(dim_a, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "cross_attention":
           return EmbCLSFusionTransformerASMasked_cross_attention(dim_a, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'])

    if config['model_name'] == "seq_transformer_vs_masked":
        if config['transformer'] == "simple":
           return EmbCLSFusionTransformerVSMasked(dim_v, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "gated":
           return EmbCLSFusionTransformerVSMasked_gate(dim_v, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "cross_attention":
           return EmbCLSFusionTransformerVSMasked_cross_attention(dim_v, dim_s, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'])

    if config['model_name'] == "seq_transformer_av":
        if config['transformer'] == "simple":
           return EmbCLSFusionTransformerAV(dim_a, dim_v, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "gated":
           return EmbCLSFusionTransformerAV_gate(dim_a, dim_v, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'], config['dropout'], config['max_clips'])
        elif config['transformer'] == "cross_attention":
           return EmbCLSFusionTransformerAV_cross_attention(dim_a, dim_v, num_labels, config['d_model'], config['attn_heads'], config['attn_layers'])
    raise RuntimeError(f"Unknown model: {model_name}")


def build_model(active_modalities, input_dims, num_labels, d_model=256, dropout=0.2, max_clips=200):

    model = GMU_Transformer(
        active_modalities=active_modalities,
        input_dims=input_dims,
        num_labels=num_labels,
        d_model=d_model,
        nhead=4,
        num_layers=1,
        dropout=dropout,
        max_clips=max_clips,
        use_missing_skeleton_token=True,
    )
    return model
