def remap_cloob_key(k: str) -> str:
    if k.startswith("transformer.transformer."):
        k = k.replace("transformer.transformer.", "transformer.", 1)

    if k == "transformer.positional_embedding":
        k = "positional_embedding"
    if k == "transformer.text_projection":
        k = "text_projection"
    if k == "logit_inv_tau":
        k = "logit_scale"

    if k.startswith("transformer.token_embedding."):
        k = k.replace("transformer.token_embedding.", "token_embedding.", 1)
    if k.startswith("transformer.ln_final."):
        k = k.replace("transformer.ln_final.", "ln_final.", 1)

    return k
