import torch

def build_affinity_matrix(query_features, gallery_features, k=2, alpha=0.5):
    """Build A(Fq, Fg, k) for GLPR.

    The confidence matrix C keeps reciprocal k-nearest neighbors with weight 1
    and keeps one-way neighbors with weight alpha. Multiplying C by cosine
    similarity gives an affinity matrix that propagates only high-confidence
    relations instead of all pairwise similarities.
    """
    # Similarity is computed on GPU, while the sparse propagation matrices stay on CPU to save memory.
    similarity_matrix = torch.mm(query_features.to("cuda"), gallery_features.to("cuda").t()).to("cpu")
    torch.cuda.empty_cache()

    # C(Fq, Fg, k): reciprocal neighbors are high-confidence, one-way neighbors are retained with alpha.
    query_knn_threshold = similarity_matrix.topk(k)[0][:, -1:]
    gallery_knn_threshold = similarity_matrix.t().topk(k)[0][:, -1:]

    query_to_gallery_mask = similarity_matrix >= query_knn_threshold
    gallery_to_query_mask = similarity_matrix.t() >= gallery_knn_threshold

    confidence_matrix = (query_to_gallery_mask & gallery_to_query_mask.t()).float()
    confidence_matrix.add_((query_to_gallery_mask & ~gallery_to_query_mask.t()).float() * alpha)

    del query_knn_threshold, gallery_knn_threshold, query_to_gallery_mask, gallery_to_query_mask

    affinity_matrix = similarity_matrix * confidence_matrix

    del similarity_matrix, confidence_matrix

    return affinity_matrix

def global_propagation(query_features, gallery_features, k):
    """Implement the global propagation stage of GLPR.

    Query and gallery features are concatenated into Fu = [Fq; Fg]. Each query
    or gallery sample aggregates neighbors from this union set, which reduces
    modality discrepancy by letting same-identity cross-modal samples smooth
    each other before local refinement.
    """
    if k <= 1:
        return query_features, gallery_features

    query_features = torch.nn.functional.normalize(query_features)
    gallery_features = torch.nn.functional.normalize(gallery_features)

    # Fu = [Fq; Fg], allowing both sets to absorb high-confidence global context.
    union_features = torch.cat([query_features, gallery_features], dim=0)

    query_to_union_affinity = build_affinity_matrix(query_features, union_features, k)
    query_features = torch.mm(query_to_union_affinity.to("cuda"), union_features.to("cuda")).to("cpu")
    del query_to_union_affinity

    gallery_to_union_affinity = build_affinity_matrix(gallery_features, union_features, k)
    gallery_features = torch.mm(gallery_to_union_affinity.to("cuda"), union_features.to("cuda")).to("cpu")
    del gallery_to_union_affinity

    del union_features

    return query_features, gallery_features

def local_propagation_refinement(query_features, gallery_features, k):
    """Implement the local refinement stage of GLPR.

    Global aggregation can blur fine identity details. This stage builds a block
    propagation matrix with intra-query, query-gallery, gallery-query, and
    intra-gallery affinities so local set structure can sharpen the globally
    smoothed features.
    """
    if k <= 1:
        return query_features, gallery_features

    query_features = torch.nn.functional.normalize(query_features)
    gallery_features = torch.nn.functional.normalize(gallery_features)

    query_to_query_affinity = build_affinity_matrix(query_features, query_features, k)
    query_to_gallery_affinity = build_affinity_matrix(query_features, gallery_features, k)
    gallery_to_query_affinity = build_affinity_matrix(gallery_features, query_features, k)
    gallery_to_gallery_affinity = build_affinity_matrix(gallery_features, gallery_features, k)

    # Block affinity matrix from Eq. (8): query-query, query-gallery, gallery-query, gallery-gallery.
    propagation_matrix = torch.cat(
        [
            torch.cat([query_to_query_affinity, query_to_gallery_affinity], dim=1),
            torch.cat([gallery_to_query_affinity, gallery_to_gallery_affinity], dim=1),
        ],
        dim=0,
    )

    del query_to_query_affinity, query_to_gallery_affinity, gallery_to_query_affinity, gallery_to_gallery_affinity

    propagation_matrix = torch.nn.functional.normalize(propagation_matrix, dim=1)
    union_features = torch.cat([query_features, gallery_features], dim=0)
    refined_features = torch.mm(propagation_matrix.to("cuda"), union_features.to("cuda")).to("cpu")

    query_count = query_features.shape[0]
    query_features = refined_features[:query_count]
    gallery_features = refined_features[query_count:]

    del propagation_matrix, union_features, refined_features

    return query_features, gallery_features

def GLPR(query_features: torch.Tensor, gallery_features: torch.Tensor, k):
    """Global-Local Propagation Refinement from the PRISM framework.

    GLPR is a post-processing step for extracted retrieval features. It does not
    update the neural network; it updates query/gallery feature tensors before
    distance computation.
    """
    if query_features.size(0) <= 1 or gallery_features.size(0) <= 1:
        return query_features, gallery_features

    # First reduce modality discrepancy globally, then recover local discriminability.
    query_features, gallery_features = global_propagation(query_features, gallery_features, k)
    query_features, gallery_features = local_propagation_refinement(query_features, gallery_features, k)

    return query_features, gallery_features
