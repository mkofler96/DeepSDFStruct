#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

"""
Deep Learning for Signed Distance Functions (DeepSDF)
======================================================

This submodule implements the DeepSDF approach for learning implicit neural
representations of 3D geometry. It provides complete workflows for:

- Training neural networks to represent geometry as learned SDFs
- Generating datasets from explicit geometry
- Reconstructing shapes from latent codes
- Optimizing latent codes for shape fitting

The implementation is based on "DeepSDF: Learning Continuous Signed Distance
Functions for Shape Representation" (Park et al., CVPR 2019) with extensions
for microstructured materials and lattice geometries.

Key Components
--------------

models.py
    DeepSDFModel class wrapping trained decoder networks and latent codes.
    
training.py
    Complete training pipeline including loss functions, learning rate
    schedules, and training loops.
    
data.py
    Dataset classes for loading and batching SDF samples.
    
reconstruction.py
    Methods for fitting latent codes to target geometries.
    
workspace.py
    Utilities for managing experiments, checkpoints, and results.
    
networks/
    Neural network architectures (decoders, hierarchical models).

Typical Workflow
----------------

1. Generate training data from explicit geometries::

    from DeepSDFStruct.sampling import generate_dataset
    generate_dataset(geometries, output_dir, n_samples=500000)

2. Train a DeepSDF model::

    from DeepSDFStruct.deep_sdf import training
    training.main(specs)  # specs define architecture, training params

3. Use the trained model::

    from DeepSDFStruct.SDF import SDFfromDeepSDF
    from DeepSDFStruct.deep_sdf.models import DeepSDFModel
    
    model = DeepSDFModel(decoder, latent_vectors, device)
    sdf = SDFfromDeepSDF(model, latent_code=latent_vectors[0])
    
4. Reconstruct new shapes::

    from DeepSDFStruct.deep_sdf import reconstruction
    latent_code = reconstruction.reconstruct(model, target_sdf)

For complete examples, see the example notebook and test files.
"""
