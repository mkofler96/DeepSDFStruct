# Changelog

All notable changes to DeepSDFStruct are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.3.0] - 2025-11-13

### Added
- **2D mesh extraction capabilities**: Added flexisquares module for 2D mesh generation
  - New `DeepSDFStruct/flexisquares/flexisquares.py` with 741 lines of code
  - New `DeepSDFStruct/flexisquares/tables.py` with marching squares tables
- **Design of Experiments (DOE) functionality**: New `design_of_experiments.py` module (167 lines)
- **Spline features and improvements**: Enhanced torch_spline module with 48 additional changes
- **Primitives support**: Added pre-trained model for primitives
  - New `trained_models/primitives/` directory with model weights
  - New primitives specifications in `specs.json`
- **Testing infrastructure**: Added tests for flexisquares, DOE, and pretrained models
  - `test_flexisquares.py` (80 lines)
  - `test_DOE.py` (29 lines)
  - Enhanced `test_pretrained_models.py` (31 lines)

### Changed
- Enhanced `SDF.py` with 84 lines of changes for improved functionality
- Updated mesh export capabilities with 39 additional changes
- Modified lattice structure module (134 lines of changes)
- Improved mesh handling (131 lines of changes)
- Updated flexicubes module with 36 modifications
- Documentation workflow updates in `.github/workflows/docs.yml`

### Files Modified
- 28 files changed: 3,199 insertions(+), 323 deletions(-)

---

## [v1.2.0] - 2025-10-20

### Added
- **Free-Form Deformation (FFD) support**: Random FFD data augmentation capability
- **Enhanced activation functions**: Added freedom to configure activation functions in neural networks
- **Training improvements**: Added logarithmic scale plotting for loss visualization
- **Model loading tests**: New test to verify model loading functionality
- **Training data generation**: Added comprehensive summary for training data generation

### Changed
- **Loss visualization**: Changed loss axis scale to logarithmic for better interpretation
- **Training plotting**: Removed specific plot types and added log plot to training module
- **Shape handling**: Fixed shape mismatch errors and added assertions for better validation
- **Surface sampling**: Fixed issue where surf samples were only available for trimesh meshes

### Fixed
- Shape mismatch errors in neural network operations
- Folder name inconsistencies
- Test specifications updated

### Files Modified
- 12 files changed: 257 insertions(+), 210 deletions(-)

---

## [v1.1.0] - 2025-10-15

### Added
- **Object file support**: Added capability to generate training data from .obj files
- **Hierarchical neural network architecture**: New hierarchical deep SDF decoder
  - File renamed: `deep_sdf_decoder_with_homogen.py` → `hierarchical_deep_sdf_decoder.py`
- **Local shapes support**: New `local_shapes.py` module (172 lines)
- **Structural optimization**: Complete optimization module using torchfem (128 lines)
  - New `test_structural_optimization.py` with 151 test cases
- **Mesh processing enhancements**:
  - Added functionality to remove disconnected mesh regions
  - Added early return optimization when no disconnected regions exist
  - Example disconnected mesh file for testing (1,784 lines)
- **Chair dataset**: Added example .obj files (1005.obj, 1024.obj) with README
- **Deep local shapes reconstruction**: Enhanced reconstruction capabilities

### Changed
- **Plotting overhaul**: Replaced most plotting functionality with ParaView
- **Reconstruction improvements**: Now samples from surface instead of vertices
- **Device management**: Updated CUDA device handling, removed default CUDA assumption
- **FlexiCubes fixes**: Applied fix from nv-tlabs/FlexiCubes#12
- **Logging**: Switched from file-level to module-level logging
- **Training workflow**: Removed main from training.py (should not be standalone)

### Removed
- **Homogenization functionality**: Deleted `homogenization_sdf` module
- **Deep SDF utilities**: Removed `deep_sdf/utils.py` (145 lines)
- **LibTorch executable**: Removed `create_libtorch_executable.py`
- **Autograd tests**: Removed custom backward method tests (no longer needed)
- **Example files**: Cleaned up old example.py and confidential files

### Fixed
- FlexiCubes dtype errors and implementation issues
- Loss shape errors in reconstruction
- Reshape errors in various modules
- Bounds not being treated as array
- Log frequency and test run ordering
- Various float dtype issues

### Files Modified
- 40 files changed: 6,941 insertions(+), 3,591 deletions(-)
- Major refactoring with significant additions of optimization and local shapes functionality

---

## [v1.0.9] - 2025-09-25

### Removed
- **Old torch spline implementation**: Cleaned up legacy torch spline code
  - Removed 102 lines of deprecated code from `torch_spline.py`
  - Simplified implementation after nn.Module migration

### Changed
- Minor adjustments to mesh.py and parametrization.py
- Updated test files to reflect new torch spline structure

### Files Modified
- 6 files changed: 13 insertions(+), 105 deletions(-)

---

## [v1.0.8] - 2025-09-25

### Added
- **Torch spline as nn.Module**: Major enhancement to torch_spline module
  - Added 54 lines of new functionality
  - Implemented spline as proper PyTorch neural network module
- **Improved spline validation**: Better assertions for 3D splines
- **Enhanced test coverage**: Expanded test_torch_spline.py with 108 lines of improved tests

### Changed
- **Project configuration**: Updated pyproject.toml to automatically include packages
- Minor updates to parametrization.py

### Files Modified
- 4 files changed: 116 insertions(+), 55 deletions(-)

---

## [v1.0.7] - 2025-09-25

### Changed
- Added reconstructions directory to .gitignore to exclude generated files from version control

---

## Summary Statistics

### Overall Project Growth (v1.0.7 → v1.3.0)
- **Total commits**: 50+
- **Net changes**: ~10,000+ insertions, ~4,000+ deletions
- **Major features added**: 
  - 2D mesh extraction (Flexisquares)
  - Design of Experiments
  - FFD data augmentation
  - Hierarchical neural networks
  - Structural optimization
  - Local shapes support
  - Object file training data generation

### Key Development Focus Areas
1. **Neural Network Architecture**: Enhanced DeepSDF decoder with hierarchical support
2. **Mesh Processing**: Major improvements in 2D/3D mesh generation and manipulation
3. **Optimization**: Added comprehensive structural optimization capabilities
4. **Data Handling**: Improved training data generation from various formats
5. **Visualization**: Transitioned to ParaView-based plotting
6. **Code Quality**: Removed deprecated code, improved testing, fixed numerous bugs

---

[v1.3.0]: https://github.com/mkofler96/DeepSDFStruct/compare/v1.2.0...v1.3.0
[v1.2.0]: https://github.com/mkofler96/DeepSDFStruct/compare/v1.1.0...v1.2.0
[v1.1.0]: https://github.com/mkofler96/DeepSDFStruct/compare/v1.0.9...v1.1.0
[v1.0.9]: https://github.com/mkofler96/DeepSDFStruct/compare/v1.0.8...v1.0.9
[v1.0.8]: https://github.com/mkofler96/DeepSDFStruct/compare/v1.0.7...v1.0.8
[v1.0.7]: https://github.com/mkofler96/DeepSDFStruct/releases/tag/v1.0.7
