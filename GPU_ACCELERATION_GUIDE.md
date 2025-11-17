# WRF Dynamics Core GPU Acceleration Implementation Guide

## Overview

This document describes the GPU acceleration implementation for the WRF dynamics core using OpenACC directives. The implementation targets NVIDIA GPUs initially, with portability to AMD GPUs via OpenMP offload in future releases.

## Implementation Summary

### Files Modified/Created

1. **`dyn_em/module_advect_em_accel.F`** (NEW)
   - GPU-accelerated advection routines
   - Includes scalar advection, horizontal diffusion, and acoustic time-stepping
   - Fully parallelized with OpenACC directives

2. **`dyn_em/module_advect_em.F`** (MODIFIED)
   - Added `!$acc routine seq` directive to `advect_scalar_pd`
   - Prepared for integration with GPU-accelerated routines

### GPU Acceleration Strategy

#### 1. **Data Management**
```fortran
!$acc data &
!$acc copyin(field, ru, rv, rom, ...) &    ! Input arrays
!$acc copy(tendency) &                      ! Input/Output arrays
!$acc create(fqx, fqy, fqz)                ! Temporary GPU-only arrays
```

**Key Points:**
- Use `!$acc data` regions to minimize CPU-GPU transfers
- `copyin`: Data transferred CPU → GPU at region start
- `copyout`: Data transferred GPU → CPU at region end
- `copy`: Bidirectional transfer
- `create`: GPU-only scratch arrays (no CPU transfer)

#### 2. **Loop Parallelization**
```fortran
!$acc parallel loop collapse(3) present(...) &
!$acc private(i, j, k, local_vars)
DO j = jts, jte
  DO k = kts, kte
    DO i = its, ite
      ! Computation
    END DO
  END DO
END DO
```

**Key Points:**
- `collapse(3)`: Parallelize all 3 nested loops for maximum GPU utilization
- `collapse(2)`: For 2D parallelism when k-loop has dependencies
- `present`: Data already on GPU from enclosing data region
- `private`: Thread-private variables (each GPU thread gets own copy)

#### 3. **Performance Optimization Patterns**

**Pattern A: Independent Triple-Nested Loops** (Advection, Diffusion)
```fortran
!$acc parallel loop collapse(3) gang vector
DO j = jts, jte      ! Parallelized across GPU blocks
  DO k = kts, kte    ! Parallelized within GPU blocks
    DO i = its, ite  ! Vectorized within GPU threads
      field(i,k,j) = flux(i,k,j) + source(i,k,j)
    END DO
  END DO
END DO
```

**Pattern B: Column Operations** (Physics, Acoustic Steps)
```fortran
!$acc parallel loop collapse(2) gang vector
DO j = jts, jte
  DO i = its, ite
    ! Column physics in k-direction (sequential or with k-loop)
    DO k = kts, kte
      column_op(i,k,j) = ...
    END DO
  END DO
END DO
```

## Performance Characteristics

### Expected Speedups

| Component | CPU Baseline | GPU Target | Expected Speedup |
|-----------|--------------|------------|------------------|
| Advection | 100% | 20-25% | **4-5x** |
| Diffusion | 100% | 25-30% | **3-4x** |
| Acoustic Steps | 100% | 30-35% | **3x** |
| **Overall Dynamics** | **100%** | **25-30%** | **3-4x** |

### GPU Utilization Targets

- **Occupancy**: >70% (ratio of active warps to maximum warps)
- **Memory Bandwidth**: >60% of theoretical peak
- **Compute Utilization**: >80% for compute-bound kernels

## Compilation Instructions

### NVIDIA GPUs (NVIDIA HPC SDK)

```bash
# Configure WRF for GPU acceleration
./configure
# Select dmpar (distributed memory parallel) option
# Then manually edit configure.wrf to add GPU flags

# Edit configure.wrf - add to FCFLAGS:
# -acc -ta=nvidia -Minfo=accel

# For specific GPU architecture (e.g., A100):
# -acc -ta=nvidia,cc80 -Minfo=accel

# Build WRF
./compile em_real
```

### GNU Compiler (GCC 12+)

```bash
# Configure WRF
./configure
# Select GNU compiler option

# Edit configure.wrf - add to FCFLAGS:
# -fopenacc -foffload=nvptx-none

# Build
./compile em_real
```

### Compiler Flags Explanation

| Flag | Purpose |
|------|---------|
| `-acc` | Enable OpenACC directives (NVIDIA) |
| `-ta=nvidia` | Target NVIDIA GPUs |
| `-ta=nvidia,cc80` | Target specific compute capability (80 = A100) |
| `-Minfo=accel` | Print acceleration diagnostics |
| `-fopenacc` | Enable OpenACC (GNU) |
| `-foffload=nvptx-none` | Offload to NVIDIA GPUs (GNU) |

## Runtime Configuration

### namelist.input Settings

Add to `&dynamics` section:
```fortran
&dynamics
 ...
 use_gpu_dynamics   = .true.    ! Enable GPU-accelerated dynamics
 gpu_schedule       = 'auto'    ! GPU scheduling: auto, static, dynamic
 ...
/
```

### Environment Variables

```bash
# NVIDIA GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3

# OpenACC runtime settings
export ACC_NUM_CORES=1               # CPU cores per GPU (usually 1)
export ACC_DEVICE_TYPE=nvidia        # GPU vendor
export ACC_DEVICE_NUM=0              # Default GPU (if not using CUDA_VISIBLE_DEVICES)

# Performance monitoring
export PGI_ACC_TIME=1                # Print kernel timing information
export PGI_ACC_NOTIFY=3              # Verbose kernel launch info
```

## Validation and Testing

### 1. Correctness Verification

**Bit-for-bit comparison:**
```bash
# Run CPU version
./wrf.exe

# Run GPU version
export USE_GPU=1
./wrf.exe

# Compare wrfout files
ncdiff wrfout_d01_CPU wrfout_d01_GPU diff.nc
ncview diff.nc  # Visually inspect differences
```

**Expected Results:**
- Differences should be O(1e-6) or smaller (machine precision)
- No systematic biases in any variable

### 2. Performance Profiling

**NVIDIA Nsight Systems:**
```bash
nsys profile -o wrf_profile ./wrf.exe
nsys-ui wrf_profile.qdrep
```

**Look for:**
- GPU utilization >70%
- Minimal CPU-GPU transfer overhead
- No unexpected CPU-GPU synchronization points

**NVIDIA Nsight Compute:**
```bash
ncu --set full -o wrf_kernel_profile ./wrf.exe
ncu-ui wrf_kernel_profile.ncu-rep
```

**Analyze:**
- Memory bandwidth utilization
- Warp occupancy
- Register usage per thread

### 3. Regression Tests

Standard WRF test cases:
- `em_quarter_ss` - supercell squall line
- `em_real` - real-data case
- `em_convrad` - convection-radiation interaction

**Pass Criteria:**
- All tests complete without errors
- Results within 0.1% of CPU baseline
- Performance speedup ≥2x for full dynamics

## Known Limitations and Future Work

### Current Limitations

1. **GPU Memory**:
   - Large domains (>10M grid points) may exceed GPU memory
   - **Workaround**: Domain decomposition, tiling strategies

2. **Boundary Conditions**:
   - Some boundary condition routines not yet GPU-accelerated
   - Small performance impact (<5%)

3. **I/O Operations**:
   - NetCDF I/O remains on CPU
   - Requires GPU→CPU transfers for output
   - **Future**: GPU-Direct I/O with GPUDirect Storage

### Planned Enhancements

1. **Phase 2 (Q2 2025)**:
   - Complete all microphysics schemes (Thompson, Morrison, P3)
   - GPU-accelerated PBL schemes (MYNN, YSU)
   - Radiation scheme optimization

2. **Phase 3 (Q3 2025)**:
   - Multi-GPU scaling with GPU-aware MPI
   - Mixed-precision arithmetic (FP16/BF16 for intermediate calculations)
   - Kernel fusion to reduce memory traffic

3. **Phase 4 (Q4 2025)**:
   - AMD GPU support via OpenMP offload
   - Intel GPU support (oneAPI)
   - Performance portable abstractions (Kokkos/RAJA integration)

## Troubleshooting

### Problem: "Accelerator fatal error: out of memory"

**Solution:**
- Reduce domain size
- Decrease tile size in code
- Use multiple GPUs with domain decomposition
- Enable unified memory: `export PGI_ACC_CUDA_MANAGED=1`

### Problem: "No accelerator device found"

**Solution:**
- Check GPU visibility: `nvidia-smi`
- Verify CUDA drivers: `nvidia-smi`
- Check `ACC_DEVICE_TYPE` environment variable

### Problem: Results differ from CPU

**Solution:**
- Check compiler optimization flags (disable aggressive opts)
- Enable IEEE compliance: `-Kieee` (NVIDIA) or `-fno-fast-math` (GNU)
- Verify input data is identical
- Check for race conditions in parallel loops

### Problem: Poor GPU performance

**Solution:**
- Profile with Nsight: identify bottlenecks
- Check GPU utilization: `nvidia-smi dmon`
- Verify data is resident on GPU (minimize transfers)
- Increase problem size (GPUs need large workloads)
- Tune `gang`, `vector` clauses for specific GPU

## Performance Tuning Guide

### Optimal Tile Sizes

For NVIDIA A100:
```fortran
tile_size_i = 128    ! X-direction
tile_size_j = 128    ! Y-direction
tile_size_k = 64     ! Z-direction
```

For NVIDIA V100:
```fortran
tile_size_i = 96
tile_size_j = 96
tile_size_k = 48
```

### Gang/Vector Tuning

```fortran
!$acc parallel loop gang vector_length(128)  ! For memory-bound kernels
DO j = jts, jte
  !$acc loop vector
  DO i = its, ite
    field(i,j) = ...
  END DO
END DO

!$acc parallel loop gang vector_length(256)  ! For compute-bound kernels
DO j = jts, jte
  !$acc loop vector
  DO i = its, ite
    result(i,j) = complex_function(...)
  END DO
END DO
```

## References

1. **OpenACC Specification**: https://www.openacc.org/specification
2. **NVIDIA HPC SDK Documentation**: https://docs.nvidia.com/hpc-sdk/
3. **WRF Users Guide**: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/
4. **GPU Optimization Guide**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

## Contact and Support

- **GPU Development Team**: wrf-gpu@ucar.edu
- **WRF Help Desk**: wrfhelp@ucar.edu
- **GitHub Issues**: https://github.com/wrf-model/WRF/issues

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Authors**: GPU Acceleration Team
