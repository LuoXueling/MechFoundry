# Diffused Nitsche vs body-fitted P1: comparison tables

Computed by `tables.py` from the stored per-entry records; accuracy measures re-evaluated from the stored solutions by `recompute_metrics.py`. Recomputed vs stored (`comparison.json`): 282 of 282 compared entries bit-for-bit equal in every checked metric (relative L² with and without rigid removal, rigid share, compliance, outside-octree fraction, Cauchy and von Mises relative L², s₉₅, s₉₉); maximum relative difference 0.0e+00.

### Degrees of freedom at octree depth 7

LaTeX label `tab:si-bf-dofs-d7`.

*Caption.* Degrees of freedom (in thousands) of diffused Nitsche (three per free node of the trimmed octree) and of the three body-fitted P1 meshes (three per vertex) at octree depth 7. Matched DOFs: vertex count targeted at the octree's node count (secant search, closest of up to six trials kept); matched surface: surface edge length h_min; graded: h_min at the surface, coarser inside like the octree. Flags, at the session's h_min: T, wall thinner than 4h_min; G, gaps narrower than h_min on over 1% of the surface; L, a load within 3h_min of a support; S, a support piece below 50h_min² or a partial clamp on a section thinner than 4h_min; blank, unflagged. —, no body-fitted mesh (the input surface self-intersects). Medians over all objects and over the unflagged ones; n, number of values. Details in the SI section “Error comparison with body-fitted FEM”.

|  |  |  | Body-fitted P1 |  |  |
|:---|:---:|---:|---:|---:|---:|
| **Object** | **Flags** | **Diffused Nitsche** | **Matched DOFs** | **Matched surface** | **Graded** |
| a TV remote control |  | 49.4 | 49.5 | 57.0 | 49.0 |
| a USB stick |  | 43.1 | 43.2 | 50.3 | 42.3 |
| an electric toothbrush |  | 17.2 | 17.2 | 11.0 | 17.2 |
| a coffee mug | T | 260 | 260 | 249 | 256 |
| a computer mouse |  | 108 | 108 | 237 | 108 |
| a school chair with a steel frame and a plastic seat | T, S | 83.2 | 59.6 | 33.0 | 82.7 |
| a glass-top dining table | T, S | 73.8 | 72.7 | 53.4 | 71.7 |
| a park bench | T, S | 94.4 | — | — | — |
| a bar stool | S | 86.1 | 84.4 | 46.5 | 83.9 |
| a sofa | S | 132 | 133 | 195 | 131 |
| a teddy bear |  | 116 | 116 | 192 | 116 |
| a wooden rocking horse | T | 98.4 | 97.1 | 63.0 | 95.7 |
| a pogo stick | S | 14.3 | 14.6 | 3.39 | 13.8 |
| a wooden toy train engine | G, L, S | 103 | 104 | 149 | 102 |
| a yo-yo | G | 339 | 371 | 761 | 337 |
| a jelly dessert |  | 200 | 202 | 609 | 200 |
| a loaf of bread |  | 235 | 238 | 843 | 247 |
| a hamburger | G | 237 | 238 | 837 | 248 |
| a layered birthday cake |  | 196 | 197 | 570 | 195 |
| a slice of watermelon |  | 113 | 113 | 245 | 113 |
| a giraffe | S | 36.4 | 35.3 | 23.9 | 35.1 |
| an elephant | S | 95.8 | 96.3 | 127 | 95.5 |
| a rhinoceros | S | 79.1 | 79.8 | 114 | 78.5 |
| a horse | S | 86.1 | 85.9 | 95.8 | 85.3 |
| a crocodile | S | 40.3 | 39.6 | 33.7 | 38.6 |
| a car suspension control arm with rubber bushings |  | 82.1 | 81.7 | 76.9 | 80.1 |
| a truck leaf spring |  | 16.0 | 16.0 | 7.00 | 15.2 |
| a car wheel with its rubber tyre | G | 308 | 279 | 708 | 318 |
| a car brake disc | S | 184 | — | — | — |
| a crankshaft | G | 65.3 | 64.8 | 68.9 | 64.6 |
| a road roller with a steel drum and rubber tyres | T, G, S | 101 | 101 | 64.1 | 101 |
| a jackhammer | S | 21.9 | 21.3 | 12.7 | 21.4 |
| an industrial robot arm | G | 90.3 | 91.0 | 97.8 | 87.1 |
| an adjustable wrench | T, S | 18.7 | 18.3 | 8.36 | 18.8 |
| a claw hammer |  | 27.7 | 27.2 | 18.4 | 27.2 |
| a dental crown |  | 357 | 370 | 1300 | 364 |
| a knee implant | G | 291 | 296 | 654 | 287 |
| a walking cane with a rubber tip | T, L, S | 8.27 | 8.42 | 1.99 | 8.09 |
| a human femur |  | 36.6 | 35.7 | 28.4 | 35.5 |
| a lumbar vertebra |  | 99.6 | 99.3 | 142 | 99.3 |
| a skyscraper | L | 52.8 | 53.0 | 80.8 | 51.8 |
| a thatched stone cottage | L | 219 | 221 | 720 | 218 |
| a lighthouse |  | 99.7 | 91.0 | 160 | 97.0 |
| an igloo |  | 245 | 245 | 292 | 244 |
| a log cabin | L | 264 | 270 | 814 | 263 |
| the Leaning Tower of Pisa | T, G, S | 86.0 | 182 | 228 | 127 |
| the Pantheon | L | 206 | 207 | 256 | 206 |
| Himeji Castle | G | 313 | — | — | — |
| a natural stone arch | G | 193 | 196 | 353 | 192 |
| the Taj Mahal |  | 209 | 213 | 422 | 207 |
| *Median (all objects)* |  | 99.0 | 97.1 | 127 | 97.0 |
| *n* |  | 50 | 47 | 47 | 47 |
| *Median (unflagged)* |  | 104 | 104 | 176 | 103 |
| *n* |  | 18 | 18 | 18 | 18 |

### Discrepancy at octree depth 7

LaTeX label `tab:si-bf-accuracy-d7`.

*Caption.* Discrepancy of diffused Nitsche from each body-fitted P1 solution at octree depth 7. η_rig and η_rig^H¹: relative L² and H¹ norms of the displacement difference after removing its best-fit rigid motion (in %); ℓ_N/ℓ_P1: compliance ratio (below 1: Nitsche stiffer); s₉₅, s₉₉: Nitsche-to-P1 ratios of the 95th and 99th percentiles of the von Mises stress. Flags, at the session's h_min: T, wall thinner than 4h_min; G, gaps narrower than h_min on over 1% of the surface; L, a load within 3h_min of a support; S, a support piece below 50h_min² or a partial clamp on a section thinner than 4h_min; blank, unflagged. Rows in bold (gray in the PDF and xlsx): η_rig above 10% against at least one mesh. —, no body-fitted mesh (the input surface self-intersects). Medians over all objects and over the unflagged ones; n, number of values. Details in the SI section “Error comparison with body-fitted FEM”.

|  |  | Matched DOFs |  |  |  |  | Matched surface |  |  |  |  | Graded |  |  |  |  |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Object** | **Flags** | **η_rig (%)** | **η_rig^H¹ (%)** | **ℓ_N/ℓ_P1** | **s₉₅** | **s₉₉** | **η_rig (%)** | **η_rig^H¹ (%)** | **ℓ_N/ℓ_P1** | **s₉₅** | **s₉₉** | **η_rig (%)** | **η_rig^H¹ (%)** | **ℓ_N/ℓ_P1** | **s₉₅** | **s₉₉** |
| a TV remote control |  | 3.7 | 7.0 | 0.909 | 0.924 | 0.893 | 2.9 | 5.8 | 0.935 | 0.942 | 0.910 | 2.8 | 5.7 | 0.937 | 0.943 | 0.912 |
| a USB stick |  | 1.4 | 4.4 | 0.871 | 0.886 | 0.841 | 1.1 | 3.8 | 0.886 | 0.904 | 0.847 | 1.0 | 3.6 | 0.886 | 0.907 | 0.851 |
| **an electric toothbrush** |  | 15 | 22 | 0.773 | 0.828 | 0.755 | 8.4 | 13 | 0.876 | 0.858 | 0.823 | 12 | 17 | 0.824 | 0.826 | 0.761 |
| a coffee mug | T | 4.7 | 10 | 0.869 | 0.885 | 0.842 | 4.6 | 10 | 0.872 | 0.887 | 0.840 | 4.8 | 10 | 0.868 | 0.884 | 0.837 |
| a computer mouse |  | 7.0 | 21 | 0.935 | 0.972 | 0.937 | 6.4 | 21 | 0.945 | 0.974 | 0.933 | 6.1 | 21 | 0.949 | 0.977 | 0.932 |
| **a school chair with a steel frame and a plastic seat** | T, S | 34 | 46 | 0.545 | 0.618 | 0.542 | 32 | 42 | 0.599 | 0.649 | 0.546 | 35 | 47 | 0.533 | 0.605 | 0.513 |
| **a glass-top dining table** | T, S | 16 | 28 | 0.852 | 0.893 | 0.953 | 12 | 21 | 0.907 | 0.940 | 1.003 | 15 | 26 | 0.870 | 0.909 | 0.969 |
| a park bench | T, S | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **a bar stool** | S | 15 | 30 | 0.862 | 0.911 | 0.818 | 12 | 27 | 0.865 | 0.901 | 0.831 | 15 | 31 | 0.845 | 0.888 | 0.803 |
| a sofa | S | 1.9 | 9.2 | 0.937 | 0.908 | 0.920 | 2.1 | 9.7 | 0.939 | 0.904 | 0.918 | 2.0 | 9.6 | 0.943 | 0.912 | 0.911 |
| a teddy bear |  | 1.5 | 4.2 | 0.831 | 0.878 | 0.893 | 1.6 | 4.3 | 0.837 | 0.927 | 0.877 | 1.6 | 4.3 | 0.839 | 0.928 | 0.876 |
| a wooden rocking horse | T | 9.1 | 16 | 0.857 | 0.883 | 0.888 | 8.9 | 15 | 0.857 | 0.887 | 0.881 | 10 | 17 | 0.844 | 0.873 | 0.873 |
| **a pogo stick** | S | 20 | 71 | 0.864 | 0.782 | 1.068 | 9.5 | 35 | 0.881 | 0.766 | 1.017 | 9.1 | 35 | 0.845 | 0.732 | 1.014 |
| a wooden toy train engine | G, L, S | 8.4 | 42 | 0.883 | 0.950 | 0.823 | 8.4 | 42 | 0.886 | 0.952 | 0.822 | 8.0 | 42 | 0.886 | 0.949 | 0.825 |
| **a yo-yo** | G | 21 | 61 | 0.461 | 0.676 | 0.565 | 21 | 62 | 0.454 | 0.669 | 0.554 | 21 | 61 | 0.460 | 0.673 | 0.561 |
| a jelly dessert |  | 2.7 | 23 | 0.957 | 0.958 | 0.963 | 3.2 | 22 | 0.963 | 0.948 | 0.960 | 2.6 | 23 | 0.969 | 0.962 | 0.974 |
| a loaf of bread |  | 4.8 | 23 | 0.886 | 0.924 | 0.863 | 4.5 | 23 | 0.902 | 0.938 | 0.877 | 4.2 | 23 | 0.907 | 0.944 | 0.879 |
| a hamburger | G | 7.1 | 23 | 0.911 | 0.937 | 0.859 | 7.6 | 24 | 0.908 | 0.927 | 0.843 | 5.8 | 24 | 0.930 | 0.944 | 0.869 |
| a layered birthday cake |  | 3.0 | 19 | 0.976 | 0.969 | 0.979 | 2.8 | 20 | 0.985 | 0.961 | 0.978 | 2.4 | 21 | 0.989 | 0.962 | 0.977 |
| a slice of watermelon |  | 4.0 | 21 | 0.923 | 0.951 | 0.897 | 3.7 | 20 | 0.926 | 0.949 | 0.886 | 3.4 | 20 | 0.931 | 0.953 | 0.889 |
| a giraffe | S | 2.7 | 7.5 | 1.016 | 0.946 | 1.180 | 2.1 | 6.1 | 1.030 | 0.939 | 1.146 | 2.5 | 7.1 | 1.019 | 0.932 | 1.152 |
| an elephant | S | 1.1 | 4.1 | 0.981 | 0.974 | 0.981 | 1.2 | 4.2 | 0.984 | 0.979 | 0.984 | 1.2 | 4.2 | 0.984 | 0.976 | 0.986 |
| a rhinoceros | S | 2.2 | 8.3 | 0.945 | 0.940 | 0.976 | 2.3 | 8.4 | 0.952 | 0.947 | 0.976 | 2.3 | 8.4 | 0.954 | 0.954 | 0.977 |
| **a horse** | S | 28 | 62 | 0.861 | 0.915 | 0.942 | 27 | 59 | 0.859 | 0.909 | 0.942 | 27 | 59 | 0.848 | 0.910 | 0.942 |
| a crocodile | S | 8.0 | 16 | 0.634 | 0.759 | 0.602 | 8.0 | 16 | 0.637 | 0.747 | 0.598 | 8.1 | 17 | 0.627 | 0.741 | 0.608 |
| a car suspension control arm with rubber bushings |  | 7.5 | 23 | 0.912 | 0.934 | 0.937 | 4.6 | 16 | 0.893 | 0.916 | 0.909 | 4.6 | 16 | 0.891 | 0.915 | 0.908 |
| **a truck leaf spring** |  | 24 | 37 | 0.710 | 0.693 | 0.642 | 17 | 27 | 0.793 | 0.742 | 0.727 | 22 | 33 | 0.736 | 0.689 | 0.672 |
| a car wheel with its rubber tyre | G | 7.4 | 16 | 0.822 | 0.901 | 0.816 | 8.3 | 18 | 0.799 | 0.882 | 0.795 | 7.7 | 17 | 0.816 | 0.897 | 0.811 |
| a car brake disc | S | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **a crankshaft** | G | 12 | 31 | 0.469 | 0.511 | 0.445 | 12 | 30 | 0.494 | 0.532 | 0.475 | 12 | 30 | 0.494 | 0.532 | 0.469 |
| **a road roller with a steel drum and rubber tyres** | T, G, S | 51 | 57 | 0.372 | 0.550 | 0.807 | 47 | 54 | 0.399 | 0.571 | 0.841 | 49 | 56 | 0.384 | 0.558 | 0.825 |
| **a jackhammer** | S | 6.6 | 24 | 0.547 | 0.856 | 0.539 | 11 | 32 | 0.605 | 0.821 | 0.542 | 13 | 38 | 0.549 | 0.814 | 0.494 |
| **an industrial robot arm** | G | 23 | 54 | 0.424 | 0.649 | 0.573 | 23 | 54 | 0.417 | 0.640 | 0.564 | 23 | 54 | 0.418 | 0.641 | 0.565 |
| an adjustable wrench | T, S | 2.1 | 7.1 | 1.077 | 1.099 | 1.092 | 2.6 | 8.8 | 1.094 | 1.112 | 1.105 | 2.5 | 8.7 | 1.083 | 1.103 | 1.091 |
| a claw hammer |  | 1.6 | 5.9 | 0.879 | 0.874 | 0.868 | 1.8 | 6.5 | 0.872 | 0.866 | 0.855 | 2.0 | 7.1 | 0.862 | 0.860 | 0.844 |
| a dental crown |  | 7.7 | 24 | 0.943 | 0.866 | 0.811 | 7.7 | 25 | 0.945 | 0.860 | 0.807 | 7.5 | 25 | 0.946 | 0.864 | 0.808 |
| **a knee implant** | G | 80 | 99 | 0.403 | 0.437 | 0.340 | 80 | 99 | 0.398 | 0.432 | 0.337 | 80 | 99 | 0.406 | 0.438 | 0.341 |
| **a walking cane with a rubber tip** | T, L, S | 24 | 39 | 0.921 | 0.807 | 0.741 | 18 | 30 | 0.938 | 0.862 | 0.741 | 21 | 35 | 0.925 | 0.837 | 0.703 |
| a human femur |  | 0.64 | 2.2 | 0.974 | 0.973 | 0.963 | 1.2 | 3.2 | 0.956 | 0.955 | 0.953 | 1.4 | 3.7 | 0.948 | 0.941 | 0.936 |
| a lumbar vertebra |  | 6.9 | 23 | 0.933 | 0.898 | 0.885 | 6.7 | 23 | 0.946 | 0.902 | 0.889 | 6.6 | 23 | 0.946 | 0.903 | 0.890 |
| a skyscraper | L | 0.35 | 0.37 | 0.992 | 0.981 | 0.979 | 0.34 | 0.37 | 1.004 | 0.980 | 0.983 | 0.32 | 0.34 | 1.003 | 0.981 | 0.982 |
| a thatched stone cottage | L | 0.73 | 4.9 | 0.983 | 0.995 | 1.006 | 0.62 | 5.3 | 1.012 | 0.990 | 0.987 | 0.59 | 5.4 | 1.009 | 0.992 | 0.992 |
| a lighthouse |  | 1.4 | 2.2 | 0.985 | 0.989 | 0.980 | 1.8 | 2.6 | 0.995 | 0.988 | 0.986 | 1.8 | 2.6 | 0.995 | 0.988 | 0.988 |
| an igloo |  | 2.4 | 9.1 | 0.986 | 0.965 | 0.949 | 2.4 | 9.2 | 0.988 | 0.965 | 0.948 | 2.4 | 9.3 | 0.988 | 0.965 | 0.949 |
| a log cabin | L | 2.3 | 12 | 0.916 | 0.942 | 0.886 | 2.9 | 15 | 0.909 | 0.938 | 0.883 | 2.7 | 14 | 0.913 | 0.944 | 0.883 |
| the Leaning Tower of Pisa | T, G, S | 4.4 | 4.9 | 0.946 | 0.958 | 0.962 | 4.3 | 4.9 | 0.942 | 0.952 | 0.959 | 4.3 | 4.8 | 0.968 | 1.010 | 0.977 |
| the Pantheon | L | 2.3 | 2.5 | 0.998 | 0.981 | 0.937 | 2.3 | 2.5 | 1.002 | 0.978 | 0.951 | 2.2 | 2.4 | 1.001 | 0.978 | 0.944 |
| Himeji Castle | G | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a natural stone arch | G | 9.0 | 11 | 0.686 | 0.798 | 0.700 | 9.1 | 11 | 0.685 | 0.796 | 0.692 | 9.0 | 11 | 0.687 | 0.796 | 0.694 |
| the Taj Mahal |  | 2.1 | 2.4 | 0.976 | 0.978 | 0.972 | 1.8 | 2.2 | 1.029 | 0.981 | 0.970 | 1.6 | 2.0 | 1.013 | 0.985 | 0.964 |
| *Median (all objects)* |  | 6.6 | 19 | 0.909 | 0.911 | 0.888 | 4.6 | 16 | 0.907 | 0.916 | 0.883 | 4.8 | 17 | 0.907 | 0.912 | 0.883 |
| *n* |  | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 |
| *Median (unflagged)* |  | 3.4 | 20 | 0.928 | 0.929 | 0.895 | 3.0 | 14 | 0.940 | 0.940 | 0.899 | 2.7 | 17 | 0.941 | 0.942 | 0.899 |
| *n* |  | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 |

### Computational cost at octree depth 7

LaTeX label `tab:si-bf-cost-d7`.

*Caption.* Computational cost at octree depth 7: retries and wall-clock times in seconds (GPU, after a warm-up) of the successful attempt; in parentheses, the total over all attempts where failed attempts add more than 0.5%. Solver retries: attempts after the first; mesh retries: failed TetGen calls plus rejected remeshes of the surface the kept mesh was built from (rejections inside unkept size-search trials not counted). Prep.: preparation (Nitsche: boundary conditions, SDF, octree, trimming, solver setup; P1: surface resizing, TetGen, boundary mapping, solver setup). Solve: Newton iteration to relative residual 10⁻⁶; to 10⁻²: the same solve until it first reaches 10⁻². Superscripts: <sup>C</sup>, CPU solve; <sup>R</sup>, reduced GPU solver; both not comparable with GPU times. Flags, at the session's h_min: T, wall thinner than 4h_min; G, gaps narrower than h_min on over 1% of the surface; L, a load within 3h_min of a support; S, a support piece below 50h_min² or a partial clamp on a section thinner than 4h_min; blank, unflagged. —, no body-fitted mesh (the input surface self-intersects). Medians over all objects and over the unflagged ones; n, number of values. Time medians are of successful attempts. Details in the SI section “Error comparison with body-fitted FEM”.

|  |  | Diffused Nitsche |  |  |  | Matched DOFs |  |  |  |  | Matched surface |  |  |  |  | Graded |  |  |  |  |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Object** | **Flags** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** | **Mesh retries** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** | **Mesh retries** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** | **Mesh retries** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** |
| a TV remote control |  | 0 | 0.336 | 0.725 | 0.725 | 0 | 0 | 4.46 | 0.654 | 0.654 | 0 | 0 | 4.71 | 0.924 | 0.704 | 0 | 0 | 4.58 | 0.646 | 0.646 |
| a USB stick |  | 0 | 0.305 | 0.868 | 0.603 | 0 | 0 | 4.26 | 0.844 | 0.644 | 0 | 0 | 4.36 | 0.963 | 0.710 | 0 | 0 | 4.47 | 1.09 | 0.630 |
| an electric toothbrush |  | 0 | 0.172 | 0.939 | 0.785 | 0 | 0 | 3.63 | 2.17 | 1.44 | 0 | 0 | 3.55 | 1.60 | 1.15 | 0 | 0 | 3.83 | 1.87 | 1.50 |
| a coffee mug | T | 0 | 1.40 | 4.13 | 2.22 | 0 | 0 | 14.5 | 4.36 | 1.65 | 0 | 0 | 14.1 | 4.14 | 1.58 | 0 | 0 | 16.3 | 4.25 | 1.63 |
| a computer mouse |  | 0 | 0.585 | 0.731 | 0.411 | 0 | 0 | 6.67 | 0.800 | 0.525 | 0 | 0 | 11.4 | 1.50 | 0.950 | 0 | 0 | 7.39 | 1.08 | 0.747 |
| a school chair with a steel frame and a plastic seat | T, S | 0 | 0.491 | 3.60 | 2.51 | 0 | 1 | 6.55 | 2.96 (4.13) | 1.79 | 0 | 1 | 5.78 | 2.54 (3.53) | 1.52 | 0 | 1 | 9.40 | 4.55 (5.91) | 2.29 |
| a glass-top dining table | T, S | 0 | 0.677 | 2.77 | 1.80 | 0 | 0 | 8.37 | 2.25 | 1.61 | 0 | 0 | 16.7 | 1.83 | 1.31 | 0 | 0 | 15.1 | 1.82 | 1.26 |
| a park bench | T, S | 0 | 0.695 | 3.40 | 2.27 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a bar stool | S | 0 | 0.498 | 2.64 | 1.72 | 0 | 0 | 6.47 | 2.92 | 1.77 | 0 | 1 | 5.42 | 1.93 (3.04) | 1.34 | 0 | 1 | 7.54 | 2.73 (4.11) | 1.88 |
| a sofa | S | 0 | 2.33 | 5.79 | 3.07 | 0 | 0 | 7.97 | 1.42 | 0.870 | 0 | 0 | 10.5 | 1.98 | 1.21 | 0 | 0 | 9.10 | 1.39 | 0.867 |
| a teddy bear |  | 0 | 0.603 | 3.08 | 1.89 | 0 | 0 | 7.30 | 2.82 | 2.30 | 0 | 0 | 11.4 | 4.37 | 3.57 | 0 | 0 | 8.26 | 2.81 | 2.30 |
| a wooden rocking horse | T | 0 | 0.573 | 1.42 | 1.42 | 0 | 0 | 6.79 | 2.17 | 1.51 | 0 | 0 | 5.79 | 1.58 | 1.12 | 0 | 0 | 7.87 | 2.18 | 1.51 |
| a pogo stick | S | 0 | 0.173 | 0.766 | 0.489 | 0 | 0 | 3.67 | 2.17 | 0.704 | 0 | 0 | 3.55 | 0.991 | 0.478 | 0 | 0 | 3.87 | 1.77 | 0.712 |
| a wooden toy train engine | G, L, S | 0 | 0.713 | 1.10 | 0.632 | 0 | 0 | 12.0 | 0.988 | 0.645 | 0 | 0 | 19.8 | 1.58 | 1.06 | 0 | 0 | 14.7 | 1.28 | 0.854 |
| a yo-yo | G | 0 | 1.70 | 3.10 | 1.63 | 0 | 0 | 18.4 | 4.67 | 1.93 | 0 | 0 | 33.8 | 10.3 | 7.27 | 0 | 0 | 19.3 | 3.15 | 1.85 |
| a jelly dessert |  | 0 | 2.16 | 6.21 | 5.25 | 0 | 0 | 25.1 | 2.21 | 1.68 | 0 | 0 | 58.3 | 8.80 | 7.05 | 0 | 0 | 17.5 | 3.98 | 3.25 |
| a loaf of bread |  | 0 | 1.17 | 24.5 | 1.52 | 0 | 0 | 13.2 | 11.5 | 1.62 | 0 | 0 | 36.2 | 40.1 | 5.47 | 0 | 0 | 15.1 | 11.4 | 1.60 |
| a hamburger | G | 0 | 1.18 | 21.8 | 1.53 | 0 | 0 | 13.0 | 11.8 | 1.58 | 0 | 0 | 34.7 | 43.9 | 5.33 | 0 | 0 | 14.9 | 11.7 | 1.58 |
| a layered birthday cake |  | 0 | 1.60 | 6.03 | 4.34 | 0 | 0 | 11.3 | 4.37 | 2.99 | 0 | 0 | 36.3 | 11.0 | 6.83 | 0 | 0 | 26.8 | 3.27 | 2.26 |
| a slice of watermelon |  | 0 | 0.643 | 1.41 | 0.749 | 0 | 0 | 6.86 | 1.02 | 0.577 | 0 | 0 | 11.6 | 2.00 | 1.08 | 0 | 0 | 7.95 | 0.992 | 0.563 |
| a giraffe | S | 0 | 0.260 | 1.26 | 0.865 | 0 | 0 | 4.22 | 1.58 | 1.12 | 0 | 0 | 4.08 | 1.35 | 0.956 | 0 | 0 | 4.65 | 2.08 | 1.14 |
| an elephant | S | 0 | 0.508 | 1.48 | 0.788 | 0 | 0 | 6.41 | 1.34 | 0.736 | 0 | 0 | 7.35 | 1.61 | 0.850 | 0 | 0 | 7.13 | 1.31 | 0.698 |
| a rhinoceros | S | 0 | 0.459 | 0.864 | 0.477 | 0 | 0 | 5.66 | 1.40 | 0.614 | 0 | 0 | 6.62 | 1.79 | 0.775 | 0 | 0 | 6.08 | 1.00 | 0.616 |
| a horse | S | 0 | 0.668 | 2.79 | 1.77 | 0 | 0 | 14.6 | 3.78 | 2.04 | 0 | 0 | 16.5 | 3.16 | 2.21 | 0 | 0 | 7.76 | 4.31 | 1.83 |
| a crocodile | S | 0 | 0.435 | 0.894 | 0.894 | 0 | 0 | 10.6 | 1.85 | 1.32 | 0 | 0 | 10.5 | 1.70 | 1.22 | 0 | 0 | 11.8 | 1.86 | 1.32 |
| a car suspension control arm with rubber bushings |  | 0 | 0.547 | 1.65 | 1.16 | 0 | 0 | 6.03 | 1.67 | 1.20 | 0 | 0 | 5.96 | 1.52 | 1.08 | 0 | 0 | 6.95 | 1.59 | 1.13 |
| a truck leaf spring |  | 0 | 0.188 | 0.528 | 0.349 | 0 | 0 | 3.75 | 0.999 | 0.799 | 0 | 0 | 3.70 | 0.713 | 0.410 | 0 | 0 | 3.94 | 0.956 | 0.760 |
| a car wheel with its rubber tyre | G | 0 | 1.52 | 4.58 | 2.49 | 0 | 0 | 12.3 | 3.07 | 1.67 | 0 | 0 | 23.4 | 7.89 | 4.20 | 0 | 0 | 13.0 | 3.52 | 1.92 |
| a car brake disc | S | 0 | 1.02 | 1.76 | 0.987 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a crankshaft | G | 0 | 0.462 | 0.759 | 0.759 | 0 | 0 | 5.66 | 1.17 | 0.846 | 0 | 0 | 5.72 | 0.864 | 0.864 | 0 | 0 | 5.96 | 1.20 | 0.874 |
| a road roller with a steel drum and rubber tyres | T, G, S | 0 | 1.07 | 1.78 | 0.920 | 0 | 0 | 21.5 | 3.28 | 2.35 | 0 | 0 | 32.5 | 3.04 | 2.16 | 0 | 0 | 35.3 | 5.19 | 3.96 |
| a jackhammer | S | 0 | 0.200 | 2.04 | 0.795 | 0 | 3 | 3.97 | 2.46<sup>C</sup> (33.5) | 1.70<sup>C</sup> | 0 | 3 | 3.85 | 1.52<sup>C</sup> (22.9) | 1.07<sup>C</sup> | 0 | 4 | 4.22 | 2.48<sup>C</sup> (44.8) | 1.71<sup>C</sup> |
| an industrial robot arm | G | 0 | 0.537 | 1.66 | 1.10 | 0 | 0 | 6.78 | 2.23 | 1.23 | 0 | 0 | 6.93 | 2.37 | 1.31 | 0 | 0 | 7.65 | 2.16 | 1.19 |
| an adjustable wrench | T, S | 0 | 0.188 | 0.694 | 0.495 | 0 | 0 | 3.79 | 1.57 | 0.652 | 0 | 0 | 3.54 | 0.917 | 0.462 | 0 | 0 | 3.95 | 1.28 | 0.616 |
| a claw hammer |  | 0 | 0.227 | 1.08 | 0.565 | 0 | 0 | 3.91 | 1.18 | 0.670 | 0 | 0 | 3.78 | 0.993 | 0.589 | 0 | 0 | 4.17 | 1.52 | 0.739 |
| a dental crown |  | 0 | 1.55 | 2.47 | 1.30 | 0 | 0 | 21.9 | 3.10 | 2.03 | 0 | 3 | 76.1 | 148<sup>C</sup> | 139<sup>C</sup> | 0 | 0 | 18.5 | 2.21 | 1.36 |
| a knee implant | G | 0 | 1.49 | 3.18 | 1.61 | 0 | 0 | 14.3 | 3.07 | 1.76 | 0 | 0 | 26.0 | 6.92 | 3.92 | 0 | 0 | 12.0 | 2.79 | 1.60 |
| a walking cane with a rubber tip | T, L, S | 0 | 0.142 | 0.417 | 0.275 | 0 | 0 | 3.48 | 0.654 | 0.482 | 0 | 0 | 3.44 | 0.455 | 0.341 | 0 | 0 | 3.67 | 0.667 | 0.490 |
| a human femur |  | 0 | 0.257 | 1.08 | 0.659 | 0 | 0 | 4.22 | 1.57 | 0.904 | 0 | 0 | 4.11 | 1.65 | 0.851 | 0 | 0 | 4.71 | 1.89 | 0.943 |
| a lumbar vertebra |  | 0 | 0.897 | 1.42 | 0.709 | 0 | 0 | 16.8 | 1.04 | 0.634 | 0 | 0 | 41.4 | 1.55 | 0.955 | 0 | 0 | 7.80 | 0.991 | 0.603 |
| a skyscraper | L | 0 | 0.324 | 0.528 | 0.307 | 0 | 0 | 4.53 | 0.670 | 0.432 | 0 | 0 | 5.28 | 0.834 | 0.523 | 0 | 0 | 4.88 | 0.650 | 0.421 |
| a thatched stone cottage | L | 0 | 1.08 | 2.43 | 1.29 | 0 | 0 | 11.7 | 1.72 | 0.947 | 0 | 0 | 30.3 | 5.48 | 2.94 | 0 | 0 | 13.0 | 1.67 | 0.928 |
| a lighthouse |  | 0 | 0.671 | 0.979 | 0.584 | 0 | 0 | 7.31 | 1.00 | 0.615 | 0 | 0 | 10.8 | 1.59 | 0.984 | 0 | 0 | 8.38 | 1.06 | 0.659 |
| an igloo |  | 0 | 1.41 | 1.84 | 0.943 | 0 | 0 | 12.1 | 1.68 | 1.05 | 0 | 0 | 13.9 | 1.95 | 1.21 | 0 | 0 | 13.7 | 1.62 | 1.01 |
| a log cabin | L | 0 | 1.29 | 1.83 | 0.977 | 0 | 0 | 14.5 | 1.95 | 1.20 | 0 | 0 | 34.3 | 6.09 | 3.63 | 0 | 0 | 16.1 | 1.88 | 1.15 |
| the Leaning Tower of Pisa | T, G, S | 0 | 0.593 | 0.993 | 0.555 | 2 | 1 | 22.2 | 8.48 (10.4) | 3.21 | 2 | 1 | 24.6 | 6.40 (8.66) | 1.64 | 8 | 1 | 21.6 (48.4) | 4.89 (6.53) | 1.99 |
| the Pantheon | L | 0 | 1.06 | 1.61 | 0.891 | 0 | 0 | 11.3 | 1.50 | 0.957 | 0 | 0 | 13.4 | 1.80 | 1.14 | 0 | 0 | 12.8 | 1.44 | 0.915 |
| Himeji Castle | G | 0 | 1.52 | 3.82 | 2.04 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a natural stone arch | G | 0 | 1.45 | 2.05 | 1.15 | 0 | 0 | 33.0 | 2.15 | 0.970 | 0 | 0 | 36.7 | 5.99 | 3.10 | 0 | 0 | 12.4 | 1.75 | 1.10 |
| the Taj Mahal |  | 0 | 1.03 | 2.55 | 1.36 | 0 | 0 | 11.7 | 2.03 | 1.13 | 0 | 0 | 18.8 | 3.85 | 2.15 | 0 | 0 | 13.7 | 1.88 | 1.05 |
| *Median (all objects)* |  | 0 | 0.655 | 1.71 | 0.960 | 0 | 0 | 7.31 | 1.95 | 1.13 | 0 | 0 | 11.4 | 1.83 | 1.21 | 0 | 0 | 8.26 | 1.86 | 1.13 |
| *n* |  | 50 | 50 | 50 | 50 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 |
| *Median (unflagged)* |  | 0 | 0.623 | 1.42 | 0.767 | 0 | 0 | 7.08 | 1.62 | 0.975 | 0 | 0 | 11.4 | 1.62 | 1.08 | 0 | 0 | 7.87 | 1.60 | 0.976 |
| *n* |  | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 |

### Degrees of freedom at octree depth 8

LaTeX label `tab:si-bf-dofs-d8`.

*Caption.* Degrees of freedom (in thousands) of diffused Nitsche (three per free node of the trimmed octree) and of the three body-fitted P1 meshes (three per vertex) at octree depth 8. Matched DOFs: vertex count targeted at the octree's node count (secant search, closest of up to six trials kept); matched surface: surface edge length h_min; graded: h_min at the surface, coarser inside like the octree. Flags, at the session's h_min: T, wall thinner than 4h_min; G, gaps narrower than h_min on over 1% of the surface; L, a load within 3h_min of a support; S, a support piece below 50h_min² or a partial clamp on a section thinner than 4h_min; blank, unflagged. —, no body-fitted mesh (the input surface self-intersects). Medians over all objects and over the unflagged ones; n, number of values. Details in the SI section “Error comparison with body-fitted FEM”.

|  |  |  | Body-fitted P1 |  |  |
|:---|:---:|---:|---:|---:|---:|
| **Object** | **Flags** | **Diffused Nitsche** | **Matched DOFs** | **Matched surface** | **Graded** |
| a TV remote control |  | 209 | 210 | 443 | 208 |
| a USB stick |  | 193 | 193 | 378 | 193 |
| an electric toothbrush |  | 66.7 | 68.1 | 74.3 | 65.8 |
| a coffee mug | T | 1260 | 1270 | 1970 | 1250 |
| a computer mouse |  | 445 | 446 | 1890 | 443 |
| a school chair with a steel frame and a plastic seat | T, S | 336 | 344 | 453 | 335 |
| a glass-top dining table | T, S | 293 | 217 | 193 | 293 |
| a park bench | T, S | 400 | — | — | — |
| a bar stool | S | 353 | 351 | 348 | 348 |
| a sofa | S | 544 | 569 | 1510 | 541 |
| a teddy bear |  | 486 | 453 | 1550 | 482 |
| a wooden rocking horse | T | 412 | 423 | 476 | 407 |
| a pogo stick | S | 48.0 | 47.0 | 30.6 | 46.6 |
| a wooden toy train engine | G, L, S | 438 | 455 | 1700 | 454 |
| a yo-yo | G | 1560 | 1510 | 4490 | 1550 |
| a jelly dessert |  | 831 | 829 | 4890 | 826 |
| a loaf of bread |  | 983 | 984 | 6730 | 979 |
| a hamburger | G | 1000 | 999 | 6690 | 997 |
| a layered birthday cake |  | 841 | 846 | 4520 | 840 |
| a slice of watermelon |  | 473 | 460 | 2000 | 477 |
| a giraffe | S | 145 | 152 | 181 | 141 |
| an elephant | S | 404 | 412 | 1010 | 399 |
| a rhinoceros | S | 333 | 351 | 904 | 330 |
| a horse | S | 354 | 358 | 761 | 349 |
| a crocodile | S | 169 | 170 | 244 | 167 |
| a car suspension control arm with rubber bushings |  | 353 | 358 | 588 | 350 |
| a truck leaf spring |  | 58.0 | 56.8 | 54.3 | 57.6 |
| a car wheel with its rubber tyre | G | 1410 | 1460 | 3580 | 1390 |
| a car brake disc | S | 912 | — | — | — |
| a crankshaft | G | 299 | 309 | 552 | 296 |
| a road roller with a steel drum and rubber tyres | T, G, S | 454 | 464 | 501 | 437 |
| a jackhammer | S | 87.6 | 90.8 | 105 | 84.2 |
| an industrial robot arm | G | 398 | 402 | 723 | 392 |
| an adjustable wrench | T, S | 76.9 | 44.9 | 55.2 | 76.4 |
| a claw hammer |  | 113 | 119 | 144 | 111 |
| a dental crown |  | 1480 | 1470 | 9040 | 1470 |
| a knee implant | G | 1310 | 1280 | 4950 | 1300 |
| a walking cane with a rubber tip | T, L, S | 30.6 | 29.2 | 18.2 | 30.2 |
| a human femur |  | 153 | 155 | 214 | 150 |
| a lumbar vertebra |  | 418 | 423 | 1130 | 415 |
| a skyscraper | L | 218 | 207 | 620 | 216 |
| a thatched stone cottage | L | 897 | 902 | 5710 | 894 |
| a lighthouse |  | 437 | 434 | 1480 | 443 |
| an igloo |  | 1070 | 1080 | 2470 | 1070 |
| a log cabin | L | 1130 | 1140 | 6420 | 1120 |
| the Leaning Tower of Pisa | T, G, S | 422 | 433 | 709 | 341 |
| the Pantheon | L | 935 | 968 | 2190 | 965 |
| Himeji Castle | G | 1420 | — | — | — |
| a natural stone arch | G | 825 | 835 | 2790 | 822 |
| the Taj Mahal |  | 880 | 862 | 3320 | 902 |
| *Median (all objects)* |  | 420 | 423 | 904 | 407 |
| *n* |  | 50 | 47 | 47 | 47 |
| *Median (unflagged)* |  | 441 | 440 | 1510 | 443 |
| *n* |  | 18 | 18 | 18 | 18 |

### Discrepancy at octree depth 8

LaTeX label `tab:si-bf-accuracy-d8`.

*Caption.* Discrepancy of diffused Nitsche from each body-fitted P1 solution at octree depth 8. η_rig and η_rig^H¹: relative L² and H¹ norms of the displacement difference after removing its best-fit rigid motion (in %); ℓ_N/ℓ_P1: compliance ratio (below 1: Nitsche stiffer); s₉₅, s₉₉: Nitsche-to-P1 ratios of the 95th and 99th percentiles of the von Mises stress. Flags, at the session's h_min: T, wall thinner than 4h_min; G, gaps narrower than h_min on over 1% of the surface; L, a load within 3h_min of a support; S, a support piece below 50h_min² or a partial clamp on a section thinner than 4h_min; blank, unflagged. Rows in bold (gray in the PDF and xlsx): η_rig above 10% against at least one mesh. —, no body-fitted mesh (the input surface self-intersects). <sup>†</sup> Nitsche solve converged only to 10⁻²; excluded from the medians. Medians over all objects and over the unflagged ones; n, number of values. Details in the SI section “Error comparison with body-fitted FEM”.

|  |  | Matched DOFs |  |  |  |  | Matched surface |  |  |  |  | Graded |  |  |  |  |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Object** | **Flags** | **η_rig (%)** | **η_rig^H¹ (%)** | **ℓ_N/ℓ_P1** | **s₉₅** | **s₉₉** | **η_rig (%)** | **η_rig^H¹ (%)** | **ℓ_N/ℓ_P1** | **s₉₅** | **s₉₉** | **η_rig (%)** | **η_rig^H¹ (%)** | **ℓ_N/ℓ_P1** | **s₉₅** | **s₉₉** |
| a TV remote control |  | 2.6 | 5.1 | 0.929 | 0.948 | 0.940 | 2.8 | 5.3 | 0.925 | 0.942 | 0.933 | 2.4 | 4.8 | 0.934 | 0.950 | 0.949 |
| a USB stick |  | 0.43 | 2.0 | 0.938 | 0.965 | 0.924 | 0.49 | 2.1 | 0.934 | 0.963 | 0.915 | 0.33 | 1.9 | 0.949 | 0.972 | 0.931 |
| an electric toothbrush |  | 6.1 | 10 | 1.091 | 0.950 | 1.014 | 5.2 | 8.5 | 1.077 | 0.943 | 1.025 | 2.9 | 4.9 | 1.042 | 0.942 | 0.977 |
| a coffee mug | T | 1.7 | 4.3 | 0.943 | 0.954 | 0.937 | 2.0 | 4.7 | 0.938 | 0.951 | 0.933 | 1.8 | 4.3 | 0.944 | 0.952 | 0.938 |
| a computer mouse |  | 3.4 | 16 | 0.975 | 0.997 | 0.970 | 3.1 | 14 | 0.976 | 0.999 | 0.957 | 2.8 | 15 | 0.980 | 1.003 | 0.960 |
| a school chair with a steel frame and a plastic seat | T, S | 4.2 | 5.7 | 0.932 | 0.934 | 0.917 | 5.5 | 7.5 | 0.908 | 0.917 | 0.902 | 4.3 | 5.9 | 0.926 | 0.926 | 0.917 |
| **a glass-top dining table** | T, S | 6.2 | 11 | 0.951 | 0.975 | 0.929 | 5.1 | 9.2 | 0.960 | 0.981 | 0.932 | 10 | 18 | 0.921 | 0.962 | 0.917 |
| a park bench | T, S | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a bar stool | S | 5.3 | 10 | 0.949 | 0.954 | 0.933 | 5.2 | 10 | 0.949 | 0.954 | 0.931 | 5.6 | 11 | 0.945 | 0.951 | 0.926 |
| a sofa | S | 0.58 | 5.3 | 0.997 | 0.973 | 0.963 | 0.71 | 5.3 | 0.991 | 0.966 | 0.960 | 0.49 | 5.2 | 0.998 | 0.981 | 0.964 |
| a teddy bear |  | 0.58 | 1.8 | 0.926 | 0.936 | 0.958 | 0.60 | 1.8 | 0.929 | 0.985 | 0.951 | 0.55 | 1.7 | 0.935 | 0.988 | 0.948 |
| a wooden rocking horse | T | 2.7 | 5.3 | 0.960 | 0.966 | 0.969 | 2.9 | 5.6 | 0.956 | 0.965 | 0.967 | 2.7 | 5.3 | 0.960 | 0.966 | 0.969 |
| **a pogo stick** | S | 16 | 46 | 0.957 | 0.876 | 1.042 | 16 | 46 | 0.965 | 0.886 | 1.047 | 16 | 46 | 0.957 | 0.881 | 1.042 |
| **a wooden toy train engine** | G, L, S | 11 | 40 | 0.901 | 0.971 | 0.874 | 10 | 41 | 0.905 | 0.975 | 0.869 | 9.8 | 40 | 0.914 | 0.978 | 0.876 |
| **a yo-yo** | G | 16 | 48 | 0.606 | 0.761 | 0.674 | 16 | 48 | 0.596 | 0.751 | 0.663 | 16 | 48 | 0.601 | 0.754 | 0.667 |
| a jelly dessert |  | 1.8 | 18 | 0.976 | 0.971 | 0.972 | 2.2 | 17 | 0.974 | 0.960 | 0.955 | 1.6 | 18 | 0.982 | 0.976 | 0.973 |
| a loaf of bread<sup>†</sup> |  | 2.0 | 16 | 0.972 | 0.998 | 0.965 | 2.7 | 16 | 0.935 | 0.956 | 0.927 | 2.3 | 17 | 0.943 | 0.966 | 0.930 |
| a hamburger | G | 5.3 | 19 | 0.939 | 0.963 | 0.917 | 5.6 | 19 | 0.933 | 0.954 | 0.905 | 3.8 | 19 | 0.955 | 0.973 | 0.935 |
| a layered birthday cake |  | 1.4 | 13 | 0.990 | 0.981 | 0.994 | 1.6 | 14 | 0.991 | 0.974 | 0.987 | 1.3 | 15 | 0.992 | 0.976 | 0.986 |
| a slice of watermelon |  | 2.7 | 15 | 0.947 | 0.968 | 0.935 | 1.9 | 14 | 0.960 | 0.971 | 0.933 | 1.5 | 15 | 0.967 | 0.978 | 0.938 |
| a giraffe | S | 0.84 | 3.0 | 1.048 | 0.988 | 1.034 | 0.89 | 3.1 | 1.051 | 0.988 | 1.032 | 0.87 | 2.9 | 1.032 | 0.984 | 1.029 |
| an elephant | S | 0.42 | 2.5 | 0.993 | 0.997 | 1.003 | 0.49 | 2.6 | 0.994 | 0.997 | 1.002 | 0.46 | 2.7 | 0.996 | 0.998 | 1.002 |
| a rhinoceros | S | 0.67 | 4.8 | 0.991 | 0.984 | 1.005 | 0.75 | 4.7 | 0.992 | 0.997 | 1.003 | 0.70 | 4.8 | 0.993 | 0.993 | 1.003 |
| **a horse** | S | 26 | 54 | 0.967 | 0.968 | 0.981 | 26 | 54 | 0.949 | 0.966 | 0.969 | 26 | 54 | 0.951 | 0.966 | 0.969 |
| a crocodile | S | 1.9 | 5.6 | 0.905 | 0.901 | 0.912 | 2.3 | 5.9 | 0.891 | 0.894 | 0.898 | 2.2 | 5.9 | 0.891 | 0.901 | 0.891 |
| a car suspension control arm with rubber bushings |  | 4.2 | 11 | 0.968 | 0.974 | 0.979 | 4.1 | 11 | 0.964 | 0.972 | 0.975 | 4.2 | 11 | 0.969 | 0.975 | 0.980 |
| **a truck leaf spring** |  | 14 | 24 | 0.831 | 0.856 | 0.727 | 14 | 24 | 0.832 | 0.857 | 0.729 | 15 | 25 | 0.824 | 0.859 | 0.718 |
| a car wheel with its rubber tyre | G | 2.8 | 7.5 | 0.924 | 0.950 | 0.925 | 3.1 | 8.1 | 0.918 | 0.946 | 0.917 | 2.9 | 7.9 | 0.925 | 0.950 | 0.923 |
| a car brake disc | S | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a crankshaft | G | 5.4 | 14 | 0.769 | 0.797 | 0.753 | 4.7 | 12 | 0.801 | 0.820 | 0.783 | 4.6 | 12 | 0.806 | 0.824 | 0.785 |
| **a road roller with a steel drum and rubber tyres** | T, G, S | 37 | 42 | 0.493 | 0.654 | 0.592 | 37 | 42 | 0.491 | 0.655 | 0.592 | 37 | 42 | 0.489 | 0.653 | 0.589 |
| **a jackhammer** | S | 10 | 16 | 0.841 | 0.983 | 0.836 | 11 | 16 | 0.834 | 0.984 | 0.829 | 11 | 17 | 0.831 | 0.984 | 0.820 |
| **an industrial robot arm** | G | 20 | 46 | 0.556 | 0.737 | 0.607 | 20 | 47 | 0.549 | 0.733 | 0.600 | 20 | 47 | 0.553 | 0.736 | 0.605 |
| an adjustable wrench | T, S | 0.27 | 2.2 | 1.007 | 1.008 | 1.002 | 0.27 | 2.2 | 1.007 | 1.006 | 1.004 | 0.31 | 2.3 | 1.005 | 1.004 | 1.001 |
| a claw hammer |  | 0.42 | 1.8 | 0.951 | 0.955 | 0.968 | 0.45 | 1.9 | 0.949 | 0.953 | 0.962 | 0.39 | 1.7 | 0.953 | 0.955 | 0.971 |
| a dental crown |  | 4.5 | 16 | 0.966 | 0.920 | 0.886 | 4.6 | 16 | 0.967 | 0.917 | 0.884 | 4.5 | 16 | 0.968 | 0.920 | 0.884 |
| **a knee implant** | G | 65 | 80 | 0.456 | 0.487 | 0.428 | 65 | 81 | 0.451 | 0.480 | 0.422 | 65 | 80 | 0.458 | 0.486 | 0.427 |
| a walking cane with a rubber tip | T, L, S | 7.6 | 13 | 0.958 | 0.899 | 0.836 | 6.4 | 11 | 0.967 | 0.913 | 0.868 | 9.0 | 15 | 0.937 | 0.889 | 0.830 |
| a human femur |  | 0.18 | 0.94 | 0.992 | 0.989 | 0.981 | 0.26 | 1.0 | 0.990 | 0.985 | 0.979 | 0.10 | 0.89 | 0.995 | 0.992 | 0.983 |
| a lumbar vertebra |  | 4.1 | 16 | 0.961 | 0.941 | 0.934 | 3.9 | 16 | 0.965 | 0.939 | 0.933 | 3.8 | 16 | 0.966 | 0.941 | 0.934 |
| a skyscraper | L | 0.13 | 0.16 | 0.994 | 0.994 | 0.992 | 0.10 | 0.14 | 1.003 | 0.993 | 0.990 | 0.081 | 0.13 | 1.003 | 0.992 | 0.994 |
| a thatched stone cottage | L | 0.70 | 4.1 | 0.985 | 0.984 | 0.982 | 0.53 | 4.3 | 0.998 | 0.986 | 0.975 | 0.47 | 4.4 | 0.997 | 0.987 | 0.976 |
| a lighthouse |  | 0.90 | 1.4 | 0.994 | 0.997 | 0.991 | 1.6 | 2.2 | 0.996 | 0.991 | 0.992 | 1.3 | 1.9 | 0.996 | 0.992 | 0.992 |
| an igloo |  | 2.5 | 7.9 | 0.971 | 0.965 | 0.952 | 1.5 | 6.4 | 0.991 | 0.981 | 0.977 | 1.5 | 6.8 | 0.989 | 0.981 | 0.977 |
| a log cabin | L | 1.5 | 8.6 | 0.945 | 0.954 | 0.965 | 2.0 | 11 | 0.933 | 0.954 | 0.936 | 1.8 | 11 | 0.939 | 0.960 | 0.935 |
| the Leaning Tower of Pisa | T, G, S | 3.7 | 4.2 | 0.964 | 0.961 | 0.958 | 3.7 | 4.2 | 0.965 | 0.958 | 0.956 | 4.6 | 5.4 | 0.969 | 0.965 | 0.969 |
| the Pantheon | L | 0.96 | 1.1 | 0.992 | 0.984 | 0.973 | 1.2 | 1.3 | 1.004 | 0.991 | 0.981 | 0.96 | 1.1 | 1.002 | 0.988 | 0.985 |
| Himeji Castle | G | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a natural stone arch | G | 6.7 | 8.3 | 0.776 | 0.867 | 0.797 | 6.7 | 8.4 | 0.773 | 0.864 | 0.792 | 6.7 | 8.3 | 0.776 | 0.865 | 0.796 |
| the Taj Mahal |  | 1.2 | 1.4 | 0.980 | 0.987 | 0.989 | 0.95 | 1.3 | 1.013 | 0.992 | 0.989 | 0.77 | 1.2 | 1.003 | 0.996 | 0.990 |
| *Median (all objects)* |  | 2.7 | 8.4 | 0.958 | 0.964 | 0.955 | 3.0 | 8.4 | 0.960 | 0.961 | 0.944 | 2.8 | 8.1 | 0.956 | 0.966 | 0.948 |
| *n* |  | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 46 |
| *Median (unflagged)* |  | 2.5 | 10 | 0.968 | 0.965 | 0.968 | 1.9 | 8.5 | 0.967 | 0.971 | 0.957 | 1.5 | 6.8 | 0.969 | 0.976 | 0.971 |
| *n* |  | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 |

### Computational cost at octree depth 8

LaTeX label `tab:si-bf-cost-d8`.

*Caption.* Computational cost at octree depth 8: retries and wall-clock times in seconds (GPU, after a warm-up) of the successful attempt; in parentheses, the total over all attempts where failed attempts add more than 0.5%. Solver retries: attempts after the first; mesh retries: failed TetGen calls plus rejected remeshes of the surface the kept mesh was built from (rejections inside unkept size-search trials not counted). Prep.: preparation (Nitsche: boundary conditions, SDF, octree, trimming, solver setup; P1: surface resizing, TetGen, boundary mapping, solver setup). Solve: Newton iteration to relative residual 10⁻⁶; to 10⁻²: the same solve until it first reaches 10⁻². Superscripts: <sup>C</sup>, CPU solve; <sup>R</sup>, reduced GPU solver; both not comparable with GPU times; <sup>†</sup>, converged only to 10⁻², excluded from the median. Flags, at the session's h_min: T, wall thinner than 4h_min; G, gaps narrower than h_min on over 1% of the surface; L, a load within 3h_min of a support; S, a support piece below 50h_min² or a partial clamp on a section thinner than 4h_min; blank, unflagged. —, no body-fitted mesh (the input surface self-intersects). Medians over all objects and over the unflagged ones; n, number of values. Time medians are of successful attempts. Details in the SI section “Error comparison with body-fitted FEM”.

|  |  | Diffused Nitsche |  |  |  | Matched DOFs |  |  |  |  | Matched surface |  |  |  |  | Graded |  |  |  |  |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Object** | **Flags** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** | **Mesh retries** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** | **Mesh retries** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** | **Mesh retries** | **Solver retries** | **Prep. (s)** | **Solve (s)** | **To 10⁻² (s)** |
| a TV remote control |  | 0 | 1.15 | 2.66 | 2.66 | 0 | 0 | 8.74 | 2.63 | 1.96 | 0 | 0 | 17.2 | 5.58 | 4.07 | 0 | 0 | 9.77 | 2.56 | 1.90 |
| a USB stick |  | 0 | 1.08 | 3.97 | 2.02 | 0 | 0 | 8.01 | 2.69 | 1.91 | 0 | 0 | 15.0 | 6.96 | 3.75 | 0 | 0 | 9.17 | 3.44 | 1.87 |
| an electric toothbrush |  | 0 | 0.505 | 3.22 | 2.17 | 0 | 0 | 4.29 | 5.26 | 3.62 | 0 | 0 | 4.49 | 4.53 | 3.17 | 0 | 0 | 4.68 | 9.73 | 8.05 |
| a coffee mug | T | 0 | 6.25 | 29.8 | 22.1 | 0 | 0 | 73.0 | 24.4 | 17.5 | 0 | 0 | 77.4 | 127<sup>C</sup> | 114<sup>C</sup> | 0 | 0 | 80.1 | 23.8 | 16.9 |
| a computer mouse |  | 0 | 2.36 | 2.86 | 1.51 | 0 | 0 | 18.7 | 2.75 | 1.71 | 0 | 0 | 72.3 | 437<sup>C</sup> | 423<sup>C</sup> | 0 | 0 | 55.6 | 3.58 | 2.41 |
| a school chair with a steel frame and a plastic seat | T, S | 1 | 1.79 (3.52) | 14.1 (17.6) | 7.12 | 0 | 1 | 10.9 | 11.5 (14.5) | 6.94 | 0 | 1 | 14.2 | 20.6 (24.5) | 8.96 | 0 | 1 | 38.5 | 13.7 (16.7) | 6.90 |
| a glass-top dining table | T, S | 0 | 1.55 | 8.85 | 5.90 | 0 | 1 | 7.89 | 4.16 (6.31) | 2.75 | 0 | 1 | 6.75 | 4.42 (6.48) | 2.86 | 0 | 1 | 33.3 | 6.10 (8.82) | 4.09 |
| a park bench | T, S | 1 | 2.38 (4.78) | 12.4 (16.6) | 8.28 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a bar stool | S | 1 | 1.78 (3.52) | 10.4 (14.0) | 6.73 | 0 | 1 | 14.5 | 7.91 (11.3) | 5.30 | 0 | 1 | 14.2 | 7.81 (11.2) | 5.23 | 0 | 1 | 24.6 | 10.4 (13.7) | 5.24 |
| a sofa | S | 0 | 2.64 | 9.56 | 3.51 | 0 | 0 | 25.2 | 6.31 | 3.74 | 0 | 0 | 58.4 | 158<sup>C</sup> | 147<sup>C</sup> | 0 | 0 | 30.5 | 19.6 | 10.8 |
| a teddy bear |  | 0 | 2.36 | 12.7 | 7.67 | 0 | 0 | 22.3 | 12.1 | 9.81 | 0 | 0 | 59.8 | 1743<sup>C</sup> | 1436<sup>C</sup> | 0 | 0 | 37.1 | 13.4 | 10.9 |
| a wooden rocking horse | T | 0 | 2.10 | 6.55 | 6.55 | 0 | 1 | 16.5 | 10.2 (14.3) | 6.93 | 0 | 1 | 18.7 | 11.1 (15.7) | 7.55 | 0 | 1 | 27.0 | 9.56 (13.4) | 6.60 |
| a pogo stick | S | 1 | 0.400 (0.793) | 3.20 (5.77) | 0.748 | 0 | 1 | 3.79 | 5.98 (7.12) | 0.800 | 0 | 0 | 3.50 | 1.97 | 0.620 | 0 | 1 | 4.32 | 4.79 (5.90) | 0.792 |
| a wooden toy train engine | G, L, S | 0 | 2.95 | 4.40 | 2.40 | 0 | 0 | 44.2 | 4.20 | 2.68 | 0 | 0 | 248 | 263<sup>C</sup> | 245<sup>C</sup> | 0 | 0 | 29.8 | 6.07 | 4.10 |
| a yo-yo | G | 0 | 7.45 | 15.4 | 7.92 | 0 | 0 | 137 | 270<sup>C</sup> | 248<sup>C</sup> | 0 | 0 | 181 | 874<sup>C</sup> | 824<sup>C</sup> | 0 | 0 | 161 | 101<sup>C</sup> | 91.2<sup>C</sup> |
| a jelly dessert |  | 0 | 3.97 | 12.6 | 9.61 | 0 | 0 | 54.7 | 11.7 | 9.51 | 0 | 1 | 378 | 71.5<sup>R</sup> (2284) | 50.1<sup>R</sup> | 0 | 0 | 75.6 | 10.1 | 6.23 |
| a loaf of bread |  | 6 | 4.74 (32.6) | 1991<sup>C</sup><sup>†</sup> (8765) | 170<sup>C</sup> | 0 | 0 | 40.2 | 47.4 | 6.32 | 0 | 1 | 281 | 365<sup>R</sup> (2549) | 40.7<sup>R</sup> | 0 | 0 | 53.4 | 59.9 | 7.17 |
| a hamburger | G | 0 | 4.96 | 110 | 6.10 | 0 | 0 | 88.5 | 53.3 | 6.53 | 0 | 1 | 347 | 338<sup>R</sup> (1473) | 38.8<sup>R</sup> | 0 | 0 | 51.0 | 51.5 | 6.26 |
| a layered birthday cake |  | 0 | 3.80 | 16.1 | 10.3 | 0 | 0 | 67.6 | 9.95 | 6.15 | 0 | 1 | 408 | 67.7<sup>R</sup> (1259) | 36.4<sup>R</sup> | 0 | 0 | 82.2 | 9.36 | 5.82 |
| a slice of watermelon |  | 0 | 2.59 | 5.44 | 5.44 | 0 | 0 | 35.5 | 3.77 | 2.01 | 2 | 0 | 164 (472) | 911<sup>C</sup> | 885<sup>C</sup> | 0 | 0 | 96.3 | 8.66 | 6.99 |
| a giraffe | S | 0 | 0.945 | 4.56 | 3.20 | 0 | 1 | 6.94 | 6.27 (8.20) | 2.53 | 0 | 1 | 8.15 | 7.31 (9.42) | 2.95 | 0 | 1 | 10.4 | 8.48 (10.3) | 3.15 |
| an elephant | S | 0 | 1.99 | 6.39 | 3.20 | 0 | 0 | 15.4 | 7.72 | 2.76 | 0 | 0 | 38.6 | 21.2 | 15.2 | 0 | 0 | 22.3 | 7.60 | 2.69 |
| a rhinoceros | S | 0 | 1.71 | 3.63 | 1.90 | 0 | 0 | 13.6 | 5.49 | 3.83 | 0 | 0 | 34.2 | 15.3 | 10.6 | 0 | 0 | 17.1 | 5.16 | 2.08 |
| a horse | S | 1 | 1.86 (3.74) | 21.0 (32.1) | 7.20 | 0 | 1 | 13.9 | 13.4 (16.8) | 5.42 | 0 | 1 | 68.6 | 51.6 (59.0) | 11.5 | 0 | 1 | 22.1 | 18.3 (21.7) | 5.27 |
| a crocodile | S | 0 | 1.18 | 5.45 | 3.74 | 0 | 1 | 8.80 | 6.37 (8.86) | 3.49 | 0 | 1 | 25.2 | 7.84 (11.2) | 3.96 | 0 | 1 | 11.4 | 6.03 (8.06) | 3.13 |
| a car suspension control arm with rubber bushings |  | 0 | 1.90 | 7.12 | 5.13 | 0 | 0 | 13.6 | 10.5 | 4.62 | 0 | 0 | 22.7 | 18.0 | 7.91 | 0 | 0 | 25.6 | 8.33 | 4.53 |
| a truck leaf spring |  | 0 | 0.452 | 1.89 | 1.43 | 0 | 0 | 3.85 | 1.94 | 1.51 | 0 | 0 | 3.83 | 1.91 | 1.47 | 0 | 0 | 4.19 | 2.02 | 1.57 |
| a car wheel with its rubber tyre | G | 0 | 6.78 | 23.1 | 11.7 | 0 | 0 | 125 | 244<sup>C</sup> | 210<sup>C</sup> | 0 | 0 | 409 | 558<sup>C</sup> | 500<sup>C</sup> | 0 | 0 | 68.2 | 57.1<sup>C</sup> | 44.0<sup>C</sup> |
| a car brake disc | S | 0 | 4.85 | 9.57 | 5.31 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a crankshaft | G | 0 | 1.57 | 4.82 | 3.29 | 0 | 0 | 13.5 | 4.57 | 3.22 | 0 | 0 | 19.7 | 8.41 | 5.86 | 0 | 0 | 11.9 | 4.20 | 2.96 |
| a road roller with a steel drum and rubber tyres | T, G, S | 0 | 3.66 | 15.4 | 4.85 | 0 | 1 | 16.8 | 13.3 (17.8) | 6.67 | 0 | 1 | 65.2 | 10.6 (17.2) | 7.18 | 0 | 1 | 38.1 | 9.14 (13.1) | 5.97 |
| a jackhammer | S | 1 | 0.571 (1.15) | 34.2 (35.5) | 25.9 | 0 | 3 | 4.77 | 9.35<sup>C</sup> (48.4) | 6.25<sup>C</sup> | 0 | 3 | 5.12 | 11.3<sup>C</sup> (49.4) | 7.61<sup>C</sup> | 0 | 3 | 6.46 | 8.39<sup>C</sup> (56.4) | 5.61<sup>C</sup> |
| an industrial robot arm | G | 0 | 2.08 | 7.78 | 5.34 | 0 | 0 | 15.0 | 11.8 | 5.09 | 0 | 0 | 28.1 | 22.7 | 14.0 | 0 | 0 | 27.0 | 9.17 | 4.87 |
| an adjustable wrench | T, S | 0 | 0.573 | 2.71 | 1.45 | 0 | 0 | 3.81 | 1.84 | 1.02 | 0 | 0 | 4.02 | 4.45 | 1.63 | 0 | 0 | 5.26 | 3.26 | 1.46 |
| a claw hammer |  | 0 | 0.739 | 3.34 | 1.76 | 0 | 0 | 5.81 | 3.82 | 2.39 | 0 | 0 | 6.69 | 4.49 | 1.98 | 0 | 0 | 7.12 | 3.62 | 1.62 |
| a dental crown |  | 0 | 6.58 | 9.94 | 5.15 | 0 | 0 | 53.1 | 193<sup>C</sup> | 182<sup>C</sup> | 0 | 0 | 387 | 1496<sup>C</sup> | 1088<sup>C</sup> | 0 | 0 | 72.1 | 71.0<sup>C</sup> | 62.0<sup>C</sup> |
| a knee implant | G | 0 | 5.96 | 15.2 | 7.41 | 0 | 3 | 46.8 | 114<sup>C</sup> | 105<sup>C</sup> | 0 | 0 | 190 | 794<sup>C</sup> | 575<sup>C</sup> | 0 | 0 | 76.6 | 14.2 | 7.83 |
| a walking cane with a rubber tip | T, L, S | 0 | 0.315 | 0.947 | 0.621 | 0 | 0 | 3.56 | 1.24 | 0.855 | 0 | 0 | 3.42 | 0.969 | 0.686 | 0 | 0 | 3.81 | 1.25 | 0.872 |
| a human femur |  | 0 | 0.871 | 4.00 | 2.37 | 0 | 0 | 6.96 | 4.05 | 1.91 | 0 | 0 | 9.19 | 5.56 | 3.61 | 0 | 0 | 9.19 | 3.99 | 2.57 |
| a lumbar vertebra |  | 0 | 2.17 | 5.10 | 1.83 | 0 | 0 | 16.1 | 5.19 | 3.73 | 0 | 0 | 43.5 | 14.8 | 5.98 | 0 | 0 | 31.3 | 5.05 | 3.63 |
| a skyscraper | L | 0 | 1.12 | 1.95 | 1.05 | 0 | 0 | 9.90 | 1.75 | 1.07 | 0 | 0 | 23.8 | 5.34 | 3.12 | 0 | 0 | 10.3 | 1.84 | 1.13 |
| a thatched stone cottage | L | 0 | 4.16 | 9.93 | 5.12 | 0 | 0 | 36.5 | 6.89 | 3.69 | 0 | 0 | 232 | 2796<sup>C</sup> | 1487<sup>C</sup> | 0 | 0 | 46.5 | 6.53 | 3.50 |
| a lighthouse |  | 0 | 2.19 | 4.20 | 2.24 | 0 | 0 | 43.0 | 4.76 | 2.92 | 0 | 0 | 57.6 | 158<sup>C</sup> | 148<sup>C</sup> | 0 | 0 | 23.3 | 5.06 | 2.97 |
| an igloo |  | 0 | 5.45 | 7.61 | 3.97 | 0 | 0 | 67.6 | 9.33 | 6.11 | 0 | 0 | 129 | 271<sup>C</sup> | 243<sup>C</sup> | 0 | 0 | 72.6 | 6.88 | 4.14 |
| a log cabin | L | 0 | 5.40 | 8.28 | 4.39 | 0 | 0 | 51.7 | 8.79 | 5.22 | 0 | 0 | 265 | 1759<sup>C</sup> | 1178<sup>C</sup> | 0 | 0 | 56.7 | 10.3 | 5.94 |
| the Leaning Tower of Pisa | T, G, S | 0 | 2.17 | 4.73 | 2.62 | 0 | 1 | 15.2 | 12.1 (16.1) | 3.00 | 0 | 1 | 26.4 | 20.3 (26.7) | 10.2 | 8 | 1 | 13.5 (82.2) | 21.1 (24.0) | 4.62 |
| the Pantheon | L | 0 | 4.46 | 7.60 | 4.23 | 0 | 0 | 35.3 | 8.60 | 5.63 | 0 | 0 | 78.1 | 225<sup>C</sup> | 210<sup>C</sup> | 0 | 0 | 86.4 | 7.87 | 4.82 |
| Himeji Castle | G | 0 | 6.76 | 19.7 | 10.7 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| a natural stone arch | G | 0 | 4.47 | 7.31 | 3.89 | 1 | 0 | 43.2 | 9.53 | 6.77 | 0 | 0 | 155 | 490<sup>C</sup> | 470<sup>C</sup> | 0 | 0 | 93.4 | 7.18 | 4.42 |
| the Taj Mahal |  | 0 | 3.93 | 11.3 | 6.04 | 0 | 0 | 162 | 11.2 | 5.72 | 0 | 0 | 124 | 1228<sup>C</sup> | 1164<sup>C</sup> | 0 | 0 | 269 | 14.6 | 7.86 |
| *Median (all objects)* |  | 0 | 2.18 | 7.60 | 4.62 | 0 | 0 | 16.1 | 7.91 | 3.83 | 0 | 0 | 43.5 | 21.2 | 11.5 | 0 | 0 | 29.8 | 8.39 | 4.62 |
| *n* |  | 50 | 50 | 49 | 50 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 | 47 |
| *Median (unflagged)* |  | 0 | 2.28 | 5.10 | 3.31 | 0 | 0 | 20.5 | 5.23 | 3.68 | 0 | 0 | 58.7 | 69.6 | 38.5 | 0 | 0 | 34.2 | 7.60 | 4.34 |
| *n* |  | 18 | 18 | 17 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 |


## Definitions and provenance

The tables carry no notes; these definitions complement the captions, and symbols follow the SI text ("Error comparison with body-fitted FEM").

**Degrees of freedom.** Diffused Nitsche: 3 × the free (inner) nodes of the trimmed octree at the depth compared.
Body-fitted P1: 3 × the tetrahedral mesh's vertices. Matched DOFs: vertex count targeted at the Nitsche node
count (secant search of up to six trials over the edge length s*, the closest trial kept). Matched surface: surface triangles of edge
length h_min. Graded: h_min at the surface, interior sized by c × the local octree leaf size (TetGen -m), vertex
count targeted as for matched DOFs. Eleven entries (eight matched-DOF, three graded) end outside 5 % of the
node count.

**Discrepancy measures** (against that P1 solution, on the P1 mesh's second-order quadrature over the body Ω;
e = u_N − u_P1). r* is the best-fit infinitesimal rigid motion of e (six modes, weighted least squares at the
quadrature points).
- η_rig = ‖e − r*‖_L² / ‖u_P1‖_L², the rigid-removed relative L² discrepancy (the headline measure).
- η_rig^H¹ = ‖e − r*‖_H¹ / ‖u_P1‖_H¹, with ‖v‖²_H¹ = ∫|v|² + ∫|∇v|²_F, the same r* (fitted in L², not refitted),
  the same quadrature points and each method's own displacement gradient (P1: constant per tetrahedron; diffused
  Nitsche: the trilinear gradient of its octree cell), the arrays the stress measures use. An infinitesimal
  rotation has a constant skew gradient, so removing r* also changes the gradient part. The norm mixes units
  (lengths in metres): for parts much smaller than 1 m the gradient term dominates. The gradient-only ratio and
  the discrepancy without rigid removal (η) are in the xlsx.
- ℓ_N/ℓ_P1 = ℓ(u_N)/ℓ(u_P1), the compliance ratio, ℓ the work of the P1 nodal loads; below 1 means diffused
  Nitsche is stiffer.
- s₉₅, s₉₉: Nitsche-to-P1 ratios of the volume-weighted 95th and 99th percentiles of the von Mises stress (Cauchy
  stress from each method's own gradient, the same neo-Hookean material).

**Cost** (wall-clock seconds, GPU-synchronized, after an untimed warm-up in the same process). A value is that of
the successful attempt (for P1, on the kept mesh); a value in parentheses is the total over all attempts, shown
where failed attempts add more than 0.5 %. Successful size-search trials that were not kept are in neither.
- Diffused Nitsche solver retries: fallback paths run after the first (GPU multigrid, more Krylov iterations,
  a second multigrid implementation, the same with a 500E penalty cap, diagonal scaling, direct linear solves on
  the CPU, the same with 10 load steps and line search, the same at the session tolerance; skipped paths not
  counted). Every path is a full rerun, so a Nitsche retry repeats preparation and solve. Preparation: surface
  load, case parse, boundary-condition resolution, SDF and octree, solver build and trimming, case build; over
  all attempts, the sum over the paths run with the case build counted once (it is measured once and added into
  every path's preparation).
- P1 mesh retries: failed TetGen calls (crash, error or 30-min time-out, followed by the next switch variant on
  the matched-surface mesh — surface preservation toggled, then a looser quality bound —, by a 15 % larger size
  in a size-search trial, or by the graded search repeated without surface preservation), plus remeshes redone
  with a tighter distance bound and fallbacks to the source surface, in the kept mesh's build or a failed one. A
  rejected remesh counts as one retry. Failed size-search trials (30-min TetGen time-outs included) are retries;
  successful trials that were not kept are not, and nothing in them is counted. Meshing is the only part of P1
  preparation that is retried.
- P1 solver retries: failed GPU solver paths (multigrid, more Krylov iterations, a second multigrid
  implementation, diagonal scaling; each its own record) plus failed CPU-solver runs (before the reduced GPU
  solver).
- P1 preparation: surface resizing (with its self-intersection checks and any remesh redo), sizing field,
  TetGen, boundary mapping, boundary-condition transfer and solver setup of the kept mesh; over all attempts,
  plus the recorded time of every failed TetGen call. (The stored `prep_seconds_incl_failed_tetgen` also adds
  failed self-intersection checks, which are either inside the kept build's resizing time already or belong to
  unused trials; it is in the xlsx as "preparation, all attempts, stored".)
- Solve: Newton iteration to the common relative tolerance 10⁻⁶; over all attempts, plus every failed attempt's
  recorded time (diffused Nitsche: its solve phases; P1: solver setup and solve until it failed, without the
  boundary mapping and transfer the attempt repeated) plus the wall time of every failed CPU-solver run (GNU time;
  includes its assembly and factorization). To 10⁻²: the same successful solve up to its first Newton iterate at
  the session tolerance 10⁻² (from the Newton log).
- CPU solves (P1: direct-factorization Newton solver; diffused Nitsche: direct linear solves) and reduced-GPU
  solves (full Newton, torch tangent and JAX-AMG) are marked in the cost tables; their times are not comparable
  with GPU times.

**Not recorded** (so not in the all-attempts totals): the boundary mapping and transfer a failed P1 solve attempt
redid before its setup; the resizing and setup time of a failed TetGen call's build; the warm-ups (excluded by
design); a session rerun of the batch driver (natural stone arch: its orchestrator was killed after a heap
corruption and an idle hang, and it resumed from the recorded attempts, so the wall time lost to the crashed job
is in no field); the CPU solver's in-run switch from modified to full Newton is inside its one run's time; the
time to 10⁻² of a failed attempt. No repair round ran.

**Medians** as in `summarize.py`: over every object with a value and over the unflagged ones; n per column.
Entries that converged only to a looser tolerance are left out of the metric medians; CPU and reduced-GPU times
are in the time medians. Time and retry medians are of the successful-attempt values. One deviation from
summarize.py: the loaf of bread's depth-8 Nitsche solve time (to 10⁻² only) is left out of the Nitsche solve
median (n 49 instead of 50; median over all objects 7.60 s instead of 7.61 s, over unflagged 5.10 s instead of
5.27 s).

