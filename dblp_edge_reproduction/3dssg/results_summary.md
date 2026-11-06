# 3DSSG results (mean ± std over seeds)

## Base models (test link-prediction AUROC)

| edge set | gnn | seeds | Et AUROC | Et AUPR |
|---|---|---|---|---|
| all | rgcn | 1 | 0.938 | 0.946 |
| spatial_support | gcn | 1 | 0.873 | 0.856 |
| spatial_support | rgcn | 1 | 0.881 | 0.883 |

## all | rgcn | delete out 2.5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.938 | 0.680 | 2.024 | 191.696 | 20,480 |
| retrain |  | 1 | 0.936 | 0.495 | 1.067 | 745.771 | 1,725,568 |

## spatial_support | gcn | delete out 2.5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.873 | 0.756 | 1.521 | 33.781 | 20,480 |
| gradient_ascent |  | 1 | 0.864 | 0.584 | 1.076 | 0.988 | 84,928 |
| retrain |  | 1 | 0.873 | 0.591 | 1.073 | 62.387 | 84,928 |

## spatial_support | rgcn | delete in 0.5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.849 | 0.966 | 2.072 | 114.683 | 20,480 |
| gradient_ascent |  | 1 | 0.774 | 0.672 | 1.286 | 3.046 | 987,328 |
| retrain |  | 1 | 0.885 | 0.765 | 1.180 | 310.019 | 987,328 |

## spatial_support | rgcn | delete in 2.5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0 | 1 | 0.863 | 0.779 | 1.234 | 118.562 | 20,480 |
| gnndelete | 0.2 | 1 | 0.824 | 0.915 | 1.751 | 111.034 | 20,480 |
| gnndelete | 0.4 | 1 | 0.804 | 0.928 | 1.964 | 113.511 | 20,480 |
| gnndelete | 0.5 | 1 | 0.793 | 0.928 | 2.038 | 116.750 | 20,480 |
| gnndelete | 0.6 | 1 | 0.780 | 0.927 | 2.105 | 110.557 | 20,480 |
| gnndelete | 0.8 | 1 | 0.723 | 0.925 | 2.239 | 102.690 | 20,480 |
| gnndelete | 1 | 1 | 0.685 | 0.906 | 1.866 | 104.248 | 20,480 |
| gradient_ascent |  | 1 | 0.752 | 0.676 | 1.322 | 3.099 | 987,328 |
| retrain |  | 1 | 0.872 | 0.778 | 1.206 | 315.741 | 987,328 |

## spatial_support | rgcn | delete in 5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.759 | 0.920 | 2.078 | 107.909 | 20,480 |
| gradient_ascent |  | 1 | 0.734 | 0.713 | 1.383 | 3.225 | 987,328 |
| retrain |  | 1 | 0.851 | 0.808 | 1.283 | 299.027 | 987,328 |

## spatial_support | rgcn | delete node 100

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.878 | 0.982 | 2.376 | 112.954 | 20,480 |
| gradient_ascent |  | 1 | 0.864 | 0.988 | 2.858 | 3.462 | 987,328 |
| retrain |  | 1 | 0.886 | 0.980 | 2.327 | 315.075 | 987,328 |

## spatial_support | rgcn | delete out 0.5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.881 | 0.949 | 2.278 | 123.359 | 20,480 |
| gradient_ascent |  | 1 | 0.780 | 0.575 | 1.244 | 3.880 | 987,328 |
| retrain |  | 1 | 0.881 | 0.586 | 1.112 | 314.418 | 987,328 |

## spatial_support | rgcn | delete out 2.5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.881 | 0.811 | 2.177 | 117.438 | 20,480 |
| gradient_ascent |  | 1 | 0.779 | 0.574 | 1.248 | 3.205 | 987,328 |
| retrain |  | 1 | 0.880 | 0.594 | 1.116 | 315.596 | 987,328 |

## spatial_support | rgcn | delete out 5

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.881 | 0.750 | 2.142 | 112.488 | 20,480 |
| gradient_ascent |  | 1 | 0.765 | 0.575 | 1.277 | 2.899 | 987,328 |
| retrain |  | 1 | 0.869 | 0.610 | 1.150 | 307.974 | 987,328 |

## spatial_support | rgcn | delete scene 10

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.881 | 0.981 | 2.211 | 103.668 | 20,480 |
| gradient_ascent |  | 1 | 0.869 | 0.957 | 1.764 | 2.923 | 987,328 |
| retrain |  | 1 | 0.884 | 0.972 | 1.902 | 300.953 | 987,328 |

## spatial_support | rgcn | delete scene 50

| method | λ | seeds | Et AUROC ↑ | Ed AUROC | MI ratio ↑ | time (s) | trainable params |
|---|---|---|---|---|---|---|---|
| gnndelete | 0.5 | 1 | 0.880 | 0.982 | 2.311 | 120.434 | 20,480 |
| gradient_ascent |  | 1 | 0.864 | 0.958 | 1.847 | 3.177 | 987,328 |
| retrain |  | 1 | 0.887 | 0.976 | 1.973 | 312.577 | 987,328 |
