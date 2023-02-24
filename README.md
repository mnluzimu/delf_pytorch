# DELF Pytorch Version

### Ex1: resnet18

**fine-tuning**

```
python3 -m example.finetuning ./models --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet18' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 3 --query-size=2000 --pool-size=22000 --batch-size 1 --image-size 900 --epochs 100
```

```
>>>> used networks: 
  (meta): dict( 
     architecture: resnet18
     pooling: gem
     outputdim: 512
     mean: [0.485, 0.456, 0.406]
     std: [0.229, 0.224, 0.225]
  )

# 0
>> roxford5k: mAP E: 30.83, M: 21.53, H: 5.16
>> roxford5k: mP@k[1, 5, 10] E: [60.29 47.35 40.27], M: [60.   44.57 35.84], H: [17.14 12.29  9.38]
>> rparis6k: mAP E: 61.0, M: 46.89, H: 21.97
>> rparis6k: mP@k[1, 5, 10] E: [95.71 92.57 89.29], M: [95.71 93.71 91.43], H: [72.86 64.86 57.71]

# 42
>> roxford5k: mAP E: 69.64, M: 52.79, H: 26.51
>> roxford5k: mP@k[1, 5, 10] E: [88.24 79.93 75.22], M: [87.14 78.57 73.14], H: [55.71 44.57 37.71]
>> rparis6k: mAP E: 79.71, M: 61.71, H: 33.42
>> rparis6k: mP@k[1, 5, 10] E: [94.29 93.14 91.86], M: [95.71 95.43 94.  ], H: [81.43 76.86 72.71]

# 100
>> roxford5k: mAP E: 69.74, M: 52.38, H: 26.21
>> roxford5k: mP@k[1, 5, 10] E: [86.76 79.63 74.93], M: [84.29 79.43 73.86], H: [60.   45.14 38.14]
>> rparis6k: mAP E: 80.01, M: 61.78, H: 33.7
>> rparis6k: mP@k[1, 5, 10] E: [95.71 93.71 92.71], M: [95.71 96.29 94.86], H: [84.29 77.71 74.29]
```



**attention**

```
python3 -m example.attention ./attention_models --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet18' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 3 --query-size=2000 --pool-size=22000 --batch-size 1 --image-size 900 --epochs 100 --param-directory './models'
```

```
# 59
>> roxford5k: mAP E: 68.78, M: 52.15, H: 24.43
>> roxford5k: mP@k[1, 5, 10] E: [86.76 81.1  74.93], M: [85.71 78.76 73.05], H: [52.86 43.43 37.14]
>> rparis6k: mAP E: 77.6, M: 60.16, H: 32.2
>> rparis6k: mP@k[1, 5, 10] E: [94.29 92.95 91.38], M: [94.29 96.57 94.71], H: [81.43 77.71 71.43]

# 100
>> roxford5k: mAP E: 68.35, M: 52.22, H: 25.27
>> roxford5k: mP@k[1, 5, 10] E: [85.29 80.51 74.78], M: [84.29 79.24 73.52], H: [55.71 43.86 38.  ]
>> rparis6k: mAP E: 78.31, M: 60.46, H: 32.32
>> rparis6k: mP@k[1, 5, 10] E: [95.71 93.24 91.67], M: [95.71 96.57 95.43], H: [81.43 78.   72.71]
```



**extract features**

```
python3 -m example.extract_features ./features --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet18' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 3 --query-size=2000 --pool-size=22000 --batch-size 1 --image-size 900 --epochs 100 --param-directory './attention_models' --feature-num 2000 --alpha-thresh 0
```



**voc tree**

```
python3 -m voc_tree.main ./trees_delf --root-directory ./features/retrieval-SfM-120k_resnet18_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-06_nnum3_qsize2000_psize22000_bsize1_uevery1_imsize900 --test-datasets 'roxford5k,rparis6k' --tree-height 6 --tree-branch 10  --output output_delf.txt --rerank-num 150
```



```
# resnet18
>> Training finished in 10819.56745171547 s
>> roxford5k: mAP E: 39.68, M: 32.2, H: 13.99
>> roxford5k: mP@k[1, 5, 10] E: [52.94 50.59 47.98], M: [57.14 51.43 48.03], H: [35.71 26.29 22.57]
>> Training finished in 30835.063895702362 s
>> rparis6k: mAP E: 69.02, M: 51.45, H: 20.83
>> rparis6k: mP@k[1, 5, 10] E: [84.29 84.57 84.86], M: [88.57 86.86 87.86], H: [61.43 54.86 49.  ]
```



### Ex2: resnet18 refined

change pooling to spoc

**fine-tuning**

```
python3 -m example.finetuning ./models --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet18' --pool 'spoc' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 --batch-size 3 --image-size 900 --epochs 100
```

```
>> roxford5k: mAP E: 38.09, M: 26.38, H: 7.55
>> roxford5k: mP@k[1, 5, 10] E: [58.82 50.88 45.16], M: [61.43 45.71 38.14], H: [18.57 13.05 11.9 ]
>> rparis6k: mAP E: 63.15, M: 46.99, H: 19.88
>> rparis6k: mP@k[1, 5, 10] E: [94.29 90.86 87.43], M: [94.29 92.86 90.  ], H: [72.86 59.14 53.14]
```

**attention**

```
python3 -m example.attention ./attention_models --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet18' --pool 'spoc' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 --batch-size 3 --image-size 900 --epochs 100 --param-directory './models'
```

```
>> roxford5k: mAP E: 39.39, M: 27.94, H: 8.9
>> roxford5k: mP@k[1, 5, 10] E: [61.76 52.5  47.22], M: [65.71 46.95 40.95], H: [20.   14.   12.57]
>> rparis6k: mAP E: 65.17, M: 48.66, H: 20.41
>> rparis6k: mP@k[1, 5, 10] E: [95.71 91.14 88.43], M: [95.71 93.71 90.57], H: [77.14 57.71 53.14]
```

**extract features**

```
python3 -m example.extract_features ./features --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet18' --pool 'spoc' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 --batch-size 3 --image-size 900 --epochs 100 --param-directory './attention_models' --feature-num 2000 --alpha-thresh 0
```

**voc tree**

```
python3 -m voc_tree.main ./trees_delf --root-directory ./features/retrieval-SfM-120k_resnet18_spoc_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-06_nnum5_qsize2000_psize22000_bsize3_uevery1_imsize900 --test-datasets 'roxford5k,rparis6k' --tree-height 6 --tree-branch 10  --output output_delf.txt --rerank-num 150
```



```
>> Training finished in 10644.076338291168 s
>> roxford5k: mAP E: 35.14, M: 24.26, H: 4.53
>> roxford5k: mP@k[1, 5, 10] E: [48.53 46.18 42.5 ], M: [48.57 40.   35.5 ], H: [17.14 12.86 10.35]
>> Training finished in 16055.188648939133 s
>> rparis6k: mAP E: 56.8, M: 42.62, H: 17.21
>> rparis6k: mP@k[1, 5, 10] E: [78.57 75.71 75.29], M: [80.   78.   76.71], H: [57.14 47.14 42.  ]
```



### Ex3: resnet34

**fine tuning**

```
python3 -m example.finetuning ./models --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet34' --pool 'spoc' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 --batch-size 3 --image-size 900 --epochs 100
```

```
>> roxford5k: mAP E: 56.68, M: 44.18, H: 20.4
>> roxford5k: mP@k[1, 5, 10] E: [76.47 70.   64.12], M: [77.14 68.   61.98], H: [38.57 29.43 27.24]
>> rparis6k: mAP E: 71.09, M: 54.76, H: 26.29
>> rparis6k: mP@k[1, 5, 10] E: [94.29 91.43 89.14], M: [94.29 94.29 92.57], H: [78.57 68.   61.71]
```

**attention**

```
python3 -m example.attention ./attention_models --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet34' --pool 'spoc' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 --batch-size 3 --image-size 900 --epochs 100 --param-directory './models'
```

**extract features**

```
python3 -m example.extract_features ./features --gpu-id '3' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'resnet34' --pool 'spoc' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 --batch-size 3 --image-size 900 --epochs 100 --param-directory './attention_models' --feature-num 1000 --alpha-thresh 0
```

**voc tree**

```
python3 -m voc_tree.main ./trees_delf --root-directory ./features/retrieval-SfM-120k_resnet3_spoc_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-06_nnum5_qsize2000_psize22000_bsize3_uevery1_imsize900 --test-datasets 'roxford5k,rparis6k' --tree-height 6 --tree-branch 10  --output output_delf.txt --rerank-num 150
```

```
>> Training finished in 9179.60101556778 s
>> roxford5k: mAP E: 22.64, M: 17.75, H: 2.83
>> roxford5k: mP@k[1, 5, 10] E: [33.82 29.71 28.09], M: [32.86 27.43 25.14], H: [7.14 7.14 5.86]
>> Training finished in 11491.79721069336 s
>> rparis6k: mAP E: 35.18, M: 27.5, H: 10.21
>> rparis6k: mP@k[1, 5, 10] E: [60.   54.   50.86], M: [60. 56. 54.], H: [28.57 28.86 26.  ]
```





### SIFT

#### SIFT on server

**extract SIFT features**

```
python3 -m voc_tree.SIFT.get_descriptor ./features --test-datasets 'roxford5k,rparis6k'
```

**build voc tree**

```
python3 -m voc_tree.main ./trees --root-directory ./features/sift --test-datasets 'roxford5k,rparis6k' --tree-height 6 --tree-branch 10  --output output_sift.txt --rerank-num 150
```

 **SIFT + voctree**

```
# branchs = 10
# height= 6
>> roxford5k: mAP E: 19.54, M: 14.48, H: 3.13
>> roxford5k: mP@k[1, 5, 10] E: [50.   42.5  36.03], M: [51.43 39.43 32.9 ], H: [15.71  8.29  6.  ]
>> rparis6k: mAP E: 36.88, M: 28.51, H: 11.48
>> rparis6k: mP@k[1, 5, 10] E: [84.29 78.   72.14], M: [87.14 80.86 76.71], H: [60.   40.57 32.71]
```

**SIFT + voctree + RANSAC**

```
# branchs = 10
# height= 6
# n_rerank = 150
# Homo
>> roxford5k: mAP E: 23.81, M: 18.02, H: 6.38
>> roxford5k: mP@k[1, 5, 10] E: [60.29 48.97 42.21], M: [64.29 49.43 41.86], H: [25.71 14.29 10.57]
>> rparis6k: mAP E: 37.74, M: 28.55, H: 11.19
>> rparis6k: mP@k[1, 5, 10] E: [88.57 81.71 76.71], M: [94.29 86.29 82.  ], H: [65.71 44.86 36.14]
```

```
# branchs = 10
# height= 6
# n_rerank = 150
# FundamentalMatrix
>> roxford5k: mAP E: 22.0, M: 15.89, H: 3.26
>> roxford5k: mP@k[1, 5, 10] E: [55.88 45.29 39.56], M: [58.57 43.71 36.9 ], H: [20.   10.    7.71]
>> rparis6k: mAP E: 36.71, M: 28.26, H: 10.99
>> rparis6k: mP@k[1, 5, 10] E: [85.71 81.43 73.71], M: [94.29 85.43 78.71], H: [58.57 39.71 34.  ]
```

```
# branchs = 10
# height= 6
# n_rerank = 200
# Homo
>> roxford5k: mAP E: 25.16, M: 17.76, H: 5.02
>> roxford5k: mP@k[1, 5, 10] E: [61.76 49.9  43.38], M: [62.86 48.57 41.29], H: [25.71 12.57  8.86]
>> rparis6k: mAP E: 37.57, M: 28.51, H: 11.17
>> rparis6k: mP@k[1, 5, 10] E: [90.   82.86 77.  ], M: [94.29 88.57 83.  ], H: [70.   47.14 36.29]
```



#### SIFT on my computer

**oxford5k: SIFT + voctree**

```
# branchs = 10
# height = 6
>> oxford5k: mAP E: 33.33, M: 29.41, H: 14.88
>> oxford5k: mP@k[1, 5, 10] E: [85.45 52.36 40.82], M: [100.    63.64  49.64], H: [58.18 30.18 23.64]
```

**oxford5k: SIFT + voctree + hamming embedding**

```
# branchs = 10
# height = 6
# ht = 10
>> oxford5k: mAP E: 11.19, M: 9.37, H: 4.45
>> oxford5k: mP@k[1, 5, 10] E: [76.36 22.91 11.45], M: [100.    29.09  14.55], H: [29.09  6.18  3.45]
```

```
# branchs = 10
# height = 6
# ht = 20
>> oxford5k: mAP E: 19.51, M: 16.28, H: 7.52
>> oxford5k: mP@k[1, 5, 10] E: [76.36 43.64 32.55], M: [100.    54.55  41.27], H: [49.09 19.27 11.82]
```

```
# branchs = 10
# height = 6
# ht = 30
>> oxford5k: mAP E: 25.68, M: 21.91, H: 9.87
>> oxford5k: mP@k[1, 5, 10] E: [81.82 46.18 36.36], M: [100.    56.73  44.73], H: [49.09 24.   15.64]
```

```
# branchs = 10
# height = 6
# ht = 40
>> oxford5k: mAP E: 27.84, M: 24.19, H: 11.14
>> oxford5k: mP@k[1, 5, 10] E: [81.82 46.55 37.76], M: [100.    58.91  46.55], H: [52.73 26.18 19.27]
```

```
# branchs = 10
# height = 6
# ht = 50
>> oxford5k: mAP E: 29.6, M: 25.62, H: 11.76
>> oxford5k: mP@k[1, 5, 10] E: [81.82 49.09 38.85], M: [100.    61.09  47.64], H: [52.73 28.   19.82]
```

```
# branchs = 10
# height = 6
# ht = 60
>> oxford5k: mAP E: 29.96, M: 25.9, H: 11.81
>> oxford5k: mP@k[1, 5, 10] E: [81.82 49.09 39.36], M: [100.    61.45  48.  ], H: [52.73 28.36 19.45]
```

```
# branchs = 10
# height = 6
# ht = 70
>> oxford5k: mAP E: 30.3, M: 26.22, H: 11.98
>> oxford5k: mP@k[1, 5, 10] E: [81.82 49.82 39.91], M: [100.    61.82  49.09], H: [52.73 29.09 20.55]
```







**oxford5k: SIFT + voctree + RANSAC**

```
# branchs = 10
# height = 6
# Homo
# n_rerank = 50
>> oxford5k: mAP E: 37.14, M: 31.83, H: 16.54
>> oxford5k: mP@k[1, 5, 10] E: [87.27 58.55 44.64], M: [100.    67.64  53.78], H: [63.64 35.67 26.21]
```

```
# branchs = 10
# height = 6
# Homo
# n_rerank = 100
>> oxford5k: mAP E: 37.3, M: 32.38, H: 16.59
>> oxford5k: mP@k[1, 5, 10] E: [85.45 58.91 44.58], M: [100.    67.64  54.18], H: [69.09 36.82 26.03]
```

```
# branchs = 10
# height = 6
# Homo
# n_rerank = 200
>> oxford5k: mAP E: 37.73, M: 32.45, H: 16.35
>> oxford5k: mP@k[1, 5, 10] E: [83.64 60.36 46.73], M: [100.    69.45  55.23], H: [65.45 36.97 25.54]
```

```
# branchs = 10
# height = 6
# Homo
# n_rerank = 400
>> oxford5k: mAP E: 37.53, M: 31.97, H: 16.79
>> oxford5k: mP@k[1, 5, 10] E: [87.27 58.55 45.  ], M: [100.    67.27  54.55], H: [67.27 37.61 26.88]
```



```
# branchs = 10
# height = 6
# FundamentalMatrix
# n_rerank = 50
>> oxford5k: mAP E: 36.86, M: 31.7, H: 16.01
>> oxford5k: mP@k[1, 5, 10] E: [87.27 58.18 45.01], M: [100.    68.    53.45], H: [65.45 34.24 25.41]
```

```
# branchs = 10
# height = 6
# FundamentalMatrix
# n_rerank = 100
>> oxford5k: mAP E: 37.6, M: 32.39, H: 16.4
>> oxford5k: mP@k[1, 5, 10] E: [87.27 58.18 44.58], M: [100.    68.    53.45], H: [65.45 34.73 26.08]
```

```
# branchs = 10
# height = 6
# FundamentalMatrix
# n_rerank = 200
>> oxford5k: mAP E: 38.19, M: 32.47, H: 15.55
>> oxford5k: mP@k[1, 5, 10] E: [87.27 58.18 45.29], M: [100.    68.36  53.27], H: [65.45 34.42 25.52]
```

```
# branchs = 10
# height = 6
# FundamentalMatrix
# n_rerank = 400
>> oxford5k: mAP E: 37.06, M: 31.01, H: 14.22
>> oxford5k: mP@k[1, 5, 10] E: [85.45 56.73 43.74], M: [100.    66.18  51.09], H: [63.64 32.   22.36]
```



```
# branchs = 10
# height = 6
# FundamentalMatrix
# n_rerank = 300
>> oxford5k: mAP E: 37.25, M: 31.37, H: 14.78
>> oxford5k: mP@k[1, 5, 10] E: [85.45 57.82 43.56], M: [100.    67.27  51.27], H: [61.82 33.09 23.27]
```

