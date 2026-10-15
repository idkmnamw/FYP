# Completed main experiment — saved output snapshot

Source: repository commit `dff59b6`, before file organisation and supplementary analyses. These are the user-run outputs, not a new training run. Full original notebook and figures remain in Git history. The local saved model, selection and data are not included in Git.

Keep the original LightGBM selection and validation cutoff for this completed experiment. Supplementary checks added later must not be presented as prespecified benchmarks or used to retune the exposed test period.

## Original cell 6

```text
fold_1: 66,384 training rows; 31,971 validation rows
  lr_c01: AP=0.2596; F1@0.5=0.3287
  lr_c1: AP=0.2627; F1@0.5=0.3302
  rf_depth8: AP=0.2827; F1@0.5=0.3191
  rf_depth16: AP=0.2932; F1@0.5=0.3470
  xgb_depth3: AP=0.2896; F1@0.5=0.3056
  xgb_depth5: AP=0.2859; F1@0.5=0.3218
  lgb_leaves15: AP=0.2966; F1@0.5=0.3220
  lgb_leaves31: AP=0.2885; F1@0.5=0.3232
fold_2: 101,515 training rows; 31,753 validation rows
  lr_c01: AP=0.3137; F1@0.5=0.3334
  lr_c1: AP=0.3183; F1@0.5=0.3446
  rf_depth8: AP=0.3372; F1@0.5=0.3363
  rf_depth16: AP=0.3630; F1@0.5=0.4087
  xgb_depth3: AP=0.3742; F1@0.5=0.4137
  xgb_depth5: AP=0.3997; F1@0.5=0.4326
  lgb_leaves15: AP=0.3859; F1@0.5=0.4265
  lgb_leaves31: AP=0.3936; F1@0.5=0.4352
fold_3: 138,296 training rows; 54,666 validation rows
  lr_c01: AP=0.3088; F1@0.5=0.3677
  lr_c1: AP=0.3074; F1@0.5=0.3673
  rf_depth8: AP=0.3428; F1@0.5=0.3509
  rf_depth16: AP=0.3736; F1@0.5=0.3975
  xgb_depth3: AP=0.3525; F1@0.5=0.3768
  xgb_depth5: AP=0.3706; F1@0.5=0.3934
  lgb_leaves15: AP=0.3780; F1@0.5=0.3928
  lgb_leaves31: AP=0.3817; F1@0.5=0.4102

             candidate               family   mean_ap    std_ap  scored_folds  \
3         lgb_leaves31             LightGBM  0.354562  0.057560             3   
2         lgb_leaves15             LightGBM  0.353507  0.049414             3   
9           xgb_depth5              XGBoost  0.352078  0.059131             3   
6           rf_depth16        Random Forest  0.343250  0.043693             3   
8           xgb_depth3              XGBoost  0.338756  0.043957             3   
7            rf_depth8        Random Forest  0.320872  0.033204             3   
5                lr_c1  Logistic Regression  0.296153  0.029482             3   
4               lr_c01  Logistic Regression  0.294050  0.029907             3   
1  Current-change rule             Baseline  0.120633  0.026743             3   
0      Always no surge             Baseline  0.105784  0.010496             3   

    mean_f1  mean_recall  mean_precision  total_fit_seconds  
3  0.389521     0.663780        0.278966           2.965296  
2  0.380420     0.693219        0.265359           2.454417  
9  0.382601     0.669785        0.271368           2.441482  
6  0.384409     0.678183        0.268736          26.177096  
8  0.365340     0.714848        0.248765           1.969861  
7  0.335440     0.863315        0.208264          10.693934  
5  0.347410     0.541704        0.278509           2.411096  
4  0.343278     0.537337        0.275766           2.089309  
1  0.191334     0.193216        0.189724           0.000000  
0  0.000000     0.000000        0.000000           0.000000  
```

## Original cell 8

```text
Selected: lgb_leaves31 | Validation-selected score cutoff: 0.6023499784696148

                 model   cutoff   rows  positives  prevalence  \
0         lgb_leaves31  0.60235  43594       3911    0.089714   
1    lgb_leaves31 @0.5  0.50000  43594       3911    0.089714   
2      Always no surge      NaN  43594       3911    0.089714   
3  Current-change rule      NaN  43594       3911    0.089714   

   predicted_surges  average_precision  precision    recall        f1     tn  \
0             11534           0.222652   0.222126  0.655075  0.331758  30711   
1             15665           0.222652   0.203320  0.814370  0.325398  27203   
2                 0           0.089714   0.000000  0.000000  0.000000  39683   
3              4415           0.101548   0.175764  0.198415  0.186404  36044   

      fp    fn    tp  
0   8972  1349  2562  
1  12480   726  3185  
2      0  3911     0  
3   3639  3135   776  
1177
```

## Original cell 10

```text
                 model   cutoff   rows  positives  prevalence  \
0         lgb_leaves31  0.60235  57456       7107    0.123695   
1    lgb_leaves31 @0.5  0.50000  57456       7107    0.123695   
2      Always no surge      NaN  57456       7107    0.123695   
3  Current-change rule      NaN  57456       7107    0.123695   

   predicted_surges  average_precision  precision    recall        f1     tn  \
0             12077           0.331978   0.324253  0.551006  0.408257  42188   
1             16462           0.331978   0.295468  0.684396  0.412746  38751   
2                 0           0.123695   0.000000  0.000000  0.000000  50349   
3              7093           0.150438   0.225715  0.225271  0.225493  44857   

      fp    fn    tp  
0   8161  3191  3916  
1  11598  2243  4864  
2      0  7107     0  
3   5492  5506  1601  
                             criterion  model_value  reference_value  \
0  Mean fold AP exceeds both baselines     0.354562         0.120633   
1       Test AP exceeds both baselines     0.331978         0.150438   
2       Test F1 exceeds both baselines     0.408257         0.225493   

   assessable   met  
0        True  True  
1        True  True  
2        True  True  
```

## Original cell 12

```text
                   model      breakdown project_group   rows  positives  \
0           lgb_leaves31  project_group       Chicken   5256        179   
1           lgb_leaves31  project_group   Cooking Oil   5461          1   
2           lgb_leaves31  project_group          Eggs  10668        536   
3           lgb_leaves31  project_group         Flour   2965         21   
4           lgb_leaves31  project_group          Rice    982         16   
5           lgb_leaves31  project_group         Sugar   4652          0   
6           lgb_leaves31  project_group    Vegetables  27472       6354   
183      Always no surge  project_group       Chicken   5256        179   
184      Always no surge  project_group   Cooking Oil   5461          1   
185      Always no surge  project_group          Eggs  10668        536   
186      Always no surge  project_group         Flour   2965         21   
187      Always no surge  project_group          Rice    982         16   
188      Always no surge  project_group         Sugar   4652          0   
189      Always no surge  project_group    Vegetables  27472       6354   
366  Current-change rule  project_group       Chicken   5256        179   
367  Current-change rule  project_group   Cooking Oil   5461          1   
368  Current-change rule  project_group          Eggs  10668        536   
369  Current-change rule  project_group         Flour   2965         21   
370  Current-change rule  project_group          Rice    982         16   
371  Current-change rule  project_group         Sugar   4652          0   
372  Current-change rule  project_group    Vegetables  27472       6354   

     prevalence  predicted_surges  average_precision  precision    recall  \
0      0.034056                59           0.234911   0.237288  0.078212   
1      0.000183                 0           0.000656   0.000000  0.000000   
2      0.050244              1452           0.201433   0.189394  0.513060   
3      0.007083                22           0.073312   0.136364  0.142857   
4      0.016293                 2           0.215022   1.000000  0.125000   
5      0.000000                 0                NaN   0.000000       NaN   
6      0.231290             10542           0.353568   0.343578  0.570035   
183    0.034056                 0           0.034056   0.000000  0.000000   
184    0.000183                 0           0.000183   0.000000  0.000000   
185    0.050244                 0           0.050244   0.000000  0.000000   
186    0.007083                 0           0.007083   0.000000  0.000000   
187    0.016293                 0           0.016293   0.000000  0.000000   
188    0.000000                 0                NaN   0.000000       NaN   
189    0.231290                 0           0.231290   0.000000  0.000000   
366    0.034056               174           0.027553   0.028736  0.027933   
367    0.000183                 1           0.000190   0.000000  0.000000   
368    0.050244               565           0.046055   0.033628  0.035448   
369    0.007083                33           0.006210   0.000000  0.000000   
370    0.016293                11           0.011517   0.000000  0.000000   
371    0.000000                 0                NaN   0.000000       NaN   
372    0.231290              6309           0.228938   0.249960  0.248190   

           f1     tn    fp    fn    tp state district  
0    0.117647   5032    45   165    14   NaN      NaN  
1    0.000000   5460     0     1     0   NaN      NaN  
2    0.276660   8955  1177   261   275   NaN      NaN  
3    0.139535   2925    19    18     3   NaN      NaN  
4    0.222222    966     0    14     2   NaN      NaN  
5         NaN   4652     0     0     0   NaN      NaN  
6    0.428741  14198  6920  2732  3622   NaN      NaN  
183  0.000000   5077     0   179     0   NaN      NaN  
184  0.000000   5460     0     1     0   NaN      NaN  
185  0.000000  10132     0   536     0   NaN      NaN  
186  0.000000   2944     0    21     0   NaN      NaN  
187  0.000000    966     0    16     0   NaN      NaN  
188       NaN   4652     0     0     0   NaN      NaN  
189  0.000000  21118     0  6354     0   NaN      NaN  
366  0.028329   4908   169   174     5   NaN      NaN  
367  0.000000   5459     1     1     0   NaN      NaN  
368  0.034514   9586   546   517    19   NaN      NaN  
369  0.000000   2911    33    21     0   NaN      NaN  
370  0.000000    955    11    16     0   NaN      NaN  
371       NaN   4652     0     0     0   NaN      NaN  
372  0.249072  16386  4732  4777  1577   NaN      NaN  
<Figure size 1100x400 with 2 Axes>
```
