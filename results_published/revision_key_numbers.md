# HyPhysML revision — key numbers for the manuscript

Seeds: [42, 7, 13, 99, 2024, 17, 88, 55, 101, 314]; test n per seed: [1332]; N_OPTUNA=200 (nested, per seed)

## Table 2 (10-seed means)

               R2_mean  R2_std  RMSE_mean  MAE_mean  MAPE_mean  NSE_mean
Model                                                                   
HyPhysML        0.9890  0.0007     2.3355    1.7914     2.3946    0.9890
GBR             0.9886  0.0008     2.3779    1.8362     2.4551    0.9886
XGBoost         0.9886  0.0007     2.3796    1.8309     2.4525    0.9886
HistGBR         0.9883  0.0007     2.4108    1.8618     2.4924    0.9883
LightGBM        0.9882  0.0009     2.4185    1.8524     2.4707    0.9882
SVR             0.9880  0.0008     2.4336    1.8591     2.4915    0.9880
MLP             0.9877  0.0009     2.4677    1.9192     2.5579    0.9877
RF              0.9861  0.0011     2.6218    2.0013     2.6230    0.9861
Extra Trees     0.9850  0.0013     2.7198    2.0503     2.6625    0.9850
Decision Tree   0.9733  0.0016     3.6321    2.6868     3.4476    0.9733
KNN             0.9696  0.0023     3.8773    3.0024     3.7610    0.9696
Ridge           0.9542  0.0011     4.7638    3.8284     4.7687    0.9542
Obenaus         0.9053  0.0030     6.8486    5.3749     6.5177    0.9053
Rizk            0.7205  0.0128    11.7631    9.4897    11.5755    0.7205

Best of 14: HyPhysML. HyPhysML R²=0.9890 ± 0.0007, RMSE=2.335 kV, MAPE=2.39%
RMSE reduction vs Obenaus: 65.9%;  ΔRMSE vs XGBoost: 0.044 kV
Seed-level 95% CI of the mean R² (HyPhysML): [0.9886, 0.9894]
Seed-42 split: R²=0.9890, observation-bootstrap 95% CI [0.9877, 0.9901], n=1332
Friedman chi2(13) = 126.24, P = 1.17e-20

## Table 3
                vs  wins_of_10  mean_dR2  dR2_CI95_lo  dR2_CI95_hi  mean_dRMSE_kV  p_one_sided  p_two_sided significant_bonferroni  cliffs_delta effect
               GBR          10  0.000405     0.000286     0.000527       0.042456     0.000977     0.001953                   True          0.32  small
           XGBoost          10  0.000418     0.000309     0.000523       0.044123     0.000977     0.001953                   True          0.34 medium
           HistGBR          10  0.000722     0.000613     0.000831       0.075396     0.000977     0.001953                   True          0.50  large
          LightGBM          10  0.000801     0.000662     0.000930       0.083011     0.000977     0.001953                   True          0.50  large
               SVR          10  0.000947     0.000724     0.001160       0.098171     0.000977     0.001953                   True          0.62  large
               MLP          10  0.001288     0.001019     0.001538       0.132212     0.000977     0.001953                   True          0.78  large
                RF          10  0.002875     0.002629     0.003131       0.286373     0.000977     0.001953                   True          1.00  large
       Extra Trees          10  0.003938     0.003545     0.004333       0.384299     0.000977     0.001953                   True          1.00  large
     Decision Tree          10  0.015625     0.014812     0.016362       1.296633     0.000977     0.001953                   True          1.00  large
               KNN          10  0.019370     0.018075     0.020480       1.541859     0.000977     0.001953                   True          1.00  large
             Ridge          10  0.034795     0.034225     0.035385       2.428335     0.000977     0.001953                   True          1.00  large
           Obenaus          10  0.083668     0.082033     0.085420       4.513165     0.000977     0.001953                   True          1.00  large
              Rizk          10  0.268449     0.260910     0.275982       9.427641     0.000977     0.001953                   True          1.00  large
       HyPhysML-MC          10  0.001453     0.001310     0.001604       0.149812     0.000977     0.001953          n/a (variant)          0.88  large
    HyPhysML-noHPO          10  0.000479     0.000401     0.000548       0.050126     0.000977     0.001953          n/a (variant)          0.36 medium
HyPhysML-MeanStack          10  0.000641     0.000528     0.000740       0.067061     0.000977     0.001953          n/a (variant)          0.42 medium

## Meta-weights (mean, sd)
               mean     std     min     max  n_negative_of_10
XGBoost      0.1696  0.0744  0.0510  0.2668                 0
LightGBM     0.1688  0.0499  0.0724  0.2423                 0
GBR          0.3472  0.1157  0.1372  0.5161                 0
HistGBR      0.2132  0.1055  0.0695  0.3916                 0
RF          -0.0960  0.0340 -0.1529 -0.0301                10
Extra Trees  0.2060  0.0310  0.1361  0.2423                 0
KNN         -0.0069  0.0089 -0.0219  0.0078                 8
intercept   -0.1556  0.0619 -0.2882 -0.0921                10

## Auxiliary physics regression
             Regression Coefficient    mean     sd     min     max expected_sign sign_as_expected_seeds                        identifiable
   with type indicators      log_CD  0.0076 0.0000  0.0075  0.0077      positive                     10 no (collinear with type indicators)
   with type indicators      log_AD  0.0211 0.0003  0.0206  0.0216      positive                     10 no (collinear with type indicators)
   with type indicators     log_SDD -0.2567 0.0004 -0.2574 -0.2559      negative                     10                                 yes
   with type indicators      log_RH -1.1366 0.0046 -1.1461 -1.1312      negative                     10                                 yes
   with type indicators       Aging -0.0053 0.0000 -0.0054 -0.0053      negative                     10                                 yes
   with type indicators       log_J -0.1276 0.0008 -0.1292 -0.1269      negative                     10                                 yes
   with type indicators           K -0.0022 0.0000 -0.0022 -0.0022             -                      -                                 yes
without type indicators      log_CD  1.0190 0.0103  1.0062  1.0381      positive                     10                                 yes
without type indicators      log_AD  0.2844 0.0062  0.2711  0.2958      positive                     10                                 yes
without type indicators     log_SDD -0.2568 0.0006 -0.2578 -0.2558      negative                     10                                 yes
without type indicators      log_RH -1.1369 0.0051 -1.1471 -1.1291      negative                     10                                 yes
without type indicators       Aging -0.0053 0.0000 -0.0054 -0.0053      negative                     10                                 yes
without type indicators       log_J -0.1277 0.0007 -0.1292 -0.1268      negative                     10                                 yes
without type indicators           K -0.0022 0.0000 -0.0023 -0.0022             -                      -                                 yes

## Monotonicity audit (% of adjacent steps where predicted FOV rises)
      Model Variable  pct_steps_increasing  pct_steps_increasing_gt_0_1kV  pct_points_any_violation  max_increase_kV  data_pct_steps_increasing
   HyPhysML    Aging                12.187                         10.433                    36.562            3.887                      15.87
   HyPhysML        J                 2.639                          1.466                    10.555            1.479                       4.47
   HyPhysML       RH                 0.005                          0.005                     0.015            0.533                       2.04
   HyPhysML      SDD                 0.000                          0.000                     0.000           -3.251                       0.38
HyPhysML-MC    Aging                 0.000                          0.000                     0.000           -0.004                      15.87
HyPhysML-MC        J                 0.000                          0.000                     0.000           -0.059                       4.47
HyPhysML-MC       RH                 0.000                          0.000                     0.000           -3.258                       2.04
HyPhysML-MC      SDD                 0.000                          0.000                     0.000           -6.113                       0.38
    XGBoost    Aging                11.674                          9.927                    35.023            2.857                      15.87
    XGBoost        J                 2.815                          1.667                    11.259            2.005                       4.47
    XGBoost       RH                 0.000                          0.000                     0.000           -0.574                       2.04
    XGBoost      SDD                 0.000                          0.000                     0.000           -2.819                       0.38

## Noise (HyPhysML)
   Model           mode Feature  Noise_pct  R2_mean   R2_sd  dR2_mean  dR2_sd  RMSE_mean  pct_clipped  n
HyPhysML raw_propagated   Aging          5  0.98807 0.00078  -0.00090 0.00042    2.42918     12.57958 50
HyPhysML raw_propagated   Aging         10  0.98715 0.00081  -0.00183 0.00047    2.52226     12.63664 50
HyPhysML raw_propagated   Aging         15  0.98600 0.00083  -0.00298 0.00050    2.63273     12.80931 50
HyPhysML raw_propagated   Aging         20  0.98425 0.00094  -0.00473 0.00052    2.79188     12.72523 50
HyPhysML raw_propagated      RH          5  0.98622 0.00202  -0.00275 0.00192    2.60670      0.00000 50
HyPhysML raw_propagated      RH         10  0.98499 0.00216  -0.00399 0.00203    2.72104      0.00000 50
HyPhysML raw_propagated      RH         15  0.98132 0.00235  -0.00765 0.00233    3.03697      0.00000 50
HyPhysML raw_propagated      RH         20  0.97616 0.00193  -0.01282 0.00187    3.43447      0.00000 50
HyPhysML raw_propagated     SDD          5  0.98610 0.00075  -0.00287 0.00032    2.62260      0.00000 50
HyPhysML raw_propagated     SDD         10  0.98135 0.00077  -0.00762 0.00057    3.03841      0.00000 50
HyPhysML raw_propagated     SDD         15  0.97575 0.00119  -0.01322 0.00087    3.46423      0.00000 50
HyPhysML raw_propagated     SDD         20  0.96938 0.00160  -0.01960 0.00135    3.89350      0.01051 50
HyPhysML  single_column   Aging          5  0.98867 0.00074  -0.00030 0.00023    2.36740      0.00000 50
HyPhysML  single_column   Aging         10  0.98865 0.00077  -0.00033 0.00024    2.36992      0.00000 50
HyPhysML  single_column   Aging         15  0.98861 0.00078  -0.00036 0.00025    2.37361      0.00000 50
HyPhysML  single_column   Aging         20  0.98833 0.00079  -0.00065 0.00026    2.40281      0.00000 50
HyPhysML  single_column      RH          5  0.98810 0.00091  -0.00087 0.00064    2.42586      0.00000 50
HyPhysML  single_column      RH         10  0.98791 0.00094  -0.00106 0.00067    2.44523      0.00000 50
HyPhysML  single_column      RH         15  0.98698 0.00095  -0.00200 0.00086    2.53849      0.00000 50
HyPhysML  single_column      RH         20  0.98565 0.00084  -0.00332 0.00084    2.66512      0.00000 50
HyPhysML  single_column     SDD          5  0.98897 0.00068  -0.00001 0.00002    2.33650      0.00000 50
HyPhysML  single_column     SDD         10  0.98897 0.00067  -0.00001 0.00003    2.33636      0.00000 50
HyPhysML  single_column     SDD         15  0.98897 0.00068  -0.00001 0.00002    2.33655      0.00000 50
HyPhysML  single_column     SDD         20  0.98896 0.00068  -0.00001 0.00002    2.33703      0.00000 50

## Learning curve
 frac  n_train  R2_train_mean  R2_train_sd  R2_test_mean  R2_test_sd     gap
  0.1    532.0        0.99518      0.00056       0.98293     0.00067 0.01225
  0.2   1064.0        0.99396      0.00048       0.98600     0.00026 0.00796
  0.3   1597.0        0.99279      0.00057       0.98666     0.00032 0.00613
  0.5   2662.0        0.99274      0.00027       0.98759     0.00074 0.00514
  0.7   3726.0        0.99245      0.00059       0.98816     0.00073 0.00429
  1.0   5324.0        0.99238      0.00044       0.98888     0.00069 0.00351

## Learning-curve diagnostic
                                                  protocol  fold     R2                                  types_in_validation                                  RH_levels_in_validation
unshuffled KFold on type-ordered rows (old learning curve)     1 0.9617                                        [np.int64(1)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
unshuffled KFold on type-ordered rows (old learning curve)     2 0.9328                           [np.int64(1), np.int64(2)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
unshuffled KFold on type-ordered rows (old learning curve)     3 0.9713                           [np.int64(2), np.int64(3)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
unshuffled KFold on type-ordered rows (old learning curve)     4 0.9741                           [np.int64(3), np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
unshuffled KFold on type-ordered rows (old learning curve)     5 0.9407                                        [np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
                                  shuffled KFold (seed 42)     1 0.9893 [np.int64(1), np.int64(2), np.int64(3), np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
                                  shuffled KFold (seed 42)     2 0.9887 [np.int64(1), np.int64(2), np.int64(3), np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
                                  shuffled KFold (seed 42)     3 0.9881 [np.int64(1), np.int64(2), np.int64(3), np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
                                  shuffled KFold (seed 42)     4 0.9888 [np.int64(1), np.int64(2), np.int64(3), np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]
                                  shuffled KFold (seed 42)     5 0.9888 [np.int64(1), np.int64(2), np.int64(3), np.int64(4)] [np.int64(65), np.int64(70), np.int64(80), np.int64(90)]

## Ablation
                    R2_mean   R2_std  RMSE_mean  MAPE_mean  Time_fit_mean  dR2_vs_full
Model                                                                                 
HyPhysML            0.98897  0.00072    2.33545    2.39463      1474.9879      0.00000
HyPhysML-MC         0.98752  0.00061    2.48526    2.57095       538.0455     -0.00145
XGBoost             0.98856  0.00067    2.37958    2.45255         6.2787     -0.00042
HyPhysML-noHPO      0.98850  0.00076    2.38558    2.41818       165.9021     -0.00048
HyPhysML-MeanStack  0.98833  0.00076    2.40251    2.44438       344.9133     -0.00064

## Per type (mean over seeds)
       n_test  R2_mean   R2_sd  RMSE_mean  MAE_mean  MAPE_mean
Type                                                          
Smp_1     333   0.9894  0.0012     2.2152    1.7042     2.2038
Smp_2     333   0.9860  0.0014     2.4056    1.8265     2.7180
Smp_3     333   0.9849  0.0017     2.4643    1.8942     2.5652
Smp_4     333   0.9911  0.0011     2.2391    1.7407     2.0915

## Residuals
                                    Scope     n  mean_kV  sd_kV  Shapiro_W  Shapiro_p
                         seed 42 test set  1332   0.0663 2.3008     0.9898        0.0
pooled, 10 seeds (Shapiro on random 5000) 13320   0.0127 2.3362     0.9925        0.0

## Data-level physics
Factor  n_adjacent_steps  pct_steps_FOV_decreases  pct_cells_strictly_monotone  median_step_kV
   SDD              4992                    99.62                        98.86         -10.524
    RH              4992                    97.96                        93.99          -9.543
 Aging              4992                    84.13                        52.76          -6.926
     J              5120                    95.53                        82.89          -5.704
     K              5120                    65.74                         2.19          -1.412

         mean       std       min       max      by
1   -0.254100  0.070300 -0.487600 -0.134800  Sample
2   -0.286100  0.073800 -0.495700 -0.156800  Sample
3   -0.247600  0.063100 -0.449600 -0.127600  Sample
4   -0.236500  0.064200 -0.434600 -0.128400  Sample
65  -0.193700  0.034400 -0.308900 -0.127600      RH
70  -0.224900  0.040900 -0.354600 -0.134800      RH
80  -0.293800  0.059700 -0.495700 -0.173300      RH
90  -0.311900  0.063400 -0.487600 -0.166000      RH
0   -0.242700  0.060900 -0.446800 -0.128400   Aging
15  -0.245800  0.062900 -0.495700 -0.127600   Aging
30  -0.260300  0.073600 -0.493100 -0.136800   Aging
45  -0.275500  0.078100 -0.487600 -0.130700   Aging
ALL -0.256082  0.070387 -0.495678 -0.127596     all

## Extrapolation
                Factor Held_out_level                 Position       Model  n_test      R2    RMSE     MAE    MAPE     NSE
random 80/20 (seed 42)              -            interpolation    HyPhysML    1332  0.9885  2.3468  1.7905  2.4020  0.9885
random 80/20 (seed 42)              -            interpolation HyPhysML-MC    1332  0.9849  2.6951  2.0280  2.7386  0.9849
random 80/20 (seed 42)              -            interpolation     XGBoost    1332  0.9880  2.3984  1.8183  2.4370  0.9880
random 80/20 (seed 42)              -            interpolation     Obenaus    1332  0.8999  6.9359  5.4511  6.5836  0.8999
                   SDD           0.03 boundary (extrapolation)    HyPhysML    1664  0.8484  7.1166  6.2425  6.2884  0.8484
                   SDD           0.03 boundary (extrapolation) HyPhysML-MC    1664  0.6835 10.2813  9.9789 10.0481  0.6835
                   SDD           0.03 boundary (extrapolation)     XGBoost    1664  0.8454  7.1854  6.3562  6.3260  0.8454
                   SDD           0.03 boundary (extrapolation)     Obenaus    1664 -0.0903 19.0830 15.8638 14.8471 -0.0903
                   SDD           0.06                 interior    HyPhysML    1664  0.8425  7.2430  6.0614  6.7331  0.8425
                   SDD           0.06                 interior HyPhysML-MC    1664  0.6801 10.3218  9.9807 11.3741  0.6801
                   SDD           0.06                 interior     XGBoost    1664  0.8436  7.2179  6.0244  6.6404  0.8436
                   SDD           0.06                 interior     Obenaus    1664  0.8342  7.4316  6.0618  6.7803  0.8342
                   SDD            0.1                 interior    HyPhysML    1664  0.8088  8.1950  7.2271 10.0268  0.8088
                   SDD            0.1                 interior HyPhysML-MC    1664  0.4935 13.3374 13.0242 17.8402  0.4935
                   SDD            0.1                 interior     XGBoost    1664  0.8294  7.7417  6.7636  9.3605  0.8294
                   SDD            0.1                 interior     Obenaus    1664  0.8908  6.1944  4.7772  6.0902  0.8908
                   SDD           0.14 boundary (extrapolation)    HyPhysML    1664  0.8889  6.2497  5.0608  8.3538  0.8889
                   SDD           0.14 boundary (extrapolation) HyPhysML-MC    1664  0.7363  9.6305  8.9787 13.9448  0.7363
                   SDD           0.14 boundary (extrapolation)     XGBoost    1664  0.8917  6.1705  5.0039  8.2269  0.8917
                   SDD           0.14 boundary (extrapolation)     Obenaus    1664  0.8292  7.7513  6.3568 10.8598  0.8292
                    RH             65 boundary (extrapolation)    HyPhysML    1664  0.7123 10.5978  9.8977  9.7813  0.7123
                    RH             65 boundary (extrapolation) HyPhysML-MC    1664  0.6867 11.0585 10.2889 10.1422  0.6867
                    RH             65 boundary (extrapolation)     XGBoost    1664  0.7179 10.4933  9.7139  9.5895  0.7179
                    RH             65 boundary (extrapolation)     Obenaus    1664  0.7426 10.0248  8.0214  7.7752  0.7426
                    RH             70                 interior    HyPhysML    1664  0.7437  9.4710  8.5942  9.6819  0.7437
                    RH             70                 interior HyPhysML-MC    1664  0.6506 11.0568 10.4094 11.8958  0.6506
                    RH             70                 interior     XGBoost    1664  0.7723  8.9253  7.9951  9.1603  0.7723
                    RH             70                 interior     Obenaus    1664  0.8619  6.9511  5.2706  5.6753  0.8619
                    RH             80                 interior    HyPhysML    1664  0.8560  6.9902  5.8495  8.4996  0.8560
                    RH             80                 interior HyPhysML-MC    1664  0.3458 14.8998 14.3254 19.9922  0.3458
                    RH             80                 interior     XGBoost    1664  0.4846 13.2253 12.5108 17.4571  0.4846
                    RH             80                 interior     Obenaus    1664  0.8702  6.6370  5.2907  7.7482  0.8702
                    RH             90 boundary (extrapolation)    HyPhysML    1664  0.8995  5.7120  5.0076  7.4753  0.8995
                    RH             90 boundary (extrapolation) HyPhysML-MC    1664  0.8764  6.3334  5.7323  8.5773  0.8764
                    RH             90 boundary (extrapolation)     XGBoost    1664  0.8947  5.8453  5.1337  7.7114  0.8947
                    RH             90 boundary (extrapolation)     Obenaus    1664  0.7156  9.6083  8.0076 10.8311  0.7156
                 Aging              0 boundary (extrapolation)    HyPhysML    1664  0.8209  9.0847  7.4921  9.3917  0.8209
                 Aging              0 boundary (extrapolation) HyPhysML-MC    1664  0.9837  2.7439  2.1420  2.5072  0.9837
                 Aging              0 boundary (extrapolation)     XGBoost    1664  0.8549  8.1769  6.7013  8.3863  0.8549
                 Aging              0 boundary (extrapolation)     Obenaus    1664  0.5292 14.7309 12.2890 13.2489  0.5292
                 Aging             15                 interior    HyPhysML    1664  0.9374  5.3944  4.3856  4.7403  0.9374
                 Aging             15                 interior HyPhysML-MC    1664  0.9841  2.7161  2.1339  2.4841  0.9841
                 Aging             15                 interior     XGBoost    1664  0.9440  5.1016  4.1555  4.5904  0.9440
                 Aging             15                 interior     Obenaus    1664  0.8614  8.0233  6.5869  7.1999  0.8614
                 Aging             30                 interior    HyPhysML    1664  0.9783  3.0676  2.4478  3.1846  0.9783
                 Aging             30                 interior HyPhysML-MC    1664  0.8402  8.3216  7.8281  9.9378  0.8402
                 Aging             30                 interior     XGBoost    1664  0.9699  3.6117  2.8962  3.7549  0.9699
                 Aging             30                 interior     Obenaus    1664  0.8936  6.7896  5.5121  6.7407  0.8936
                 Aging             45 boundary (extrapolation)    HyPhysML    1664  0.7686  9.2830  8.6457 12.3214  0.7686
                 Aging             45 boundary (extrapolation) HyPhysML-MC    1664  0.6715 11.0601 10.5600 14.9152  0.6715
                 Aging             45 boundary (extrapolation)     XGBoost    1664  0.7553  9.5451  8.9505 12.7970  0.7553
                 Aging             45 boundary (extrapolation)     Obenaus    1664  0.7625  9.4041  7.9421 12.0790  0.7625
                Sample              1    unseen insulator type    HyPhysML    1664  0.9197  6.1421  5.4154  6.0437  0.9197
                Sample              1    unseen insulator type HyPhysML-MC    1664  0.8272  9.0099  8.3462  9.6110  0.8272
                Sample              1    unseen insulator type     XGBoost    1664  0.8303  8.9305  8.2493  9.3840  0.8303
                Sample              1    unseen insulator type     Obenaus    1664  0.8783  7.5607  6.2618  7.3128  0.8783
                Sample              2    unseen insulator type    HyPhysML    1664  0.8488  7.9127  7.2890 10.0323  0.8488
                Sample              2    unseen insulator type HyPhysML-MC    1664  0.8700  7.3358  6.7083  9.3255  0.8700
                Sample              2    unseen insulator type     XGBoost    1664  0.8756  7.1765  6.5356  9.1666  0.8756
                Sample              2    unseen insulator type     Obenaus    1664  0.6609 11.8503 10.0534 13.9118  0.6609
                Sample              3    unseen insulator type    HyPhysML    1664  0.9554  4.2455  3.4526  4.2865  0.9554
                Sample              3    unseen insulator type HyPhysML-MC    1664  0.9700  3.4815  2.6464  3.6606  0.9700
                Sample              3    unseen insulator type     XGBoost    1664  0.9633  3.8513  2.9785  3.9856  0.9633
                Sample              3    unseen insulator type     Obenaus    1664  0.8529  7.7078  5.8221  7.1919  0.8529
                Sample              4    unseen insulator type    HyPhysML    1664  0.8829  8.0926  6.3628  6.5736  0.8829
                Sample              4    unseen insulator type HyPhysML-MC    1664  0.8921  7.7691  6.0731  6.3283  0.8921
                Sample              4    unseen insulator type     XGBoost    1664  0.9147  6.9065  5.3914  5.6498  0.9147
                Sample              4    unseen insulator type     Obenaus    1664  0.7153 12.6206 10.1464 10.6032  0.7153

## HPO/test overlap of the OLD protocol
 Seed  n_train  n_test  test_in_old_HPO_subset  pct  test_in_new_HPO_data
   42     5324    1332                    1093 82.1                     0
    7     5324    1332                    1063 79.8                     0
   13     5324    1332                    1052 79.0                     0
   99     5324    1332                    1073 80.6                     0
 2024     5324    1332                    1072 80.5                     0
   17     5324    1332                    1082 81.2                     0
   88     5324    1332                    1053 79.1                     0
   55     5324    1332                    1057 79.4                     0
  101     5324    1332                    1074 80.6                     0
  314     5324    1332                    1069 80.3                     0

SHAP (full stack) methods: {'XGBoost': 'KernelExplainer', 'LightGBM': 'TreeExplainer', 'GBR': 'TreeExplainer', 'HistGBR': 'TreeExplainer', 'RF': 'TreeExplainer', 'Extra Trees': 'TreeExplainer', 'KNN': 'KernelExplainer'}; additivity error 0.1591 kV