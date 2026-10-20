# EDA summary (auto-generated)
Rows: 303  |  Features: 13
Class counts: {0: 164, 1: 139}
Missing (raw): {'ca': 4, 'thal': 2}

## Descriptive statistics (numeric)
```
          age  trestbps    chol  thalach  oldpeak      ca
count  303.00    303.00  303.00   303.00   303.00  299.00
mean    54.44    131.69  246.69   149.61     1.04    0.67
std      9.04     17.60   51.78    22.88     1.16    0.94
min     29.00     94.00  126.00    71.00     0.00    0.00
25%     48.00    120.00  211.00   133.50     0.00    0.00
50%     56.00    130.00  241.00   153.00     0.80    0.00
75%     61.00    140.00  275.00   166.00     1.60    1.00
max     77.00    200.00  564.00   202.00     6.20    3.00
```

## Correlation with target (sorted by |r|)
```
thal        0.526
ca          0.460
exang       0.432
oldpeak     0.425
thalach    -0.417
cp          0.414
slope       0.339
sex         0.277
age         0.223
restecg     0.169
trestbps    0.151
chol        0.085
fbs         0.025
```