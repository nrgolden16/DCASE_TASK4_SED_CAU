## DCASE task 4 
### Semi-Supervised Sound Event Detection and Time Localization research

### SOTA Model
```
        best student/teacher val_metrics: 1.468 / 1.478
   training took 186.76 mins
   test starts!
      test result is out!
      [student] psds1: 0.4072, psds2: 0.6196
                event_macro_f1: 0.497, event_micro_f1: 0.495,
                segment_macro_f1: 0.753, segment_micro_f1: 0.791, intersection_f1: 0.725
      [teacher] psds1: 0.4211, psds2: 0.6440
                event_macro_f1: 0.521, event_micro_f1: 0.516,
                segment_macro_f1: 0.773, segment_micro_f1: 0.806, intersection_f1: 0.741
date & time of end is : 2022-09-26 06:04:06
```

### Best Model (CRNN + BAM 계열 Convolution Attention Module)
```
        best student/teacher val_metrics: 1.417 / 1.431
   training took 309.52 mins
   test starts!
      test result is out!
      [student] psds1: 0.4241, psds2: 0.6318
                event_macro_f1: 0.521, event_micro_f1: 0.523, 
                segment_macro_f1: 0.766, segment_micro_f1: 0.804, intersection_f1: 0.739
      [teacher] psds1: 0.4422, psds2: 0.6492
                event_macro_f1: 0.533, event_micro_f1: 0.536, 
                segment_macro_f1: 0.773, segment_micro_f1: 0.808, intersection_f1: 0.744
date & time of end is : 2022-10-09 17:56:51
```
