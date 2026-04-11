# UPLOTS

Preprocess the raw dataset in the foulder   '/UPLOTS/Data/datasets/':

 - data_etth.ipynb
 - data energy.ipynb
 - pems0408.ipynb

run our work in UPLOTS:

```
nohup python -u main.py --gpu 4 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_etth evening_peak_etth morning_peak_energy \
 evening_peak_energy morning_peak_pems04  evening_peak_pems04 morning_peak_pems08 evening_peak_pems08 \
--sample 0 --train --epoch 1000 --batch 8 > mix_gpt2_train_etthmpep_energympep_pems04mpep_pems08mpep_1000_mask0.0.log 2>&1  && \
(nohup python -u main.py --gpu 4 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_etth \
--sample 0 --milestone 1000 > test_etthmp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 3 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file evening_peak_etth --sample 0 --milestone 1000 > \
test_etthep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 7 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file morning_peak_energy --sample 0 --milestone 1000 > \
test_energymp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 1 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file evening_peak_energy --sample 0 --milestone 1000 > \
test_energyep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 4 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file morning_peak_pems04 --sample 0 --milestone 1000 > \
test_pems04mp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 3 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file evening_peak_pems04 --sample 0 --milestone 1000 > \
test_pems04ep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 7 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file morning_peak_pems08 --sample 0 --milestone 1000 > \
test_pems08mp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & \
nohup python -u main.py --gpu 1 --name etthmpep_energympep_pems04mpep_pems08mpep \
--config_file evening_peak_pems08 --sample 0 --milestone 1000 > \
test_pems08ep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 &) &
```

We have provide some reproduced baselines, you can run baselines in the folder '/UPLOTS/baselines/'.


The full downstream forecasting task results are as follows.
### PhaseFormer MSE↓

|Ratio|ETTh_MP orig|ETTh_MP aug|ETTh_EP orig|ETTh_EP aug|Energy_MP orig|Energy_MP aug|Energy_EP orig|Energy_EP aug|
|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|1%|0.3439|**0.2804**|0.2667|**0.2147**|0.2963|**0.2700**|0.3388|**0.3060**|
|5%|0.3043|**0.2805**|0.2380|**0.2144**|0.2868|**0.2699**|0.3262|**0.3063**|
|10%|0.2995|**0.2793**|0.2315|**0.2141**|0.2832|**0.2698**|0.3202|**0.3049**|
|30%|0.2909|**0.2783**|0.2214|**0.2134**|0.2764|**0.2671**|0.3116|**0.3029**|
|90%|0.2836|**0.2745**|0.2174|**0.2116**|0.2661|**0.2638**|0.3030|**0.2983**|

### PhaseFormer MAE↓

|Ratio|ETTh_MP orig|ETTh_MP aug|ETTh_EP orig|ETTh_EP aug|Energy_MP orig|Energy_MP aug|Energy_EP orig|Energy_EP aug|
|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|1%|0.4066|**0.3682**|0.3419|**0.3032**|0.3306|**0.3010**|0.3671|**0.3305**|
|5%|0.3778|**0.3682**|0.3207|**0.3028**|0.3196|**0.3019**|0.3520|**0.3296**|
|10%|0.3745|**0.3680**|0.3145|**0.3026**|0.3173|**0.3031**|0.3480|**0.3294**|
|30%|0.3717|**0.3678**|0.3077|**0.3019**|0.3124|**0.2985**|0.3416|**0.3261**|
|90%|0.3695|**0.3651**|0.3053|**0.3005**|0.3029|**0.2978**|0.3326|**0.3235**|

### SparseTSF MSE↓

|Ratio|ETTh_MP orig|ETTh_MP aug|ETTh_EP orig|ETTh_EP aug|Energy_MP orig|Energy_MP aug|Energy_EP orig|Energy_EP aug|
|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|1%|0.4840|**0.2899**|0.3671|**0.2208**|0.3792|**0.2832**|0.4305|**0.3201**|
|5%|0.3861|**0.2899**|0.3024|**0.2208**|0.3045|**0.2837**|0.3467|**0.3202**|
|10%|0.3356|**0.2898**|0.2663|**0.2207**|0.2914|**0.2835**|0.3316|**0.3198**|
|30%|0.2969|**0.2898**|0.2350|**0.2205**|0.2831|**0.2831**|0.3208|**0.3197**|
|90%|0.2898|**0.2898**|0.2208|**0.2206**|0.2814|0.2822|0.3186|0.3190|

### SparseTSF MAE↓

|Ratio|ETTh_MP orig|ETTh_MP aug|ETTh_EP orig|ETTh_EP aug|Energy_MP orig|Energy_MP aug|Energy_EP orig|Energy_EP aug|
|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|1%|0.4855|**0.3705**|0.4055|**0.3087**|0.3775|**0.3157**|0.4144|**0.3468**|
|5%|0.4336|**0.3705**|0.3662|**0.3087**|0.3455|**0.3151**|0.3808|**0.3462**|
|10%|0.4021|**0.3705**|0.3414|**0.3086**|0.3350|**0.3152**|0.3704|**0.3466**|
|30%|0.3752|**0.3704**|0.3188|**0.3086**|0.3202|**0.3155**|0.3548|**0.3465**|
|90%|0.3703|0.3705|0.3084|0.3085|0.3187|**0.3166**|0.3509|**0.3483**|
