# UPLOTS

Preprocess the raw dataset in the foulder   '/UPLOTS/Data/datasets/':

 - data_etth.ipynb
 - data energy.ipynb
 - pems0408.ipynb

run our work in UPLOTS:

```
nohup python -u main.py --gpu 4 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_etth evening_peak_etth morning_peak_energy  evening_peak_energy morning_peak_pems04  evening_peak_pems04 morning_peak_pems08 evening_peak_pems08 --sample 0 --train --epoch 1000 --batch 8 > mix_gpt2_train_etthmpep_energympep_pems04mpep_pems08mpep_1000_mask0.0.log 2>&1  && (nohup python -u main.py --gpu 4 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_etth --sample 0 --milestone 1000 > test_etthmp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 3 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_etth --sample 0 --milestone 1000 > test_etthep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 7 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_energy --sample 0 --milestone 1000 > test_energymp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 1 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_energy --sample 0 --milestone 1000 > test_energyep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 4 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_pems04 --sample 0 --milestone 1000 > test_pems04mp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 3 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_pems04 --sample 0 --milestone 1000 > test_pems04ep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 7 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file morning_peak_pems08 --sample 0 --milestone 1000 > test_pems08mp_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 & nohup python -u main.py --gpu 1 --name etthmpep_energympep_pems04mpep_pems08mpep --config_file evening_peak_pems08 --sample 0 --milestone 1000 > test_pems08ep_gpt2mix_2layer_mile1000_mask0.0.log 2>&1 &) &
```


run baselines in the folder '/UPLOTS/baselines/'.


