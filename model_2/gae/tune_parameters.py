#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:yuyi
# datetime:2019/9/25 13:55
from gae.train import train_and_test
from gae.util import plot_all
from gae.util import set_configs
import pandas as pd
import time

# configs = set_configs(weight_decay=[0], weight_init=['truncated_normal'], if_BN=[True, False])
configs = set_configs(weight_decay=[0, 1e-4, 5e-4],
                      learning_rate=[0.0005, 0.001, 0.005],
                      if_BN=[True])
all_best_result = []
for config in configs:
    best_result = train_and_test(config)
    all_best_result.append(best_result)
    plot_all(result_path=config['result_path'], title_name=config['result_name'])
pd.DataFrame(all_best_result,
             columns=['result_name'] +
                     ['recall_in@%d' % (i+1) for i in range(20)] +
                     ['recall_out@%d' % (i+1) for i in range(20)]).\
    to_csv('./results/all_best_result.csv', index=False)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
