# -*- coding:utf8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import os


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class PlotDrawer:

    def __init__(self, filename='agt_13_performance_records.json'):
        self.filename = filename

    def load_performance_records(self, filepath):
        records = []
        for subpath in os.listdir(filepath):
            filename = os.path.join(filepath, subpath)

            data = json.load(open(filename, 'r'))
            numbers = {'x': [], 'success_rate': [], 'ave_turns': [], 'ave_rewards': [],
                       'ave_emo': [], 'ave_hit': [], 'total_hit': []}
            keylist = [int(key) for key in data['success_rate'].keys()]
            keylist.sort()

            for key in keylist:
                if int(key) > -1:
                    numbers['x'].append(int(key))
                    numbers['success_rate'].append(data['success_rate'][str(key)])
                    numbers['ave_turns'].append(data['ave_turns'][str(key)])
                    numbers['ave_rewards'].append(data['ave_reward'][str(key)])

            records.append(numbers)

        return records

    def get_mean_scores(self, records):
        mean_scores = {'x': [], 'success_rate': [], 'ave_turns': [], 'ave_rewards': []}
        lower_scores = {'x': [], 'success_rate': [], 'ave_turns': [], 'ave_rewards': []}
        upper_scores = {'x': [], 'success_rate': [], 'ave_turns': [], 'ave_rewards': []}

        keylist = [int(key) for key in records[0]['x']]
        keylist.sort()

        for key in keylist:
            if int(key) > -1:
                mean_scores['x'].append(int(key) + 1)
                lower_scores['x'].append(int(key) + 1)
                upper_scores['x'].append(int(key) + 1)

                sr_items = [r['success_rate'][int(key)] for r in records]
                mean_scores['success_rate'].append(np.mean(sr_items))
                std = np.std(sr_items) / 1.5
                lower_scores['success_rate'].append(np.mean(sr_items) - std)
                upper_scores['success_rate'].append(np.mean(sr_items) + std)

                at_items = [r['ave_turns'][int(key)] for r in records]
                mean_scores['ave_turns'].append(np.mean(at_items))
                lower_scores['ave_turns'].append(np.min(at_items))
                upper_scores['ave_turns'].append(np.max(at_items))

                ar_items = [r['ave_rewards'][int(key)] for r in records]
                mean_scores['ave_rewards'].append(np.mean(ar_items))
                lower_scores['ave_rewards'].append(np.min(ar_items))
                upper_scores['ave_rewards'].append(np.max(ar_items))

        return mean_scores, lower_scores, upper_scores

    def moving_average(self, interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        k = np.convolve(interval, window, 'same')
        return k

    def draw_learning_curve(self, cmax, cmin, cmean, c, name, marker='s', linestyle='-', ax=None, dpoint=[199, 399, 497],
                            window_size=5):

        # plt.plot(cmax['x'],
        #          self.moving_average(cmean['success_rate'], window_size=window_size), label=name, color=c, lw=1.2,
        #          linestyle=linestyle, markevery=10)  # , markevery=10, marker=marker, markersize=4)
        #
        # plt.fill_between(cmean['x'],
        #                  self.moving_average(cmax['success_rate'], window_size=window_size),
        #                  self.moving_average(cmin['success_rate'], window_size=window_size), color=c, alpha=0.05)

        plt.plot(cmax['x'],
                 cmean['success_rate'], label=name, color=c, lw=1.2,
                 linestyle=linestyle, markevery=10)
        plt.fill_between(cmean['x'],
                         cmax['success_rate'],cmin['success_rate'], color=c, alpha=0.05)

        plt.show()

        res_string = name + ' & '
        for dp in dpoint:
            res_string += str(round(self.moving_average(cmean['success_rate'], window_size=window_size)[dp], 4)) + ' & '
            res_string += str(round(self.moving_average(cmean['ave_rewards'], window_size=window_size)[dp], 2)) + ' & '
            res_string += str(round(cmean['ave_turns'][dp], 2)) + ' & '

        res_string += '\\\\'

        return res_string


if __name__ == '__main__':
    drawer = PlotDrawer()

    fig = plt.figure(figsize=(8, 5))
    ax_succ = fig.add_subplot(111)

    plt.axvline(x=200, ls="--", c="gray", alpha=0.5)  # 添加垂直直线
    plt.axvline(x=400, ls="--", c="gray", alpha=0.5)  # 添加垂直直线

    window_size = 5

    # 绘制DQN的曲线
    # dat = drawer.load_performance_records('./deep_dialog/checkpoints/movie_esdp_DQN/run_4')
    # succ_m, succ_l, succ_h = drawer.get_mean_scores(dat)
    # messages = drawer.draw_learning_curve(succ_h, succ_l, succ_m, '#6699CC', 'DQN', marker='o', ax=ax_succ, linestyle='-', window_size=window_size)
    # print(messages)

    # 绘制ESDP的曲线
    dat = drawer.load_performance_records('./deep_dialog/checkpoints/taxi_esdp_07_20/DQN(E)')
    succ_m, succ_l, succ_h = drawer.get_mean_scores(dat)
    messages = drawer.draw_learning_curve(succ_h, succ_l, succ_m, '#99CC33', 'ESDP', marker='o', ax=ax_succ,
                                          linestyle='-', window_size=window_size)
    print(messages)

    ax_succ.set_xlabel('Training Epoch', fontsize=12)
    ax_succ.set_ylabel('Success Rate', fontsize=12)

    # plt.ylim(0, 0.9)
    # plt.xlim(0, 250)
    # plt.legend(loc=4, fontsize=12)
    #
    # plt.savefig('./res.png')
