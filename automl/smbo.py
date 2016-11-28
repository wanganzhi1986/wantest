#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from acq_func import  GaussianAcquisitionFunction, RandomForestAcquisitionFunction
from runhistory import RunHistory
import copy
from collections import Counter

class RunType(object):
    '''
       class to define numbers for status types.
       Makes life easier in select_runs
    '''
    SUCCESS = 1
    TIMEOUT = 2
    CENSORED = 3


class StatusType(object):

    """
        class to define numbers for status types
    """
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5


class AbstractSmboSolver(object):

    def __init__(self,evaluator,
                 instances,
                 run_algorithm,
                 incumbent=None,
                 runhistory=None,
                 time_bound=None,
                 max_iters=20,
                 verbose=True,
                 acq_fun_kind="ei",
                 impute_state=None,
                 success_state=None
                 ):
        self.evaluator = evaluator
        self.instances = instances
        self.run_algorithm = run_algorithm
        self.incumbent = incumbent
        self.runhistory = runhistory
        self.time_bound = time_bound
        self.max_iters = max_iters
        self.acq_fun_kind = acq_fun_kind
        self.verbose = verbose
        self.impute_state = impute_state
        self.success_state = success_state


    def run(self):
        iteration = 1
        start = time.time()

        while True:
            time_spend = time.time() - start
            if self.time_bound and time_spend > self.time_bound:
                break
            if iteration > self.max_iters:
                break
            X, y = self.transform_runhistory(self.runhistory)
            # 选择下一个参数
            challengers = self.choose_next(X, y)

            self.incumbent, inc_perf = self.intensify(
                challengers=challengers,
                incumbent=self.incumbent,
                run_history=self.runhistory,
                time_bound=max(0.01, time_spend))
            iteration += 1

        return self.incumbent


    # 选择下一个运行的参数
    def choose_next(self, X, y):
        raise NotImplementedError

    # 初始值
    def init_design(self):
        raise NotImplementedError

    # 选择最好的参数值
    def intensify(self, challengers, run_history, incumbent=None, time_bound=None):
        if incumbent is None:
            incumbent = challengers[0]
            # Line 1 + 2
            for chall_indx, challenger in enumerate(challengers):
                if challenger == incumbent:

                    continue
                # self.logger.debug("Intensify on %s", challenger)
                # if hasattr(challenger, 'origin'):
                #     self.logger.debug(
                #         "Configuration origin: %s", challenger.origin)
                inc_runs = run_history.get_runs_for_config(incumbent)

                # Line 3
                # First evaluate incumbent on a new instance
                if len(inc_runs) < self.maxR:
                    # Line 4
                    # find all instances that have the most runs on the inc
                    inc_inst = [s.instance for s in inc_runs]
                    inc_inst = list(Counter(inc_inst).items())
                    inc_inst.sort(key=lambda x: x[1], reverse=True)
                    max_runs = inc_inst[0][1]
                    inc_inst = set([x[0] for x in inc_inst if x[1] == max_runs])

                    available_insts = (self.instances - inc_inst)

                    # if all instances were used n times, we can pick an instances
                    # from the complete set again

                    # Line 6 (Line 5 is further down...)
                    if available_insts:
                        # Line 5 (here for easier code)
                        next_instance = self.rs.choice(list(available_insts))
                        # Line 7
                        status, cost, dur, res = self.tae_runner.start(config=incumbent,
                                                                       instance=next_instance,
                                                                       seed=next_seed,
                                                                       cutoff=self.cutoff,
                                                                       instance_specific=self.instance_specifics.get(
                                                                           next_instance, "0"))

                        num_run += 1
                    else:
                        self.logger.debug(
                            "No further instance-seed pairs for incumbent available.")

                # Line 8
                N = 1

                inc_inst_seeds = set(run_history.get_runs_for_config(incumbent))
                inc_perf = aggregate_func(incumbent, run_history, inc_inst_seeds)

                # Line 9
                while True:
                    chall_inst_seeds = set(map(lambda x: (
                        x.instance, x.seed), run_history.get_runs_for_config(challenger)))

                    # Line 10
                    # 最优的参数配置运行的实例数和随机选的参数配置实例之间的差别:最好的有的实例
                    # 而随机的没有这些实例,然后将这些实例分成两部分，一部分是分给随机的变量去运行，一部分仍然是
                    # 是作为一个差别的实例集
                    missing_runs = list(inc_inst_seeds - chall_inst_seeds)

                    # Line 11
                    # 将实例打散以便随机选择
                    self.rs.shuffle(missing_runs)
                    # 要运行的实例，如果N越大的话，那么随机的配置实际上会运行更多的最佳配置所运行的实例，随着N的不断
                    # 的增大，
                    to_run = missing_runs[:min(N, len(missing_runs))]
                    # Line 13 (Line 12 comes below...)
                    # 差别的实例
                    missing_runs = missing_runs[min(N, len(missing_runs)):]
                    # 从最佳的配置的实例中去除差别的实例，此时保留的是和随机配置相同的实例
                    inst_seed_pairs = list(inc_inst_seeds - set(missing_runs))

                    # 最佳最配在这些实例上运行，获得它的评价信息
                    inc_sum_cost = sum_cost(config=incumbent, instance_seed_pairs=inst_seed_pairs,
                                            run_history=run_history)

                    # Line 12
                    # 经过这些迭代后，随机的配置运行的实例个数要大于或等于inst_seed_pairs的个数
                    for instance, seed in to_run:
                        # Run challenger on all <config,seed> to run
                        if self.run_obj_time:
                            chall_inst_seeds = set(map(lambda x: (
                                x.instance, x.seed), run_history.get_runs_for_config(challenger)))
                            chal_sum_cost = sum_cost(config=challenger, instance_seed_pairs=chall_inst_seeds,
                                                     run_history=run_history)

                            # 将最佳配置的使用的时间和随机的配置使用的时间之间的差距动态的调整每个配置的截止时间，
                            # 如果
                            cutoff = min(self.cutoff,
                                         (inc_sum_cost - chal_sum_cost) *
                                         self.Adaptive_Capping_Slackfactor)

                            if cutoff < 0:  # no time left to validate challenger
                                self.logger.debug(
                                    "Stop challenger itensification due to adaptive capping.")
                                break

                        else:
                            cutoff = self.cutoff

                        # 使用此时的配置在单个实例上的运行结果
                        status, cost, dur, res = self.tae_runner.start(config=challenger,
                                                                       instance=instance,
                                                                       seed=seed,
                                                                       cutoff=cutoff,
                                                                       instance_specific=self.instance_specifics.get(
                                                                           instance, "0"))
                        num_run += 1

                    # we cannot use inst_seed_pairs here since we could have less runs
                    # for the challenger due to the adaptive capping
                    # 此时的len(chall_inst_seeds) = len(to_runs)+ len(上次chall_inst_seeds)
                    chall_inst_seeds = set(map(lambda x: (
                        x.instance, x.seed), run_history.get_runs_for_config(challenger)))

                    if len(chall_inst_seeds) < len(inst_seed_pairs):
                        # challenger got not enough runs because of exhausted config budget
                        self.logger.debug("No (or not enough) valid runs for challenger.")
                        break
                    else:
                        # aggregate_fun:将每个参数配置在相同的实例上的消耗的代价进行累加
                        chal_perf = aggregate_func(challenger, run_history, chall_inst_seeds)
                        inc_perf = aggregate_func(incumbent, run_history, chall_inst_seeds)
                        # Line 15
                        if chal_perf > inc_perf:
                            # Incumbent beats challenger
                            self.logger.debug("Incumbent (%.4f) is better than challenger (%.4f) on %d runs." % (
                                inc_perf, chal_perf, len(inst_seed_pairs)))
                            break

                    # Line 16
                    if len(chall_inst_seeds) == len(inc_inst_seeds):
                        # Challenger is as good as incumbent -> change incumbent

                        n_samples = len(inst_seed_pairs)
                        self.logger.info("Challenger (%.4f) is better than incumbent (%.4f) on %d runs." % (
                            chal_perf / n_samples, inc_perf / n_samples, n_samples))
                        self.logger.info(
                            "Changing incumbent to challenger: %s" % (challenger))
                        incumbent = challenger
                        inc_perf = chal_perf
                        self.stats.inc_changed += 1
                        self.traj_logger.add_entry(train_perf=inc_perf,
                                                   incumbent_id=self.stats.inc_changed,
                                                   incumbent=incumbent)
                        break
                    # Line 17
                    else:
                        # challenger is not worse, continue
                        N = 2 * N

                if chall_indx >= 1 and num_run > self.run_limit:
                    self.logger.debug(
                        "Maximum #runs for intensification reached")
                    break
                elif chall_indx >= 1 and time.time() - self.start_time - time_bound >= 0:
                    self.logger.debug("Timelimit for intensification reached ("
                                      "used: %f sec, available: %f sec)" % (
                                          time.time() - self.start_time, time_bound))
                    break

            # output estimated performance of incumbent
            inc_runs = run_history.get_runs_for_config(incumbent)
            inc_perf = aggregate_func(incumbent, run_history, inc_runs)
            self.logger.info("Updated estimated performance of incumbent on %d runs: %.4f" % (
                len(inc_runs), inc_perf))

            return incumbent, inc_perf

    # 将运行的历史数据进行转换
    def transform_runhistory(self, runhistory, impute_censored_data=False):
        assert isinstance(runhistory, RunHistory)

        # consider only successfully finished runs
        s_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.run_info),
                                        select=RunType.SUCCESS)
        # Store a list of instance IDs
        s_instance_id_list = [k.instance_id for k in s_run_list.keys()]

        X, Y = self._build_matrix(run_list=s_run_list, runhistory=runhistory,
                                  instances=s_instance_id_list)

        # Also get TIMEOUT runs
        t_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.run_info),
                                        select=RunType.TIMEOUT)
        t_instance_id_list = [k.instance_id for k in s_run_list.keys()]

        tX, tY = self._build_matrix(run_list=t_run_list, runhistory=runhistory,
                                    instances=t_instance_id_list)

        # if we don't have successful runs,
        # we have to return all timeout runs
        if not s_run_list:
            return tX, tY

        if impute_censored_data:
            # Get all censored runs
            c_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.run_info),
                                            select=RunType.CENSORED)
            if len(c_run_list) == 0:
                self.logger.debug("No censored data found, skip imputation")
            else:
                # Store a list of instance IDs
                c_instance_id_list = [k.instance_id for k in c_run_list.keys()]

                cen_X, cen_Y = self._build_matrix(run_list=c_run_list,
                                                  runhistory=runhistory,
                                                  instances=c_instance_id_list)

                # Also impute TIMEOUTS
                cen_X = np.vstack((cen_X, tX))
                cen_Y = np.concatenate((cen_Y, tY))

                imp_Y = self.imputor.impute(censored_X=cen_X, censored_y=cen_Y,
                                            uncensored_X=X, uncensored_y=Y)

                # Shuffle data to mix censored and imputed data
                X = np.vstack((X, cen_X))
                Y = np.concatenate((Y, imp_Y))
        else:
            # If we do not impute,we also return TIMEOUT data
            X = np.vstack((X, tX))
            Y = np.concatenate((Y, tY))

        return X, Y

    # 选择参数运行的信息
    def __select_runs(self, rh_data, select):
        '''
        select runs of a runhistory
        '''
        new_dict = dict()
        if self.impute_state is None:
            self.impute_state = [StatusType.TIMEOUT, ]

        if self.success_states is None:
            self.success_states = [StatusType.SUCCESS, ]

        if select == RunType.SUCCESS:
            for run in rh_data.keys():
                if rh_data[run].status in self.success_states:
                    new_dict[run] = rh_data[run]
        elif select == RunType.TIMEOUT:
            for run in rh_data.keys():
                if (rh_data[run].status == StatusType.TIMEOUT and
                        rh_data[run].time >= self.cutoff_time):
                    new_dict[run] = rh_data[run]
        elif select == RunType.CENSORED:
            for run in rh_data.keys():
                if rh_data[run].status in self.impute_state \
                        and rh_data[run].time < self.cutoff_time:
                    new_dict[run] = rh_data[run]
        else:
            err_msg = "select must be in (%s), but is %s" % \
                      (",".join(["%d" % t for t in
                                 [RunType.SUCCESS, RunType.TIMEOUT,
                                  RunType.CENSORED]]), select)
            self.logger.critical(err_msg)
            raise ValueError(err_msg)

        return new_dict


    def _build_matrix(self, run_list, runhistory, instances):
        raise NotImplementedError




class SMBO(object):

    def __init__(self,
                 evaluator,
                 run_algorithm,
                 instances,
                 init_design=None,
                 incumbent=None,
                 runhistory=None,
                 time_bound=None,
                 max_iters=20,
                 verbose=True,
                 acq_fun_kind="ei",
                 acq_model_kind="gp",
                 instance_features=None
                 ):
        # 运行的目标程序
        self.run_algorithm = run_algorithm
        # 进行参数选择的模型
        # 运行的实例
        self.instances = instances
        # 最好的参数值
        self.incumbent = incumbent
        # 程序运行的历史信息
        self.runhistory = runhistory
        #程序的评价器
        self.evaluator = evaluator
        self.init_design = init_design
        self.max_iters = max_iters
        self.time_bound = time_bound
        self.verbose = True
        self.acq_func_kind = acq_fun_kind
        self.acq_model_kind = acq_model_kind
        self.feature_instaces  = instance_features

        if self.init_design is None:
            self.init_design = self.get_init_design()

        self.model = None
        self.acquisition_function = None
        if self.max_iters is None and self.time_bound is None:
            raise ValueError("applaction must given limit")

        if self.acq_model_kind not in ["smac", "gp"]:
            raise ValueError("acq kind:%s is invalid"%self.acq_model_kind)


    def run(self):



    def _fit(self, X, y):
        # Main BO loop

        return self.incumbent


    # 获得选择模型的初始实例
    def get_init_design(self):

        if self.acq_model_kind == "gp":
            self.acquisition_function = GaussianAcquisitionFunction(model=self.model,acq_kind=self.acq_func_kind )
        if self.acq_model_kind == "smac":
            self.acquisition_function = RandomForestAcquisitionFunction(acq_kind=self.acq_func_kind)



    def choose_next(self, model, X, y):
        self.model = model.fit(X, y)

    # 确定最好的参数值
    def intensify(self, challengers, run_history, incumbent=None, time_bound=None):
        if incumbent is None:
            incumbent = challengers[0]

            # Line 1 + 2
            for chall_indx, challenger in enumerate(challengers):
                if challenger == incumbent:
                    self.logger.warning(
                        "Challenger was the same as the current incumbent; Skipping challenger")
                    continue
                self.logger.debug("Intensify on %s", challenger)
                if hasattr(challenger, 'origin'):
                    self.logger.debug(
                        "Configuration origin: %s", challenger.origin)
                inc_runs = run_history.get_runs_for_config(incumbent)

                # Line 3
                # First evaluate incumbent on a new instance
                if len(inc_runs) < self.maxR:
                    # Line 4
                    # find all instances that have the most runs on the inc
                    inc_inst = [s.instance for s in inc_runs]
                    inc_inst = list(Counter(inc_inst).items())
                    inc_inst.sort(key=lambda x: x[1], reverse=True)
                    max_runs = inc_inst[0][1]
                    inc_inst = set([x[0] for x in inc_inst if x[1] == max_runs])

                    available_insts = (self.instances - inc_inst)

                    # if all instances were used n times, we can pick an instances
                    # from the complete set again
                    if not self.deterministic and not available_insts:
                        available_insts = self.instances

                    # Line 6 (Line 5 is further down...)
                    if self.deterministic:
                        next_seed = 0
                    else:
                        next_seed = self.rs.randint(low=0, high=MAXINT,
                                                    size=1)[0]

                    if available_insts:
                        # Line 5 (here for easier code)
                        next_instance = self.rs.choice(list(available_insts))
                        # Line 7
                        status, cost, dur, res = self.tae_runner.start(config=incumbent,
                                                                       instance=next_instance,
                                                                       seed=next_seed,
                                                                       cutoff=self.cutoff,
                                                                       instance_specific=self.instance_specifics.get(
                                                                           next_instance, "0"))

                        num_run += 1
                    else:
                        self.logger.debug(
                            "No further instance-seed pairs for incumbent available.")

                # Line 8
                N = 1

                inc_inst_seeds = set(run_history.get_runs_for_config(incumbent))
                inc_perf = aggregate_func(incumbent, run_history, inc_inst_seeds)

                # Line 9
                while True:
                    chall_inst_seeds = set(map(lambda x: (
                        x.instance, x.seed), run_history.get_runs_for_config(challenger)))

                    # Line 10
                    # 最优的参数配置运行的实例数和随机选的参数配置实例之间的差别:最好的有的实例
                    # 而随机的没有这些实例,然后将这些实例分成两部分，一部分是分给随机的变量去运行，一部分仍然是
                    # 是作为一个差别的实例集
                    missing_runs = list(inc_inst_seeds - chall_inst_seeds)

                    # Line 11
                    # 将实例打散以便随机选择
                    self.rs.shuffle(missing_runs)
                    # 要运行的实例，如果N越大的话，那么随机的配置实际上会运行更多的最佳配置所运行的实例，随着N的不断
                    # 的增大，
                    to_run = missing_runs[:min(N, len(missing_runs))]
                    # Line 13 (Line 12 comes below...)
                    # 差别的实例
                    missing_runs = missing_runs[min(N, len(missing_runs)):]
                    # 从最佳的配置的实例中去除差别的实例，此时保留的是和随机配置相同的实例
                    inst_seed_pairs = list(inc_inst_seeds - set(missing_runs))

                    # 最佳最配在这些实例上运行，获得它的评价信息
                    inc_sum_cost = sum_cost(config=incumbent, instance_seed_pairs=inst_seed_pairs,
                                            run_history=run_history)

                    # Line 12
                    # 经过这些迭代后，随机的配置运行的实例个数要大于或等于inst_seed_pairs的个数
                    for instance, seed in to_run:
                        # Run challenger on all <config,seed> to run
                        if self.run_obj_time:
                            chall_inst_seeds = set(map(lambda x: (
                                x.instance, x.seed), run_history.get_runs_for_config(challenger)))
                            chal_sum_cost = sum_cost(config=challenger, instance_seed_pairs=chall_inst_seeds,
                                                     run_history=run_history)

                            # 将最佳配置的使用的时间和随机的配置使用的时间之间的差距动态的调整每个配置的截止时间，
                            # 如果
                            cutoff = min(self.cutoff,
                                         (inc_sum_cost - chal_sum_cost) *
                                         self.Adaptive_Capping_Slackfactor)

                            if cutoff < 0:  # no time left to validate challenger
                                self.logger.debug(
                                    "Stop challenger itensification due to adaptive capping.")
                                break

                        else:
                            cutoff = self.cutoff

                        # 使用此时的配置在单个实例上的运行结果
                        status, cost, dur, res = self.tae_runner.start(config=challenger,
                                                                       instance=instance,
                                                                       seed=seed,
                                                                       cutoff=cutoff,
                                                                       instance_specific=self.instance_specifics.get(
                                                                           instance, "0"))
                        num_run += 1

                    # we cannot use inst_seed_pairs here since we could have less runs
                    # for the challenger due to the adaptive capping
                    # 此时的len(chall_inst_seeds) = len(to_runs)+ len(上次chall_inst_seeds)
                    chall_inst_seeds = set(map(lambda x: (
                        x.instance, x.seed), run_history.get_runs_for_config(challenger)))

                    if len(chall_inst_seeds) < len(inst_seed_pairs):
                        # challenger got not enough runs because of exhausted config budget
                        self.logger.debug("No (or not enough) valid runs for challenger.")
                        break
                    else:
                        # aggregate_fun:将每个参数配置在相同的实例上的消耗的代价进行累加
                        chal_perf = aggregate_func(challenger, run_history, chall_inst_seeds)
                        inc_perf = aggregate_func(incumbent, run_history, chall_inst_seeds)
                        # Line 15
                        if chal_perf > inc_perf:
                            # Incumbent beats challenger
                            self.logger.debug("Incumbent (%.4f) is better than challenger (%.4f) on %d runs." % (
                                inc_perf, chal_perf, len(inst_seed_pairs)))
                            break

                    # Line 16
                    if len(chall_inst_seeds) == len(inc_inst_seeds):
                        # Challenger is as good as incumbent -> change incumbent

                        n_samples = len(inst_seed_pairs)
                        self.logger.info("Challenger (%.4f) is better than incumbent (%.4f) on %d runs." % (
                            chal_perf / n_samples, inc_perf / n_samples, n_samples))
                        self.logger.info(
                            "Changing incumbent to challenger: %s" % (challenger))
                        incumbent = challenger
                        inc_perf = chal_perf
                        self.stats.inc_changed += 1
                        self.traj_logger.add_entry(train_perf=inc_perf,
                                                   incumbent_id=self.stats.inc_changed,
                                                   incumbent=incumbent)
                        break
                    # Line 17
                    else:
                        # challenger is not worse, continue
                        N = 2 * N

                if chall_indx >= 1 and num_run > self.run_limit:
                    self.logger.debug(
                        "Maximum #runs for intensification reached")
                    break
                elif chall_indx >= 1 and time.time() - self.start_time - time_bound >= 0:
                    self.logger.debug("Timelimit for intensification reached ("
                                      "used: %f sec, available: %f sec)" % (
                                          time.time() - self.start_time, time_bound))
                    break

            # output estimated performance of incumbent
            inc_runs = run_history.get_runs_for_config(incumbent)
            inc_perf = aggregate_func(incumbent, run_history, inc_runs)
            self.logger.info("Updated estimated performance of incumbent on %d runs: %.4f" % (
                len(inc_runs), inc_perf))

            return incumbent, inc_perf


