from verl import DataProto
import torch
import time
import numpy as np


def is_continuous_subsequence(sub, main):
    len_sub = len(sub)
    len_main = len(main)
    if len_sub > len_main:
        return False
    for i in range(len_main - len_sub + 1):
        if main[i:i + len_sub] == sub:
            return True
    return False

def get_subtrace_idx(sub, main, main_idx):
    len_main = len(main)
    len_sub = len(sub)
    for i in range(len_main - len_sub + 1):
        if main[i:i + len_sub] == sub:
            return main_idx[i:i + len_sub]
    return None

def find_max_subset_multisets(grouped_traces, grouped_trace_idxs, group_num_list):
    all_traces = []
    for set_idx, trace_list in enumerate(grouped_traces):
        for trace_idx, trace in enumerate(trace_list):
            all_traces.append({
                'set_name': group_num_list[set_idx],
                'trace_name': trace_idx,
                'trace': trace,
                'trace_idx': grouped_trace_idxs[set_idx][trace_idx],
            })

    valid_traces = []
    for trace_info in all_traces:
        for other_trace in all_traces:
            if trace_info['set_name'] == other_trace['set_name']:
                continue
            if is_continuous_subsequence(
                    sub=trace_info['trace'],
                    main=other_trace['trace']
            ):
                valid_traces.append({
                    'set_name': trace_info['set_name'],
                    'trace_name': trace_info['trace_name'],
                    'trace': trace_info['trace'],
                    'trace_idx': trace_info['trace_idx'],
                    'is_continuous_subset_of': {
                        'set_name': other_trace['set_name'],
                        'trace_name': other_trace['trace_name'],
                        'trace': other_trace['trace'],
                        'trace_idx': other_trace['trace_idx'],
                    }
                })
                break
    return valid_traces

def extract_low_entropy_traces(entropys, response_strs, trace_len=5, threshold=0.2):
    low_entropy_traces = []
    low_entropy_trace_idxs = []
    i = 0
    n = len(entropys)
    while i < n:
        if entropys[i] < threshold:
            current_trace = []
            current_trace_idx = []

            while i < n and entropys[i] < threshold:
                current_trace.append(response_strs[i])
                current_trace_idx.append(i)
                i += 1

            if len(current_trace) >= trace_len:
                low_entropy_traces.append(current_trace)
                low_entropy_trace_idxs.append(current_trace_idx)
        else:
            i += 1
    return low_entropy_traces, low_entropy_trace_idxs

def revise_adv(
        data: DataProto,
        trace_length=5,
) -> DataProto:
    start_time = time.time()
    response_tokenlevel_str_lists = data.non_tensor_batch['response_tokenlevel_str_list'].tolist()
    accs = data.non_tensor_batch['acc'].tolist()
    uids = list(set(data.non_tensor_batch['uid']))
    response_masks = data.batch['response_mask'].tolist()
    entropys = data.batch['entropys'].tolist()
    advantages = data.batch['advantages'].tolist()
    trace_len = trace_length5

    uid2metric_groups = {uids[i]: [] for i in range(len(uids))}
    for idx, uid in enumerate(data.non_tensor_batch['uid']):
        metric_uid = {}
        metric_uid['uid'] = uid
        metric_uid['global_id'] = idx
        metric_uid['response_mask'] = response_masks[idx]
        metric_uid['acc'] = accs[idx]
        metric_uid['response_tokenlevel_str_list'] = response_tokenlevel_str_lists[idx]
        metric_uid['entropys'] = entropys[idx]
        metric_uid['advantages'] = advantages[idx]
        uid2metric_groups[uid].append(metric_uid)

    for uid in uids:
        round_start_time = time.time()
        metric_uids = uid2metric_groups[uid]
        right_grouped_low_entropy_traces = []
        right_grouped_low_entropy_trace_idxs = []
        right_grouped_group_nums = []

        wrong_grouped_low_entropy_traces = []
        wrong_grouped_low_entropy_trace_idxs = []
        wrong_grouped_group_nums = []

        group_num2response_tokenlevel_dict = {}
        group_num2global_id = {}
        group_num2revise_list = {}
        group_num2response_len_dict = {}
        entropys_list = []
        right_num = 0
        wrong_num = 0
        for metric_uid in metric_uids:
            response_mask = metric_uid['response_mask']
            entropys = metric_uid['entropys']
            acc = metric_uid['acc']
            if acc:
                right_num += 1
            else:
                wrong_num += 1

        group_num = 0
        for metric_uid in metric_uids:
            global_id = metric_uid['global_id']
            response_mask = metric_uid['response_mask']
            max_response_len = len(response_mask)
            acc = metric_uid['acc']
            entropys = metric_uid['entropys']
            advantages = metric_uid['advantages']
            response_tokenlevel_str_list = metric_uid['response_tokenlevel_str_list']

            response_len = 0
            revise_list = []
            for mask in response_mask:
                if str(mask) != '1':
                    break
                response_len += 1
            entropys = entropys[:response_len]

            sorted_entropy_list = sorted(entropys)
            threshold_entropy = sorted_entropy_list[min(round(len(entropys) * 0.8), len(entropys) - 1)]
            for entropy in entropys:
                if entropy <= threshold_entropy:
                    if acc:
                        revise_list.append(1 / right_num)
                    else:
                        revise_list.append(1 / wrong_num)
                else:
                    revise_list.append(1)
            advantages = advantages[:response_len]
            response_tokenlevel_str_list = response_tokenlevel_str_list[:response_len]
            group_num2response_tokenlevel_dict[group_num] = response_tokenlevel_str_list
            group_num2global_id[group_num] = global_id
            group_num2response_len_dict[group_num] = response_len

            revise_list += [0 for x in range(max_response_len - response_len)]
            group_num2revise_list[group_num] = revise_list

            low_entropy_traces, low_entropy_trace_idxs = extract_low_entropy_traces(entropys,
                                                                                    response_tokenlevel_str_list,
                                                                                    trace_len=trace_len,
                                                                                    threshold=threshold_entropy)

            if acc:
                right_grouped_low_entropy_traces.append(low_entropy_traces)
                right_grouped_low_entropy_trace_idxs.append(low_entropy_trace_idxs)
                right_grouped_group_nums.append(group_num)
            else:
                wrong_grouped_low_entropy_traces.append(low_entropy_traces)
                wrong_grouped_low_entropy_trace_idxs.append(low_entropy_trace_idxs)
                wrong_grouped_group_nums.append(group_num)
            group_num += 1
        group_num_list = right_grouped_group_nums + wrong_grouped_group_nums
        all_subset_traces = find_max_subset_multisets(
            right_grouped_low_entropy_traces + wrong_grouped_low_entropy_traces,
            right_grouped_low_entropy_trace_idxs + wrong_grouped_low_entropy_trace_idxs, group_num_list)

        wrong_grouped_num_base = len(right_grouped_low_entropy_traces)

        trace2connection_dict = {}
        trace2group_idx_map = {}

        for item in all_subset_traces:
            subset_group_num = item['set_name']
            subset_trace = tuple(item['trace'])
            sub_trace_idx = item['trace_idx']

            if len(subset_trace) < trace_len:
                continue

            source_group_num = item['is_continuous_subset_of']['set_name']
            source_trace = tuple(item['is_continuous_subset_of']['trace'])
            source_trace_idx = item['is_continuous_subset_of']['trace_idx']
            is_add = False

            for mini_subset_trace in list(trace2connection_dict.keys()):
                connection = trace2connection_dict[mini_subset_trace]
                group_idx_map = trace2group_idx_map[mini_subset_trace]
                if is_continuous_subsequence(mini_subset_trace, subset_trace):
                    start = sub_trace_idx[0]
                    end = sub_trace_idx[-1] + 1

                    assert group_num2response_tokenlevel_dict[int(subset_group_num)][start:end] == list(subset_trace)
                    start = source_trace_idx[0]
                    end = source_trace_idx[-1] + 1
                    assert group_num2response_tokenlevel_dict[int(source_group_num)][start:end] == list(source_trace)

                    _sub_trace_idx = get_subtrace_idx(mini_subset_trace, subset_trace, sub_trace_idx)
                    assert len(_sub_trace_idx) == len(mini_subset_trace)
                    _source_trace_idx = get_subtrace_idx(mini_subset_trace, source_trace, source_trace_idx)
                    assert len(_source_trace_idx) == len(mini_subset_trace)

                    group_idx_map.update({subset_group_num: _sub_trace_idx})
                    group_idx_map.update({source_group_num: _source_trace_idx})

                    trace2group_idx_map[mini_subset_trace] = group_idx_map
                    connection.update([subset_group_num, source_group_num])
                    trace2connection_dict[mini_subset_trace] = connection
                    is_add = True
                    break
                elif is_continuous_subsequence(subset_trace, mini_subset_trace):
                    for group_num in group_idx_map:
                        mini_subset_trace_idx = group_idx_map[group_num]
                        mini_subset_trace_idx = get_subtrace_idx(subset_trace, mini_subset_trace, mini_subset_trace_idx)
                        assert len(mini_subset_trace_idx) == len(subset_trace)
                        group_idx_map.update({group_num: mini_subset_trace_idx})

                    _source_trace_idx = get_subtrace_idx(subset_trace, source_trace, source_trace_idx)
                    assert len(_source_trace_idx) == len(subset_trace)
                    group_idx_map.update({subset_group_num: sub_trace_idx})
                    group_idx_map.update({source_group_num: _source_trace_idx})

                    trace2group_idx_map.pop(mini_subset_trace)
                    trace2group_idx_map[subset_trace] = group_idx_map
                    connection.update([subset_group_num, source_group_num])
                    trace2connection_dict.pop(mini_subset_trace)
                    trace2connection_dict[subset_trace] = connection

                    is_add = True
                    break
            if not is_add:
                trace2connection_dict[subset_trace] = set([subset_group_num, source_group_num])
                group_idx_map = {}
                group_idx_map[subset_group_num] = sub_trace_idx
                _source_trace_idx = get_subtrace_idx(subset_trace, source_trace, source_trace_idx)
                group_idx_map[source_group_num] = _source_trace_idx
                trace2group_idx_map[subset_trace] = group_idx_map

        for k, v in trace2connection_dict.items():
            flag = 1  # 1 for right; 0 for both; -1 for wrong
            right_occur = False
            wrong_occur = False

            group_right_num = 0
            group_wrong_num = 0
            for group_num in v:
                global_id = group_num2global_id[group_num]
                acc = data.non_tensor_batch['acc'][global_id]
                if acc:
                    right_occur = True
                    group_right_num += 1
                else:
                    wrong_occur = True
                    group_wrong_num += 1
            if right_occur and wrong_occur:
                flag = 0
            elif wrong_occur:
                flag = -1

            for group_num in v:
                global_id = group_num2global_id[group_num]
                acc = data.non_tensor_batch['acc'][global_id]
                start = trace2group_idx_map[k][group_num][0]
                end = trace2group_idx_map[k][group_num][-1] + 1
                revise_list = group_num2revise_list[group_num]
                revise_arr = np.array(revise_list, dtype=np.float32)
                slice_arr = revise_arr[start:end].copy()

                if flag == 0:
                    new_val = 0.0
                else:
                    scale = (1.0 / right_num * group_right_num) if acc else (1.0 / wrong_num * group_wrong_num)
                    new_val = np.maximum(slice_arr, scale)
                revise_arr[start:end] = new_val
                group_num2revise_list[group_num] = revise_arr.tolist()

        for group_num, revise_list in group_num2revise_list.items():
            # update advantages
            response_len = group_num2response_len_dict[group_num]
            global_id = group_num2global_id[group_num]
            advantages_tensor = data.batch['advantages'][global_id].clone()
            if torch.sum(torch.abs(advantages_tensor)) < 1e-8:
                continue
            revise_tensor = torch.tensor(revise_list, dtype=advantages_tensor.dtype, device=advantages_tensor.device)

            advantages = data.batch.pop('advantages')
            advantages[global_id] = revise_tensor * advantages_tensor
            data.batch.update({'advantages': advantages})
    return data
