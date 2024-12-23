import math

import pandas as pd
import os
import json
from tqdm.auto import tqdm

class IREvaluation:
    def __init__(self, predict_path, label_path, sep=" ", to_save_path=None):
        self.predict_df = pd.read_csv(predict_path, sep=sep)
        self.label_df = pd.read_csv(label_path, sep=sep)

        self.evaluation_result, self.no_judge = self.parse()

        filepath, expansion = os.path.splitext(predict_path)
        save_path = filepath + "-evaluation_result.json"


        if len(self.no_judge) != 0:
            tmp = [f"query id: {self.no_judge} don't evaluate"]
            tmp.extend(self.evaluation_result)
            self.evaluation_result = tmp



        if to_save_path != None:
            save_path = to_save_path
        with open(save_path, 'w') as f:
            json.dump(self.evaluation_result, f, ensure_ascii=False, indent=4)


    def parse(self):
        evaluation_result = []

        no_judge = []

        groups = self.predict_df.groupby("query_ids")
        for query_id, predict in tqdm(groups):

            label = self.label_df[self.label_df["query_ids"] == query_id]

            query_evaluation_res = {}


            num_points = predict["query_ids"].size
            max_good_points = label[label["relevance"] != 0]["query_ids"].size
            if max_good_points == 0:
                no_judge.append(query_id)
                continue

            num_good_points = self.get_num_good_points(predict, label[label["relevance"] != 0])

            recall = num_good_points/max_good_points
            p_at = self.get_P_at(predict, label[label["relevance"] != 0])
            local_mrr = self.get_local_mrr(predict, label[label["relevance"] != 0])
            AP = self.get_AP(predict, label[label["relevance"] != 0])
            ndcg10, ndcg20, ndcg = self.get_NDCG(predict, label)

            query_evaluation_res["query ids"] = query_id
            query_evaluation_res["Num Points"] = num_points
            query_evaluation_res["Max Good Points"] = max_good_points
            query_evaluation_res["Num Good Points"] = num_good_points
            query_evaluation_res["local MRR"] = local_mrr
            query_evaluation_res["Recall"] = recall
            query_evaluation_res["AP"] = AP
            query_evaluation_res["NDCG@10"] = ndcg10
            query_evaluation_res["NDCG@20"] = ndcg20
            query_evaluation_res["NDCG"] = ndcg

            for i in range(1, 101): query_evaluation_res[f"P@{i}"] = p_at[i-1]
            evaluation_result.append(query_evaluation_res)


        evaluation_result = self.get_summary(evaluation_result)



        return evaluation_result, no_judge
    def get_num_good_points(self, predict, label):
        num_good_points = 0
        for target_doc_id in label["doc_ids"].values:
            num_good_points += predict[predict["doc_ids"] == target_doc_id]["doc_ids"].size

        return num_good_points

    def get_P_at(self, predict, label, n=100):
        predict_doc_ids = predict["doc_ids"].values
        p_at = []
        num_correct = 0
        for i in range(n):
            num = i + 1
            if i < len(predict_doc_ids):
                doc_id = predict_doc_ids[i]
                num_correct += label[label["doc_ids"] == doc_id]["doc_ids"].size
            p_at.append(num_correct / num)

        return p_at

    def get_local_mrr(self, predict, label):
        predict_doc_ids = predict["doc_ids"].values
        for i in range(len(predict_doc_ids)):
            doc_id = predict_doc_ids[i]
            if label[label["doc_ids"] == doc_id]["doc_ids"].size == 1:
                return 1/(i+1)
        return 0

    def get_AP(self, predict, label):
        predict_doc_ids = predict["doc_ids"].values
        sum_pat = 0
        num_pat = 0
        num_correct = 0
        for i in range(len(predict_doc_ids)):
            doc_id = predict_doc_ids[i]
            if label[label["doc_ids"] == doc_id]["doc_ids"].size == 1:
                num_correct += 1
                num = i + 1
                sum_pat += num_correct / num
                num_pat += 1

        return sum_pat / num_pat if num_pat != 0 else 0

    def get_summary(self, evaluation_result):
        summary = {}
        summary["query ids"] = "SUMMARY"
        summary["Num Points"] = 0
        summary["Max Good Points"] = 0
        summary["Num Good Points"] = 0
        summary["MRR"] = 0
        summary["Recall"] = 0
        summary["MAP"] = 0
        summary["NDCG@10"] = 0
        summary["NDCG@20"] = 0
        summary["NDCG"] = 0

        for i in range(1, 101): summary[f"P@{i}"] = 0
        nums = len(evaluation_result)
        for data in evaluation_result:
            summary["Num Points"] += data["Num Points"]
            summary["Max Good Points"] += data["Max Good Points"]
            summary["Num Good Points"] += data["Num Good Points"]
            summary["MRR"] += data["local MRR"]
            summary["Recall"] += data["Recall"]
            summary["MAP"] += data["AP"]
            summary["NDCG@10"] += data["NDCG@10"]
            summary["NDCG@20"] += data["NDCG@20"]
            summary["NDCG"] += data["NDCG"]
            for i in range(1, 101): summary[f"P@{i}"] += data[f"P@{i}"]


        summary["MRR"] /= nums
        summary["Recall"] /= nums
        summary["MAP"] /= nums
        summary["NDCG@10"] /= nums
        summary["NDCG@20"] /= nums
        summary["NDCG"] /= nums

        for i in range(1, 101): summary[f"P@{i}"] /= nums
        evaluation_result.append(summary)
        return evaluation_result

    def get_NDCG(self, predict, label):
        def get_DCG(predict=predict):
            predict_relevance = predict["relevance"].values
            sum_dcg = 0
            res = []
            for i in range(len(predict_relevance)):
                rel = predict_relevance[i]
                a = (2**rel - 1)
                b = math.log2(i+2)
                sum_dcg +=  a/ b

                res.append(sum_dcg)

            return res

        predict_doc_ids = predict["doc_ids"].values

        relevance = []
        for i in range(len(predict_doc_ids)):
            doc_id = predict_doc_ids[i]
            if label[label["doc_ids"] == doc_id]["doc_ids"].size == 1:
                rel = label[label["doc_ids"] == doc_id]["relevance"].values[0]
            else:
                rel = 0
            relevance.append(rel)

        predict['relevance'] = relevance

        dcgs = get_DCG(predict)
        _predict = predict.sort_values(by='relevance', ascending=False)
        idcgs = get_DCG(_predict)

        ndcg_at_10 = dcgs[9]/idcgs[9] if idcgs[9] != 0 else 0
        ndcg_at_20 = dcgs[19] / idcgs[19] if idcgs[19] != 0 else 0
        ndcg = dcgs[-1] / idcgs[-1] if idcgs[-1] != 0 else 0
        return ndcg_at_10, ndcg_at_20, ndcg





if __name__ == '__main__':
    res = IREvaluation("../tiaocan/termtiaocan/k1=1.2_b=0.75", "./truth", to_save_path=f"../ndcg.json")
