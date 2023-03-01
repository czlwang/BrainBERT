import json

featurename2label = {"onset_finetuning":"Onset",
                     "speech_finetuning":"Speech",
                     "pitch_finetuning":"Pitch",
                     "rms_finetuning":"Volume",
                     "brightness_finetuning": "brightness"}

subject2label = {"subject1": "subject-1",
                 "subject2": "subject-2",
                 "subject3": "subject-3",
                 "subject4": "subject-4",
                 "subject4": "subject-4",
                 "subject5": "subject-5",
                 "subject6": "subject-6",
                 "subject7": "subject-7"
                 }

path = "dataset_stats/all_results.json"
with open(path, "r") as f:
    all_results = json.load(f)

features = [f for f in all_results]
print(" & ".join(features) + "\\\\")

for feature in features:
    print(feature)
    p,n,total = 0,0,0
    for subject in all_results[feature]:
        feature_results = all_results[feature][subject]
        pi = feature_results["positive"]
        ni = feature_results["negative"]
        p += pi
        n += ni
        total += pi + ni
    print("positive", p, "negative", n, "total", total)

for subject in all_results["rms_finetuning"]:
    row_nums = [subject2label[subject]]
    for feature in features:
        feature_results = all_results[feature][subject]
        p = feature_results["positive"]
        n = feature_results["negative"]
        row_nums.append(f'{p:,}')
        row_nums.append(f'{n:,}')
    row_str = " & ".join(row_nums) + "\\\\"
    print(row_str)

