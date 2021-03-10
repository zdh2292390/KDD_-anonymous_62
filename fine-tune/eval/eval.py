#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import sklearn.metrics
import numpy as np


# In[ ]:


def evaluate(tasks, berts, domains, runs=10):
    for task in tasks:
        for bert in berts:
            for domain in domains:            
                scores=[]
                for run in range(1, runs+1):
                    DATA_DIR=os.path.join(task, domain)
                    OUTPUT_DIR=os.path.join("run", bert+"_"+task, domain, str(run) )
                    if os.path.exists(os.path.join(OUTPUT_DIR, "predictions.json") ):
                        if "rrc" in task:
                            ret = get_ipython().getoutput('python eval/evaluate-v1.1.py $DATA_DIR/test.json $OUTPUT_DIR/predictions.json')
                            score=json.loads(ret[0])
                            scores.append([score["exact_match"], score["f1"] ] )
                        elif "ae" in task:
                            ret = get_ipython().getoutput('python eval/evaluate_ae.py --pred_json $OUTPUT_DIR/predictions.json')
                            scores.append(float(ret[0])*100 )
                        elif "asc" in task:
                            with open(os.path.join(OUTPUT_DIR, "predictions.json") ) as f:
                                results=json.load(f)
                            y_true=results['label_ids']
                            y_pred=[np.argmax(logit) for logit in results['logits'] ]
                            p_macro, r_macro, f_macro, _=sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
                            f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
                            scores.append([100*sklearn.metrics.accuracy_score(y_true, y_pred), 100*f_macro ] )
                        else:
                            raise Exception("unknown task")
                scores=np.array(scores)
                m=scores.mean(axis=0)
                
                if len(scores.shape)>1:
                    for iz, score in enumerate(m):
                        print(task, ":", bert, domain, "metric", iz, round(score, 2) )
                else:
                    print(task, ":", bert, domain, round(m,2) )
                print


# In[ ]:

tasks=["ae"]
berts=["pt"]
domains=["laptop"]
runs = 10
evaluate(tasks, berts, domains, runs)

