

#run this code with "celery -A kmeans_workers worker --loglevel=info" on the worker machines

#Then this machine will become a worker, and will be able to run the app task, 
#i.e. the kmeans_tasks function, whenever the broker requests it.

import celery
import numpy as np
from copy import deepcopy
import json

app = celery.Celery('kmeans_workers',
                        broker='amqp://myguest:myguestpwd@PROD-JOB-844fd7d2202ac4da.elb.us-east-2.amazonaws.com',
                        backend='amqp://myguest:myguestpwd@PROD-JOB-844fd7d2202ac4da.elb.us-east-2.amazonaws.com')

@app.task
def kmeans_tasks(task, **kwargs):
    json_dump=kwargs['json_dump']
    json_load = json.loads(json_dump)
    if task=='estep':
        C = np.asarray(json_load["C"])
        X = np.asarray(json_load["X"])
        results=estep(C,X)
        return results
    elif task =='mstep':
        X = np.asarray(json_load["X"])
        n_clusters = json_load["n_clusters"]
        n_features = json_load["n_features"]
        #n_clusters=kwargs['n_clusters']
        #n_features=kwargs['n_features']
        results=mstep(X,n_clusters,n_features)
        return results
    else:
        raise ValueError('undefined task')

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64): 
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def estep(C,X):
    print('*** we are in the E-Step ***')
    #print('C',C)
    #print("shape of X:", len(X))
    for i in range(len(X)):
        #print(X[i])
        distances = dist(X[i]["points"], C)
        cluster = np.argmin(distances)
        X[i]["label"] = cluster
        #print(X[i].label)
    print(" *** Labels updated. E-step done ***")    
    return json.dumps({'X':deepcopy(X)},cls=NumpyEncoder)    


def mstep(X,n_clusters,n_features):
    print('*** we are in the M-Step ***')
    C = [None]*np.int(n_clusters)
    for i in range(n_clusters):
        points = [X[j]["points"] for j in range(len(X)) if X[j]["label"] == i]
        C[i] = np.mean(points, axis=0) 
        if (np.any(np.isnan(C[i]))):
            C[i]=np.zeros(n_features)
    print(" *** Cluster centers updated. M-step done ***")        
    return json.dumps({'C':deepcopy(C)},cls=NumpyEncoder)

