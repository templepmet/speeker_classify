import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score,f1_score
 
y_true = [0,0,0,1,1,1,1,1,1,1]
y_pred = [1,0,0,1,1,1,0,1,1,0]
 
# 混同行列作成
print('混同行列\n{}'.format(confusion_matrix(y_true,y_pred)))
 
# 正解率
print('正解率: {0:.3f}'.format(accuracy_score(y_true, y_pred)))
 
# 適合率算出
print('適合率: {0:.3f}'.format(precision_score(y_true,y_pred)))
 
# 再現率算出
print('再現率: {0:.3f}'.format(recall_score(y_true,y_pred)))
 
# F1値算出
print('F1: {0:.3f}'.format(f1_score(y_true,y_pred)))