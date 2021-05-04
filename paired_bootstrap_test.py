b = 10000 # we draw b such samples: x1,...,x1000
n = 50 # size of the bootstrap sample
def BOOTSTRAP(n, b, test_y_preds): #returns p-value(x)
  d_X_1tob = [] 
  delta_X =  test_F1_scores[0]- test_F1_scores[1] # IN TOTAL, how much better does algorithm A do than B on x
  n_examples = len(test_y_preds[0])

  s = 0  
  print('delta_X:', delta_X)

  for i in range(0, b):
    # Draw a bootstrap sample x(i) of size n and compute F1 macro
    set_Xb = np.random.choice(range(len(test_y_preds[0])), n)
    pred_A = [test_y_preds[0][i] for i in set_Xb]
    pred_A = np.asarray(pred_A)
    pred_B = [test_y_preds[1][i] for i in set_Xb]
    pred_B = np.asarray(pred_B)
    true_labels = [np.asarray(y_test[i]) for i in set_Xb]

    #print(true_labels)
    cA = classification_report(true_labels, pred_A, target_names= class_labels , output_dict = True, zero_division = 1)
    cB = classification_report(true_labels, pred_B, target_names= class_labels , output_dict = True, zero_division = 1)
    F1_micro_A = cA['macro avg']['f1-score']
    F1_macro_B = cB['macro avg']['f1-score']
    delta_xi = F1_micro_A - F1_macro_B
    d_X_1tob.append(delta_xi)  #delta: how much better does algorithm A do than B on x(i)
  
  for dx in d_X_1tob:
    if dx > (2 * delta_X):
      s += 1    
  
  #onesided empirical p-value
  print('the number of times A does better on b = {} is {}'.format(b, s) )
  p_val = s/b      
  return d_X_1tob , p_val

#Testing bootstrap
d_X_1tob , p_val = BOOTSTRAP(n, b,test_y_preds)

print('p_value', p_val)