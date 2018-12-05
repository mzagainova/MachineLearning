import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_training_data_binary(num):
  if num == 1:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, -1]

  elif num == 2:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [0, i-5, 1]
      data[i+5] = [0, i+1, -1]

  elif num == 3:
    data = np.zeros((10,3))
    data[0] = [3, 2, 1]
    data[1] = [6, 2, 1]
    data[2] = [3, 6, 1]
    data[3] = [4, 4, 1]
    data[4] = [5, 4, 1]
    data[5] = [-1, -2, -1]
    data[6] = [-2, -4, -1]
    data[7] = [-3, -3, -1]
    data[8] = [-4, -2, -1]
    data[9] = [-4, -4, -1]
  elif num == 4:
    data = np.zeros((10,3))
    data[0] = [-1, 1, 1]
    data[1] = [-2, 2, 1]
    data[2] = [-3, 5, 1]
    data[3] = [-3, -1, 1]
    data[4] = [-2, 1, 1]
    data[5] = [3, -6, -1]
    data[6] = [0, -2, -1]
    data[7] = [-1, -7, -1]
    data[8] = [1, -10, -1]
    data[9] = [0, -8, -1]

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return data

def generate_training_data_multi(num):
  if num == 1:
    data = np.zeros((20,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, 2]
      data[i+10] = [0, i-5, 3]
      data[i+15] = [0, i+1, 4]
    Y = 4

  elif num == 2:
    data = np.zeros((15,3))
    data[0] = [-5, -5, 1]
    data[1] = [-3, -2, 1]
    data[2] = [-5, -3, 1]
    data[3] = [-5, -4, 1]
    data[4] = [-2, -9, 1]
    data[5] = [0, 6, 2]
    data[6] = [-1, 3, 2]
    data[7] = [-2, 1, 2]
    data[8] = [1, 7, 2]
    data[9] = [1, 5, 2]
    data[10] = [6, 3, 3]
    data[11] = [9, 2, 3]
    data[12] = [10, 4, 3]
    data[13] = [8, 1, 3]
    data[14] = [9, 0, 3]
    Y = 3

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return [data, Y]

def plot_training_data_binary(data):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

def plot_training_data_multi(data):
  colors = ['b', 'r', 'g', 'm']
  shapes = ['+', 'o', '*', '.']

  for item in data:
    plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

def svm_test_brute(W,b,x):
    d = ( W[0]*x[0]) + (W[1]*x[1]) + b
    if d > 0:
        return 1
    else:
        return -1

def svm_train_brute(training_data):
    # separate into pos/neg samples
    pos = []
    neg = []
    for i in data:
        if i[2] == 1:
            pos.append(i)
        elif i[2] == -1:
            neg.append(i)
    pos = np.array(pos)
    neg = np.array(neg)

    # find all pairs of support vectors
    sv_pairs = []
    for p in pos:
        for n in neg:
            w1 = p[0:2] - n[0:2]
            mid = (p[0:2] + n[0:2]) / 2

            W = np.array([w1[0],w1[1]])
            b = -(W[0] * mid[0] + W[1] * mid[1])

            posPred = [ svm_test_brute(W,b,i) for i in pos[:,0:2] ]
            negPred = [ svm_test_brute(W,b,i) for i in neg[:,0:2] ]

            if (np.unique(posPred)[0] != np.unique(negPred)[0] and len(np.unique(posPred)) == 1 and len(np.unique(negPred)) == 1):
                sv_pairs.append([W,b,p,n])

    sv_pairs = np.array(sv_pairs)
    # find best of these
    maxMargin2 = 0
    minDist2 = 0
    v2 = [[0,0],0,[0,0],[0,0]]
    if(len(sv_pairs) > 0):
        maxMargin2 = compute_margin(data, sv_pairs[0,0], sv_pairs[0,1])
        minDist2 = np.abs(distance_point_to_hyperplane(sv_pairs[0,2], sv_pairs[0,0], sv_pairs[0,1]))
        sv2 = sv_pairs[0]
        for i in sv_pairs:
            margin = compute_margin(data, i[0], i[1])
            dist = np.abs(distance_point_to_hyperplane(i[2], i[0], i[1]))
            if margin > maxMargin2 and dist < minDist2:
                maxMargin2 = margin
                minDist2 = dist
                sv2 = i

    # find all triplets of support vectors
    sv_triplets = get_triplets(pos,neg)
    sv_triplets.extend(get_triplets(neg,pos))
    sv_triplets = np.array(sv_triplets)

    # find best of these
    maxMargin3 = 0
    minDist3 = 0
    v3 = [[0,0],0,[0,0],[0,0],[0,0]]

    if len(sv_triplets) > 0:
        maxMargin3 = compute_margin(data, sv_triplets[0,0], sv_triplets[0,1])
        minDist3 = np.abs(distance_point_to_hyperplane(sv_triplets[0,2], sv_triplets[0,0], sv_triplets[0,1]))
        sv3 = sv_triplets[0]
        for i in sv_triplets:
            margin = compute_margin(data, i[0], i[1])
            dist = np.abs(distance_point_to_hyperplane(i[2], i[0], i[1]))
            # mind distance needed here?
            if margin > maxMargin3 and dist < minDist3:
                maxMargin3 = margin
                minDist3 = dist
                sv3 = i

    # find best from pairs and triplets
    if maxMargin2 >= maxMargin3:
        return (sv2[0], sv2[1], sv2[2:4])
    else:
        return (sv3[0], sv3[1], sv3[2:5])

def get_triplets(pos, neg):
    # find all triplets of support vectors
    sv_triplets = []
    for i in range(0, len(pos)-1):
        for j in range(i+1, len(pos)):
            for k in range(0, len(neg)):
                w1 = pos[i, 0:2] - pos[j, 0:2]
                w1_v = w1 / np.sqrt(w1.dot(w1))
                w2 = neg[k, 0:2] - pos[j,0:2]

                proj = w2.dot(w1_v)

                mid = ((proj * w1_v) + pos[j,0:2] + neg[k,0:2]) / 2
                b = -1 * (mid[1] * w1_v[0]) + (mid[0] * w1_v[1])
                W = [-w1_v[1], w1_v[0]]

                posPred = [ svm_test_brute(W,b,i) for i in pos[:,0:2] ]
                negPred = [ svm_test_brute(W,b,i) for i in neg[:,0:2] ]

                if (np.unique(negPred)[0] != np.unique(posPred)[0] and len(np.unique(posPred)) == 1) and (len(np.unique(negPred)) == 1 ):
                    sv_triplets.append([W,b,pos[i],pos[j],neg[k]])
    return sv_triplets

def distance_point_to_hyperplane(pt, W, b):
    sum = (pt[0] * W[0]) + (pt[1] * W[1]) + b
    num = np.sqrt((W[0] * W[0]) + (W[1] * W[1]))
    return sum / num

def compute_margin(data, w, b):
    min = np.abs(distance_point_to_hyperplane(data[0], w, b))
    for i in data:
        dist = np.abs(distance_point_to_hyperplane(i, w, b))
        if dist < min:
            min = dist
    return dist


data = generate_training_data_binary(1)
#plot_training_data_binary(data)
[w,b,S] = svm_train_brute(data)
