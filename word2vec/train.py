#!/usr/bin/env python
import snowflake.connector
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

seed = 999

#def getdata(sqlquery):

   #return None


def msgvec(sentence, wvmodel): # sentence is a list of strings.
   if len(sentence) == 0:
      return np.zeros(400)

   sentvec = np.zeros(400) #same dimension as word2vec vector

   for eachword in sentence:
      vec = wvmodel[eachword]
      sentvec += vec
   return sentvec / len(sentence)


def learningcurve_depth(max_depth, X_train, y_train, X_test, y_test):
   depth_list = list(range(1, max_depth+1))
   acc_list_train = []
   acc_list_test = []
   for i in range(1, max_depth+1):
      clf = RandomForestClassifier(max_depth=i, min_samples_leaf=10)
      clf.fit(X_train, y_train)

      ## trainset
      y_pred_train = clf.predict(X_train)
      trainaccscore = accuracy_score(y_pred_train, y_train)
      acc_list_train.append(trainaccscore)

      ## testset
      y_pred_test = clf.predict(X_test)
      testaccscore = accuracy_score(y_pred_test, y_test)
      acc_list_test.append(testaccscore)

   plt.xlabel('Depth')
   plt.ylabel('Acc')
   plt.plot(depth_list, acc_list_train, color='red', label='train set acc')
   plt.plot(depth_list, acc_list_test, color='green', label='test set acc')
   #plt.show()
   plt.savefig('demo.png')



ctx = snowflake.connector.connect(user='MLSERVICEDEV',
   password='2%}!.XRGhdgtkT8%B8]6mLiq',
   account='canvas.us-east-1'
)

cs = ctx.cursor()

try:
   print('fetching...')
   cs.execute("SELECT MESSAGE.MESSAGEBODY, CONVERSATIONMESSAGE.RESPONSESECONDS FROM APP.CANVAS_TENANTDB.MESSAGE INNER JOIN APP.CANVAS_TENANTDB.CONVERSATIONMESSAGE ON APP.CANVAS_TENANTDB.MESSAGE.MESSAGEID=APP.CANVAS_TENANTDB.CONVERSATIONMESSAGE.MESSAGEID WHERE APP.CANVAS_TENANTDB.MESSAGE.SENTDATETIME BETWEEN '01/01/2018' AND '12/31/2018'")
   print('done fetching')
   all = cs.fetchall()
   sentences = []
   resp = []
   for eachmsg in all:
      if eachmsg[0] == None:
         print('find none')
      sentences.append(eachmsg[0].split())
      resp.append(eachmsg[1])

   ## For testing 2019
   print('fetching...')
   cs.execute(
      "SELECT MESSAGE.MESSAGEBODY, CONVERSATIONMESSAGE.RESPONSESECONDS FROM APP.CANVAS_TENANTDB.MESSAGE INNER JOIN APP.CANVAS_TENANTDB.CONVERSATIONMESSAGE ON APP.CANVAS_TENANTDB.MESSAGE.MESSAGEID=APP.CANVAS_TENANTDB.CONVERSATIONMESSAGE.MESSAGEID WHERE APP.CANVAS_TENANTDB.MESSAGE.SENTDATETIME BETWEEN '01/01/2019' AND '03/01/2019'")
   print('done fetching')
   all_test = cs.fetchall()
   sentences_test = []
   resp_test = []
   for eachmsg in all_test:
      if eachmsg[0] == None:
         print('find none')
      sentences_test.append(eachmsg[0].split())
      resp_test.append(eachmsg[1])


   print(len(sentences))
   print(len(sentences_test))
   combined_sentences = sentences + sentences_test
   print(len(combined_sentences))

   # train model
   #print(sentences)
   model = Word2Vec(combined_sentences, size=400, window=5, min_count=1)
   # summarize vocabulary
   words = list(model.wv.vocab)
   print(len(words))
   # save model
   model.save('model.bin')
   # load model
   new_model = Word2Vec.load('model.bin')

   ## Check binary
   respbin = [1 if i else 0 for i in resp]
   respbin_test = [1 if i else 0 for i in resp_test]

   ## Check 30 min
   #respbin = [0 if not i or i > 1800 else 1 for i in resp]
   #respbin_test = [0 if not i or i > 1800 else 1 for i in resp_test]

   X = []
   for eachsent in sentences:
      X.append(msgvec(eachsent, new_model))

   # For testing 2019
   X_test = []
   for eachsent in sentences_test:
      X_test.append(msgvec(eachsent, new_model))

   X = np.asarray(X)
   y = np.asarray(respbin)

   # For testing 2019
   X_test = np.asarray(X_test)
   y_test = np.asarray(respbin_test)

   print('baseline acc= '+ str(np.mean(y)))
   print('baseline acc 2019 testing= ' + str(np.mean(y_test)))
   X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.333, stratify=y, random_state=seed)

   ## For testing time influence
   #num = int(len(X)*0.666667)
   #X_train = X[:num]
   #X_test = X[num:]
   #y_train = y[:num]
   #y_test = y[num:]


   clf = RandomForestClassifier(max_depth=10, min_samples_leaf=10)

   #clf.fit(X_train, y_train)
   #y_pred = clf.predict(X_test)
   #print(classification_report(y_test, y_pred))

   ## Fro testing 2019
   clf.fit(X, y)
   y_pred_test = clf.predict(X_test)
   print(classification_report(y_test, y_pred_test))




   ## Learning Curve
   #learningcurve_depth(20, X_train, y_train,X_test, y_test)





finally:
   cs.close()

ctx.close()