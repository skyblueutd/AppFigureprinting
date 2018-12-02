﻿This projects focus on App Fingerprinting to identify an app (e.g. a bank app) by listening to the encrypted TCP traffic.
The dataset was collected by running multiple Android apps on a Samsung Galaxy S device. Totally 30,000 apps were randomly selected from three different categories in Google Play Store. The categories include Finance, Communication, and Social.
Since the parameters of TCP/IP packages are quite different,  we use the statistical features (min,max,percentile,skew,kurtosis) of the packages as the training feature of machine learning process.
The test accuracy is about 80%. It is interesting that the idea of training machine learning algorithms with statistical features works well here.
