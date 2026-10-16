# Fraud_Detection_ANN


Property	                      Details
Problem type	              Binary classification
Dataset size	              7,483,766 rows
Number of original columns	24, including the target
Target column	              is_fraud
Negative class	            False — legitimate transaction
Positive class	            True — fraudulent transaction
Model framework	            TensorFlow / Keras



Data Preprocessing: handle duplicates , missing values , convert all values to numeric , Standard Scaler for normalization , encoding
Dataset split : 70% train , 15 test , 15 validation

ANN arch : 

    Preprocessed transaction features
              |
    Dense layer — 128 neurons
      ReLU activation
              |
    Dense layer — 64 neurons
      ReLU activation
              |
    Dense layer — 32 neurons
      ReLU activation
              |
    Output — 1 neuron (Sigmoid probability)
            
            **Parameter        	Previously discussed setting	                What it means**
            Hidden                 layers	128 → 64 → 32             neurons	Candidate network architecture
            Hidden                   activation	ReLU	              Learns non-linear patterns
            Output                  activation	Sigmoid            	Produces one fraud score
            Loss               function	Binary cross-entropy	      Measures prediction error for binary classification
            Optimizer	                     Adam	                    Updates model weights
            Batch size	                    512	                     Number of records processed per training batch
            Maximum epochs	                 20	                     Upper limit on complete passes through the training data
            Early stopping      	Monitor validation PR-AUC	        Can stop training when validation performance stops improving
            Regularization	      Dropout and/or class weights	    Previously discussed; actual use needs verification


RESULT: 
        **  Metric	            Reported result**
          Accuracy	                0.94
          Fraud precision	          0.82
          Fraud recall	            0.87
          Fraud F1-score	          0.8468
          PR-AUC	                  0.9408
          ROC-AUC	                  0.9777
          
          The reported confusion matrix for that run was:        TP - 5712 ; TN - 1313 ; FP - 279 ; FN - 196
