
def blight_model():
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    X_train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    test_data = pd.read_csv('readonly/test.csv')
    address = pd.read_csv('readonly/addresses.csv')
    
    X_train = X_train[(X_train['compliance']==0) | (X_train['compliance']==1)]
    
    latlons = pd.read_csv('readonly/latlons.csv')
    address = address.set_index('address').join(latlons.set_index('address'), how='left')
    
    #join address and location to X_train, X_test
    X_train = X_train.set_index('ticket_id').join(address.set_index('ticket_id'))
    test_data = test_data.set_index('ticket_id').join(address.set_index('ticket_id'))
    
    #remove null values for hearing date rows
    X_train = X_train[~X_train['hearing_date'].isnull()]
    
    train_remove_list = [
            'balance_due',
            'collection_status',
            'compliance_detail',
            'payment_amount',
            'payment_date',
            'payment_status'
        ]
    
    #remove non existing features in Test Data
    X_train.drop(train_remove_list, axis=1, inplace=True)
    
    string_remove = ['violator_name', 'zip_code', 'country', 'city',
            'inspector_name', 'violation_street_number', 'violation_street_name',
            'violation_zip_code', 'violation_description',
            'mailing_address_str_number', 'mailing_address_str_name',
            'non_us_str_code', 'agency_name', 'state', 'disposition',
            'ticket_issued_date', 'hearing_date', 'grafitti_status', 'violation_code'
        ]
    
    X_train.drop(string_remove, axis=1, inplace=True)
    test_data.drop(string_remove, axis=1, inplace=True)

    X_train.lat.fillna(method='pad', inplace=True)
    X_train.lon.fillna(method='pad', inplace=True)
    test_data.lat.fillna(method='pad', inplace=True)
    test_data.lon.fillna(method='pad', inplace=True)

    y_train = X_train.compliance
    X_train.drop('compliance', axis=1)
    
    X_test = test_data
    #Scale features
    scalar = MinMaxScaler()
    X_train_s = scalar.fit_transform(X_train)
    X_test_s = scalar.transform(X_test)
    
    clf = MLPClassifier(hidden_layer_sizes = [100,10],
                      alpha=0.001,
                      random_state=0,
                      solver='lbfgs',
                      verbose=0)
    
    clf.fit(X_train_s, y_train)
    y_predProb = clf.predict_proba(X_test_s)[:,1]
    
    test = pd.read_csv('readonly/test.csv', encoding = "ISO-8859-1")
    test['compliance'] = y_predProb
    test.set_index('ticket_id', inplace=True)
        
    return test.compliance
