from sklearn import model_selection
import xgboost as xgb
#import lightgbm as lgb
#from catboost import CatBoostRegressor

## build train and test data for modeling
X_train = train_dt1
y_train = np.log1p(train_dt1["Hammer_price"].values)
X_test = test_dt1

dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.001,
          'max_depth': 10, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}
    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 20000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb

# Training XGB
pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")

sub_xgb = pd.DataFrame()
sub_xgb["Hammer_price"] = pred_test_xgb
pred_test_xgb

submission = pd.read_csv("C:/Users/HOSUB/Desktop/auction_master/Auction_submission_en.csv")
submission["Hammer_price"]=pred_test_xgb
submission.to_csv('C:/Users/HOSUB/Desktop/auction_master/Auction_submission_181202_2.csv', index=False)