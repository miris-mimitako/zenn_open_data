import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
import random

class CalcError:
  def __init__(self) -> None:
    pass

  def main(self,true_y=0, pred_y=0):
    if not type(true_y).__module__ ==np.__name__ and not type(pred_y).__module__==np.__name__: # True is demo only
      true_y = np.linspace(0,100,100) # generate value @demo only
      pred_y = [x * random.gauss(1.,0.02) for x in true_y] # generate value @demo only
    res_mse = self.mse_(true_y, pred_y)
    res_rmse = self.rmse_(true_y, pred_y)
    res_msle = self.msle_(true_y, pred_y)
    res_rmsle = self.rmsle_(true_y, pred_y)
    res_mae = self.mae_(true_y, pred_y)
    res_r2 = self.r2_(true_y, pred_y)
    dic_result = {"MSE":res_mse, "RMSE":res_rmse, "MSLE":res_msle, "RMSLE":res_rmsle, "MAE":res_mae, "R2":res_r2}
    return dic_result

  def mse_(self,true_y, pred_y):
    # Calculate MSE
    mse = mean_squared_error(true_y, pred_y)
    print ("MSE: ", mse)
    return mse

  def rmse_(self,true_y, pred_y):
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    print ("RMSE: ", rmse)
    return rmse

  def msle_(self,true_y, pred_y):
    # Calculate MSLE
    msle = mean_squared_log_error(true_y,pred_y)
    print("MSLE: ", msle)
    return msle

  def rmsle_(self,true_y, pred_y):
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(true_y,pred_y))
    print("RMSLE: ", rmsle)
    return rmsle

  def mae_(self,true_y, pred_y):
    # Calculate MAE
    mae = mean_absolute_error(true_y, pred_y)
    print("MAE: ", mae)
    return mae

  def r2_(self,true_y, pred_y):
    # Calculate R2
    r2 = r2_score(true_y, pred_y)
    print("R2: ", r2)
    return r2

  def __del__(self):
    pass

if __name__=="__main__":
  CE = CalcError() # Generate instance
  res = CE.main() # Empty is demo only, You must define true_y and pred_y.
  print (res) # demo only
