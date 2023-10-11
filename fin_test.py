import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')

day=0
id_list=[]
for i in range(len(train)):
    tmp=0
    date_list=train.date.tolist()
    if i<(len(train)-1) and date_list[i]!=date_list[i+1]: tmp=1
    id_list.append(day)
    day+=tmp

train.insert(1, "qonlang", id_list)
day=0
id_list=[]
for i in range(len(trade)):
    tmp=0
    date_list=trade.date.tolist()
    if i<(len(trade)-1) and date_list[i]!=date_list[i+1]: tmp=1
    id_list.append(day)
    day+=tmp

trade.insert(1, "qonlang", id_list)
trade = trade.set_index(trade.columns[1])
trade.index.names = ['']

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, 
    environment = e_trade_gym)

print(df_account_value_a2c)
print(df_actions_a2c)
df_account_value_a2c.to_csv('account_value')
df_actions_a2c

def process_df_for_mvo(df):
  return df.pivot(index="date", columns="tic", values="close")

def StockReturnsComputing(StockPrice, Rows, Columns): 
  import numpy as np 
  StockReturn = np.zeros([Rows-1, Columns]) 
  for j in range(Columns):        # j: Assets 
    for i in range(Rows-1):     # i: Daily Prices 
      StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100 
      
  return StockReturn

StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

TradeData.to_numpy()

#compute asset returns
arStockPrices = np.asarray(StockData)
[Rows, Cols]=arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

#compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis = 0)
covReturns = np.cov(arReturns, rowvar=False)
 
#set precision for printing results
np.set_printoptions(precision=3, suppress = True)

#display mean returns and variance-covariance matrix of returns
#print('Mean returns of assets in k-portfolio 1\n', meanReturns)
#print('Variance-Covariance matrix of returns\n', covReturns)

from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
#mvo_weights

LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
Initial_Portfolio

Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
#MVO_result

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'

df_dji = YahooDownloader(
    start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["dji"]
).fetch_data()

df_dji = df_dji[["date", "close"]]
fst_day = df_dji["close"][0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
)

result = pd.DataFrame(
    {
        "a2c": df_result_a2c["account_value"],
        "mvo": MVO_result["Mean Var"],
        #"dji": dji["close"],
    }
)

plt.rcParams["figure.figsize"] = (15,5)
plt.figure()
plt.plot(result)
plt.savefig('result.png')
result.plot()