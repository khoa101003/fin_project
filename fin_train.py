import pandas as pd
import numpy as np
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

df_raw = pd.read_csv('df_raw.csv')
train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')


check_and_make_directories([TRAINED_MODEL_DIR])

#train = pd.read_csv('train_data.csv')
#train = pd.DataFrame(train)

# If you are not using the data generated from part 1 of this tutorial, make sure 
# it has the columns and index in the form that could be make into the environment. 
# Then you can comment and skip the following two lines.
#print(train)
day=0
id_list=[]
for i in range(len(train)):
    tmp=0
    date_list=train.date.tolist()
    if i<(len(train)-1) and date_list[i]!=date_list[i+1]: tmp=1
    id_list.append(day)
    day+=tmp

train.insert(1, "qonlang", id_list)
print(len(id_list))
train = train.set_index(train.columns[1])
train.index.names = ['']


print(train)


stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
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


print(train.close.values)
print(train.date.unique())

e_train_gym = StockTradingEnv(df = train, **env_kwargs)


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

agent = DRLAgent(env = env_train)

agent = DRLAgent(env = env_train)
model_a2c = agent.get_model("a2c")


tmp_path = RESULTS_DIR + '/a2c'
new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model_a2c.set_logger(new_logger_a2c)

trained_a2c = agent.train_model(model=model_a2c, 
                             tb_log_name='a2c',
                             total_timesteps=50000)

trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")