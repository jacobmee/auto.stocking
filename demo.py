#A股票行情数据获取演示   https://github.com/mpquant/Ashare
from  Ashare import *


df=get_price('000300.XSHG',frequency='60m',count=60)  #分钟线行情，只支持从当前时间往前推，可用'1m','5m','15m','30m','60m'
print('沪深300 60分钟线\n',df)

#df=get_price('000688.XSHG',frequency='60m',count=60)  #分钟线行情，只支持从当前时间往前推，可用'1m','5m','15m','30m','60m'
#print('科创50 60分钟线\n',df)

#股市行情数据获取和作图 -2
from  Ashare import *          #股票数据库    https://github.com/mpquant/Ashare
from  MyTT import *            #myTT麦语言工具函数指标库  https://github.com/mpquant/MyTT
    
# 证券代码兼容多种格式 通达信，同花顺，聚宽
# sh000001 (000001.XSHG)    sz399006 (399006.XSHE)   sh600519 ( 600519.XSHG ) 

df=get_price('000001.XSHG',frequency='1d',count=120)      #获取今天往前120天的日线实时行情
print('上证指数日线行情\n',df.tail(5))

#-------有数据了，下面开始正题 -------------
CLOSE=df.close.values;         OPEN=df.open.values           #基础数据定义，只要传入的是序列都可以 
HIGH=df.high.values;           LOW=df.low.values             #例如  CLOSE=list(df.close) 都是一样     

MA5=MA(CLOSE,5)                                #获取5日均线序列
MA10=MA(CLOSE,10)                              #获取10日均线序列
up,mid,lower=BOLL(CLOSE)                       #获取布林带指标数据

#-------------------------作图显示-----------------------------------------------------------------
import matplotlib.pyplot as plt ;  
from matplotlib.ticker 
import MultipleLocator

plt.figure(figsize=(15,8))  
plt.plot(CLOSE,label='SHZS');    plt.plot(up,label='UP');           #画图显示 
plt.plot(mid,label='MID');       plt.plot(lower,label='LOW');
plt.plot(MA10,label='MA10',linewidth=0.5,alpha=0.7);
plt.show()